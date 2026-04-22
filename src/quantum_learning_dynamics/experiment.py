"""High-level orchestration API for quantum dynamics PAC-learning.

Abstracts the underlying quantum compilation, delegating hardware routing 
to specific quantum execution engines based on the mathematical dimension 
of the requested Hamiltonian.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from scipy.linalg import expm

from ._types import AlphaVector, InputX, LabelVector
from .features.engines import FeatureEngine, KernelEngine
from .hamiltonians.base import HamiltonianModel
from .learners.kernel_ridge import KernelRidgeLearner
from .learners.lasso import LassoLearner
from .observables.base import Observable
from .results import ExperimentResult


class Experiment:
    """End-to-end PAC-learning orchestrator for quantum dynamics.

    Dynamically routes classical requests into the appropriate quantum 
    circuits based on the physical dimension :math:`d` of the Hamiltonian.

    Parameters
    ----------
    model : HamiltonianModel
        The physical Hamiltonian defining the target system.
    observable : Observable
        The Hermitian measurement operator.
    method : {"lasso", "kernel"}
        The ML objective. "lasso" targets explicit feature extraction. 
        "kernel" targets pairwise Gram matrix evaluation.
    tau : float
        Total evolution time :math:`\\tau`.
    r_steps : int
        Number of Trotter steps used in the discretization.
    trotter_order : int, default=2
        The order of the Suzuki-Trotter decomposition (1 or 2).
    lasso_alpha : float, default=1e-3
        L1 regularization penalty applied to the Lasso learner.
    kernel_alpha : float, default=0.1
        L2 regularization penalty applied to the Kernel Ridge learner.
    execution_mode : {"emulator", "hardware"}, default="emulator"
        "emulator" relies on optimized exact statevector math with optional 
        binomial noise. "hardware" submits batched jobs to Qiskit primitives.
    shots : int, optional
        Number of measurement shots. Required for "hardware" mode. If None 
        in "emulator" mode, exact algebraic expectations are computed.
    sampler : Any, optional
        Qiskit V2 Sampler instance. If None and mode is "hardware", defaults 
        to Qiskit's `StatevectorSampler`.
    seed : int, default=42
        Global random seed for reproducible sampling, extraction, and learning.
    """

    def __init__(
        self,
        model: HamiltonianModel,
        observable: Observable,
        method: str,
        tau: float,
        r_steps: int,
        *,
        trotter_order: int = 2,
        lasso_alpha: float = 1e-3,
        kernel_alpha: float = 0.1,
        execution_mode: str = "emulator",
        shots: Optional[int] = None,
        sampler: Optional[Any] = None,
        seed: int = 42,
    ) -> None:
        if method not in ("lasso", "kernel"):
            raise ValueError(f"method must be 'lasso' or 'kernel', got {method!r}")
        if execution_mode not in ("emulator", "hardware"):
            raise ValueError(f"execution_mode must be 'emulator' or 'hardware', got {execution_mode!r}")
        if execution_mode == "hardware" and shots is None:
            raise ValueError("execution_mode='hardware' strictly requires an integer shot count.")
        if method == "kernel" and model.d == 1:
            raise ValueError("The 'kernel' method requires an inhomogeneous model (d > 1).")
        if observable.num_qubits() != model.num_qubits:
            raise ValueError(
                f"observable.num_qubits={observable.num_qubits()} does not match "
                f"model.num_qubits={model.num_qubits}"
            )

        if execution_mode == "hardware" and sampler is None:
            from qiskit.primitives import StatevectorSampler
            sampler = StatevectorSampler()

        self.model = model
        self.observable = observable
        self.method = method
        self.tau = float(tau)
        self.r_steps = int(r_steps)
        self.trotter_order = int(trotter_order)
        self.execution_mode = execution_mode
        self.shots = shots
        self.seed = int(seed)

        ss = np.random.SeedSequence(self.seed)
        sampling_seed, engine_seed, learner_seed = ss.spawn(3)
        self._sampling_rng = np.random.default_rng(sampling_seed)
        self._engine_rng = np.random.default_rng(engine_seed)

        if self.method == "lasso":
            self.learner = LassoLearner(
                alpha=lasso_alpha, 
                random_state=int(learner_seed.generate_state(1)[0])
            )
            self.engine = FeatureEngine(
                model=self.model,
                trotter_order=self.trotter_order,
                execution_mode=self.execution_mode,
                shots=self.shots,
                sampler=sampler,
                rng=self._engine_rng,
            )
        elif self.method == "kernel":
            self.learner = KernelRidgeLearner(alpha=kernel_alpha, precomputed_kernel=True)
            self.engine = KernelEngine(
                model=self.model,
                trotter_order=self.trotter_order,
                execution_mode=self.execution_mode,
                shots=self.shots,
                sampler=sampler,
                rng=self._engine_rng,
            )

    def compute_exact_labels(
        self, X_list: Sequence[InputX], alpha_star: AlphaVector, tau: float
    ) -> LabelVector:
        """Computes exact continuous dynamics via dense Hamiltonian matrices."""
        O = self.observable.to_sparse_pauli_op().to_matrix()
        labels = np.empty(len(X_list), dtype=np.float64)
        for i, x in enumerate(X_list):
            U = self.model.exact_unitary(x, alpha_star, tau)
            psi = U[:, 0]
            labels[i] = np.real(np.conj(psi) @ O @ psi)
        return labels

    def compute_trotter_labels(
        self, X_list: Sequence[InputX], alpha_star: AlphaVector, tau: float, r_steps: int
    ) -> LabelVector:
        """Computes discretized dynamics matching the circuit extraction order."""
        O = self.observable.to_sparse_pauli_op().to_matrix()
        dt = tau / r_steps
        labels = np.empty(len(X_list), dtype=np.float64)
        
        _h_cache = {}
        zero_alpha = np.zeros(self.model.d, dtype=np.float64)

        for i, x in enumerate(X_list):
            x_key = tuple(x)
            if x_key not in _h_cache:
                H_full = self.model.hamiltonian(x, alpha_star).to_matrix()
                H_fixed = self.model.hamiltonian(x, zero_alpha).to_matrix()
                _h_cache[x_key] = (H_full, H_fixed)
            else:
                H_full, H_fixed = _h_cache[x_key]

            H_upload = H_full - H_fixed
            U_upload = expm(-1j * dt * H_upload)
            
            if self.trotter_order == 1:
                U_fixed = expm(-1j * dt * H_fixed)
                U_step = U_upload @ U_fixed
            elif self.trotter_order == 2:
                U_fixed_half = expm(-1j * dt / 2.0 * H_fixed)
                U_step = U_fixed_half @ U_upload @ U_fixed_half
            else:
                raise ValueError(f"unsupported trotter_order {self.trotter_order}")

            U_trot = np.linalg.matrix_power(U_step, r_steps)
            psi = U_trot[:, 0]
            labels[i] = np.real(np.conj(psi) @ O @ psi)
            
        return labels

    def run(self, num_train: int, num_test: int) -> ExperimentResult:
        """Executes the complete PAC-learning pipeline."""
        alpha_star = self.model.sample_alpha(self._sampling_rng)
        X_train = [self.model.sample_x(self._sampling_rng) for _ in range(num_train)]
        X_test = [self.model.sample_x(self._sampling_rng) for _ in range(num_test)]

        y_train = self.compute_trotter_labels(X_train, alpha_star, self.tau, self.r_steps)
        y_true_exact = self.compute_exact_labels(X_test, alpha_star, self.tau)
        y_true_trotter = self.compute_trotter_labels(X_test, alpha_star, self.tau, self.r_steps)

        if self.method == "lasso":
            train_data = self.engine.extract(X_train, self.tau, self.r_steps, self.observable)
            test_data = self.engine.extract(X_test, self.tau, self.r_steps, self.observable)
            self.learner.fit(train_data, y_train)
            y_pred = self.learner.predict(test_data)
            
        elif self.method == "kernel":
            K_train = self.engine.compute_gram(X_train, None, self.tau, self.r_steps, self.observable)
            K_test = self.engine.compute_gram(X_test, X_train, self.tau, self.r_steps, self.observable)
            self.learner.fit(K_train, y_train)
            y_pred = self.learner.predict(K_test)

        mse_exact = float(np.mean((y_pred - y_true_exact) ** 2))
        mse_trotter = float(np.mean((y_pred - y_true_trotter) ** 2))

        return ExperimentResult(
            y_true_exact=y_true_exact,
            y_true_trotter=y_true_trotter,
            y_pred=y_pred,
            mse_exact=mse_exact,
            mse_trotter=mse_trotter,
            X_test=X_test,
            metadata={
                "method": self.method,
                "mode": self.execution_mode,
                "tau": self.tau,
                "r_steps": self.r_steps,
                "num_qubits": self.model.num_qubits,
                "d": self.model.d,
                "num_train": num_train,
                "num_test": num_test,
                "shots": self.shots,
                "seed": self.seed,
            },
        )