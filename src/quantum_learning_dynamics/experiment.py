"""High-level experiment API — the one-call interface to the paper's pipeline.

Typical usage (the "wow factor"):

    from quantum_learning_dynamics import Experiment
    from quantum_learning_dynamics.hamiltonians import InhomogeneousTFIM
    from quantum_learning_dynamics.observables import StaggeredMagnetization

    exp = Experiment(
        model       = InhomogeneousTFIM(num_qubits=4),
        observable  = StaggeredMagnetization(num_qubits=4, normalize=True),
        method      = "kernel-tensor",     # explicit — not inferred from d
        tau         = 0.5,
        r_steps     = 2,
        seed        = 42,
    )
    result = exp.run(num_train=200, num_test=50)
    result.y_true_exact, result.y_true_trotter, result.y_pred

Method names
------------
The method string IS the pipeline.  No inference from ``model.d``:

* ``"lasso-b_l"``     — shared-register circuit + Hadamard-test b_l
                        extractor + Lasso.  Requires ``model.d == 1``.
* ``"lasso-tensor"``  — separate-registers circuit + meshgrid tensor
                        extractor + Lasso on the flattened tensor.
                        Requires ``model.d > 1``.
* ``"kernel-tensor"`` — separate-registers circuit + meshgrid tensor
                        extractor + Kernel Ridge on Gram = B @ B.T.
                        Requires ``model.d > 1``.

If a chosen method is incompatible with the model's ``d``, ``__init__``
raises ``ValueError`` with a precise hint.

Label generation
----------------
Exact and Trotterised labels are generated inside :class:`Experiment`,
not in a separate dynamics module.  Exact labels come from
``model.exact_unitary`` + dense matmul; Trotterised labels are produced
through the same A(U) circuit the extractor uses, preserving the
A(U)-consistent convention that the legacy
``compute_au_labels_inhomogeneous`` established.

Seed propagation
----------------
The optional ``seed`` parameter drives the ONLY source of randomness in
the experiment.  It is passed to:

* :meth:`HamiltonianModel.sample_x` / :meth:`sample_alpha` via ``self.rng``
* the feature extractor (e.g. HadamardBLExtractor's epsilon_b noise)
* sklearn learners that accept ``random_state`` (LassoLearner)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.linalg import expm

from ._types import AlphaVector, FeatureMatrix, InputX, LabelVector
from .circuits.separate_registers import SeparateRegistersBuilder
from .circuits.shared_register import SharedRegisterBuilder
from .features.hadamard_b_l import HadamardBLExtractor
from .features.meshgrid_tensor import MeshgridTensorExtractor
from .hamiltonians.base import HamiltonianModel
from .learners.base import Learner
from .learners.kernel_ridge import KernelRidgeLearner
from .learners.lasso import LassoLearner
from .observables.base import Observable
from .results import ExperimentResult


class Experiment:
    """End-to-end PAC learning experiment for quantum dynamics.

    Routing map (method string -> (builder_cls, extractor_cls, learner_cls, allowed_d)):
        "hadamard_lasso"   : shared  + HadamardBL + Lasso       (d == 1)
        "meshgrid_lasso"   : separate + Meshgrid + Lasso        (d >  1)
        "meshgrid_kernel"  : separate + Meshgrid + KernelRidge  (d >  1)
    """

    _METHOD_REQUIRES_D: Dict[str, str] = {
        "hadamard_lasso": "=1",
        "meshgrid_lasso": ">1",
        "meshgrid_kernel": ">1",
    }

    _GLOBAL_H_CACHE = {}  # Local cache for Hamiltonian matrices during label computation, keyed by input graph configuration.

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
        epsilon_b: float = 0.0,
        seed: int = 0,
    ) -> None:
        self._validate_method_vs_d(method, model.d)
        self.model = model
        self.observable = observable
        self.method = method
        self.tau = float(tau)
        self.r_steps = int(r_steps)
        self.trotter_order = int(trotter_order)
        self.seed = int(seed)

        # Derive independent RNG streams for sampling / extractor / learner.
        ss = np.random.SeedSequence(self.seed)
        sampling_seed, extractor_seed, learner_seed = ss.spawn(3)
        self._sampling_rng = np.random.default_rng(sampling_seed)
        self._extractor_seed = extractor_seed
        self._learner_seed = int(learner_seed.generate_state(1)[0])

        # Route to concrete builder + extractor + learner.
        self.builder, self.extractor, self.learner = self._route(
            method=method,
            epsilon_b=epsilon_b,
            lasso_alpha=lasso_alpha,
            kernel_alpha=kernel_alpha,
        )

        if observable.num_qubits() != model.num_qubits:
            raise ValueError(
                f"observable.num_qubits={observable.num_qubits()} does not match "
                f"model.num_qubits={model.num_qubits}"
            )

    # ---- routing / validation -------------------------------------------

    @classmethod
    def _validate_method_vs_d(cls, method: str, d: int) -> None:
        if method not in cls._METHOD_REQUIRES_D:
            raise ValueError(f"unknown method {method!r}")
        req = cls._METHOD_REQUIRES_D[method]
        if req == "=1" and d != 1:
            raise ValueError(f"method={method!r} requires d == 1, got d={d}")
        if req == ">1" and d <= 1:
            raise ValueError(f"method={method!r} requires d > 1, got d={d}")

    def _route(self, method, epsilon_b, lasso_alpha, kernel_alpha):
        if method == "hadamard_lasso":
            builder = SharedRegisterBuilder(model=self.model, trotter_order=self.trotter_order)
            extractor = HadamardBLExtractor(
                builder=builder,
                epsilon_b=epsilon_b,
                rng=np.random.default_rng(self._extractor_seed),
            )
            learner = LassoLearner(alpha=lasso_alpha, random_state=self._learner_seed)
        elif method == "meshgrid_lasso":
            builder = SeparateRegistersBuilder(model=self.model, trotter_order=self.trotter_order)
            extractor = MeshgridTensorExtractor(
                builder=builder,
                rng=np.random.default_rng(self._extractor_seed),
            )
            learner = LassoLearner(alpha=lasso_alpha, random_state=self._learner_seed)
        elif method == "meshgrid_kernel":
            builder = SeparateRegistersBuilder(model=self.model, trotter_order=self.trotter_order)
            extractor = MeshgridTensorExtractor(
                builder=builder,
                rng=np.random.default_rng(self._extractor_seed),
            )
            learner = KernelRidgeLearner(alpha=kernel_alpha)
        else:
            raise ValueError(f"unknown method {method!r}")
        
        return builder, extractor, learner

    # ---- label computation ----------------------------------------------

    def compute_exact_labels(
        self,
        X_list: Sequence[InputX],
        alpha_star: AlphaVector,
        tau: float,
    ) -> LabelVector:
        """Ground-truth labels: y_i = <ψ₀| U†(x_i, α*) O U(x_i, α*) |ψ₀>.

        Dense-matrix path (fine for small n; switch to Statevector for n > ~12).
        """
        O = self.observable.to_sparse_pauli_op().to_matrix()
        labels = np.empty(len(X_list), dtype=np.float64)
        for i, x in enumerate(X_list):
            U = self.model.exact_unitary(x, alpha_star, tau)
            psi = U[:, 0]                       # |ψ₀⟩ = |0…0⟩ ⇒ first column
            labels[i] = np.real(np.conj(psi) @ O @ psi)
        return labels

    def compute_trotter_labels(
        self,
        X_list: Sequence[InputX],
        alpha_star: AlphaVector,
        tau: float,
        r_steps: int,
    ) -> LabelVector:
        """A(U)-consistent Trotter labels.

        Splits H(x, α) = H_fixed(x) + H_upload(α)  by subtracting H(x, 0):
            H_fixed  = H(x, α=0)        -- ZZ / mass / electric-field / etc.
            H_upload = H(x, α) - H_fixed -- Σ_k α_k P_k with physical angle α·τ/r

        Assembles U_step at the same Trotter order as the feature-extraction
        circuit (self.builder.trotter_order):
            order == 1 (shared  / first-order):
                U_step = exp(-i dt H_upload) · exp(-i dt H_fixed)
                        — matches RZZ(+2dt)·D·G·D per step in SharedRegisterBuilder.
            order == 2 (separate / symmetric Suzuki-2):
                U_step = exp(-i dt/2 H_fixed) · exp(-i dt H_upload)
                       · exp(-i dt/2 H_fixed)
                        — matches RZZ(+dt)·D·G·D·RZZ(+dt) in SeparateRegistersBuilder.
        U_trotter = U_step ** r_steps.
        """
        O = self.observable.to_sparse_pauli_op().to_matrix()
        dt = tau / r_steps
        order = getattr(self.builder, "trotter_order", 1)

        labels = np.empty(len(X_list), dtype=np.float64)
        
        # Local cache for Hamiltonian matrices to avoid redundant Qiskit overhead
        _h_cache = {}
        zero_alpha = np.zeros(self.model.d, dtype=np.float64)

        for i, x in enumerate(X_list):
            # Create a hashable key for the input graph configuration
            x_key = tuple(x) 
            
            # Only build the heavy Qiskit matrices if we haven't seen this graph yet
            if x_key not in _h_cache:
                H_full   = self.model.hamiltonian(x, alpha_star).to_matrix()
                H_fixed  = self.model.hamiltonian(x, zero_alpha).to_matrix()
                _h_cache[x_key] = (H_full, H_fixed)
            else:
                H_full, H_fixed = _h_cache[x_key]

            H_upload = H_full - H_fixed
            U_upload = expm(-1j * dt * H_upload)
            
            if order == 1:
                U_fixed = expm(-1j * dt * H_fixed)
                U_step  = U_upload @ U_fixed
            elif order == 2:
                U_fixed_half = expm(-1j * dt / 2.0 * H_fixed)
                U_step       = U_fixed_half @ U_upload @ U_fixed_half
            else:
                raise ValueError(f"unsupported trotter_order {order}")

            U_trot = np.linalg.matrix_power(U_step, r_steps)
            psi = U_trot[:, 0]
            labels[i] = np.real(np.conj(psi) @ O @ psi)
            
        return labels

    # ---- feature extraction --------------------------------------------

    def _extract_features(self, X_list: Sequence[InputX]) -> FeatureMatrix:
        """Stack per-sample feature vectors into (n_samples, n_features)."""
        rows: List[np.ndarray] = []
        for x in X_list:
            b = self.extractor.extract(x, self.tau, self.r_steps, self.observable)
            rows.append(np.ascontiguousarray(b, dtype=np.float64).ravel())
        return np.stack(rows, axis=0)

    # ---- full pipeline --------------------------------------------------

    def run(self, num_train: int, num_test: int) -> ExperimentResult:
        """Sample (α*, X_train, X_test), extract features, fit, predict, score."""
        # 1. Ground-truth α* and input samples.
        alpha_star = self.model.sample_alpha(self._sampling_rng)
        X_train = [self.model.sample_x(self._sampling_rng) for _ in range(num_train)]
        X_test  = [self.model.sample_x(self._sampling_rng) for _ in range(num_test)]

        # 2. Training labels — Trotter (matches what the extractor sees).
        y_train = self.compute_trotter_labels(X_train, alpha_star, self.tau, self.r_steps)

        # 3. Test ground truth — both exact and Trotter, so we can decompose
        #    generalization error vs. Trotter-discretization error.
        y_true_exact   = self.compute_exact_labels  (X_test, alpha_star, self.tau)
        y_true_trotter = self.compute_trotter_labels(X_test, alpha_star, self.tau, self.r_steps)

        # 4. Features.
        B_train = self._extract_features(X_train)
        B_test  = self._extract_features(X_test)

        # 5. Fit and predict.
        self.learner.fit(B_train, y_train)
        y_pred = self.learner.predict(B_test)

        # 6. MSEs.
        mse_exact   = float(np.mean((y_pred - y_true_exact)   ** 2))
        mse_trotter = float(np.mean((y_pred - y_true_trotter) ** 2))

        return ExperimentResult(
            y_true_exact   = y_true_exact,
            y_true_trotter = y_true_trotter,
            y_pred         = y_pred,
            mse_exact      = mse_exact,
            mse_trotter    = mse_trotter,
            X_test         = X_test,
            metadata       = {
                "method":      self.method,
                "tau":         self.tau,
                "r_steps":     self.r_steps,
                "num_qubits":  self.model.num_qubits,
                "d":           self.model.d,
                "num_train":   num_train,
                "num_test":    num_test,
                "alpha_star":  np.asarray(alpha_star, dtype=np.float64),
                "seed":        self.seed,
                "model_cls":   type(self.model).__name__,
                "observable_cls": type(self.observable).__name__,
                "builder_cls": type(self.builder).__name__,
                "trotter_order": getattr(self.builder, "trotter_order", 1),
            },
        )