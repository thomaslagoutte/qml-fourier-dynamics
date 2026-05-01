"""Execution engines for Fourier feature extraction and quantum overlap computation.

This module abstracts the physical execution environment from the quantum circuit
construction. It natively supports three execution pathways:
1. Exact noiseless statevector simulation (``shots=None``).
2. Hardware-realistic Binomial shot-noise emulation (``shots=N`` in emulator mode).
3. Batched physical hardware execution via Qiskit V2 Primitives (``hardware`` mode).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm
from qiskit.quantum_info import Statevector
from qiskit import transpile
from qiskit.circuit import Parameter

from .._types import FeatureMatrix, GramMatrix, InputX
from ..circuits.kernel_overlap import KernelOverlapBuilder
from ..circuits.separate_registers import SeparateRegistersBuilder
from ..circuits.shared_register import SharedRegisterBuilder
from ..hamiltonians.base import HamiltonianModel
from ..observables.base import Observable


class _EngineBase:
    """Abstract base class for quantum execution engines.

    Manages the underlying physical model, the execution environment configuration,
    and mathematical hardware noise injection.

    Parameters
    ----------
    model : HamiltonianModel
        The physical Hamiltonian driving the system.
    execution_mode : {"emulator", "hardware"}
        The simulation backend to target.
    shots : int, optional
        Number of measurement shots. If None, exact amplitudes are evaluated.
    sampler : Any, optional
        A Qiskit V2 Sampler primitive. Required only for "hardware" mode.
    rng : np.random.Generator
        Random number generator for reproducible shot-noise emulation.
    """

    def __init__(
        self,
        model: HamiltonianModel,
        execution_mode: str,
        shots: Optional[int],
        sampler: Optional[Any],
        rng: np.random.Generator,
    ) -> None:
        self.model = model
        self.execution_mode = execution_mode
        self.shots = shots
        self.sampler = sampler
        self.rng = rng

    def _apply_shot_noise(self, exact_value: float) -> float:
        """Injects statistically rigorous shot noise into an exact amplitude.

        Mimics the canonical Hadamard test estimator:
        :math:`\\mathrm{Re}(v) = 2 P(ht=0) - 1`.

        Parameters
        ----------
        exact_value : float
            The noiseless expectation value bounded in [-1.0, 1.0].

        Returns
        -------
        float
            The noisy estimator. Returns ``exact_value`` if shots are None.
        """
        if self.shots is None:
            return exact_value

        p_0 = np.clip((1.0 + exact_value) / 2.0, 0.0, 1.0)
        n_0 = self.rng.binomial(n=self.shots, p=p_0)
        return (2.0 * n_0 / self.shots) - 1.0


class FeatureEngine(_EngineBase):
    """Engine for explicit Fourier feature extraction.

    Evaluates the tensor :math:`b(x)` representing the Fourier coefficients
    of the observable dynamics. Dynamically routes requests to Shared or Separate
    register circuit builders based on the Hamiltonian's dimension ``d``.
    """

    def __init__(
        self,
        model: HamiltonianModel,
        trotter_order: int,
        execution_mode: str,
        shots: Optional[int],
        sampler: Optional[Any],
        rng: np.random.Generator,
    ) -> None:
        super().__init__(model, execution_mode, shots, sampler, rng)

        if self.model.d == 1:
            self.builder = SharedRegisterBuilder(
                model=self.model, trotter_order=trotter_order
            )
        else:
            self.builder = SeparateRegistersBuilder(
                model=self.model, trotter_order=trotter_order
            )

    def extract(
        self, X_list: Sequence[InputX], tau: float, r_steps: int, observable: Observable, show_progress: bool = True
    ) -> FeatureMatrix:
        """Extract the flattened feature matrix for a batch of input graphs.

        Exploits the linearity of the Fourier map :math:`b(x) = \sum_h c_h b_h(x)`
        by independently extracting the tensor for each Pauli term.

        Parameters
        ----------
        X_list : Sequence[InputX]
            A sequence of input structures or graphs.
        tau : float
            Total evolution time.
        r_steps : int
            Number of Trotter steps.
        observable : Observable
            The observable operator to measure.

        Returns
        -------
        FeatureMatrix
            A 2D array of shape `(n_samples, n_features)` containing the
            flattened Fourier features.
        """
        if self.model.d == 1:
            m_terms = (
                len(self.model.upload_paulis)
                if hasattr(self.model, "upload_paulis")
                else self.model.num_qubits
            )
            max_freq = 2 * r_steps * m_terms
        else:
            max_freq = 2 * r_steps

        freq_dim = (2 * max_freq + 1) ** self.model.d
        B = np.zeros((len(X_list), freq_dim), dtype=np.float64)

        for i, x in enumerate(X_list):
            for pauli_str, coeff in observable.to_sparse_pauli_op().to_list():
                if abs(coeff) < 1e-12:
                    continue

                if self.execution_mode == "emulator":
                    b_pauli = self._extract_emulator(
                        x, tau, r_steps, pauli_str, max_freq, show_progress=show_progress # <-- Added
                    )
                else:
                    b_pauli = self._extract_hardware(
                        x, tau, r_steps, pauli_str, max_freq, show_progress=show_progress
                    )

                B[i] += np.real(coeff) * b_pauli

        return B

    def _extract_emulator(
        self, x: InputX, tau: float, r_steps: int, pauli: str, max_freq: int, show_progress: bool = True
    ) -> np.ndarray:
        """Evaluate features via optimized statevector simulation."""
        qc, freqs = self.builder.build_aup(
            # ...
        )
        sv = Statevector(qc).data
        n_s = len(freqs[0])

        b_flat = np.zeros((2 * max_freq + 1) ** self.model.d, dtype=np.float64)

        # The fast-moving granular loop!
        for flat_idx in tqdm(range(len(b_flat)), desc=f"Emulator Features ({pauli})", leave=False, disable=not show_progress):
            sv_idx = self._map_to_sv_index(flat_idx, freqs, max_freq, n_s)
            exact_val = np.real(sv[sv_idx])
            b_flat[flat_idx] = self._apply_shot_noise(exact_val)

        return b_flat
    
    def _extract_hardware(
        self, x: InputX, tau: float, r_steps: int, pauli: str, max_freq: int, show_progress: bool = True
    ) -> np.ndarray:
        """Evaluate features via batched execution on Qiskit primitives using Parameterized caching."""
        
        # Initialize the transpile cache if it doesn't exist
        if not hasattr(self, "_transpiled_cache"):
            self._transpiled_cache = {}

        cache_key = (tuple(x), pauli, r_steps)
        
        # We only build and transpile if we haven't seen this specific graph and Pauli before
        if cache_key not in self._transpiled_cache:
            # Create the algebraic parameter for tau
            tau_param = Parameter('tau')
            
            base_qc, freqs = self.builder.build_aup(
                num_qubits=self.model.num_qubits,
                x=x,
                tau=tau_param,  # Pass the Parameter object instead of the float
                r_steps=r_steps,
                pauli=pauli,
                execution_mode="hardware_base",
            )
            ht_control = [qr for qr in base_qc.qregs if qr.name == "ht_control"][0]
            creg = base_qc.cregs[0]
            n_s = len(freqs[0])

            raw_qcs = []
            dim = (2 * max_freq + 1) ** self.model.d
            
            # Build the 9,261 uncompiled circuits algebraically
            for flat_idx in tqdm(range(dim), desc=f"Building Grid ({pauli})", leave=False, disable=not show_progress):
                target_tuple = self._map_to_freq_tuple(flat_idx, max_freq, n_s)
                target = target_tuple[0] if self.model.d == 1 else target_tuple

                qc_l = base_qc.copy()
                self.builder._append_freq_selector(qc_l, ht_control, freqs, target)
                qc_l.h(ht_control[0])
                qc_l.measure(ht_control[0], creg[0])
                raw_qcs.append(qc_l)

            # Transpile the parameterized skeletons exactly ONCE
            aer_basis = [
                'id', 'rz', 'sx', 'x', 'y', 'z', 'h', 's', 'sdg', 'rx', 'ry', 
                'cx', 'cy', 'cz', 'crx', 'cry', 'crz', 'ccx', 'mcx', 'measure'
            ]
            
            for _ in tqdm(range(1), desc="Transpiling Grid (Runs ONCE per graph)", leave=False, disable=not show_progress):
                transpiled_qcs = transpile(raw_qcs, basis_gates=aer_basis, optimization_level=0)
            
            if not isinstance(transpiled_qcs, list):
                transpiled_qcs = [transpiled_qcs]
                
            self._transpiled_cache[cache_key] = (transpiled_qcs, tau_param)

        # -----------------------------------------------------------------
        # Execution Phase (happens instantly on subsequent tau steps)
        # -----------------------------------------------------------------
        transpiled_qcs, tau_param = self._transpiled_cache[cache_key]
        
        # Prepare V2 PUBs by binding the actual float tau to the pre-compiled circuits
        pubs = [(qc, {tau_param: tau}) for qc in transpiled_qcs]

        b_flat = np.zeros(len(pubs), dtype=np.float64)
        chunk_size = 500  

        for chunk_start in tqdm(
            range(0, len(pubs), chunk_size), 
            desc=f"GPU Sim (tau={tau:.2f})", 
            leave=False, 
            disable=not show_progress
        ):
            chunk_pubs = pubs[chunk_start : chunk_start + chunk_size]
            job = self.sampler.run(chunk_pubs, shots=self.shots)
            result = job.result()

            for j in range(len(chunk_pubs)):
                bit_array = result[j].data.creg.array
                n1 = int(np.sum(bit_array))
                n0 = self.shots - n1
                b_flat[chunk_start + j] = (n0 - n1) / self.shots

        return b_flat

    def _map_to_sv_index(
        self, flat_idx: int, freqs: List[Any], max_freq: int, n_s: int
    ) -> int:
        """Map a flattened rank-d tensor index to Qiskit's statevector coordinate."""
        d = len(freqs)
        N = self.model.num_qubits

        grid_shape = tuple([2 * max_freq + 1] * d)
        coords = np.unravel_index(flat_idx, grid_shape)

        sv_idx = 0
        for k, coord in enumerate(coords):
            l_k = coord - max_freq
            u_k = int(l_k) % (1 << n_s)
            shift = N + 1 + (k * n_s)
            sv_idx += u_k << shift

        return sv_idx

    def _map_to_freq_tuple(
        self, flat_idx: int, max_freq: int, n_s: int
    ) -> Tuple[int, ...]:
        """Convert a flattened index into the unsigned binary frequency tuple."""
        d = self.model.d
        grid_shape = tuple([2 * max_freq + 1] * d)
        coords = np.unravel_index(flat_idx, grid_shape)

        u_tuple = []
        for coord in coords:
            l_k = coord - max_freq
            u_k = int(l_k) % (1 << n_s)
            u_tuple.append(u_k)

        return tuple(u_tuple)


class KernelEngine(_EngineBase):
    """Engine for True Quantum Overlap Kernel evaluation.

    Directly computes the pairwise Gram matrix :math:`K(x_i, x_j)` utilizing
    the Figure 8 overlap circuit design, bypassing the exponential scaling of
    the full Fourier tensor extraction.
    """

    def __init__(
        self,
        model: HamiltonianModel,
        trotter_order: int,
        execution_mode: str,
        shots: Optional[int],
        sampler: Optional[Any],
        rng: np.random.Generator,
    ) -> None:
        super().__init__(model, execution_mode, shots, sampler, rng)
        self.builder = KernelOverlapBuilder(
            model=model, trotter_order=trotter_order
        )

    def compute_gram(
        self,
        X1: Sequence[InputX],
        X2: Optional[Sequence[InputX]],
        tau: float,
        r_steps: int,
        observable: Observable,
    ) -> GramMatrix:
        """Compute the Gram matrix between samples in X1 and X2.

        Parameters
        ----------
        X1 : Sequence[InputX]
            The first sequence of input samples.
        X2 : Sequence[InputX], optional
            The second sequence of input samples. If None, computes the
            symmetric matrix :math:`K(X_1, X_1)`.
        tau : float
            Total evolution time.
        r_steps : int
            Number of Trotter steps.
        observable : Observable
            The observable operator.

        Returns
        -------
        GramMatrix
            A 2D array of shape `(len(X1), len(X2))` containing overlap values.
        """
        symmetric = X2 is None
        X2_eval = X1 if symmetric else X2

        if self.execution_mode == "emulator":
            return self._compute_gram_emulator(
                X1, X2_eval, symmetric, tau, r_steps, observable
            )
        else:
            return self._compute_gram_hardware(
                X1, X2_eval, symmetric, tau, r_steps, observable
            )

    def _compute_gram_emulator(
        self,
        X1: Sequence[InputX],
        X2: Sequence[InputX],
        symmetric: bool,
        tau: float,
        r_steps: int,
        observable: Observable,
    ) -> GramMatrix:
        """Compute the Gram matrix via optimized O(M) statevector caching."""
        N1, N2 = len(X1), len(X2)
        K = np.zeros((N1, N2), dtype=np.float64)

        sv_cache: Dict[Tuple[Tuple, str], np.ndarray] = {}
        all_graphs = set(tuple(x) for x in X1).union(set(tuple(x) for x in X2))
        paulis = [
            (p, c)
            for p, c in observable.to_sparse_pauli_op().to_list()
            if abs(c) > 1e-12
        ]

        state_builder = SeparateRegistersBuilder(
            model=self.model,
            trotter_order=getattr(self.builder, "trotter_order", 2),
        )

        for x_tuple in all_graphs:
            x_list = list(x_tuple)
            for pauli_str, _ in paulis:
                qc, _ = state_builder.build_aup(
                    num_qubits=self.model.num_qubits,
                    x=x_list,
                    tau=tau,
                    r_steps=r_steps,
                    pauli=pauli_str,
                    execution_mode="statevector",
                )
                sv_cache[(x_tuple, pauli_str)] = Statevector(qc).data

        for i, x_i in enumerate(X1):
            start_j = i if symmetric else 0
            for j in range(start_j, N2):
                x_j = X2[j]

                k_val = 0.0
                for pauli_h, c_h in paulis:
                    for pauli_hp, c_hp in paulis:
                        sv_i = sv_cache[(tuple(x_i), pauli_h)]
                        sv_j = sv_cache[(tuple(x_j), pauli_hp)]

                        exact_overlap = np.vdot(sv_i, sv_j).real
                        noisy_overlap = self._apply_shot_noise(exact_overlap)

                        k_val += np.real(c_h) * np.real(c_hp) * noisy_overlap

                K[i, j] = k_val
                if symmetric and i != j:
                    K[j, i] = k_val

        return K

    def _compute_gram_hardware(
        self,
        X1: Sequence[InputX],
        X2: Sequence[InputX],
        symmetric: bool,
        tau: float,
        r_steps: int,
        observable: Observable,
    ) -> GramMatrix:
        """Compute the Gram matrix via batched primitive submissions with Parameterized caching."""
        from qiskit.circuit import Parameter
        from tqdm import tqdm

        N1, N2 = len(X1), len(X2)
        K = np.zeros((N1, N2), dtype=np.float64)
        paulis = [
            (p, c) for p, c in observable.to_sparse_pauli_op().to_list() if abs(c) > 1e-12
        ]

        # 1. Generate a robust cache key that survives across Experiment re-instantiations
        x1_tup = tuple(tuple(x) for x in X1)
        x2_tup = tuple(tuple(x) for x in X2) if not symmetric else None
        pauli_tup = tuple(p[0] for p in paulis)
        cache_key = (x1_tup, x2_tup, r_steps, pauli_tup)

        # Use class-level caching so it survives the Jupyter Notebook loop
        if not hasattr(self.__class__, "_transpiled_kernel_cache"):
            self.__class__._transpiled_kernel_cache = {}

        # 2. Build and Transpile ONCE
        if cache_key not in self.__class__._transpiled_kernel_cache:
            tau_param = Parameter('tau')
            raw_qcs = []
            job_map = []

            pairs = []
            for i in range(N1):
                start_j = i if symmetric else 0
                for j in range(start_j, N2):
                    pairs.append((i, j))

            for i, j in tqdm(pairs, desc="Building Kernel Skeletons", leave=False):
                x_i = X1[i]
                x_j = X2[j]

                for h_idx, (pauli_h, _) in enumerate(paulis):
                    for hp_idx, (pauli_hp, _) in enumerate(paulis):
                        qc = self.builder.build_overlap(
                            num_qubits=self.model.num_qubits,
                            x1=x_i,
                            x2=x_j,
                            tau=tau_param,  # Bind the algebraic parameter
                            r_steps=r_steps,
                            pauli1=pauli_h,
                            pauli2=pauli_hp,
                        )
                        raw_qcs.append(qc)
                        job_map.append((i, j, h_idx, hp_idx))

            aer_basis = [
                'id', 'rz', 'sx', 'x', 'y', 'z', 'h', 's', 'sdg', 'rx', 'ry', 
                'cx', 'cy', 'cz', 'crx', 'cry', 'crz', 'ccx', 'mcx', 'measure'
            ]
            
            for _ in tqdm(range(1), desc="Transpiling Kernel Pairs (Runs ONCE)", leave=False):
                transpiled_qcs = transpile(raw_qcs, basis_gates=aer_basis, optimization_level=0)
            
            if not isinstance(transpiled_qcs, list):
                transpiled_qcs = [transpiled_qcs]
                
            self.__class__._transpiled_kernel_cache[cache_key] = (transpiled_qcs, job_map, tau_param)

        # -----------------------------------------------------------------
        # 3. Execution Phase (Instantly binds the float tau)
        # -----------------------------------------------------------------
        transpiled_qcs, job_map, tau_param = self.__class__._transpiled_kernel_cache[cache_key]
        
        # We attach the specific float value of tau for this loop iteration
        pubs = [(qc, {tau_param: float(tau)}) for qc in transpiled_qcs]

        job = self.sampler.run(pubs, shots=self.shots)
        result = job.result()

        for k, (i, j, h_idx, hp_idx) in enumerate(job_map):
            bit_array = result[k].data.creg.array
            n1 = int(np.sum(bit_array))
            n0 = self.shots - n1
            noisy_overlap = (n0 - n1) / self.shots

            c_h = paulis[h_idx][1]
            c_hp = paulis[hp_idx][1]

            weight = np.real(c_h) * np.real(c_hp)
            K[i, j] += weight * noisy_overlap
            if symmetric and i != j:
                K[j, i] += weight * noisy_overlap

        return K