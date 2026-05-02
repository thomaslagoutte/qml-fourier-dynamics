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


# ---------------------------------------------------------------------
# Module-level helper: basis-gates list for ``transpile`` that protects
# the inline MCX cascades produced by :mod:`circuits._controlled_ops`
# from being unrolled into long CX chains.
# ---------------------------------------------------------------------
#
# Qiskit ≤ 1.x accepts ``mcx`` directly in ``basis_gates``.
# Qiskit ≥ 2.x rejects it ("non-standard gate") and demands a Target
# with a ``custom_name_mapping``.  We probe once at import time and
# cache the right form so the transpile call inside the hardware path
# is portable across the user's GPU box (Qiskit 1.x with Aer GPU) and
# the local CI (Qiskit 2.x).

_CACHED_AER_BASIS: Optional[List[str]] = None


def _aer_basis_for_inline_assembly() -> List[str]:
    """Return the basis-gates list that preserves inline c-A(U,P) gates.

    ``mcx`` and ``ccx`` are kept in the basis so that the MCX cascades
    inside :func:`append_dctrl_cVp` / etc. are *not* eagerly decomposed
    into CX chains during transpilation.  When Qiskit's ``transpile``
    refuses ``mcx`` in ``basis_gates`` (Qiskit ≥ 2.x), we drop it from
    the list — at ``optimization_level=0`` the transpiler does not
    unroll high-level gates anyway, so the MCX instructions still
    survive intact.
    """
    global _CACHED_AER_BASIS
    if _CACHED_AER_BASIS is not None:
        return _CACHED_AER_BASIS

    candidate = [
        "id", "rz", "sx", "x", "y", "z", "h", "s", "sdg",
        "rx", "ry", "cx", "cy", "cz", "crx", "cry", "crz",
        "ccx", "mcx", "measure",
    ]
    # Probe whether the installed Qiskit accepts ``mcx`` in basis_gates.
    try:
        from qiskit import QuantumCircuit
        probe = QuantumCircuit(2)
        probe.h(0)
        transpile(probe, basis_gates=candidate, optimization_level=0)
        _CACHED_AER_BASIS = candidate
    except (ValueError, TypeError):
        # Qiskit ≥ 2.x: drop ``mcx`` (transpile at level 0 will not
        # unroll an unknown high-level gate, so MCX survives).
        _CACHED_AER_BASIS = [g for g in candidate if g != "mcx"]
    return _CACHED_AER_BASIS


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
    """Engine for explicit Fourier feature extraction."""

    # Class-level transpile cache for the LASSO hardware path.
    _LASSO_HARDWARE_CACHE: Dict[Any, Tuple[List[Any], Parameter]] = {}

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

        paulis = [
            (p, c)
            for p, c in observable.to_sparse_pauli_op().to_list()
            if abs(c) > 1e-12
        ]

        B = np.zeros((len(X_list), freq_dim), dtype=np.float64)

        if self.execution_mode == "emulator":
            for i, x in enumerate(X_list):
                for pauli_str, coeff in paulis:
                    b_pauli = self._extract_emulator(
                        x, tau, r_steps, pauli_str, max_freq, show_progress=show_progress
                    )
                    B[i] += np.real(coeff) * b_pauli
            return B

        # Hardware mode: Evaluate per graph/pauli (to save Qiskit processing overhead)
        for i, x in enumerate(X_list):
            for pauli_str, coeff in paulis:
                b_pauli = self._extract_hardware(
                    x, tau, r_steps, pauli_str, max_freq, show_progress=show_progress
                )
                B[i] += np.real(coeff) * b_pauli

        return B

    def _extract_emulator(
        self, x: InputX, tau: float, r_steps: int, pauli: str, max_freq: int, show_progress: bool = True
    ) -> np.ndarray:
        qc, freqs = self.builder.build_aup(
            num_qubits=self.model.num_qubits,
            x=x,
            tau=tau,
            r_steps=r_steps,
            pauli=pauli,
            execution_mode="statevector",
        )
        sv = Statevector(qc).data
        n_s = len(freqs[0])

        b_flat = np.zeros((2 * max_freq + 1) ** self.model.d, dtype=np.float64)

        for flat_idx in tqdm(range(len(b_flat)), desc=f"Emulator Features ({pauli})", leave=False, disable=not show_progress):
            sv_idx = self._map_to_sv_index(flat_idx, freqs, max_freq, n_s)
            exact_val = np.real(sv[sv_idx])
            b_flat[flat_idx] = self._apply_shot_noise(exact_val)

        return b_flat
    
    def _extract_hardware(
        self, x: InputX, tau: float, r_steps: int, pauli: str, max_freq: int, show_progress: bool = True
    ) -> np.ndarray:
        """Evaluate features via batched execution on Qiskit primitives using Parameterized caching."""
        
        cache_key = (tuple(x), pauli, r_steps)
        cache = self.__class__._LASSO_HARDWARE_CACHE
        
        # Build and transpile only if not cached
        if cache_key not in cache:
            tau_param = Parameter('tau')
            
            base_qc, freqs = self.builder.build_aup(
                num_qubits=self.model.num_qubits,
                x=x,
                tau=tau_param,
                r_steps=r_steps,
                pauli=pauli,
                execution_mode="hardware_base",
            )
            ht_control = [qr for qr in base_qc.qregs if qr.name == "ht_control"][0]
            creg = base_qc.cregs[0]
            n_s = len(freqs[0])

            raw_qcs = []
            dim = (2 * max_freq + 1) ** self.model.d
            
            for flat_idx in tqdm(range(dim), desc=f"Building Grid ({pauli})", leave=False, disable=not show_progress):
                target_tuple = self._map_to_freq_tuple(flat_idx, max_freq, n_s)
                target = target_tuple[0] if self.model.d == 1 else target_tuple

                qc_l = base_qc.copy()
                self.builder._append_freq_selector(qc_l, ht_control, freqs, target)
                qc_l.h(ht_control[0])
                qc_l.measure(ht_control[0], creg[0])
                raw_qcs.append(qc_l)

            aer_basis = _aer_basis_for_inline_assembly()
            
            for _ in tqdm(range(1), desc="Transpiling Grid", leave=False, disable=not show_progress):
                transpiled_qcs = transpile(raw_qcs, basis_gates=aer_basis, optimization_level=0)
            
            if not isinstance(transpiled_qcs, list):
                transpiled_qcs = [transpiled_qcs]
                
            cache[cache_key] = (transpiled_qcs, tau_param)

        # Execution Phase 
        transpiled_qcs, tau_param = cache[cache_key]
        
        # Defensive PUB construction for Qiskit V2
        pubs = [
            (qc, {tau_param: float(tau)}) if len(qc.parameters) > 0 else (qc,)
            for qc in transpiled_qcs
        ]

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
    """Engine for True Quantum Overlap Kernel evaluation."""

    _KERNEL_HARDWARE_CACHE: Dict[Any, Tuple[List[Any], List[Tuple[int, int, int, int]], Parameter]] = {}

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
        N1, N2 = len(X1), len(X2)
        K = np.zeros((N1, N2), dtype=np.float64)
        paulis = [
            (p, c)
            for p, c in observable.to_sparse_pauli_op().to_list()
            if abs(c) > 1e-12
        ]

        x1_sig    = tuple(tuple(x) for x in X1)
        x2_sig    = tuple(tuple(x) for x in X2) if not symmetric else None
        pauli_sig = tuple(p for p, _ in paulis)
        cache_key = (x1_sig, x2_sig, r_steps, pauli_sig,
                     self.model.num_qubits, self.model.d)

        cache = self.__class__._KERNEL_HARDWARE_CACHE

        if cache_key not in cache:
            tau_param = Parameter("tau")
            raw_qcs   = []
            job_map: List[Tuple[int, int, int, int]] = []

            pairs = []
            for i in range(N1):
                start_j = i if symmetric else 0
                for j in range(start_j, N2):
                    pairs.append((i, j))

            for i, j in tqdm(
                pairs, desc="Building Kernel Skeletons", leave=False
            ):
                x_i = X1[i]
                x_j = X2[j]
                for h_idx, (pauli_h, _) in enumerate(paulis):
                    for hp_idx, (pauli_hp, _) in enumerate(paulis):
                        qc = self.builder.build_overlap(
                            num_qubits=self.model.num_qubits,
                            x1=x_i,
                            x2=x_j,
                            tau=tau_param,
                            r_steps=r_steps,
                            pauli1=pauli_h,
                            pauli2=pauli_hp,
                        )
                        raw_qcs.append(qc)
                        job_map.append((i, j, h_idx, hp_idx))

            aer_basis = _aer_basis_for_inline_assembly()
            with tqdm(
                total=1,
                desc=f"Transpiling {len(raw_qcs)} Kernel Circuits",
                leave=False,
            ) as bar:
                transpiled_qcs = transpile(
                    raw_qcs, basis_gates=aer_basis, optimization_level=0
                )
                bar.update(1)

            if not isinstance(transpiled_qcs, list):
                transpiled_qcs = [transpiled_qcs]

            cache[cache_key] = (transpiled_qcs, job_map, tau_param)

        transpiled_qcs, job_map, tau_param = cache[cache_key]
        
        pubs = [
            (qc, {tau_param: float(tau)}) if len(qc.parameters) > 0 else (qc,)
            for qc in transpiled_qcs
        ]

        chunk_size = 500
        overlaps = np.zeros(len(pubs), dtype=np.float64)

        for chunk_start in tqdm(
            range(0, len(pubs), chunk_size),
            desc=f"GPU Sim Kernel (τ={tau:.2f})",
            leave=False,
        ):
            chunk_pubs = pubs[chunk_start : chunk_start + chunk_size]
            job = self.sampler.run(chunk_pubs, shots=self.shots)
            result = job.result()
            for j_local in range(len(chunk_pubs)):
                bit_array = result[j_local].data.creg.array
                n1 = int(np.sum(bit_array))
                n0 = self.shots - n1
                overlaps[chunk_start + j_local] = (n0 - n1) / self.shots

        for k, (i, j, h_idx, hp_idx) in enumerate(job_map):
            c_h = paulis[h_idx][1]
            c_hp = paulis[hp_idx][1]
            weight = np.real(c_h) * np.real(c_hp)
            K[i, j] += weight * overlaps[k]
            if symmetric and i != j:
                K[j, i] += weight * overlaps[k]

        return K