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
    """Engine for explicit Fourier feature extraction.

    Evaluates the tensor :math:`b(x)` representing the Fourier coefficients
    of the observable dynamics. Dynamically routes requests to Shared or Separate
    register circuit builders based on the Hamiltonian's dimension ``d``.

    Class-level caches
    ------------------
    :attr:`_LASSO_HARDWARE_CACHE` survives ``Experiment`` re-instantiation
    so a τ sweep over many ``Experiment`` objects amortises the build /
    transpile cost to **zero** after the first τ value.  Cache key is the
    full ``(X_list signature, Pauli signature, r_steps, num_qubits, d)``
    tuple — independent of τ, since τ is bound as a runtime
    :class:`Parameter`.
    """

    # Class-level transpile cache for the LASSO hardware path.  Indexed
    # by (X_signature, Pauli_signature, r_steps, num_qubits, d) → tuple
    # of (transpiled_qcs, job_map, tau_param).  Survives
    # ``Experiment(...)`` re-instantiation across an outer τ-sweep loop.
    _LASSO_HARDWARE_CACHE: Dict[Any, Tuple[List[Any], List[Tuple[int, int, int]], Parameter]] = {}

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

        Hardware-mode dispatch
        ----------------------
        In ``"hardware"`` mode this method **does not** loop over
        (graph, Pauli) and call the per-pair ``_extract_hardware`` —
        instead it dispatches once into :meth:`_extract_hardware_batch`,
        which builds every ``(i, P_h, target_freq)`` circuit, transpiles
        them all in **a single parallel call**, and submits them in a
        **single chunked Sampler run**.  This is the structural twin of
        :meth:`KernelEngine._compute_gram_hardware`.  The transpile and
        gate-build costs are then amortised across the entire
        ``X_list × paulis × (4r+1)^d`` batch — and across τ values via
        a class-level cache (see :data:`_LASSO_HARDWARE_CACHE`).

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

        paulis = [
            (p, c)
            for p, c in observable.to_sparse_pauli_op().to_list()
            if abs(c) > 1e-12
        ]

        if self.execution_mode == "emulator":
            # Per-(graph, Pauli) extraction — statevector eval is cheap
            # enough that the loop is fine and avoids materialising
            # huge tensors in memory.
            B = np.zeros((len(X_list), freq_dim), dtype=np.float64)
            for i, x in enumerate(X_list):
                for pauli_str, coeff in paulis:
                    b_pauli = self._extract_emulator(
                        x, tau, r_steps, pauli_str, max_freq, show_progress=show_progress
                    )
                    B[i] += np.real(coeff) * b_pauli
            return B

        # ---- Hardware: batched across the entire (X_list × paulis) ----
        return self._extract_hardware_batch(
            X_list=X_list,
            tau=tau,
            r_steps=r_steps,
            paulis=paulis,
            max_freq=max_freq,
            freq_dim=freq_dim,
            show_progress=show_progress,
        )

    def _extract_emulator(
        self, x: InputX, tau: float, r_steps: int, pauli: str, max_freq: int, show_progress: bool = True
    ) -> np.ndarray:
        """Evaluate features via optimized statevector simulation."""
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

        # The fast-moving granular loop!
        for flat_idx in tqdm(range(len(b_flat)), desc=f"Emulator Features ({pauli})", leave=False, disable=not show_progress):
            sv_idx = self._map_to_sv_index(flat_idx, freqs, max_freq, n_s)
            exact_val = np.real(sv[sv_idx])
            b_flat[flat_idx] = self._apply_shot_noise(exact_val)

        return b_flat
    
    def _extract_hardware_batch(
        self,
        X_list: Sequence[InputX],
        tau: float,
        r_steps: int,
        paulis: List[Tuple[str, complex]],
        max_freq: int,
        freq_dim: int,
        show_progress: bool = True,
    ) -> FeatureMatrix:
        """Batched hardware extraction across the full (X_list × paulis) cross-product.

        Architecture (mirrors :meth:`KernelEngine._compute_gram_hardware`)
        -------------------------------------------------------------------
        1. **Class-level transpile cache** (:data:`_LASSO_HARDWARE_CACHE`)
           keyed on ``(X_list signature, Pauli signature, r_steps,
           num_qubits, d)``.  Survives ``Experiment`` re-instantiation, so
           every τ value past the first amortises the entire build +
           transpile cost to zero.

        2. **One mega-batch of parameterised circuits** for every
           ``(i, h, target_freq)`` index.  τ is a single
           :class:`Parameter`; binding the float ``tau`` is microseconds.

        3. **Single transpile call** for the whole batch.  Qiskit's
           ``transpile`` parallelises across circuits (worker pool), so
           one call of ``N_total`` circuits is a few × faster than
           ``N_pauli × N_train`` separate calls.

        4. **Single chunked Sampler run** (chunk size 500).  Eliminates
           the per-call sampler / IPC / GPU-context overhead × 15 that
           dominated the previous per-(graph, Pauli) loop.

        5. Aggregate counts into the feature matrix
           ``B[i, flat_idx] = Σ_h c_h · ((n0 − n1) / shots)``.

        Parameters
        ----------
        X_list : Sequence[InputX]
            Sequence of input graphs to extract features for.
        tau : float
            Concrete evolution time bound to the parameterised skeleton.
        r_steps : int
            Trotter step count.
        paulis : list of (str, complex)
            Filtered list of Pauli terms ``(pauli_str, coefficient)`` —
            zero coefficients already removed.
        max_freq : int
            Half-bandwidth of the feature grid.
        freq_dim : int
            Total feature dimension :math:`(4r+1)^d` (already computed
            by the caller).
        show_progress : bool, default True
            tqdm visibility flag.

        Returns
        -------
        FeatureMatrix
            ``(len(X_list), freq_dim)`` real-valued Fourier features.
        """
        # ---- (1) Class-level cache key -----------------------------------
        x_sig     = tuple(tuple(x) for x in X_list)
        pauli_sig = tuple(p for p, _ in paulis)
        cache_key = (x_sig, pauli_sig, r_steps,
                     self.model.num_qubits, self.model.d)

        cache = self.__class__._LASSO_HARDWARE_CACHE
        if cache_key not in cache:
            # ---- (2) Build all parameterised circuits in one pass --------
            tau_param = Parameter("tau")
            raw_qcs   = []
            job_map: List[Tuple[int, int, int]] = []  # (i, p_idx, flat_idx)

            outer_iter = tqdm(
                list(enumerate(X_list)),
                desc=f"Building Grid ({len(X_list)}×{len(paulis)} graph/Pauli)",
                leave=False,
                disable=not show_progress,
            )
            for i, x in outer_iter:
                for p_idx, (pauli_str, _) in enumerate(paulis):
                    base_qc, freqs = self.builder.build_aup(
                        num_qubits=self.model.num_qubits,
                        x=x,
                        tau=tau_param,           # algebraic parameter
                        r_steps=r_steps,
                        pauli=pauli_str,
                        execution_mode="hardware_base",
                    )
                    ht_control = [
                        qr for qr in base_qc.qregs if qr.name == "ht_control"
                    ][0]
                    creg = base_qc.cregs[0]
                    n_s = len(freqs[0])

                    for flat_idx in range(freq_dim):
                        target_tuple = self._map_to_freq_tuple(
                            flat_idx, max_freq, n_s
                        )
                        target = (
                            target_tuple[0]
                            if self.model.d == 1
                            else target_tuple
                        )

                        qc_l = base_qc.copy()
                        self.builder._append_freq_selector(
                            qc_l, ht_control, freqs, target
                        )
                        qc_l.h(ht_control[0])
                        qc_l.measure(ht_control[0], creg[0])
                        raw_qcs.append(qc_l)
                        job_map.append((i, p_idx, flat_idx))

            # ---- (3) Single transpile call (Qiskit auto-parallelises) ----
            aer_basis = _aer_basis_for_inline_assembly()
            with tqdm(
                total=1,
                desc=f"Transpiling {len(raw_qcs)} Circuits (ONCE per X_list)",
                leave=False,
                disable=not show_progress,
            ) as bar:
                transpiled_qcs = transpile(
                    raw_qcs,
                    basis_gates=aer_basis,
                    optimization_level=0,
                )
                bar.update(1)

            if not isinstance(transpiled_qcs, list):
                transpiled_qcs = [transpiled_qcs]

            cache[cache_key] = (transpiled_qcs, job_map, tau_param)

        # ---- (4) Bind τ and run as a single chunked Sampler job ---------
        transpiled_qcs, job_map, tau_param = cache[cache_key]
        # Defensive PUB construction.  For graph configurations where
        # ``H_fixed(x, α=0) == 0`` (e.g. disconnected graphs with no
        # edges, or models whose entire τ-dependence is purely uploaded
        # via D·G·D), the symbolic ``tau`` Parameter never enters the
        # circuit and after transpilation ``qc.parameters == set()``.
        # Qiskit's V2 Sampler raises ``ValueError`` if you bind values
        # to a circuit with no parameters, so we omit the binding for
        # those circuits entirely — they are correctly τ-independent
        # and their measurement statistics are the same at any τ.
        pubs = [
            (qc, {tau_param: float(tau)}) if len(qc.parameters) > 0 else (qc,)
            for qc in transpiled_qcs
        ]

        chunk_size = 500
        b_per_circuit = np.zeros(len(pubs), dtype=np.float64)

        for chunk_start in tqdm(
            range(0, len(pubs), chunk_size),
            desc=f"GPU Sim (τ={tau:.3f}, {len(pubs)} circuits)",
            leave=False,
            disable=not show_progress,
        ):
            chunk_pubs = pubs[chunk_start : chunk_start + chunk_size]
            job = self.sampler.run(chunk_pubs, shots=self.shots)
            result = job.result()

            for j in range(len(chunk_pubs)):
                bit_array = result[j].data.creg.array
                n1 = int(np.sum(bit_array))
                n0 = self.shots - n1
                b_per_circuit[chunk_start + j] = (n0 - n1) / self.shots

        # ---- (5) Aggregate (i, P_h)-weighted contributions into B -------
        B = np.zeros((len(X_list), freq_dim), dtype=np.float64)
        for k, (i, p_idx, flat_idx) in enumerate(job_map):
            c_h = paulis[p_idx][1]
            B[i, flat_idx] += np.real(c_h) * b_per_circuit[k]

        return B

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

    # Class-level transpile cache for the kernel hardware path.  Indexed
    # by (X1_signature, X2_signature, r_steps, Pauli_signature, num_qubits, d).
    # Survives ``Experiment`` re-instantiation across an outer τ-sweep.
    _KERNEL_HARDWARE_CACHE: Dict[Any, Tuple[List[Any], List[Tuple[int, int, int, int]], Parameter]] = {}

    def _compute_gram_hardware(
        self,
        X1: Sequence[InputX],
        X2: Sequence[InputX],
        symmetric: bool,
        tau: float,
        r_steps: int,
        observable: Observable,
    ) -> GramMatrix:
        """Compute the Gram matrix via batched primitive submissions with Parameterized caching.

        Identical structural pattern to
        :meth:`FeatureEngine._extract_hardware_batch` so the LASSO/kernel
        scaling experiment exposes only the algorithmic complexity
        difference, not Qiskit-orchestration noise:

        * Class-level transpile cache keyed on
          ``(X1, X2, r, paulis, n, d)``.
        * Single mega-build of every ``(i, j, h, h')`` overlap circuit.
        * Single transpile call (Qiskit auto-parallelises).
        * Single chunked Sampler run (chunk size 500) — robust as the
          pair count grows for larger ``N``.
        """
        N1, N2 = len(X1), len(X2)
        K = np.zeros((N1, N2), dtype=np.float64)
        paulis = [
            (p, c)
            for p, c in observable.to_sparse_pauli_op().to_list()
            if abs(c) > 1e-12
        ]

        # ---- (1) Class-level cache key ---------------------------------
        x1_sig    = tuple(tuple(x) for x in X1)
        x2_sig    = tuple(tuple(x) for x in X2) if not symmetric else None
        pauli_sig = tuple(p for p, _ in paulis)
        cache_key = (x1_sig, x2_sig, r_steps, pauli_sig,
                     self.model.num_qubits, self.model.d)

        cache = self.__class__._KERNEL_HARDWARE_CACHE

        # ---- (2) Build all overlap circuits + single transpile ---------
        if cache_key not in cache:
            tau_param = Parameter("tau")
            raw_qcs   = []
            job_map: List[Tuple[int, int, int, int]] = []  # (i, j, h_idx, hp_idx)

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
                desc=f"Transpiling {len(raw_qcs)} Kernel Circuits (ONCE)",
                leave=False,
            ) as bar:
                transpiled_qcs = transpile(
                    raw_qcs, basis_gates=aer_basis, optimization_level=0
                )
                bar.update(1)

            if not isinstance(transpiled_qcs, list):
                transpiled_qcs = [transpiled_qcs]

            cache[cache_key] = (transpiled_qcs, job_map, tau_param)

        # ---- (3) Bind τ and run as a single chunked Sampler job --------
        transpiled_qcs, job_map, tau_param = cache[cache_key]
        # Defensive PUB construction.  For graph configurations where
        # ``H_fixed(x, α=0) == 0`` (e.g. disconnected graphs with no
        # edges, or models whose entire τ-dependence is purely uploaded
        # via D·G·D), the symbolic ``tau`` Parameter never enters the
        # circuit and after transpilation ``qc.parameters == set()``.
        # Qiskit's V2 Sampler raises ``ValueError`` if you bind values
        # to a circuit with no parameters, so we omit the binding for
        # those circuits entirely — they are correctly τ-independent
        # and their measurement statistics are the same at any τ.
        pubs = [
            (qc, {tau_param: float(tau)}) if len(qc.parameters) > 0 else (qc,)
            for qc in transpiled_qcs
        ]

        chunk_size = 500
        overlaps = np.zeros(len(pubs), dtype=np.float64)

        for chunk_start in tqdm(
            range(0, len(pubs), chunk_size),
            desc=f"GPU Sim Kernel (τ={tau:.3f}, {len(pubs)} circuits)",
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

        # ---- (4) Aggregate Pauli-weighted overlaps into K --------------
        for k, (i, j, h_idx, hp_idx) in enumerate(job_map):
            c_h = paulis[h_idx][1]
            c_hp = paulis[hp_idx][1]
            weight = np.real(c_h) * np.real(c_hp)
            K[i, j] += weight * overlaps[k]
            if symmetric and i != j:
                K[j, i] += weight * overlaps[k]

        return K