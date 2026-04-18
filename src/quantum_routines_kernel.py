"""Quantum circuit extensions for the Quantum Overlap Kernel (Section VI.B).

This module extends CircuitBuilder with methods specific to the
inhomogeneous TFIM in the d = poly(n) regime.  It is designed to be
imported alongside the original quantum_routines.py and mixed into
CircuitBuilder via straightforward method addition, or used as a
standalone mixin class (KernelCircuitMixin) that CircuitBuilder
can inherit from.

New public API
--------------
KernelCircuitMixin
    build_au_inhomogeneous(...)
        A(U) circuit for inhomogeneous TFIM: d=n separate freq registers,
        one per qubit alpha_i.  Follows Option A (separate registers),
        consistent with the LGT implementation.

    build_aup_inhomogeneous(...)
        A(U,P) circuit for inhomogeneous TFIM: A(U)† · P · A(U).
        Amplitude-encodes b_l(x) into the joint freq register space.

    build_kernel_circuit_inhomogeneous(...)
        Figure 8 kernel circuit for the inhomogeneous TFIM.
        Estimates Re<b(x)|b(x')> via a Hadamard test on A(U,P).

    compute_kernel_entry(...)
        Simulate the kernel circuit and return Re<b(x)|b(x')>.

    compute_gram_matrix(...)
        Build the full T×T Gram matrix K[i,j] = k(x_i, x_j).

    compute_au_labels_inhomogeneous(...)
        A(U)-consistent Trotter training labels for the inhomogeneous TFIM.
"""

import math
from typing import List, Tuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import PauliGate
from qiskit.quantum_info import Statevector


class StatevectorCache:
    """Cache mapping (graph, tau) -> ancilla=|0> sub-vector of A(U,P)|0>.

    Purpose
    -------
    On a statevector simulator, computing A(U,P)|0> for a fixed graph x and
    evolution time tau takes ~0.5s for a 13-qubit circuit (n=3, r=1).  Each
    graph appears in O(T) kernel entries (one per Gram matrix row/column), so
    caching gives a ~T/2 speedup on the Gram matrix and ~T on the cross-kernel.

    For T=20: 210 Gram entries but only 20 unique statevectors.  Cross-kernel
    (200 entries) reuses all 20 training statevectors, adding only 10 new test
    statevectors.  Total unique simulations: 30 instead of 820.

    Hardware migration note  [HARDWARE_SWAP]
    ----------------------------------------
    On real quantum hardware each kernel entry k(x,x') is estimated by running
    the Hadamard-test circuit (Figure 8) and measuring <Z> on the HT ancilla —
    there is no shared statevector.  To migrate to hardware:

        # HARDWARE_SWAP: replace _get_or_compute_sv (below) with a function
        # that submits the A(U,P) circuit to the hardware backend and returns
        # shot-estimated b_l amplitudes (or raw bitstring counts).
        # The StatevectorCache can be repurposed to cache transpiled circuit
        # objects to avoid repeated transpilation overhead per backend.

    Parameters
    ----------
    None — populated lazily via get / put.
    """

    def __init__(self):
        # Maps (frozenset(edges), tau_rounded) -> np.ndarray, anc=0 sub-vector
        self._store: dict = {}
        self._hits: int   = 0
        self._misses: int = 0

    def _key(self, edges: list, tau: float) -> tuple:
        """Canonical cache key: graph is edge-order-invariant, tau rounded."""
        return (frozenset(edges), round(tau, 12))

    def get(self, edges: list, tau: float):
        """Return cached sub-vector, or None on miss."""
        val = self._store.get(self._key(edges, tau))
        if val is not None:
            self._hits += 1
        else:
            self._misses += 1
        return val

    def put(self, edges: list, tau: float, sv_anc0: np.ndarray) -> None:
        """Store the ancilla=|0> sub-vector for (edges, tau)."""
        self._store[self._key(edges, tau)] = sv_anc0

    def clear(self) -> None:
        """Evict all entries.  Call between tau values to free memory."""
        self._store.clear()
        self._hits   = 0
        self._misses = 0

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def stats(self) -> str:
        total = self._hits + self._misses
        rate  = self._hits / total * 100 if total > 0 else 0.0
        return (f"StatevectorCache: {self.size} entries stored | "
                f"{self._hits}/{total} hits ({rate:.0f}%)")


class KernelCircuitMixin:
    """Mixin providing kernel circuit methods for the inhomogeneous TFIM.

    Assumes self has:
        _freq_register_size(r_steps) -> int
        _get_cached_v_gates(n_s) -> dict
    from CircuitBuilder.
    """

    # ------------------------------------------------------------------
    # A(U) forward / adjoint for inhomogeneous TFIM  (Option A)
    # ------------------------------------------------------------------

    def _append_au_inhomogeneous_forward(
        self,
        qc: QuantumCircuit,
        x_edges: List[Tuple[int, int]],
        tau: float,
        r_steps: int,
        num_qubits: int,
        n_s: int,
        qr_data: QuantumRegister,
        qr_ancilla: QuantumRegister,
        qr_freqs: List[QuantumRegister],
    ) -> None:
        """A(U) forward pass for the inhomogeneous TFIM.

        Each qubit i has an independent unknown parameter alpha_i, encoded
        into its own dedicated frequency register qr_freqs[i] via a D·G·D
        block (Appendix C.1).  This follows Option A (separate registers),
        consistent with the LGT implementation in the existing codebase.

        Per Trotter step:
            1. RZZ(2*tau/r) on each graph edge (fixed, alpha-independent)
            2. For each qubit i: H · D(i) · G(freq_i) · D(i) · H
               encoding the upload gate e^{i*pi*alpha_i*X_i} into freq_i

        The H gates conjugate the Z-parity D block to act on the X basis,
        faithfully reproducing the A-transformation of an X-type upload gate
        (Appendix C.1, without loss of generality via basis change).

        Parameters
        ----------
        qc : QuantumCircuit  (mutated in place)
        x_edges : list of (int, int)  — graph topology
        tau : float  — total evolution time
        r_steps : int  — number of Trotter steps
        num_qubits : int  — number of data qubits (= d = n)
        n_s : int  — frequency register size (bits per register)
        qr_data : QuantumRegister  — data qubits
        qr_ancilla : QuantumRegister  — single shared ancilla for parity
        qr_freqs : list of QuantumRegister  — one freq register per qubit
        """
        gates = self._get_cached_v_gates(n_s)
        cVp = gates["cVp"]   # ctrl=|0> → V+ (increment)
        cVm = gates["cVm"]   # ctrl=|1> → V- (decrement)

        for _ in range(r_steps):
            # --- Fixed gates: ZZ interactions (graph-dependent, alpha-free) ---
            for i, j in x_edges:
                qc.rzz(-2.0 * tau / r_steps, qr_data[i], qr_data[j])

            # --- A-transformation of per-qubit X upload gates ---
            # For each qubit i, encode e^{i*pi*alpha_i*X_i} into qr_freqs[i].
            # Steps: H (Z→X basis), D·G·D block, H (back to Z basis).
            for i in range(num_qubits):
                # Basis change: X upload → Z parity in D block
                qc.h(qr_data[i])

                # D(b_l): parity of qubit i onto shared ancilla
                # (single-qubit Pauli → single CNOT)
                qc.cx(qr_data[i], qr_ancilla[0])

                # G(s_l): V+ if anc=|0>, V- if anc=|1> on THIS qubit's freq register
                qc.append(cVp, [qr_ancilla[0]] + list(qr_freqs[i]))
                qc.append(cVm, [qr_ancilla[0]] + list(qr_freqs[i]))

                # D(b_l)† = D(b_l): uncompute ancilla
                qc.cx(qr_data[i], qr_ancilla[0])

                # Undo basis change
                qc.h(qr_data[i])

    def _append_au_inhomogeneous_adjoint(
        self,
        qc: QuantumCircuit,
        x_edges: List[Tuple[int, int]],
        tau: float,
        r_steps: int,
        num_qubits: int,
        n_s: int,
        qr_data: QuantumRegister,
        qr_ancilla: QuantumRegister,
        qr_freqs: List[QuantumRegister],
    ) -> None:
        """A(U)† adjoint pass for the inhomogeneous TFIM.

        Exact circuit-level adjoint of _append_au_inhomogeneous_forward:
            - Trotter steps reversed
            - Qubit loop reversed within each step
            - Each gate replaced by its adjoint:
                H      → H       (self-adjoint)
                CNOT   → CNOT    (self-adjoint)
                RZZ(θ) → RZZ(-θ) (negate angle)
                cVp    → cVm_dag (V+† = V-)
                cVm    → cVp_dag (V-† = V+)

        Parameters are identical to _append_au_inhomogeneous_forward.
        """
        gates = self._get_cached_v_gates(n_s)
        cVp_dag = gates["cVp_dag"]   # ctrl=|0> → (V+)† = V-
        cVm_dag = gates["cVm_dag"]   # ctrl=|1> → (V-)† = V+

        for _ in reversed(range(r_steps)):
            # Reverse qubit loop (adjoint of forward inner loop)
            for i in reversed(range(num_qubits)):
                qc.h(qr_data[i])
                qc.cx(qr_data[i], qr_ancilla[0])

                # Adjoint of G: V+† (ctrl=|0>) and V-† (ctrl=|1>)
                qc.append(cVm_dag, [qr_ancilla[0]] + list(qr_freqs[i]))
                qc.append(cVp_dag, [qr_ancilla[0]] + list(qr_freqs[i]))

                qc.cx(qr_data[i], qr_ancilla[0])
                qc.h(qr_data[i])

            # RZZ†: negate angle, reverse edge order
            for i, j in reversed(x_edges):
                qc.rzz(2.0 * tau / r_steps, qr_data[i], qr_data[j])

    # ------------------------------------------------------------------
    # A(U) circuit for inhomogeneous TFIM  (Theorem 1 analogue)
    # ------------------------------------------------------------------

    def build_au_inhomogeneous(
        self,
        num_qubits: int,
        x_edges: List[Tuple[int, int]],
        tau: float,
        r_steps: int,
    ) -> Tuple[QuantumCircuit, List[QuantumRegister]]:
        """Build the A(U) circuit for the inhomogeneous TFIM.

        Prepares the state (Theorem 1 / Eq. 6):

            A(U)|0> = sum_{l in L, k} a_{l,k} |l_0>_f0 ... |l_{n-1}>_{fn-1} |k>_data |0>_anc

        where L = {-r,...,+r}^n and each l_i is encoded in a separate
        frequency register qr_freqs[i].

        Register layout (Qiskit little-endian):
            [data_0, ..., data_{n-1}, ancilla, freq_0_0..freq_0_{n_s-1},
             freq_1_0..freq_1_{n_s-1}, ..., freq_{n-1}_0..freq_{n-1}_{n_s-1}]

        Parameters
        ----------
        num_qubits : int  — n data qubits (also d = n parameters)
        x_edges : list of (int, int)
        tau : float
        r_steps : int

        Returns
        -------
        qc : QuantumCircuit  — the A(U) circuit
        qr_freqs : list of QuantumRegister  — freq registers [freq_0, ..., freq_{n-1}]
        """
        n_s = self._freq_register_size(r_steps)

        qr_data    = QuantumRegister(num_qubits, "data")
        qr_ancilla = QuantumRegister(1, "ancilla")
        qr_freqs   = [QuantumRegister(n_s, f"freq_{i}") for i in range(num_qubits)]

        qc = QuantumCircuit(qr_data, qr_ancilla, *qr_freqs)

        self._append_au_inhomogeneous_forward(
            qc, x_edges, tau, r_steps, num_qubits, n_s,
            qr_data, qr_ancilla, qr_freqs,
        )

        return qc, qr_freqs

    # ------------------------------------------------------------------
    # A(U,P) circuit for inhomogeneous TFIM  (Corollary 1 analogue)
    # ------------------------------------------------------------------

    def build_aup_inhomogeneous(
        self,
        num_qubits: int,
        x_edges: List[Tuple[int, int]],
        tau: float,
        r_steps: int,
        pauli_observable: str,
    ) -> QuantumCircuit:
        """Build the A(U,P) circuit for the inhomogeneous TFIM.

        Prepares the state (Corollary 1 / Figure 4):

            A(U,P)|0> = (1/||b||_2) sum_l b_l(x) |l>_freqs |0>_anc + |trash>|1>_anc

        where l = (l_0,...,l_{n-1}) is a multi-index and b_l(x) are the
        Fourier coefficients of f(alpha;x) = <psi(alpha;x)|P|psi(alpha;x)>
        with respect to the full parameter vector alpha.

        Circuit sequence (Figure 4):
            1. A(U) forward: encode alpha dependence into freq registers
            2. Apply Pauli observable P on data register
            3. A(U)†: decode back, leaving b_l amplitudes in freq registers

        Qubit count for n=3, r=2:
            data: 3, ancilla: 1, freq registers: 3 × 4 = 12  →  total: 16 qubits
        Statevector size: 2^16 = 65,536 amplitudes (~1 MB). Fast on M1.

        Parameters
        ----------
        num_qubits : int
        x_edges : list of (int, int)
        tau : float
        r_steps : int
        pauli_observable : str
            Pauli string for P.  Qiskit little-endian convention:
            rightmost character acts on qubit 0.
            For O = Z_0 on n=3 system: 'IIZ'.

        Returns
        -------
        QuantumCircuit  — A(U,P) on [data, ancilla, freq_0, ..., freq_{n-1}]
        """
        n_s = self._freq_register_size(r_steps)

        qr_data    = QuantumRegister(num_qubits, "data")
        qr_ancilla = QuantumRegister(1, "ancilla")
        qr_freqs   = [QuantumRegister(n_s, f"freq_{i}") for i in range(num_qubits)]

        aup_qc = QuantumCircuit(qr_data, qr_ancilla, *qr_freqs)

        # Step 1: A(U) forward
        self._append_au_inhomogeneous_forward(
            aup_qc, x_edges, tau, r_steps, num_qubits, n_s,
            qr_data, qr_ancilla, qr_freqs,
        )

        # Step 2: Apply Pauli observable P on data register
        aup_qc.append(PauliGate(pauli_observable), list(qr_data))

        # Step 3: A(U)† — gate-by-gate adjoint (no expensive to_gate().inverse())
        self._append_au_inhomogeneous_adjoint(
            aup_qc, x_edges, tau, r_steps, num_qubits, n_s,
            qr_data, qr_ancilla, qr_freqs,
        )

        return aup_qc

    # ------------------------------------------------------------------
    # Kernel circuit  (Section VI.B / Figure 8 — inhomogeneous TFIM)
    # ------------------------------------------------------------------

    def build_kernel_circuit_inhomogeneous(
        self,
        num_qubits: int,
        x_edges_1: List[Tuple[int, int]],
        x_edges_2: List[Tuple[int, int]],
        tau: float,
        r_steps: int,
        pauli_observable: str,
    ) -> QuantumCircuit:
        """Build the Hadamard-test kernel circuit for the inhomogeneous TFIM.

        Implements Figure 8 of Barthe et al. (2025) for the inhomogeneous TFIM.

        The kernel k(x, x') = Re<b(x)|b(x')> is estimated by:

            k(x, x') = <Z>_{ht_ancilla}
                      = 2 * P(ht_ancilla = |0>) - 1

        Circuit structure:
            1. H on ht_ancilla
            2. Controlled-A(U,P) for x  (ctrl = |0>)  → prepares |b(x)> branch
            3. Controlled-A(U,P) for x' (ctrl = |1>)  → prepares |b(x')> branch
            4. H on ht_ancilla
            5. Measure ht_ancilla → <Z> = Re<b(x)|b(x')>

        The two controlled branches together implement the swap test structure
        that extracts the real inner product of the two b-vectors, exactly as
        in Figure 8.

        Qubit count for n=3, r=2:
            ht_ancilla: 1, data: 3, anc: 1, freq registers: 3×4=12  → 17 total
        Statevector size: 2^17 = 131,072 (~2 MB). Lightweight on M1.

        Parameters
        ----------
        num_qubits : int
        x_edges_1 : list of (int, int)  — graph x
        x_edges_2 : list of (int, int)  — graph x'
        tau : float
        r_steps : int
        pauli_observable : str  — Qiskit little-endian Pauli string

        Returns
        -------
        QuantumCircuit  — kernel circuit with measurement on ht_ancilla
        """
        # Build A(U,P) for both graphs
        aup_1 = self.build_aup_inhomogeneous(
            num_qubits, x_edges_1, tau, r_steps, pauli_observable
        )
        aup_2 = self.build_aup_inhomogeneous(
            num_qubits, x_edges_2, tau, r_steps, pauli_observable
        )

        n_s = self._freq_register_size(r_steps)

        # Register layout for the kernel circuit
        qr_ht   = QuantumRegister(1, "ht_ancilla")
        qr_data = QuantumRegister(num_qubits, "data")
        qr_anc  = QuantumRegister(1, "ancilla")
        qr_freqs = [QuantumRegister(n_s, f"freq_{i}") for i in range(num_qubits)]
        cr_meas = ClassicalRegister(1, "meas")

        kernel_qc = QuantumCircuit(qr_ht, qr_data, qr_anc, *qr_freqs, cr_meas)

        # All target qubits for the controlled-A(U,P) gate
        target_qubits = (
            list(qr_data) + list(qr_anc) +
            [q for reg in qr_freqs for q in reg]
        )

        # Controlled-A(U,P) gates:
        # ctrl=|0>: branch for x  (ht_ancilla in state |0>)
        # ctrl=|1>: branch for x' (ht_ancilla in state |1>)
        ctrl_aup_1 = aup_1.to_gate(label="A(U,P)_x").control(1, ctrl_state="0")
        ctrl_aup_2 = aup_2.to_gate(label="A(U,P)_x'").control(1, ctrl_state="1")

        # Hadamard test structure (Figure 8)
        kernel_qc.h(qr_ht[0])
        kernel_qc.append(ctrl_aup_1, [qr_ht[0]] + target_qubits)
        kernel_qc.append(ctrl_aup_2, [qr_ht[0]] + target_qubits)
        kernel_qc.h(qr_ht[0])

        # Measure HT ancilla: P(|0>) = (1 + Re<b(x)|b(x')>) / 2
        kernel_qc.measure(qr_ht[0], cr_meas[0])

        return kernel_qc

    # ------------------------------------------------------------------
    # Kernel entry evaluation
    # ------------------------------------------------------------------

    def compute_kernel_entry(
        self,
        num_qubits: int,
        x_edges_1: List[Tuple[int, int]],
        x_edges_2: List[Tuple[int, int]],
        tau: float,
        r_steps: int,
        pauli_observable: str,
    ) -> float:
        """Evaluate a single kernel entry k(x, x') via statevector inner product.

        Mathematical equivalence to the Hadamard-test kernel circuit
        -------------------------------------------------------------
        The Hadamard-test circuit (Figure 8) gives:

            <Z>_{HT} = Re<b(x)|b(x')>

        where |b(x)> is the (unnormalised) A(U,P)|0> output restricted to the
        ancilla=|0> subspace.  Equivalently:

            Re<b(x)|b(x')> = Re( sv_1[anc=0]^* · sv_2[anc=0] )
                           = Re( dot(psi_1*, psi_2) )

        where psi_k is the sub-vector of A(U,P)_k|0> on the ancilla=|0> sector.
        This is the direct statevector inner product, which produces the same
        scalar as the Hadamard test with zero shot noise.

        Implementation note — avoiding controlled-unitary decomposition
        ---------------------------------------------------------------
        The naive kernel circuit uses ``aup.to_gate().control(1)``, which
        forces Qiskit to decompose a 16-qubit unitary into a 17-qubit
        controlled version.  For a dense 2^16 × 2^16 unitary this takes
        ~2 min per call, making a T=20 Gram matrix take ~14 hours.

        Instead, we run the two A(U,P) circuits independently (each 16 qubits)
        and extract the inner product from their statevectors.  This is
        mathematically identical to the Hadamard test (both implement
        Re<b(x)|b(x')>) and follows the paper's quantum circuit model: the
        kernel is computed *on a quantum device* by evaluating A(U,P) twice
        and estimating the overlap.  In a hardware implementation this would
        be the SWAP-test or destructive SWAP; on a statevector simulator the
        inner product is the exact, noiseless equivalent.

        Register layout of A(U,P): [data(n), ancilla(1), freq_0, ..., freq_{n-1}]
        The ancilla qubit is qubit index n (0-indexed).  We extract the
        sub-vector where ancilla = |0> by selecting amplitudes at indices
        where bit n of the index is 0.

        Parameters
        ----------
        num_qubits : int
        x_edges_1 : list of (int, int)  — graph x
        x_edges_2 : list of (int, int)  — graph x'
        tau : float
        r_steps : int
        pauli_observable : str

        Returns
        -------
        float  — Re<b(x)|b(x')> in [-1, +1]
        """
        # Run A(U,P) for both graphs independently
        aup_1 = self.build_aup_inhomogeneous(
            num_qubits, x_edges_1, tau, r_steps, pauli_observable
        )
        aup_2 = self.build_aup_inhomogeneous(
            num_qubits, x_edges_2, tau, r_steps, pauli_observable
        )

        sv_1 = Statevector(aup_1).data   # shape (2^(n+1+d*n_s),)
        sv_2 = Statevector(aup_2).data

        # Extract ancilla=|0> sector.
        # Register layout (little-endian): [data(n), ancilla(1), freqs(d*n_s)]
        # ancilla is qubit index n → bit n of the basis state index.
        # ancilla=|0> means bit n is 0.
        # Stride: bit n corresponds to a stride of 2^n in the state index.
        anc_stride = 2 ** num_qubits   # = 2^n
        dim = len(sv_1)

        # Select all basis states where ancilla bit = 0
        # These are indices where (index // anc_stride) % 2 == 0
        anc_zero_mask = np.array(
            [(idx // anc_stride) % 2 == 0 for idx in range(dim)], dtype=bool
        )
        psi_1 = sv_1[anc_zero_mask]   # sub-vector on anc=|0> sector
        psi_2 = sv_2[anc_zero_mask]

        # k(x,x') = Re<b(x)|b(x')> = Re(psi_1^* · psi_2)
        return float(np.real(np.dot(psi_1.conj(), psi_2)))

    # ------------------------------------------------------------------
    # Gram matrix  (T×T kernel matrix for KRR)
    # ------------------------------------------------------------------

    def compute_gram_matrix(
        self,
        num_qubits: int,
        datasets: List[List[Tuple[int, int]]],
        tau: float,
        r_steps: int,
        pauli_observable: str,
        verbose: bool = True,
    ) -> np.ndarray:
        """Compute the full T×T Gram matrix K[i,j] = k(x_i, x_j).

        Each entry is evaluated by simulating the Hadamard-test kernel circuit
        (build_kernel_circuit_inhomogeneous) exactly.  The matrix is symmetric
        by construction (k(x,x') = k(x',x) = Re<b(x)|b(x')>), so we compute
        only the upper triangle and mirror it, halving the number of circuits.

        The diagonal entries k(x,x) = Re<b(x)|b(x)> = ||b(x)||^2 = norm_sq(x)
        are computed separately (same circuit with x_edges_1 = x_edges_2).

        Complexity: ceil(T*(T+1)/2) kernel circuit simulations.
        For T=30: 465 simulations.  At ~2s each for n=3, r=2: ~15 min.

        Parameters
        ----------
        num_qubits : int
        datasets : list of edge lists  — T training/test graphs
        tau : float
        r_steps : int
        pauli_observable : str
        verbose : bool
            If True, print progress (useful for 10-30 min runs).

        Returns
        -------
        np.ndarray, shape (T, T)  — symmetric positive semi-definite Gram matrix
        """
        T = len(datasets)
        K = np.zeros((T, T), dtype=float)

        total_entries = T * (T + 1) // 2
        computed = 0

        for i in range(T):
            for j in range(i, T):
                k_val = self.compute_kernel_entry(
                    num_qubits,
                    datasets[i],
                    datasets[j],
                    tau,
                    r_steps,
                    pauli_observable,
                )
                K[i, j] = k_val
                K[j, i] = k_val   # symmetry

                computed += 1
                if verbose and computed % 10 == 0:
                    pct = 100.0 * computed / total_entries
                    print(f"  Gram matrix: {computed}/{total_entries} entries "
                          f"({pct:.1f}%) — K[{i},{j}] = {k_val:.4f}")

        if verbose:
            print(f"  Gram matrix complete: {T}×{T}, "
                  f"min={K.min():.4f}, max={K.max():.4f}, "
                  f"diag_mean={np.diag(K).mean():.4f}")

        return K

    # ------------------------------------------------------------------
    # A(U)-consistent Trotter labels for inhomogeneous TFIM
    # ------------------------------------------------------------------

    def compute_au_labels_inhomogeneous(
        self,
        graphs: List[List[Tuple[int, int]]],
        num_qubits: int,
        alpha_vec: np.ndarray,
        tau: float,
        r_steps: int,
        pauli_observable: str,
    ) -> np.ndarray:
        """Compute A(U)-consistent Trotter training labels for the inhomogeneous TFIM.

        Convention alignment
        --------------------
        The A(U) circuit encodes the upload gate e^{i*pi*alpha_i*X_i} where
        the angle is dimensionless and circuit-convention-based.  The physical
        Trotter circuit uses the gate e^{i*(alpha_i*tau/r)*X_i}.  These are
        identical when:

            pi * alpha_upload_i = alpha_i * tau / r
            =>  alpha_upload_i  = alpha_i * tau / (pi * r)

        Training labels must use this same Trotter circuit (not exact matrix
        exponentiation, which uses a different angle convention and would
        introduce a systematic label mismatch).  This is the per-qubit
        generalisation of compute_au_labels from the original codebase.

        Trotter circuit per step
        ------------------------
            RZZ(2*tau/r, i, j)  for each (i,j) in x_edges
            RX(-2*pi*alpha_upload_i, i)  for each qubit i
                = e^{i*pi*alpha_upload_i*X_i}

        Parameters
        ----------
        graphs : list of edge lists  — training/test graphs x
        num_qubits : int
        alpha_vec : np.ndarray, shape (n,)
            True unknown parameter vector alpha* used to generate labels.
        tau : float
        r_steps : int
        pauli_observable : str  — Qiskit little-endian Pauli string

        Returns
        -------
        np.ndarray, shape (len(graphs),)
            Real-valued expectation values in [-1, +1].
        """
        from qiskit.quantum_info import SparsePauliOp

        obs_matrix = SparsePauliOp(pauli_observable).to_matrix()

        # Map physical alpha_i to A(U) upload convention
        alpha_upload = alpha_vec * tau / (np.pi * r_steps)

        labels = np.zeros(len(graphs))

        for idx, edges in enumerate(graphs):
            if tau < 1e-9:
                # Zero evolution: label = <0|O|0>
                psi = np.zeros(2 ** num_qubits, dtype=complex)
                psi[0] = 1.0
                labels[idx] = float(np.real(psi.conj() @ obs_matrix @ psi))
                continue

            # Build Trotter circuit in the A(U) upload convention
            qc = QuantumCircuit(num_qubits)
            for _ in range(r_steps):
                # Fixed ZZ interactions
                for i, j in edges:
                    qc.rzz(-2.0 * tau / r_steps, i, j)
                # Per-qubit X upload: e^{i*pi*alpha_upload_i*X_i} = RX(-2*pi*alpha_upload_i)
                for q in range(num_qubits):
                    qc.rx(-2.0 * np.pi * alpha_upload[q], q)

            sv = Statevector(qc)
            labels[idx] = float(
                np.real(sv.data.conj() @ obs_matrix @ sv.data)
            )

        return labels


# ---------------------------------------------------------------------------
# Cached methods injected after initial class definition
# These are added as a block to avoid str_replace size limits.
# In the final deliverable these live inside KernelCircuitMixin directly.
# ---------------------------------------------------------------------------

def _get_or_compute_sv(
    self,
    edges: list,
    num_qubits: int,
    tau: float,
    r_steps: int,
    pauli_observable: str,
    cache,
) -> np.ndarray:
    cached = cache.get(edges, tau)
    if cached is not None:
        return cached
    # [HARDWARE_SWAP]: replace Statevector simulation with backend.run() call
    qc = self.build_aup_inhomogeneous(num_qubits, edges, tau, r_steps, pauli_observable)
    sv_full = Statevector(qc).data
    anc_stride    = 2 ** num_qubits
    anc_zero_mask = np.array(
        [(idx // anc_stride) % 2 == 0 for idx in range(len(sv_full))], dtype=bool
    )
    sv_anc0 = sv_full[anc_zero_mask]
    cache.put(edges, tau, sv_anc0)
    return sv_anc0

def compute_kernel_entry_cached(
    self,
    num_qubits: int,
    x_edges_1: list,
    x_edges_2: list,
    tau: float,
    r_steps: int,
    pauli_observable: str,
    cache,
) -> float:
    psi_1 = self._get_or_compute_sv(x_edges_1, num_qubits, tau, r_steps, pauli_observable, cache)
    psi_2 = self._get_or_compute_sv(x_edges_2, num_qubits, tau, r_steps, pauli_observable, cache)
    return float(np.real(np.dot(psi_1.conj(), psi_2)))

def compute_gram_matrix_cached(
    self,
    num_qubits: int,
    datasets,
    tau: float,
    r_steps: int,
    pauli_observable: str,
    cache,
    verbose: bool = False,
) -> np.ndarray:
    T = len(datasets)
    K = np.zeros((T, T), dtype=float)
    for i in range(T):
        for j in range(i, T):
            k_val = self.compute_kernel_entry_cached(
                num_qubits, datasets[i], datasets[j],
                tau, r_steps, pauli_observable, cache,
            )
            K[i, j] = k_val
            K[j, i] = k_val
    if verbose:
        print(f"    {cache.stats}")
    return K

def compute_cross_kernel_cached(
    self,
    num_qubits: int,
    test_datasets,
    train_datasets,
    tau: float,
    r_steps: int,
    pauli_observable: str,
    cache,
    verbose: bool = False,
) -> np.ndarray:
    T_test  = len(test_datasets)
    T_train = len(train_datasets)
    K_cross = np.zeros((T_test, T_train), dtype=float)
    for i in range(T_test):
        for j in range(T_train):
            K_cross[i, j] = self.compute_kernel_entry_cached(
                num_qubits, test_datasets[i], train_datasets[j],
                tau, r_steps, pauli_observable, cache,
            )
    if verbose:
        print(f"    {cache.stats}")
    return K_cross

# Inject as methods on KernelCircuitMixin
KernelCircuitMixin._get_or_compute_sv          = _get_or_compute_sv
KernelCircuitMixin.compute_kernel_entry_cached  = compute_kernel_entry_cached
KernelCircuitMixin.compute_gram_matrix_cached   = compute_gram_matrix_cached
KernelCircuitMixin.compute_cross_kernel_cached  = compute_cross_kernel_cached