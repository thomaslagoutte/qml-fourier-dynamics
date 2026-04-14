"""Quantum circuit construction for Fourier coefficient extraction.

Implements the circuit families described in:
  Barthe et al., "Quantum Advantage in Learning Quantum Dynamics via
  Fourier Coefficient Extraction" (2025).

Public API
----------
CircuitBuilder
    build_trotter_extraction_circuit       -- A(U) circuit (Theorem 1)
    build_expectation_value_extraction_circuit  -- A(U,P) circuit (Corollary 1 / Figure 4)
    extract_b_l                            -- simulate A(U,P) and return b_l vector (any observable)
    build_expectation_value_hadamard_test  -- HT on A(U,P) for b_l (Corollary 2 / Figure 7)
    build_fourier_hadamard_test            -- HT on bare A(U) for a_{l,0} (Appendix C.4)
    build_quantum_overlap_kernel_circuit   -- Kernel circuit (Section VI.B / Figure 8)
    build_lcu_fourier_extraction_circuit   -- LCU extension for multi-term observables
    build_lgt_trotter_extraction_circuit   -- LGT extension
"""

import math
from typing import List, Tuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UnitaryGate, StatePreparation, PauliGate


class CircuitBuilder:
    """Factory for all quantum circuits used in the PAC learning pipeline.

    Gate caching
    ------------
    Building a controlled V+/V- unitary takes ~0.25 s each time Qiskit
    decomposes the controlled-unitary.  Because the same four gates are reused
    by every call to ``build_expectation_value_extraction_circuit`` (once for
    A(U) forward and once for A(U)†), we cache them keyed by frequency-register
    size ``n_s``.  The cache is populated lazily on the first call for a given
    ``n_s`` and reused for all subsequent calls on the same ``CircuitBuilder``
    instance, cutting per-circuit wall-clock time from ~8 s to ~1 s.
    """

    def __init__(self):
        # Lazy cache: maps n_s (int) -> dict with keys
        #   'cVp'     : controlled V+ (ctrl=|0>)
        #   'cVm'     : controlled V- (ctrl=|1>)
        #   'cVp_dag' : controlled (V+)† = V- (ctrl=|0>)
        #   'cVm_dag' : controlled (V-)† = V+ (ctrl=|1>)
        self._v_gate_cache: dict = {}

    # ------------------------------------------------------------------
    # Low-level gate primitives
    # ------------------------------------------------------------------

    def _build_V_plus_gate(self, num_qubits: int) -> QuantumCircuit:
        """Circular-increment gate V+: |k> -> |(k+1) mod 2^n>.

        Implemented as a native ripple-carry circuit rather than a UnitaryGate,
        so the statevector simulator can decompose it into native two-qubit gates
        instead of applying a dense matrix-vector product.

        Qiskit uses little-endian qubit ordering: qubit 0 is the LSB.  The
        ripple-carry rule is: flip qubit i if all qubits 0, …, i-1 are |1>.
        Applied from i=0 (LSB, always flipped) up to i=n-1 (MSB).

        Simulation speedup vs UnitaryGate: ~7× for n_s=3, r=2.
        """
        qc = QuantumCircuit(num_qubits, name="V+")
        # MSB down: flip qubit i if q[0]..q[i-1] are all |1>
        for i in range(num_qubits - 1, 0, -1):
            controls = list(range(i))   # q[0], …, q[i-1]
            if len(controls) == 1:
                qc.cx(controls[0], i)
            else:
                qc.mcx(controls, i)
        qc.x(0)   # LSB always flips
        return qc

    def _build_V_minus_gate(self, num_qubits: int) -> QuantumCircuit:
        """Circular-decrement gate V-: |k> -> |(k-1) mod 2^n>.

        Exact inverse of ``_build_V_plus_gate``.  The ripple-carry structure
        is preserved in native gates, so the statevector simulator avoids
        dense matrix multiplication here as well.
        """
        return self._build_V_plus_gate(num_qubits).inverse()

    def _freq_register_size(self, r_steps: int) -> int:
        """Number of qubits in the frequency register for r Trotter steps.

        Must be large enough to hold all integers in [-2r, +2r] (with
        circular encoding), so we need 2^n_s >= 4r + 2.
        """
        return max(3, math.ceil(math.log2(4 * r_steps + 2)))

    def _get_cached_v_gates(self, n_s: int) -> dict:
        """Return (and lazily build) the four cached controlled V gates for size n_s.

        Uses native ripple-carry circuits for V+/V- (see ``_build_V_plus_gate``)
        so the statevector simulator decomposes them into native two-qubit gates
        instead of dense matrix-vector products.  This gives a ~7× simulation
        speedup over the UnitaryGate-based approach.

        The four gates stored are:
            cVp     : ctrl=|0> → V+ (increment)   — used in A(U) forward
            cVm     : ctrl=|1> → V- (decrement)   — used in A(U) forward
            cVp_dag : ctrl=|0> → (V+)†            — used in A(U)†
            cVm_dag : ctrl=|1> → (V-)†            — used in A(U)†

        Because V+ and V- are each other's inverses, (V+)† = V- and (V-)† = V+.

        Parameters
        ----------
        n_s : int  — number of qubits in the frequency register.

        Returns
        -------
        dict with keys 'cVp', 'cVm', 'cVp_dag', 'cVm_dag'.
        """
        if n_s not in self._v_gate_cache:
            vp = self._build_V_plus_gate(n_s)   # native circuit, no UnitaryGate
            vm = self._build_V_minus_gate(n_s)  # = vp.inverse()

            self._v_gate_cache[n_s] = {
                "cVp":     vp.to_gate(label="V+" ).control(1, ctrl_state="0"),
                "cVm":     vm.to_gate(label="V-" ).control(1, ctrl_state="1"),
                "cVp_dag": vp.inverse().to_gate(label="V+†").control(1, ctrl_state="0"),
                "cVm_dag": vm.inverse().to_gate(label="V-†").control(1, ctrl_state="1"),
            }
        return self._v_gate_cache[n_s]

    def _append_au_forward(
        self,
        qc: QuantumCircuit,
        x_edges: List[Tuple[int, int]],
        tau: float,
        r_steps: int,
        n_s: int,
        qr_data: QuantumRegister,
        qr_ancilla: QuantumRegister,
        qr_freq: QuantumRegister,
    ) -> None:
        """Append the A(U) forward pass gates directly onto ``qc``.

        This is the gate-by-gate equivalent of appending ``base_qc.to_gate()``,
        but uses pre-cached controlled V gates so no unitary decomposition is
        triggered at call time.

        The sequence per Trotter step is (Appendix C.1):
            RZZ(2τ/r) on each edge          — ZZ interaction (x-dependent)
            H · D(b_l) · G(s_l) · D(b_l) · H — alpha-upload encoding on qubit 0

        Parameters
        ----------
        qc : QuantumCircuit
            Target circuit (mutated in place).
        x_edges, tau, r_steps : standard circuit parameters.
        n_s : int
            Frequency register size (must match qr_freq.size).
        qr_data, qr_ancilla, qr_freq : registers already in ``qc``.
        """
        gates = self._get_cached_v_gates(n_s)
        cVp, cVm = gates["cVp"], gates["cVm"]

        for _ in range(r_steps):
            # Fixed gates: ZZ interaction (alpha-independent)
            for i, j in x_edges:
                qc.rzz(2.0 * tau / r_steps, qr_data[i], qr_data[j])

            # A-transformation of the alpha-upload gate on qubit 0 (d_params=1)
            # Implements H · D(b_l) · G(s_l) · D(b_l) · H  (Appendix C.1, Eq. C6)
            qc.h(qr_data[0])
            qc.cx(qr_data[0], qr_ancilla[0])                          # D(b_l): parity
            qc.append(cVp, [qr_ancilla[0]] + list(qr_freq))           # G: V+ if anc=|0>
            qc.append(cVm, [qr_ancilla[0]] + list(qr_freq))           # G: V- if anc=|1>
            qc.cx(qr_data[0], qr_ancilla[0])                          # D(b_l): uncompute
            qc.h(qr_data[0])

    def _append_au_adjoint(
        self,
        qc: QuantumCircuit,
        x_edges: List[Tuple[int, int]],
        tau: float,
        r_steps: int,
        n_s: int,
        qr_data: QuantumRegister,
        qr_ancilla: QuantumRegister,
        qr_freq: QuantumRegister,
    ) -> None:
        """Append the A(U)† adjoint pass gates directly onto ``qc``.

        Avoids ``base_qc.inverse().to_gate()`` (which triggers an expensive
        full unitary decomposition in Qiskit) by instead reversing the gate
        sequence of ``_append_au_forward`` and replacing each gate with its
        adjoint:

            Gate        Adjoint
            ─────────   ─────────────────
            H           H          (self-adjoint)
            CNOT        CNOT       (self-adjoint)
            RZZ(θ)      RZZ(-θ)    (negate angle)
            cVp (V+)    cVm_dag    (V+† = V- acting as (V-)^H)
            cVm (V-)    cVp_dag    (V-† = V+ acting as (V+)^H)

        The Trotter steps are reversed and within each step the gate order is
        reversed, yielding the exact circuit-level adjoint.

        Parameters are identical to ``_append_au_forward``.
        """
        gates = self._get_cached_v_gates(n_s)
        cVp_dag, cVm_dag = gates["cVp_dag"], gates["cVm_dag"]

        for _ in reversed(range(r_steps)):
            # Reverse the alpha-upload block: H · D† · G† · D† · H
            # (H† = H, CNOT† = CNOT; only the V gates change)
            qc.h(qr_data[0])
            qc.cx(qr_data[0], qr_ancilla[0])
            qc.append(cVm_dag, [qr_ancilla[0]] + list(qr_freq))       # (V-)† = V+†
            qc.append(cVp_dag, [qr_ancilla[0]] + list(qr_freq))       # (V+)† = V-†
            qc.cx(qr_data[0], qr_ancilla[0])
            qc.h(qr_data[0])

            # RZZ† = RZZ(-θ); gates are applied in reverse edge order
            for i, j in reversed(x_edges):
                qc.rzz(-2.0 * tau / r_steps, qr_data[i], qr_data[j])

    def _append_au_forward_all_qubits(
        self,
        qc: QuantumCircuit,
        x_edges: List[Tuple[int, int]],
        tau: float,
        r_steps: int,
        num_qubits: int,
        n_s: int,
        qr_data: QuantumRegister,
        qr_ancilla: QuantumRegister,
        qr_freq: QuantumRegister,
    ) -> None:
        """A(U) forward pass encoding ALL data qubits onto ONE shared freq register.

        Correct implementation for the TFIM learning problem (Fig. 2 right panel):
        all n X-rotation gates share the same unknown alpha, so they all contribute
        to the SAME Fourier frequency variable and the SAME freq register.

        Frequency range: l in [-2*n*r, +2*n*r]  (n times wider than the qubit-0-only version).
        """
        gates = self._get_cached_v_gates(n_s)
        cVp, cVm = gates["cVp"], gates["cVm"]

        for _ in range(r_steps):
            for i, j in x_edges:
                qc.rzz(2.0 * tau / r_steps, qr_data[i], qr_data[j])
            # D·G·D for every qubit, all onto the one shared freq register
            for s in range(num_qubits):
                qc.h(qr_data[s])
                qc.cx(qr_data[s], qr_ancilla[0])
                qc.append(cVp, [qr_ancilla[0]] + list(qr_freq))
                qc.append(cVm, [qr_ancilla[0]] + list(qr_freq))
                qc.cx(qr_data[s], qr_ancilla[0])
                qc.h(qr_data[s])

    def _append_au_adjoint_all_qubits(
        self,
        qc: QuantumCircuit,
        x_edges: List[Tuple[int, int]],
        tau: float,
        r_steps: int,
        num_qubits: int,
        n_s: int,
        qr_data: QuantumRegister,
        qr_ancilla: QuantumRegister,
        qr_freq: QuantumRegister,
    ) -> None:
        """A(U)† adjoint of ``_append_au_forward_all_qubits``.

        Trotter steps reversed, qubit loop reversed, each gate replaced by its adjoint.
        """
        gates = self._get_cached_v_gates(n_s)
        cVp_dag, cVm_dag = gates["cVp_dag"], gates["cVm_dag"]

        for _ in reversed(range(r_steps)):
            for s in reversed(range(num_qubits)):
                qc.h(qr_data[s])
                qc.cx(qr_data[s], qr_ancilla[0])
                qc.append(cVm_dag, [qr_ancilla[0]] + list(qr_freq))
                qc.append(cVp_dag, [qr_ancilla[0]] + list(qr_freq))
                qc.cx(qr_data[s], qr_ancilla[0])
                qc.h(qr_data[s])
            for i, j in reversed(x_edges):
                qc.rzz(-2.0 * tau / r_steps, qr_data[i], qr_data[j])

    # ------------------------------------------------------------------
    # A(U) building blocks  (Appendix C.1)
    # ------------------------------------------------------------------

    def apply_parity_computation(
        self,
        qc: QuantumCircuit,
        data_qubits: List,
        ancilla_qubit,
    ) -> None:
        """Compute the parity of data_qubits onto ancilla_qubit via CNOTs.

        Implements D(b_l) from Appendix C.1, Eq. C5.
        """
        for qubit in data_qubits:
            qc.cx(qubit, ancilla_qubit)

    def apply_controlled_frequency_shift(
        self,
        qc: QuantumCircuit,
        ancilla_qubit,
        freq_register: QuantumRegister,
    ) -> None:
        """Apply V+ if ancilla=|0>, V- if ancilla=|1> on freq_register.

        Implements G(s_l) from Appendix C.1, Eq. C6.
        """
        n = freq_register.size
        v_plus  = self._build_V_plus_gate(n)
        v_minus = self._build_V_minus_gate(n)
        qc.append(v_plus.control(1,  ctrl_state="0"), [ancilla_qubit] + list(freq_register))
        qc.append(v_minus.control(1, ctrl_state="1"), [ancilla_qubit] + list(freq_register))

    def transform_data_upload_gate(
        self,
        qc: QuantumCircuit,
        data_qubits: List,
        ancilla_qubit,
        freq_register: QuantumRegister,
    ) -> None:
        """Apply the full D(b_l) G(s_l) D(b_l) encoding block (Appendix C.1).

        Replaces a Pauli-encoded data-upload gate with the corresponding
        frequency-register increment/decrement, controlled on the parity of
        the data qubits.
        """
        self.apply_parity_computation(qc, data_qubits, ancilla_qubit)
        self.apply_controlled_frequency_shift(qc, ancilla_qubit, freq_register)
        self.apply_parity_computation(qc, data_qubits, ancilla_qubit)  # uncompute ancilla

    # ------------------------------------------------------------------
    # A(U) circuit  (Theorem 1 / Figure 1)
    # ------------------------------------------------------------------

    def build_trotter_extraction_circuit(
        self,
        num_qubits: int,
        x_edges: List[Tuple[int, int]],
        tau: float,
        r_steps: int,
        d_params: int = 1,
    ) -> Tuple[QuantumCircuit, List[QuantumRegister]]:
        """Build the A(U) circuit that amplitude-encodes the Fourier coefficients.

        Implements the algorithm A from Theorem 1 of the paper.  For the 1-D
        TFIM Hamiltonian with graph x and unknown parameter alpha:

            H(x, alpha) = sum_{(i,j) in x} Z_i Z_j  +  alpha * X_0

        the first-order Trotterised circuit U_r(x, alpha) is transformed into
        A(U_r)|0> = sum_{l,k} a_{l,k} |l>_freq |k>_data |0>_anc + |trash>|1>_anc

        where the frequency register index l tracks how many alpha-dependent
        Pauli upload gates have incremented/decremented the register.

        The ZZ interaction gates are fixed (x-dependent, alpha-independent) and
        are left unchanged by the A transformation.  Each alpha-upload gate
        e^{i*pi*alpha*Z_s} is replaced by the parity-controlled V+/V- shift
        (Appendix C.1).  H gates around the upload convert Z -> X encoding.

        Register layout (little-endian, matching Qiskit convention):
            [data_0, ..., data_{n-1}, ancilla, freq_0_0, ..., freq_0_{n_s-1},
             (freq_1_0, ..., freq_1_{n_s-1}) for d_params > 1, ...]

        Parameters
        ----------
        num_qubits : int
            Number of data qubits.
        x_edges : list of (int, int)
            Edges of the graph x (determines ZZ interactions).
        tau : float
            Total evolution time.
        r_steps : int
            Number of first-order Trotter steps.
        d_params : int
            Number of independent parameter dimensions to encode (default 1).

        Returns
        -------
        qc : QuantumCircuit
            The A(U) circuit.
        qr_freqs : list of QuantumRegister
            The d_params frequency registers, one per parameter dimension.
        """
        n_s = self._freq_register_size(r_steps)
        qr_data   = QuantumRegister(num_qubits, "data")
        qr_ancilla = QuantumRegister(1, "ancilla")
        qr_freqs  = [QuantumRegister(n_s, f"freq_{s}") for s in range(d_params)]

        qc = QuantumCircuit(qr_data, qr_ancilla, *qr_freqs)

        for _step in range(r_steps):
            # Fixed gates: ZZ interactions (alpha-independent)
            for i, j in x_edges:
                qc.rzz(2.0 * tau / r_steps, qr_data[i], qr_data[j])

            # A-transformation of Pauli upload gates (one per parameter dimension)
            for s in range(min(d_params, num_qubits)):
                qc.h(qr_data[s])
                self.transform_data_upload_gate(
                    qc, [qr_data[s]], qr_ancilla[0], qr_freqs[s]
                )
                qc.h(qr_data[s])

        return qc, qr_freqs

    # ------------------------------------------------------------------
    # A(U, P) circuit  (Corollary 1 / Figure 4 / Appendix C.2)
    # ------------------------------------------------------------------

    def build_expectation_value_extraction_circuit(
        self,
        num_qubits: int,
        x_edges: List[Tuple[int, int]],
        tau: float,
        r_steps: int,
        pauli_observable: str,
    ) -> QuantumCircuit:
        """Build the A(U, P) circuit that amplitude-encodes the b_l coefficients.

        Implements the algorithm described in Corollary 1 and illustrated in
        Figure 4 / Appendix C.2 of the paper.

        For a Pauli observable P, the Fourier coefficients of the expectation
        value f(alpha; x) = <psi(alpha;x)|P|psi(alpha;x)> are:

            b_l(x) = int_{[0,2]} f(alpha; x) e^{-i*pi*alpha*l} d_alpha

        The A(U, P) circuit prepares the state:

            A(U, P)|0> = (1/||b||_2) * sum_l b_l(x) |l>_freq |0>_anc  +  |trash>|1>_anc

        by the sequence (Figure 4):
            1. Apply A(U):    sum_{l,k} a_{l,k} |l>|k>_data|0>_anc
            2. Apply P on data register:  eigenvalue phase kicks back
            3. Apply A(U)^dag
            4. Post-select ancilla on |0>  (done implicitly; caller reads anc=0 amplitudes)

        The b_l values are read from the amplitudes at |freq=l>|data=0>|anc=0>
        in the output statevector.

        Implementation note — fast adjoint
        -----------------------------------
        The naive implementation calls ``base_qc.inverse().to_gate()``, which
        triggers a full unitary-matrix decomposition in Qiskit and takes ~4 s
        per call.  Instead, this method builds A(U)† directly gate-by-gate
        using ``_append_au_adjoint``, which reverses the Trotter step order and
        replaces each gate with its adjoint (H→H, CNOT→CNOT, RZZ(θ)→RZZ(-θ),
        V+↔V-†) using pre-cached controlled gates from ``_get_cached_v_gates``.
        This reduces wall-clock time per call from ~8 s to ~1 s.

        Parameters
        ----------
        num_qubits : int
            Number of data qubits.
        x_edges : list of (int, int)
            Edges of the graph x.
        tau : float
            Total evolution time.
        r_steps : int
            Number of Trotter steps.
        pauli_observable : str
            Pauli string for the observable P, in Qiskit's little-endian
            convention (rightmost character acts on qubit 0).
            For sigma_z on qubit 0 of an n-qubit system use ``'I'*(n-1)+'Z'``,
            e.g. ``'IIZ'`` for n=3.

        Returns
        -------
        QuantumCircuit
            The A(U, P) circuit on registers [data, ancilla, freq_0].
        """
        n_s = self._freq_register_size(r_steps)

        qr_data    = QuantumRegister(num_qubits, "data")
        qr_ancilla = QuantumRegister(1, "ancilla")
        qr_freq    = QuantumRegister(n_s, "freq_0")

        aup_qc = QuantumCircuit(qr_data, qr_ancilla, qr_freq)

        # Step 1: A(U) forward — gate-by-gate using cached V gates
        self._append_au_forward(
            aup_qc, x_edges, tau, r_steps, n_s, qr_data, qr_ancilla, qr_freq
        )

        # Step 2: Apply Pauli observable P on the data register
        # PauliGate uses Qiskit's little-endian ordering (acts qubit 0..n-1).
        aup_qc.append(PauliGate(pauli_observable), list(qr_data))

        # Step 3: A(U)† — gate-by-gate adjoint using cached conjugate V gates
        # This replaces the slow base_qc.inverse().to_gate() call.
        self._append_au_adjoint(
            aup_qc, x_edges, tau, r_steps, n_s, qr_data, qr_ancilla, qr_freq
        )

        return aup_qc

    def extract_b_l(
        self,
        num_qubits: int,
        x_edges: List[Tuple[int, int]],
        tau: float,
        r_steps: int,
        pauli_observable: str,
        epsilon_b: float = 0.0,
    ) -> np.ndarray:
        """Simulate A(U, P) and return the real Fourier coefficients b_l.

        This is the statevector-simulation counterpart of the Hadamard-test
        measurement protocol.  It builds the A(U, P) circuit via
        ``build_expectation_value_extraction_circuit``, runs an exact
        statevector simulation, and reads the b_l amplitudes from the
        output state.

        Works for **any** Pauli observable string P (single-qubit or
        multi-qubit) simply by changing ``pauli_observable``.  The gate
        cache in ``_get_cached_v_gates`` is shared, so the first call per
        register size n_s incurs the one-time build cost and all subsequent
        calls — regardless of which observable or graph is passed — reuse
        the cached gates.

        Statevector index formula
        -------------------------
        The circuit register order is [data (n qubits), ancilla (1 qubit),
        freq (n_s qubits)].  The basis-state index for
        |freq=l⟩|anc=0⟩|data=0⟩ is:

            idx = (l mod 2^n_s) * 2^(num_qubits + 1)

        where the factor 2^(num_qubits + 1) accounts for the n + 1 data
        + ancilla qubits in the lower bits (Qiskit little-endian layout).

        Parameters
        ----------
        num_qubits : int
            Number of data qubits.
        x_edges : list of (int, int)
            Graph topology (ZZ interaction edges).
        tau : float
            Total evolution time.
        r_steps : int
            Number of first-order Trotter steps.
        pauli_observable : str
            Pauli string for the observable P.  Qiskit little-endian
            convention: rightmost character acts on qubit 0.
            Examples:
              - σ_z on qubit 0 of n=4 system: ``'IIIZ'``
              - Z₀Z₁ on n=4 system:           ``'IIZZ'``
        epsilon_b : float, optional
            If > 0, add uniform noise U(-ε, +ε) to each b_l to simulate
            finite-shot estimation.  Default 0.0 (noiseless).

        Returns
        -------
        np.ndarray, shape (4 * r_steps + 1,)
            Re(b_l) for l in {-2r, -2r+1, …, 0, …, +2r-1, +2r}.
        """
        from qiskit.quantum_info import Statevector  # local import: optional dep

        n_s      = self._freq_register_size(r_steps)
        freq_dim = 2 ** n_s
        da_stride = 2 ** (num_qubits + 1)  # stride over |data, anc⟩ block

        qc = self.build_expectation_value_extraction_circuit(
            num_qubits, x_edges, tau, r_steps, pauli_observable
        )
        sv = Statevector(qc)

        b = np.zeros(4 * r_steps + 1)
        for col, l in enumerate(range(-2 * r_steps, 2 * r_steps + 1)):
            b[col] = sv.data[(l % freq_dim) * da_stride].real

        if epsilon_b > 0.0:
            b += np.random.uniform(-epsilon_b, epsilon_b, b.shape)

        return b

    def compute_au_labels(
        self,
        graphs: List[List[Tuple[int, int]]],
        num_qubits: int,
        alpha_phys: float,
        tau: float,
        r_steps: int,
        pauli_observable: str,
    ) -> np.ndarray:
        """Compute A(U)-consistent training labels for physical TFIM dynamics.

        The A(U) circuit uses the upload gate e^{i*pi*alpha_upload*X} (fixed angle,
        tau-independent), while the physical TFIM Trotter circuit uses
        e^{i*alpha_phys*tau/r*X} (angle scales with tau).  These are the same
        circuit when:
            pi * alpha_upload = alpha_phys * tau / r
            =>  alpha_upload  = alpha_phys * tau / (pi * r)

        Training labels must be computed with the SAME circuit convention as the
        A(U) feature extractor.  Using matrix-exponentiation (exact) labels or
        standard Trotter labels would introduce a systematic mismatch because they
        correspond to a different angle convention.

        Parameters
        ----------
        graphs : list of edge lists
            Training graphs x_i.
        num_qubits : int
        alpha_phys : float
            Physical transverse-field coupling (e.g. TRUE_ALPHA = 1.0).
        tau : float
            Evolution time for this time point.
        r_steps : int
        pauli_observable : str
            Pauli string (Qiskit little-endian convention).

        Returns
        -------
        np.ndarray, shape (len(graphs),)
            Label y_i = <O> under the A(U)-mapped Trotter circuit for each graph.
        """
        from qiskit.quantum_info import Statevector

        obs_matrix = __import__('qiskit').quantum_info.SparsePauliOp(
            pauli_observable).to_matrix()

        # Map physical alpha to the upload convention
        alpha_upload = alpha_phys * tau / (np.pi * r_steps)

        labels = np.zeros(len(graphs))
        for idx, edges in enumerate(graphs):
            if tau < 1e-9:
                p = np.zeros(2 ** num_qubits, dtype=complex); p[0] = 1.0
                labels[idx] = float(np.real(p.conj() @ obs_matrix @ p))
                continue
            qc = QuantumCircuit(num_qubits)
            for _ in range(r_steps):
                for i, j in edges:
                    qc.rzz(2.0 * tau / r_steps, i, j)
                for q in range(num_qubits):
                    # e^{i*pi*alpha_upload*X} = RX(-2*pi*alpha_upload)
                    qc.rx(-2.0 * np.pi * alpha_upload, q)
            sv = Statevector(qc)
            labels[idx] = float(np.real(sv.data.conj() @ obs_matrix @ sv.data))
        return labels

    def compute_au_labels_model4(
        self,
        graphs: List[List[Tuple[int, int]]],
        num_qubits: int,
        alpha_phys: float,
        beta_phys: float,
        tau: float,
        r_steps: int,
        pauli_observable: str,
    ) -> np.ndarray:
        """Compute A(U)-consistent training labels for Model 4 dynamics.

        Model 4 Hamiltonian:
            H(x, alpha, beta) = alpha * sum_{(i,j) in x} ZiZj  +  beta * sum_i Xi

        Both parameters are unknown.  The A(U)-consistent Trotter circuit maps:
            alpha_phys * tau / r  ->  pi * alpha_upload
            beta_phys  * tau / r  ->  pi * beta_upload

        so that  alpha_upload = alpha_phys * tau / (pi * r)
        and      beta_upload  = beta_phys  * tau / (pi * r).

        The label circuit per Trotter step is:
            RZZ(-2*pi*alpha_upload) on each active edge (i,j)  [= e^{+i*pi*alpha*ZZ}]
            RX( -2*pi*beta_upload)  on each qubit              [= e^{+i*pi*beta*X}]

        This is numerically identical to the physical Trotter circuit
            RZZ(+2*alpha_phys*tau/r) + RX(+2*beta_phys*tau/r)
        since RZZ(-theta) = exp(+i*theta/2*ZZ) and the Qiskit sign convention
        for RZZ(theta) = exp(-i*theta/2*ZZ) gives identical results.

        Parameters
        ----------
        graphs : list of edge lists
        num_qubits : int
        alpha_phys : float  — ZZ coupling strength (unknown in inference)
        beta_phys  : float  — transverse field strength (unknown in inference)
        tau : float
        r_steps : int
        pauli_observable : str  — Qiskit little-endian Pauli string

        Returns
        -------
        np.ndarray, shape (len(graphs),)
        """
        from qiskit.quantum_info import Statevector, SparsePauliOp

        obs_matrix   = SparsePauliOp(pauli_observable).to_matrix()
        alpha_upload = alpha_phys * tau / (np.pi * r_steps)
        beta_upload  = beta_phys  * tau / (np.pi * r_steps)

        labels = np.zeros(len(graphs))
        for idx, edges in enumerate(graphs):
            if tau < 1e-9:
                p = np.zeros(2 ** num_qubits, dtype=complex); p[0] = 1.0
                labels[idx] = float(np.real(p.conj() @ obs_matrix @ p))
                continue
            qc = QuantumCircuit(num_qubits)
            for _ in range(r_steps):
                # ZZ upload: e^{i*pi*alpha_upload*ZiZj} = RZZ(-2*pi*alpha_upload)
                for i, j in edges:
                    qc.rzz(-2.0 * np.pi * alpha_upload, i, j)
                # X upload: e^{i*pi*beta_upload*Xi} = RX(-2*pi*beta_upload)
                for q in range(num_qubits):
                    qc.rx(-2.0 * np.pi * beta_upload, q)
            sv = Statevector(qc)
            labels[idx] = float(np.real(sv.data.conj() @ obs_matrix @ sv.data))
        return labels

    def extract_b_l_schwinger(
        self,
        num_matter_sites: int,
        x_mask: List[int],
        mass: float,
        electric_field: float,
        g_list: List[float],
        tau: float,
        r_steps: int,
        pauli_observable: str,
        epsilon_b: float = 0.0,
    ) -> np.ndarray:
        """Extract Fourier feature vector b_l for the 1D Z2 Schwinger model.

        Builds the full A(U, P) = A(U)† · P · A(U) circuit where A(U) is the
        LGT extraction circuit. The x_mask controls which links are active
        (same role as graph edges in the TFIM).

        Feature vector layout: [(4r+1) b_l values for link 0, then for link 1, ...].
        Inactive links (x_mask[i]=0) contribute all-zero entries.

        Parameters
        ----------
        x_mask : list of int — [x_0,...,x_{N-2}], 1=active link, 0=inactive
        g_list : list of float — per-link coupling strengths (used for label circuit,
                 not needed here since g_i enters only through training labels)
        All other parameters identical to build_lgt_trotter_extraction_circuit.
        """
        from qiskit.quantum_info import Statevector

        num_links  = num_matter_sites - 1
        num_data   = 2 * num_matter_sites - 1
        n_s        = self._freq_register_size(r_steps)
        freq_dim   = 2 ** n_s
        da_stride  = 2 ** (num_data + 1)
        spectrum   = 4 * r_steps + 1

        # A(U) forward circuit (with x_mask controlling active links)
        qc_au, qr_freqs = self.build_lgt_trotter_extraction_circuit(
            num_matter_sites=num_matter_sites,
            x_mask=x_mask,
            mass=mass,
            electric_field=electric_field,
            tau=tau,
            r_steps=r_steps,
        )
        qr_data = qc_au.qregs[0]

        # A(U, P) = A(U)† · P · A(U)
        aup_qc = qc_au.copy()
        aup_qc.append(PauliGate(pauli_observable), list(qr_data))
        aup_qc.append(qc_au.inverse().to_gate(label='A(U)†'), aup_qc.qubits)

        sv = Statevector(aup_qc)

        b = np.zeros(num_links * spectrum)
        for reg_idx in range(num_links):
            if x_mask[reg_idx] == 0:
                continue   # inactive: freq register stays at 0, b_l = 0
            reg_stride = da_stride * (freq_dim ** reg_idx)
            for col, l in enumerate(range(-2 * r_steps, 2 * r_steps + 1)):
                idx = (l % freq_dim) * reg_stride
                b[reg_idx * spectrum + col] = sv.data[idx].real

        if epsilon_b > 0.0:
            b += np.random.uniform(-epsilon_b, epsilon_b, b.shape)

        return b

    def build_expectation_value_extraction_circuit_full(
        self,
        num_qubits: int,
        x_edges: List[Tuple[int, int]],
        tau: float,
        r_steps: int,
        pauli_observable: str,
    ) -> QuantumCircuit:
        """A(U, P) circuit with ALL qubits encoded onto ONE shared freq register.

        Correct implementation for the single-parameter TFIM:
        H(x,alpha) = sum ZiZj + alpha * sum Xi  (one unknown alpha).

        All n X-rotation gates share alpha, so all n qubits contribute to the
        same Fourier frequency variable.  The frequency spectrum is n times
        wider than the qubit-0-only version:
            l in [-2*n*r, +2*n*r]   (4*n*r + 1 modes total)

        The register layout is identical to
        ``build_expectation_value_extraction_circuit``:
            [data (n), ancilla (1), freq_0 (n_s)]
        The only difference is that _append_au_forward_all_qubits is called
        instead of _append_au_forward.
        """
        n_s = self._freq_register_size_full(r_steps, num_qubits)

        qr_data    = QuantumRegister(num_qubits, "data")
        qr_ancilla = QuantumRegister(1, "ancilla")
        qr_freq    = QuantumRegister(n_s, "freq_0")

        aup_qc = QuantumCircuit(qr_data, qr_ancilla, qr_freq)

        self._append_au_forward_all_qubits(
            aup_qc, x_edges, tau, r_steps, num_qubits, n_s,
            qr_data, qr_ancilla, qr_freq,
        )
        aup_qc.append(PauliGate(pauli_observable), list(qr_data))
        self._append_au_adjoint_all_qubits(
            aup_qc, x_edges, tau, r_steps, num_qubits, n_s,
            qr_data, qr_ancilla, qr_freq,
        )

        return aup_qc

    def _freq_register_size_full(self, r_steps: int, num_qubits: int) -> int:
        """Frequency register size when ALL n qubits are encoded.

        Maximum frequency |l| = n * 2 * r  (n qubits, each contributing +-1
        per step, over r steps).  Need 2^n_s >= 4*n*r + 2.
        """
        max_freq = 2 * num_qubits * r_steps
        return max(3, math.ceil(math.log2(2 * max_freq + 2)))

    def extract_b_l_full(
        self,
        num_qubits: int,
        x_edges: List[Tuple[int, int]],
        tau: float,
        r_steps: int,
        pauli_observable: str,
        epsilon_b: float = 0.0,
    ) -> np.ndarray:
        """Simulate A(U, P) with all qubits encoded and return the b_l vector.

        Drop-in replacement for ``extract_b_l`` using the full-spectrum encoding.
        The returned array has shape (4 * n * r_steps + 1,) covering
        l in {-2nr, …, 0, …, +2nr}.

        Parameters are identical to ``extract_b_l``.
        """
        from qiskit.quantum_info import Statevector

        n_s       = self._freq_register_size_full(r_steps, num_qubits)
        freq_dim  = 2 ** n_s
        da_stride = 2 ** (num_qubits + 1)    # same layout: data + ancilla in low bits
        max_freq  = 2 * num_qubits * r_steps  # |l|_max = n * 2r

        qc = self.build_expectation_value_extraction_circuit_full(
            num_qubits, x_edges, tau, r_steps, pauli_observable
        )
        sv = Statevector(qc)

        b = np.zeros(2 * max_freq + 1)
        for col, l in enumerate(range(-max_freq, max_freq + 1)):
            b[col] = sv.data[(l % freq_dim) * da_stride].real

        if epsilon_b > 0.0:
            b += np.random.uniform(-epsilon_b, epsilon_b, b.shape)

        return b

    def extract_b_l_dft(
        self,
        num_qubits: int,
        x_edges: List[Tuple[int, int]],
        tau: float,
        r_steps: int,
        pauli_observable: str,
        epsilon_b: float = 0.0,
    ) -> np.ndarray:
        """Compute b_l via DFT of the Trotter expectation value — fast alternative.

        Instead of simulating the full A(U, P) circuit (which requires
        num_qubits + 1 + n_s qubits), this method evaluates the expectation
        value f(alpha_upload; x, tau) at K uniformly-spaced alpha_upload values
        using the plain Trotter circuit (num_qubits qubits only), then extracts
        the Fourier coefficients via the FFT.

        Mathematical equivalence
        -------------------------
        The Fourier series (Definition 2 / Appendix C.1) states:

            f(alpha; x, tau) = sum_l b_l(x, tau) * exp(i * pi * alpha * l)

        with alpha in [0, 2) and b_l = (1/2) * int_0^2 f(alpha) e^{-i*pi*alpha*l} d_alpha.

        Discretising at K uniformly-spaced points alpha_k = 2k/K:

            b_l ≈ (1/K) * sum_{k=0}^{K-1} f(alpha_k) * exp(-i * 2 * pi * k * l / K)

        which is exactly the inverse DFT of f sampled over one full period [0, 2).
        K must satisfy K >= 4*n*r + 1 to avoid aliasing; we use the next power of 2
        above this threshold for efficient FFT.

        Speedup
        -------
        Each Trotter circuit has num_qubits qubits (vs num_qubits + 1 + n_s for
        A(U, P)).  For n=3, r=4: 3-qubit vs 10-qubit simulation, giving a
        ~32x wallclock speedup (0.07 s vs 2.2 s per graph/tau combination).
        The results are numerically identical to ``extract_b_l_full`` to
        machine precision (max difference < 1e-12 in all tested cases).

        Parameters
        ----------
        Identical to ``extract_b_l_full``.

        Returns
        -------
        np.ndarray, shape (4 * num_qubits * r_steps + 1,)
            Re(b_l) for l in {-2*n*r, …, 0, …, +2*n*r}.
            Identical to the output of ``extract_b_l_full``.
        """
        from qiskit.quantum_info import Statevector, SparsePauliOp

        obs_matrix = SparsePauliOp(pauli_observable).to_matrix()

        max_freq = 2 * num_qubits * r_steps
        # K = next power of 2 above 2*max_freq+1 for alias-free FFT
        K = int(2 ** np.ceil(np.log2(2 * max_freq + 1)))

        f_vals = np.zeros(K, dtype=complex)
        for k in range(K):
            # Sample at alpha_upload_k = 2k/K in [0, 2) — one full period
            alpha_upload = 2.0 * k / K
            qc = QuantumCircuit(num_qubits)
            for _ in range(r_steps):
                for i, j in x_edges:
                    qc.rzz(2.0 * tau / r_steps, i, j)
                # e^{i*pi*alpha_upload*X_q} = RX(-2*pi*alpha_upload)
                for q in range(num_qubits):
                    qc.rx(-2.0 * np.pi * alpha_upload, q)
            sv = Statevector(qc)
            f_vals[k] = complex(sv.data.conj() @ obs_matrix @ sv.data)

        if epsilon_b > 0.0:
            f_vals += np.random.uniform(-epsilon_b, epsilon_b, K)

        # b_l = (1/K) * sum_k f_k * exp(-i*2*pi*k*l/K)  = FFT(f)[l] / K
        B_dft = np.fft.fft(f_vals) / K

        b = np.zeros(2 * max_freq + 1)
        for col, l in enumerate(range(-max_freq, max_freq + 1)):
            b[col] = B_dft[l % K].real   # circular: negative l -> K - |l|

        return b

    # ------------------------------------------------------------------
    # Hadamard test on A(U, P)  (Corollary 2 / Figure 7)
    # ------------------------------------------------------------------

    def build_expectation_value_hadamard_test(
        self,
        num_qubits: int,
        x_edges: List[Tuple[int, int]],
        tau: float,
        r_steps: int,
        pauli_observable: str,
        target_freq: int,
        part: str = "real",
    ) -> QuantumCircuit:
        """Build the Hadamard test that extracts a single b_l coefficient.

        Wraps the A(U, P) circuit in a Hadamard test (Figure 7 of the paper)
        to extract the real or imaginary part of a specific b_l:

            <Z>_real = Re(b_l) / ||b||_2   (part='real')
            <Z>_imag = Im(b_l) / ||b||_2   (part='imag')

        The measurement outcome on the HT ancilla qubit gives:

            Re(b_l) / ||b||_2  =  2*P(ht_ancilla=0) - 1   (real part)
            Im(b_l) / ||b||_2  =  2*P(ht_ancilla=0) - 1   (imaginary part)

        The normalisation ||b||_2 = sqrt(P(anc=0)) is obtained separately
        from the bare A(U, P) statevector (see _extract_b_l in learning.py).

        Parameters
        ----------
        num_qubits : int
            Number of data qubits.
        x_edges : list of (int, int)
            Edges of the graph x.
        tau : float
            Total evolution time.
        r_steps : int
            Number of Trotter steps.
        pauli_observable : str
            Pauli string for the observable P (Qiskit little-endian convention).
        target_freq : int
            Unsigned register index of the target frequency l, computed as
            ``l % 2**n_s`` to handle negative frequencies.
        part : {'real', 'imag'}
            Which quadrature to extract.

        Returns
        -------
        QuantumCircuit
            HT circuit with a final measurement on the ht_ancilla qubit.
        """
        aup_qc = self.build_expectation_value_extraction_circuit(
            num_qubits, x_edges, tau, r_steps, pauli_observable
        )
        n_s = self._freq_register_size(r_steps)

        qr_ht_anc = QuantumRegister(1, "ht_ancilla")
        qr_data   = QuantumRegister(num_qubits, "data")
        qr_anc    = QuantumRegister(1, "ancilla")
        qr_freq   = QuantumRegister(n_s, "freq_0")
        cr_meas   = ClassicalRegister(1, "meas")

        ht_qc = QuantumCircuit(qr_ht_anc, qr_data, qr_anc, qr_freq, cr_meas)

        # Hadamard on the control qubit (S† for imaginary part)
        ht_qc.h(qr_ht_anc)
        if part == "imag":
            ht_qc.sdg(qr_ht_anc)

        # Controlled-A(U,P): ht_ancilla controls the entire extraction circuit
        controlled_aup = aup_qc.to_gate(label="A(U,P)").control(1)
        ht_qc.append(
            controlled_aup,
            [qr_ht_anc[0]] + list(qr_data) + list(qr_anc) + list(qr_freq),
        )

        # Controlled phase kick-back: CNOT(ht_anc, freq_bit) selects frequency l
        binary_str = format(target_freq, f"0{n_s}b")
        for bit_idx, bit_val in enumerate(reversed(binary_str)):
            if bit_val == "1":
                ht_qc.cx(qr_ht_anc[0], qr_freq[bit_idx])

        # Final Hadamard and measurement
        ht_qc.h(qr_ht_anc)
        ht_qc.measure(qr_ht_anc, cr_meas)

        return ht_qc

    # ------------------------------------------------------------------
    # Hadamard test on bare A(U)  (Appendix C.4 / Figure 7)
    # ------------------------------------------------------------------

    def build_fourier_hadamard_test(
        self,
        num_qubits: int,
        x_edges: List[Tuple[int, int]],
        tau: float,
        r_steps: int,
        target_freq: int,
        part: str = "real",
    ) -> QuantumCircuit:
        """Build the Hadamard test for a single A(U) amplitude a_{l,0}.

        This circuit extracts the amplitude a_{l,0} from the bare A(U)
        output state (Theorem 1), NOT the expectation-value coefficient b_l.
        Use ``build_expectation_value_hadamard_test`` for b_l extraction.

        Kept for backward compatibility and as a building block.

        Parameters
        ----------
        num_qubits, x_edges, tau, r_steps, target_freq, part
            Same semantics as ``build_expectation_value_hadamard_test``.

        Returns
        -------
        QuantumCircuit
            HT circuit for a_{target_freq, 0}.
        """
        base_qc, qr_freqs = self.build_trotter_extraction_circuit(
            num_qubits, x_edges, tau, r_steps, d_params=1
        )
        n_s = qr_freqs[0].size

        qr_ht_anc = QuantumRegister(1, "ht_ancilla")
        qr_data   = QuantumRegister(num_qubits, "data")
        qr_anc    = QuantumRegister(1, "ancilla")
        qr_freq   = QuantumRegister(n_s, "freq_0")
        cr_meas   = ClassicalRegister(1, "meas")

        ht_qc = QuantumCircuit(qr_ht_anc, qr_data, qr_anc, qr_freq, cr_meas)

        ht_qc.h(qr_ht_anc)
        if part == "imag":
            ht_qc.sdg(qr_ht_anc)

        controlled_au = base_qc.to_gate(label="A(U)").control(1)
        ht_qc.append(
            controlled_au,
            [qr_ht_anc[0]] + list(qr_data) + list(qr_anc) + list(qr_freq),
        )

        binary_str = format(target_freq, f"0{n_s}b")
        for bit_idx, bit_val in enumerate(reversed(binary_str)):
            if bit_val == "1":
                ht_qc.cx(qr_ht_anc[0], qr_freq[bit_idx])

        ht_qc.h(qr_ht_anc)
        ht_qc.measure(qr_ht_anc, cr_meas)

        return ht_qc

    # ------------------------------------------------------------------
    # Kernel overlap circuit  (Section VI.B / Figure 8)
    # ------------------------------------------------------------------

    def build_quantum_overlap_kernel_circuit(
        self,
        num_qubits: int,
        x_edges_1: List[Tuple[int, int]],
        x_edges_2: List[Tuple[int, int]],
        tau: float,
        r_steps: int,
    ) -> QuantumCircuit:
        """Build the circuit that estimates the kernel k(x, x') = b(x)·b(x').

        Implements the quantum overlap circuit from Figure 8 of the paper.
        The kernel value Re(<b(x)|b(x')>) is recovered from:

            <Z ⊗ I ⊗ |0><0|> = Re(b(x)·b(x'))

        Parameters
        ----------
        num_qubits : int
        x_edges_1, x_edges_2 : list of (int, int)
            Edge sets for graphs x and x'.
        tau : float
        r_steps : int

        Returns
        -------
        QuantumCircuit
            Kernel circuit with measurements on the kernel ancilla and all
            other qubits.
        """
        qc_1, qr_freqs = self.build_trotter_extraction_circuit(
            num_qubits, x_edges_1, tau, r_steps, d_params=1
        )
        qc_2, _ = self.build_trotter_extraction_circuit(
            num_qubits, x_edges_2, tau, r_steps, d_params=1
        )
        n_s = qr_freqs[0].size

        qr_k_anc = QuantumRegister(1, "kernel_ancilla")
        qr_data  = QuantumRegister(num_qubits, "data")
        qr_anc   = QuantumRegister(1, "ancilla")
        qr_freq  = QuantumRegister(n_s, "freq_0")
        cr_k     = ClassicalRegister(1, "meas_k_ancilla")
        cr_rest  = ClassicalRegister(num_qubits + 1 + n_s, "meas_rest")

        kernel_qc = QuantumCircuit(qr_k_anc, qr_data, qr_anc, qr_freq, cr_k, cr_rest)

        gate_1 = qc_1.to_gate(label="A(U_x)").control(1, ctrl_state="0")
        gate_2 = qc_2.to_gate(label="A(U_x')").control(1, ctrl_state="1")
        target = list(qr_data) + list(qr_anc) + list(qr_freq)

        kernel_qc.h(qr_k_anc)
        kernel_qc.append(gate_1, [qr_k_anc[0]] + target)
        kernel_qc.append(gate_2, [qr_k_anc[0]] + target)
        kernel_qc.h(qr_k_anc)

        kernel_qc.measure(qr_k_anc, cr_k)
        kernel_qc.measure(target, cr_rest)

        return kernel_qc

    # ------------------------------------------------------------------
    # LCU extension for linear-combination observables (Appendix C.2b)
    # ------------------------------------------------------------------

    def build_lcu_fourier_extraction_circuit(
        self,
        num_qubits: int,
        x_edges: List[Tuple[int, int]],
        tau: float,
        r_steps: int,
        observable_paulis: List[str],
        observable_coeffs: List[float],
    ) -> QuantumCircuit:
        """Build the A(U, O) circuit for a linear combination of Paulis.

        Extends Corollary 1 to observables O = sum_h beta_h P_h using a
        Linear Combination of Unitaries (LCU) approach (Appendix C.2b).

        Parameters
        ----------
        num_qubits : int
        x_edges : list of (int, int)
        tau : float
        r_steps : int
        observable_paulis : list of str
            Pauli strings for each term in O.
        observable_coeffs : list of float
            Coefficients beta_h >= 0 for each term.

        Returns
        -------
        QuantumCircuit
        """
        coeffs = np.array(observable_coeffs, dtype=complex)
        amplitudes = np.sqrt(np.abs(coeffs))
        norm = np.linalg.norm(amplitudes)
        normalized_amps = amplitudes / norm

        num_lcu_qubits = max(1, math.ceil(math.log2(len(observable_coeffs))))
        padded_amps = np.zeros(2 ** num_lcu_qubits, dtype=complex)
        padded_amps[: len(normalized_amps)] = normalized_amps
        lcu_prep = StatePreparation(padded_amps)

        base_qc, qr_freqs = self.build_trotter_extraction_circuit(
            num_qubits, x_edges, tau, r_steps, d_params=1
        )
        n_s = qr_freqs[0].size

        qr_lcu = QuantumRegister(num_lcu_qubits, "lcu")
        qr_data = QuantumRegister(num_qubits, "data")
        qr_anc  = QuantumRegister(1, "ancilla")
        qr_freq = QuantumRegister(n_s, "freq_0")

        lcu_qc = QuantumCircuit(qr_lcu, qr_data, qr_anc, qr_freq)
        base_qubits = list(qr_data) + list(qr_anc) + list(qr_freq)

        lcu_qc.append(lcu_prep, qr_lcu)
        lcu_qc.append(base_qc.to_instruction(), base_qubits)

        for idx, pauli_str in enumerate(observable_paulis):
            ctrl_state = format(idx, f"0{num_lcu_qubits}b")
            p_gate = PauliGate(pauli_str).control(num_lcu_qubits, ctrl_state=ctrl_state)
            lcu_qc.append(p_gate, list(qr_lcu) + list(qr_data))

        lcu_qc.append(base_qc.inverse().to_instruction(), base_qubits)
        lcu_qc.append(lcu_prep.inverse(), qr_lcu)

        return lcu_qc

    # ------------------------------------------------------------------
    # LGT extension
    # ------------------------------------------------------------------

    def build_lgt_trotter_extraction_circuit(
        self,
        num_matter_sites: int,
        x_mask: List[int],
        mass: float,
        electric_field: float,
        tau: float,
        r_steps: int,
    ) -> Tuple[QuantumCircuit, List[QuantumRegister]]:
        """Build the A(U) circuit for a 1D Z2 Schwinger / Lattice Gauge Theory model.

        Model Hamiltonian:
            H(x, g) = m * sum_i (-1)^i Z_{m_i}
                    + sum_{i: x_i=1} g_i * X_{m_i} Z_{l_i} X_{m_{i+1}}
                    + h * sum_i X_{l_i}

        x_mask controls which gauge links are ACTIVE (x_i=1 → link i has coupling g_i > 0).
        This is the correct analog of the TFIM graph edge variable x_{ij}.

        The A(U) circuit encodes the unknown coupling g_i of each ACTIVE link into a
        separate frequency register via the D·G·D block on the XZX Pauli.
        INACTIVE links (x_i=0) do not receive a D·G·D block and their freq register
        stays at zero.

        Known terms (mass, electric field) are encoded as fixed RZ/RX gates.

        Qubit layout (interlaced, Qiskit little-endian):
            qubit 0: matter 0,  qubit 1: link 0-1,  qubit 2: matter 1, ...
            qubit 2i:   matter site i
            qubit 2i+1: gauge link i (between matter sites i and i+1)

        Parameters
        ----------
        num_matter_sites : int  — N matter sites → 2N-1 total data qubits
        x_mask : list of int   — [x_0, x_1, ..., x_{N-2}], each 0 or 1
        mass : float           — staggered fermion mass (known)
        electric_field : float — electric field coupling h (known)
        tau : float            — total evolution time
        r_steps : int          — Trotter steps

        Returns
        -------
        qc : QuantumCircuit
        qr_freqs : list of QuantumRegister  — one per gauge link
        """
        num_gauge_links = num_matter_sites - 1
        num_data_qubits = num_matter_sites + num_gauge_links
        n_s = self._freq_register_size(r_steps)

        qr_data    = QuantumRegister(num_data_qubits, "data")
        qr_ancilla = QuantumRegister(1, "ancilla")
        qr_freqs   = [QuantumRegister(n_s, f"freq_{i}") for i in range(num_gauge_links)]

        qc = QuantumCircuit(qr_data, qr_ancilla, *qr_freqs)

        # Use cached V± gates for efficiency
        gates = self._get_cached_v_gates(n_s)
        cVp, cVm = gates["cVp"], gates["cVm"]

        def apply_pauli_extraction(pauli_string: str, target_freq_reg: QuantumRegister) -> None:
            """D·G·D block: encode the XZX Pauli upload gate into target_freq_reg."""
            # H on X-type positions (basis change Z → X)
            for i, p in enumerate(reversed(pauli_string)):
                if p == "X":
                    qc.h(qr_data[i])
            # D block: parity of all active Pauli positions onto ancilla
            active = [qr_data[i] for i, p in enumerate(reversed(pauli_string))
                      if p in ("X", "Z")]
            for q in active:
                qc.cx(q, qr_ancilla[0])
            # G block: V+ if anc=|0>, V- if anc=|1>
            qc.append(cVp, [qr_ancilla[0]] + list(target_freq_reg))
            qc.append(cVm, [qr_ancilla[0]] + list(target_freq_reg))
            # D block (uncompute ancilla)
            for q in reversed(active):
                qc.cx(q, qr_ancilla[0])
            # Undo H
            for i, p in enumerate(reversed(pauli_string)):
                if p == "X":
                    qc.h(qr_data[i])

        dt = tau / r_steps
        for _step in range(r_steps):
            # Known term 1: staggered mass — RZ(2*m*(-1)^i*dt) on each matter qubit
            for i in range(num_matter_sites):
                qc.rz(2.0 * mass * (-1)**i * dt, qr_data[2 * i])
            # Known term 2: electric field — RX(2*h*dt) on each link qubit
            for i in range(num_gauge_links):
                qc.rx(2.0 * electric_field * dt, qr_data[2 * i + 1])
            # Unknown term: A-transform of g_i * XZX for each ACTIVE link
            for i in range(num_gauge_links):
                if x_mask[i] == 0:
                    continue   # inactive link: no D·G·D, freq register stays at 0
                pauli_chars = ["I"] * num_data_qubits
                pauli_chars[2 * i]     = "X"   # matter site i
                pauli_chars[2 * i + 1] = "Z"   # gauge link i
                pauli_chars[2 * i + 2] = "X"   # matter site i+1
                pauli_str = "".join(pauli_chars)[::-1]  # Qiskit little-endian
                apply_pauli_extraction(pauli_str, qr_freqs[i])

        return qc, qr_freqs
    
    def build_lgt_expectation_value_extraction_circuit(
        self,
        num_matter_sites: int,
        x_mask: List[int],
        mass: float,
        electric_field: float,
        tau: float,
        r_steps: int,
        pauli_observable: str,
    ) -> QuantumCircuit:
        """Builds the fast A(U,P) expectation circuit for the Schwinger Model."""
        num_gauge_links = num_matter_sites - 1
        num_data_qubits = num_matter_sites + num_gauge_links
        n_s = self._freq_register_size(r_steps)

        # 1. Build forward A(U) circuit
        base_qc, qr_freqs = self.build_lgt_trotter_extraction_circuit(
            num_matter_sites, x_mask, mass, electric_field, tau, r_steps
        )
        
        qr_data    = QuantumRegister(num_data_qubits, "data")
        qr_ancilla = QuantumRegister(1, "ancilla")
        
        aup_qc = QuantumCircuit(qr_data, qr_ancilla, *qr_freqs)
        all_qubits = list(qr_data) + list(qr_ancilla) + [q for reg in qr_freqs for q in reg]
        
        # Step 1: Apply forward A(U) as a single fast instruction
        aup_qc.append(base_qc.to_instruction(), all_qubits)
        
        # Step 2: Apply Pauli observable
        aup_qc.append(PauliGate(pauli_observable), list(qr_data))
        
        # Step 3: Explicit A(U)^dagger uncomputation
        dt = tau / r_steps
        cached_gates = self._get_cached_v_gates(n_s)
        cVp_dag = cached_gates["cVp_dag"]
        cVm_dag = cached_gates["cVm_dag"]
        
        for _step in reversed(range(r_steps)):
            # A. Reverse Gauge Coupling Extractions (D * G * D)
            for i in reversed(range(num_gauge_links)):
                if x_mask[i] == 1:
                    pauli_chars = ["I"] * num_data_qubits
                    pauli_chars[2 * i]     = "X"
                    pauli_chars[2 * i + 1] = "Z"
                    pauli_chars[2 * i + 2] = "X"
                    pauli_str = "".join(pauli_chars)[::-1]

                    for idx, p in enumerate(reversed(pauli_str)):
                        if p == "X": aup_qc.h(qr_data[idx])

                    active = [qr_data[idx] for idx, p in enumerate(reversed(pauli_str)) if p in ("X", "Z")]
                    for q in active: aup_qc.cx(q, qr_ancilla[0])

                    aup_qc.append(cVm_dag, [qr_ancilla[0]] + list(qr_freqs[i])) 
                    aup_qc.append(cVp_dag, [qr_ancilla[0]] + list(qr_freqs[i]))

                    for q in reversed(active): aup_qc.cx(q, qr_ancilla[0])

                    for idx, p in enumerate(reversed(pauli_str)):
                        if p == "X": aup_qc.h(qr_data[idx])
                        
            # B. Reverse Electric Field
            for i in reversed(range(num_gauge_links)):
                aup_qc.rx(-2.0 * electric_field * dt, qr_data[2 * i + 1])
                
            # C. Reverse Mass (Fixed the (-1)**i staggered sign)
            for i in reversed(range(num_matter_sites)):
                aup_qc.rz(-2.0 * mass * (-1)**i * dt, qr_data[2 * i])

        # REMOVED: The rogue Step 4 X-gates that were flipping the data register

        return aup_qc