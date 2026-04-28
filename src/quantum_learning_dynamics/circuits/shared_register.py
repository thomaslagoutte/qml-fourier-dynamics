"""Shared-register Fourier-extraction circuit for the ``d = 1`` regime.

Construction (Corollary 1 of Barthe et al., 2025)
-------------------------------------------------
When the parameter vector has a single component (:math:`d = 1`) — as in
the homogeneous TFIM — the Fourier coefficients :math:`b_l(x)` of the
observable curve :math:`c_\\alpha(x)` can all be amplitude-encoded in a
single shared frequency register of size

.. math::

    n_s = \\lceil \\log_2(4\\,L + 2) \\rceil,

where :math:`L` is the largest representable Fourier index supported by
the circuit (number of re-uploads of :math:`\\alpha`).

First-order Trotter step (one per ``r_steps`` iterations)
---------------------------------------------------------
1. :math:`R_{ZZ}(2\\,dt)` on every edge ``(i, j)`` in ``x`` — the fixed
   (``alpha``-independent) physical dynamics.
2. :math:`D \\cdot G \\cdot D` applied to each qubit of the data
   register:

   * :math:`D = H` on the active data qubit;
   * :math:`G = cV^{+}\\,cV^{-}` with the data qubit as the control and
     the shared frequency register as the target.

The continuous :math:`\\alpha` parameter does not appear in the circuit:
its periodic dependence is entirely encoded as :math:`\\pm 1` shifts of
the frequency register.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import PauliGate

from ._controlled_ops import (
    append_ctrl_pauli_string,
    append_ctrl_rzz,
    append_dctrl_cVm,
    append_dctrl_cVm_dag,
    append_dctrl_cVp,
    append_dctrl_cVp_dag,
)
from .base import CircuitBuilder, ExecutionMode, TargetFreq


class SharedRegisterBuilder(CircuitBuilder):
    """Fourier-extraction builder for ``d = 1`` Hamiltonians.

    All :math:`\\alpha`-uploaded Pauli rotations share a single frequency
    register of width :math:`n_s`. The extractor
    :class:`~quantum_learning_dynamics.features.engines.FeatureEngine`
    reads the entire Fourier vector :math:`b(x) \\in \\mathbb{R}^{4r+1}`
    from one statevector evaluation (``"statevector"`` mode) or from a
    batch of per-frequency Hadamard tests (``"hardware"`` mode).

    Parameters
    ----------
    model : HamiltonianModel
        Model with ``model.d == 1``.
    trotter_order : {1, 2}, default 2
        Order of the Trotter product formula.
    """

    def __init__(self, model, trotter_order: int = 2):
        super().__init__(model=model, trotter_order=trotter_order)
        if trotter_order not in (1, 2):
            raise ValueError("trotter_order must be 1 or 2")

    # ------------------------------------------------------------------
    # Register sizing
    # ------------------------------------------------------------------

    def freq_register_size(self, num_qubits: int, r_steps: int) -> int:
        """Qubit count :math:`n_s` of the shared frequency register.

        With ``d = 1`` the frequency bandwidth scales with both the
        number of re-uploads per step (proportional to ``num_qubits`` for
        the transverse-field terms) and the number of Trotter steps
        ``r_steps``.
        """
        return max(3, math.ceil(math.log2(4 * num_qubits * r_steps + 2)))

    # ------------------------------------------------------------------
    # D·G·D parity-fold / frequency-shift block
    # ------------------------------------------------------------------

    def _dgd_forward(self, qc, data, freq, cache) -> None:
        """Apply :math:`H \\cdot cV^{+} \\cdot cV^{-} \\cdot H` per qubit.

        Each data qubit acts as the control for the shared frequency
        register. The two polarities of :math:`cV^{\\pm}` encode the
        even / odd parity branches of the original Pauli-:math:`X`
        rotation into :math:`\\pm 1` shifts of the frequency register.
        """
        for q in range(len(data)):
            qc.h(data[q])
            qc.append(cache["cVp"], [data[q], *freq])
            qc.append(cache["cVm"], [data[q], *freq])
            qc.h(data[q])

    def _dgd_adjoint(self, qc, data, freq, cache) -> None:
        """Exact inverse of :meth:`_dgd_forward`.

        Reverse qubit order, reverse :math:`V^{\\pm}` order, and dagger
        each controlled shift.
        """
        for q in reversed(range(len(data))):
            qc.h(data[q])
            qc.append(cache["cVm_dag"], [data[q], *freq])
            qc.append(cache["cVp_dag"], [data[q], *freq])
            qc.h(data[q])

    # ------------------------------------------------------------------
    # Inline-controlled D·G·D — the HT-controlled twin of the above.
    # ------------------------------------------------------------------
    #
    # Mathematical contract: ``_ctrl_dgd_*`` is the ``ht``-controlled
    # version of ``_dgd_*``.  We exploit the identity
    #
    #     c-(Φ⁻¹ · M · Φ)   =   Φ⁻¹ · c-M · Φ      (Φ unconditional)
    #
    # to leave the outer ``H(data[q])`` basis changes unconditional and
    # only control the inner cV± gates (where the parity control is
    # ``data[q]``, not ``anc``).  This avoids paying for a c-H per
    # qubit per Trotter step, which would otherwise need a CRY-CX-CRY
    # decomposition and double the gate count.

    def _ctrl_dgd_forward(self, qc, ht, data, freq) -> None:
        """``ht``-controlled twin of :meth:`_dgd_forward`.

        Per qubit ``q`` the conditional block is
        ``c-(H · cV+ · cV- · H)`` with the inner cV± gates promoted to
        their doubly-controlled (ht, data[q]) variants and the outer
        H gates left unconditional.
        """
        for q in range(len(data)):
            qc.h(data[q])                                       # unconditional Φ
            append_dctrl_cVp(qc, ht, data[q], freq)             # c-cV+
            append_dctrl_cVm(qc, ht, data[q], freq)             # c-cV-
            qc.h(data[q])                                       # unconditional Φ⁻¹

    def _ctrl_dgd_adjoint(self, qc, ht, data, freq) -> None:
        """``ht``-controlled twin of :meth:`_dgd_adjoint`.

        Reverses qubit order, swaps :math:`cV^{\\pm}` order, and uses
        the doubly-controlled (cV±)† variants on the inner gates.
        """
        for q in reversed(range(len(data))):
            qc.h(data[q])
            append_dctrl_cVm_dag(qc, ht, data[q], freq)
            append_dctrl_cVp_dag(qc, ht, data[q], freq)
            qc.h(data[q])

    # ------------------------------------------------------------------
    # Forward / adjoint Trotter evolutions
    # ------------------------------------------------------------------

    def _append_au_forward(self, qc, data, freq, edges, tau, r_steps) -> None:
        """Append :math:`\\mathcal{A}(U)` (forward direction) for ``r_steps`` Trotter steps."""
        dt = tau / r_steps
        cache = self.gate_cache.get(len(freq))
        for _ in range(r_steps):
            if self.trotter_order == 1:
                for (i, j) in edges:
                    qc.rzz(+2.0 * dt, data[i], data[j])
                self._dgd_forward(qc, data, freq, cache)
            elif self.trotter_order == 2:
                for (i, j) in edges:
                    qc.rzz(+1.0 * dt, data[i], data[j])
                self._dgd_forward(qc, data, freq, cache)
                for (i, j) in edges:
                    qc.rzz(+1.0 * dt, data[i], data[j])

    def _append_au_adjoint(self, qc, data, freq, edges, tau, r_steps) -> None:
        """Append :math:`\\mathcal{A}(U)^{\\dagger}` (adjoint direction) for ``r_steps`` Trotter steps."""
        dt = tau / r_steps
        cache = self.gate_cache.get(len(freq))
        rev_edges = list(reversed(list(edges)))
        for _ in range(r_steps):
            if self.trotter_order == 1:
                self._dgd_adjoint(qc, data, freq, cache)
                for (i, j) in rev_edges:
                    qc.rzz(-2.0 * dt, data[i], data[j])
            elif self.trotter_order == 2:
                for (i, j) in rev_edges:
                    qc.rzz(-1.0 * dt, data[i], data[j])
                self._dgd_adjoint(qc, data, freq, cache)
                for (i, j) in rev_edges:
                    qc.rzz(-1.0 * dt, data[i], data[j])

    # ------------------------------------------------------------------
    # Inline-controlled Trotter evolutions (used by hardware mode).
    # ------------------------------------------------------------------

    def _append_ctrl_au_forward(self, qc, ht, data, freq, edges, tau, r_steps) -> None:
        """Inline ``ht``-controlled :math:`\\mathcal{A}(U)` (forward).

        Direct twin of :meth:`_append_au_forward`: every elementary gate
        is replaced by its singly- or doubly-controlled inline form.
        Mathematically identical to wrapping the full block in
        ``.control(1)``, but synthesised entirely in the
        ``CX / CRZ / CCX / MCX`` basis — no Quantum-Shannon decomposition.
        """
        dt = tau / r_steps
        for _ in range(r_steps):
            if self.trotter_order == 1:
                for (i, j) in edges:
                    append_ctrl_rzz(qc, ht, data[i], data[j], +2.0 * dt)
                self._ctrl_dgd_forward(qc, ht, data, freq)
            elif self.trotter_order == 2:
                for (i, j) in edges:
                    append_ctrl_rzz(qc, ht, data[i], data[j], +1.0 * dt)
                self._ctrl_dgd_forward(qc, ht, data, freq)
                for (i, j) in edges:
                    append_ctrl_rzz(qc, ht, data[i], data[j], +1.0 * dt)

    def _append_ctrl_au_adjoint(self, qc, ht, data, freq, edges, tau, r_steps) -> None:
        """Inline ``ht``-controlled :math:`\\mathcal{A}(U)^{\\dagger}` (adjoint)."""
        dt = tau / r_steps
        rev_edges = list(reversed(list(edges)))
        for _ in range(r_steps):
            if self.trotter_order == 1:
                self._ctrl_dgd_adjoint(qc, ht, data, freq)
                for (i, j) in rev_edges:
                    append_ctrl_rzz(qc, ht, data[i], data[j], -2.0 * dt)
            elif self.trotter_order == 2:
                for (i, j) in rev_edges:
                    append_ctrl_rzz(qc, ht, data[i], data[j], -1.0 * dt)
                self._ctrl_dgd_adjoint(qc, ht, data, freq)
                for (i, j) in rev_edges:
                    append_ctrl_rzz(qc, ht, data[i], data[j], -1.0 * dt)

    # ------------------------------------------------------------------
    # Public circuit factories
    # ------------------------------------------------------------------

    def build_au(
        self,
        num_qubits: int,
        x: Sequence[Tuple[int, int]],
        tau: float,
        r_steps: int,
    ) -> Tuple[QuantumCircuit, List[QuantumRegister]]:
        """Construct :math:`\\mathcal{A}(U)^{\\dagger}\\,\\mathcal{A}(U)` on the data register.

        This is the bare Fourier-extraction sandwich *without* an
        observable insertion; useful mainly as a sanity-check primitive.

        Returns
        -------
        QuantumCircuit
            Circuit on ``data + anc + freq`` registers.
        list of QuantumRegister
            Length-1 list containing the shared frequency register.
        """
        edges = x
        n_s = self.freq_register_size(num_qubits, r_steps)
        data = QuantumRegister(num_qubits, "data")
        anc  = QuantumRegister(1, "anc")
        freq = QuantumRegister(n_s, "freq")
        qc   = QuantumCircuit(data, anc, freq)

        self._append_au_forward(qc, data, freq, edges, tau, r_steps)
        self._append_au_adjoint(qc, data, freq, edges, tau, r_steps)
        return qc, [freq]

    def build_aup(
        self,
        num_qubits: int,
        x: Sequence[Tuple[int, int]],
        tau: float,
        r_steps: int,
        pauli: str,
        *,
        execution_mode: ExecutionMode = "statevector",
        target_freq: Optional[TargetFreq] = None,
    ) -> Tuple[QuantumCircuit, List[QuantumRegister]]:
        """Construct :math:`\\mathcal{A}(U, P)` for a single Pauli observable.

        Circuit structure (``"statevector"`` mode)
        ------------------------------------------
        ``A_forward → PauliGate(P) on data → A_adjoint``. The Fourier
        coefficient :math:`b_l` is read from the ``(data=0, anc=0,
        freq = l \\bmod 2^{n_s})`` amplitude of the statevector.

        Circuit structure (``"hardware"`` mode)
        ---------------------------------------
        Implements the per-frequency Hadamard test of Corollary 2 /
        Figure 7 via **inline controlled-gate assembly**:

        1. The HT-controlled :math:`\\mathcal{A}(U, P)` is built
           **gate-by-gate** on the outer circuit using the controlled
           primitives in :mod:`._controlled_ops` — c-RZZ via 2 CX +
           1 CRZ, doubly-controlled cV± via MCX cascades, and
           c-PauliGate(P) as a sequence of ``cx``/``cy``/``cz``.
           Mathematically identical to wrapping the inner block in
           ``.control(1)``, but **never invokes HighLevelSynthesis'
           controlled-unitary fallback** (Quantum-Shannon decomposition),
           which was the dominant runtime bottleneck for n ≥ 3.
        2. The HT skeleton is therefore
           ``H → c-A(U,P) (inline) → freq-selector CX ladder → H →
           measure(ht_control)``.  The CX ladder shifts the interference
           slot from :math:`f = 0` to :math:`f = l` so the final H
           projects :math:`\\mathrm{Re}(b_l)`.

        Parameters
        ----------
        num_qubits : int
            Number of data qubits.
        x : sequence of (int, int)
            Edge list encoding the fixed part of the Hamiltonian.
        tau, r_steps, pauli
            See :meth:`CircuitBuilder.build_aup`.
        execution_mode : {"statevector", "hardware_base", "hardware"}
            Execution-mode discriminant; see
            :data:`~.base.ExecutionMode`.
        target_freq : int or (int,), optional
            Required in ``"hardware"`` mode. Unsigned int or length-1
            tuple representing :math:`u = l \\bmod 2^{n_s}`. Ignored in
            the other modes.

        Returns
        -------
        QuantumCircuit
            The assembled circuit.
        list of QuantumRegister
            Length-1 list containing the shared frequency register.
        """
        if execution_mode not in ("statevector", "hardware", "hardware_base"):
            raise ValueError(
                f"execution_mode must be 'statevector', 'hardware', or "
                f"'hardware_base', got {execution_mode!r}"
            )

        edges = x
        n_s = self.freq_register_size(num_qubits, r_steps)
        data = QuantumRegister(num_qubits, "data")
        anc  = QuantumRegister(1, "anc")
        freq = QuantumRegister(n_s, "freq")

        if execution_mode == "statevector":
            # Bare circuit; all amplitudes read simultaneously downstream.
            qc = QuantumCircuit(data, anc, freq)
            self._append_au_forward(qc, data, freq, edges, tau, r_steps)
            qc.append(PauliGate(pauli), list(data))
            self._append_au_adjoint(qc, data, freq, edges, tau, r_steps)
            return qc, [freq]

        # ---- "hardware" / "hardware_base" shared prefix ----------------
        # Inline-controlled assembly: build c-A(U, P) gate-by-gate using
        # the controlled-rotation and doubly-controlled-cV± primitives
        # in :mod:`._controlled_ops`.  Mathematically identical to
        # ``inner.to_gate().control(1, annotated=True)`` but
        # **synthesised entirely in the CX / CRZ / CCX / MCX basis**
        # without ever invoking ``HighLevelSynthesis``'s controlled-
        # unitary fallback (Quantum-Shannon decomposition), which was
        # the dominant bottleneck for n ≥ 3 hardware-mode runs.
        #
        # The ``anc`` register is allocated for layout consistency with
        # build_au but is unused by the d=1 D·G·D block — only ``data``
        # qubits act as parity controls of cV±.

        ht_control = QuantumRegister(1, "ht_control")
        creg = ClassicalRegister(1, "creg")
        qc = QuantumCircuit(ht_control, data, anc, freq, creg)
        ht = ht_control[0]

        qc.h(ht)
        # c-A(U) (forward) ─ c-PauliGate(P) ─ c-A(U)† (adjoint)
        # — the entire HT-controlled inner block, inlined.
        self._append_ctrl_au_forward(qc, ht, data, freq, edges, tau, r_steps)
        append_ctrl_pauli_string(qc, ht, list(data), pauli)
        self._append_ctrl_au_adjoint(qc, ht, data, freq, edges, tau, r_steps)

        if execution_mode == "hardware_base":
            # Return prefix only; caller appends the cheap per-frequency tail.
            return qc, [freq]

        # ---- "hardware" mode: finalise the HT for a single target_freq.
        if target_freq is None:
            raise ValueError(
                "execution_mode='hardware' requires target_freq "
                "(unsigned int u = l mod 2^n_s) for the per-frequency "
                "Hadamard test."
            )
        if isinstance(target_freq, tuple):
            if len(target_freq) != 1:
                raise ValueError(
                    f"SharedRegisterBuilder (d=1) expects a scalar "
                    f"target_freq or a length-1 tuple, got {target_freq!r}"
                )
            target_u = int(target_freq[0])
        else:
            target_u = int(target_freq)
        if not 0 <= target_u < (1 << n_s):
            raise ValueError(
                f"target_freq={target_u} out of range "
                f"[0, 2^n_s={1 << n_s})"
            )

        # Frequency-selector CX ladder: shift |f⟩ → |f ⊕ l⟩ on the c=1
        # branch so the final H projects Re(b_l) rather than Re(b_0).
        self._append_freq_selector(qc, ht_control, [freq], target_u)
        qc.h(ht_control[0])
        qc.measure(ht_control[0], creg[0])

        return qc, [freq]
