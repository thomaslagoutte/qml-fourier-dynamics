"""Separate-registers Fourier-extraction circuit for the ``d > 1`` regime.

Construction
------------
For Hamiltonians whose parameter vector has :math:`d > 1` components —
e.g. the inhomogeneous TFIM or the :math:`Z_2` Schwinger model — every
component :math:`\\alpha_k` gets its own frequency register. The rank-:math:`d`
Fourier tensor

.. math::

    b_{l_0,\\,\\ldots,\\,l_{d-1}}(x)
        \\;=\\; \\sum_{k} a_{\\,l_0,\\ldots,l_{d-1},\\,k}

is read by the statevector-mode extractor from the
``(data=0, anc=0, freq_0=l_0, ..., freq_{d-1}=l_{d-1})`` amplitude
and by the hardware-mode extractor through one per-frequency Hadamard
test per grid point.

Weight-:math:`k` uploads
------------------------
Some models (``SchwingerZ2Model``) upload multi-qubit Pauli strings such
as :math:`XZX` on ``(m_l, \\text{link}_l, m_{l+1})``. The ``D`` block in
:meth:`_dgd_forward` / :meth:`_dgd_adjoint` applies the appropriate
Clifford basis changes (``H`` for :math:`X`, :math:`S^\\dagger H` for
:math:`Y`) and then folds the parity of all active qubits onto the
shared ancilla ``anc[0]`` via a CNOT cascade. The :math:`cV^{\\pm}`
shifts are then controlled on that ancilla and target the frequency
register associated with the relevant :math:`\\alpha` component.

Total statevector size
----------------------
``2 ** (num_qubits + 1 + d * n_s)``.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import PauliGate

from ._controlled_ops import (
    append_ctrl_pauli_string,
    append_ctrl_rxx,
    append_ctrl_ryy,
    append_ctrl_rzz,
    append_dctrl_cVm,
    append_dctrl_cVm_dag,
    append_dctrl_cVp,
    append_dctrl_cVp_dag,
)
from .base import CircuitBuilder, ExecutionMode, TargetFreq


class SeparateRegistersBuilder(CircuitBuilder):
    """Fourier-extraction builder for ``d > 1`` Hamiltonians.

    One frequency register of width :math:`n_s` per component of
    :math:`\\alpha`. Shared by the inhomogeneous TFIM and the Schwinger
    models; also serves as the base class for the overlap-kernel builder.

    Parameters
    ----------
    model : HamiltonianModel
        Model with ``model.d >= 1``.
    trotter_order : {1, 2}, default 2
        Order of the Trotter product formula.
    """

    def __init__(self, model, trotter_order: int = 2):
        super().__init__(model=model, trotter_order=trotter_order)
        if trotter_order not in (1, 2):
            raise ValueError("trotter_order must be 1 or 2")

    def freq_register_size(self, num_qubits: int, r_steps: int) -> int:
        """Qubit count :math:`n_s` per (per-parameter) frequency register.

        In the ``d > 1`` regime each :math:`\\alpha_k` contributes exactly
        one upload per Trotter step, so the bandwidth depends only on
        ``r_steps``: :math:`n_s = \\max(3, \\lceil \\log_2(4\\,r + 2) \\rceil)`.
        """
        return max(3, math.ceil(math.log2(4 * r_steps + 2)))

    # ------------------------------------------------------------------
    # D·G·D parity-fold / frequency-shift block
    # ------------------------------------------------------------------

    def _dgd_forward(self, qc, data, anc, freqs, cache, x) -> None:
        """Parity-fold / frequency-shift routed per active :math:`\\alpha` component.

        Only applies the shift for Pauli uploads that are *active* for
        the current input :math:`x`. For the Schwinger model, this
        respects the gauge-link activation mask — severed links contribute
        no hopping term and therefore no frequency shift.
        """
        ones_alpha = np.ones(self.model.d)
        zero_alpha = np.zeros(self.model.d)
        H_up = self.model.hamiltonian(x, ones_alpha) - self.model.hamiltonian(x, zero_alpha)
        active_paulis = {p for p, c in H_up.to_list() if abs(c) > 1e-12}

        for k, pauli_str in enumerate(self.model.upload_paulis):
            if pauli_str not in active_paulis:
                continue

            # Basis change: fold the weight-k Pauli string onto computational
            # parity. X → HZH, Y → S†HZHS; Z needs no rotation.
            active = []
            for i, p in enumerate(reversed(pauli_str)):
                if p == 'X':
                    qc.h(data[i])
                    active.append(data[i])
                elif p == 'Y':
                    qc.sdg(data[i])
                    qc.h(data[i])
                    active.append(data[i])
                elif p == 'Z':
                    active.append(data[i])

            # Parity cascade onto the ancilla.
            for q in active:
                qc.cx(q, anc[0])
            qc.append(cache["cVp"], [anc[0], *freqs[k]])
            qc.append(cache["cVm"], [anc[0], *freqs[k]])
            for q in reversed(active):
                qc.cx(q, anc[0])

            # Undo the basis change.
            for i, p in enumerate(reversed(pauli_str)):
                if p == 'X':
                    qc.h(data[i])
                elif p == 'Y':
                    qc.h(data[i])
                    qc.s(data[i])

    def _dgd_adjoint(self, qc, data, anc, freqs, cache, x) -> None:
        """Exact inverse of :meth:`_dgd_forward`.

        Iterates the upload Pauli list in reverse and uses the adjoint
        controlled shifts :math:`(cV^{\\pm})^{\\dagger}`.
        """
        ones_alpha = np.ones(self.model.d)
        zero_alpha = np.zeros(self.model.d)
        H_up = self.model.hamiltonian(x, ones_alpha) - self.model.hamiltonian(x, zero_alpha)
        active_paulis = {p for p, c in H_up.to_list() if abs(c) > 1e-12}

        for k, pauli_str in reversed(list(enumerate(self.model.upload_paulis))):
            if pauli_str not in active_paulis:
                continue

            active = []
            for i, p in enumerate(reversed(pauli_str)):
                if p == 'X':
                    qc.h(data[i])
                    active.append(data[i])
                elif p == 'Y':
                    qc.sdg(data[i])
                    qc.h(data[i])
                    active.append(data[i])
                elif p == 'Z':
                    active.append(data[i])

            for q in active:
                qc.cx(q, anc[0])
            qc.append(cache["cVm_dag"], [anc[0], *freqs[k]])
            qc.append(cache["cVp_dag"], [anc[0], *freqs[k]])
            for q in reversed(active):
                qc.cx(q, anc[0])

            for i, p in enumerate(reversed(pauli_str)):
                if p == 'X':
                    qc.h(data[i])
                elif p == 'Y':
                    qc.h(data[i])
                    qc.s(data[i])

    # ------------------------------------------------------------------
    # Inline-controlled D·G·D — HT-controlled twin of the above.
    # ------------------------------------------------------------------
    #
    # Mathematical contract (cf. ``shared_register._ctrl_dgd_*``):
    #
    #     c-(Φ⁻¹ · M · Φ)  =  Φ⁻¹ · c-M · Φ      (Φ unconditional)
    #
    # The basis-change H / S† / S on the data qubits stay unconditional;
    # the inner (CX(q, anc) · cV± · CX(q, anc)) block gets every gate
    # promoted to its singly- or doubly-controlled form on ``ht``:
    #
    #     CX(q, anc)            →  CCX(ht, q, anc)
    #     cVp(anc, freqs[k])    →  doubly-controlled (ht, anc) cV+
    #     cVm(anc, freqs[k])    →  doubly-controlled (ht, anc) cV-
    #
    # No QSD fallback is invoked.

    def _ctrl_dgd_forward(self, qc, ht, data, anc, freqs, x) -> None:
        """``ht``-controlled twin of :meth:`_dgd_forward`."""
        ones_alpha = np.ones(self.model.d)
        zero_alpha = np.zeros(self.model.d)
        H_up = self.model.hamiltonian(x, ones_alpha) - self.model.hamiltonian(x, zero_alpha)
        active_paulis = {p for p, c in H_up.to_list() if abs(c) > 1e-12}

        for k, pauli_str in enumerate(self.model.upload_paulis):
            if pauli_str not in active_paulis:
                continue

            # Unconditional basis change (Φ).
            active = []
            for i, p in enumerate(reversed(pauli_str)):
                if p == 'X':
                    qc.h(data[i])
                    active.append(data[i])
                elif p == 'Y':
                    qc.sdg(data[i])
                    qc.h(data[i])
                    active.append(data[i])
                elif p == 'Z':
                    active.append(data[i])

            # Controlled inner block (M).  CX(q, anc) → CCX(ht, q, anc).
            for q in active:
                qc.ccx(ht, q, anc[0])
            append_dctrl_cVp(qc, ht, anc[0], freqs[k])
            append_dctrl_cVm(qc, ht, anc[0], freqs[k])
            for q in reversed(active):
                qc.ccx(ht, q, anc[0])

            # Unconditional inverse basis change (Φ⁻¹).
            for i, p in enumerate(reversed(pauli_str)):
                if p == 'X':
                    qc.h(data[i])
                elif p == 'Y':
                    qc.h(data[i])
                    qc.s(data[i])

    def _ctrl_dgd_adjoint(self, qc, ht, data, anc, freqs, x) -> None:
        """``ht``-controlled twin of :meth:`_dgd_adjoint`."""
        ones_alpha = np.ones(self.model.d)
        zero_alpha = np.zeros(self.model.d)
        H_up = self.model.hamiltonian(x, ones_alpha) - self.model.hamiltonian(x, zero_alpha)
        active_paulis = {p for p, c in H_up.to_list() if abs(c) > 1e-12}

        for k, pauli_str in reversed(list(enumerate(self.model.upload_paulis))):
            if pauli_str not in active_paulis:
                continue

            active = []
            for i, p in enumerate(reversed(pauli_str)):
                if p == 'X':
                    qc.h(data[i])
                    active.append(data[i])
                elif p == 'Y':
                    qc.sdg(data[i])
                    qc.h(data[i])
                    active.append(data[i])
                elif p == 'Z':
                    active.append(data[i])

            for q in active:
                qc.ccx(ht, q, anc[0])
            append_dctrl_cVm_dag(qc, ht, anc[0], freqs[k])
            append_dctrl_cVp_dag(qc, ht, anc[0], freqs[k])
            for q in reversed(active):
                qc.ccx(ht, q, anc[0])

            for i, p in enumerate(reversed(pauli_str)):
                if p == 'X':
                    qc.h(data[i])
                elif p == 'Y':
                    qc.h(data[i])
                    qc.s(data[i])

    # ------------------------------------------------------------------
    # Fixed (α-independent) Hamiltonian evolution
    # ------------------------------------------------------------------

    def _append_fixed_hamiltonian(self, qc, data, x, dt, sign=1.0):
        """Compile the ``alpha``-independent Hamiltonian terms natively.

        Each Pauli string in :math:`H_{\\text{fixed}}(x) = H(x, 0)` is
        mapped to its corresponding single- or two-qubit native rotation
        (``rx``, ``ry``, ``rz``, ``rxx``, ``ryy``, ``rzz``). This avoids
        the sparse-matrix fallback that generic ``SparsePauliOp``
        evolution triggers and keeps the circuit depth predictable.
        """
        zero_alpha = np.zeros(self.model.d)
        H_fixed = self.model.hamiltonian(x, zero_alpha)

        for pauli_str, coeff in H_fixed.to_list():
            if abs(coeff) < 1e-12:
                continue
            theta = 2.0 * sign * dt * np.real(coeff)

            p_list = list(reversed(pauli_str))
            active_q = [(i, p) for i, p in enumerate(p_list) if p != 'I']

            if len(active_q) == 1:
                q0, p0 = active_q[0]
                if p0 == 'Z':
                    qc.rz(theta, data[q0])
                elif p0 == 'X':
                    qc.rx(theta, data[q0])
                elif p0 == 'Y':
                    qc.ry(theta, data[q0])
            elif len(active_q) == 2:
                q0, p0 = active_q[0]
                q1, p1 = active_q[1]
                if p0 == 'Z' and p1 == 'Z':
                    qc.rzz(theta, data[q0], data[q1])
                elif p0 == 'X' and p1 == 'X':
                    qc.rxx(theta, data[q0], data[q1])
                elif p0 == 'Y' and p1 == 'Y':
                    qc.ryy(theta, data[q0], data[q1])

    # ------------------------------------------------------------------
    # Forward / adjoint Trotter evolutions
    # ------------------------------------------------------------------

    def _append_au_forward(self, qc, data, anc, freqs, x, tau, r_steps) -> None:
        """Append :math:`\\mathcal{A}(U)` (forward) for ``r_steps`` Trotter steps."""
        dt = tau / r_steps
        cache = self.gate_cache.get(len(freqs[0]))
        for _ in range(r_steps):
            if self.trotter_order == 1:
                self._append_fixed_hamiltonian(qc, data, x, dt, 1.0)
                self._dgd_forward(qc, data, anc, freqs, cache, x)
            elif self.trotter_order == 2:
                self._append_fixed_hamiltonian(qc, data, x, dt / 2.0, 1.0)
                self._dgd_forward(qc, data, anc, freqs, cache, x)
                self._append_fixed_hamiltonian(qc, data, x, dt / 2.0, 1.0)

    def _append_au_adjoint(self, qc, data, anc, freqs, x, tau, r_steps) -> None:
        """Append :math:`\\mathcal{A}(U)^{\\dagger}` (adjoint) for ``r_steps`` Trotter steps."""
        dt = tau / r_steps
        cache = self.gate_cache.get(len(freqs[0]))
        for _ in range(r_steps):
            if self.trotter_order == 1:
                self._dgd_adjoint(qc, data, anc, freqs, cache, x)
                self._append_fixed_hamiltonian(qc, data, x, dt, -1.0)
            elif self.trotter_order == 2:
                self._append_fixed_hamiltonian(qc, data, x, dt / 2.0, -1.0)
                self._dgd_adjoint(qc, data, anc, freqs, cache, x)
                self._append_fixed_hamiltonian(qc, data, x, dt / 2.0, -1.0)

    # ------------------------------------------------------------------
    # Inline-controlled Trotter evolutions (used by hardware mode).
    # ------------------------------------------------------------------

    def _append_ctrl_fixed_hamiltonian(self, qc, ht, data, x, dt, sign=1.0) -> None:
        """``ht``-controlled twin of :meth:`_append_fixed_hamiltonian`.

        Maps each native rotation in :math:`H_{\\text{fixed}}(x)` to its
        singly-controlled inline form:

        +-------------------+----------------------------------------+
        | rz / rx / ry      | ``crz / crx / cry``  (native)          |
        +-------------------+----------------------------------------+
        | rzz               | :func:`append_ctrl_rzz` (CX·CRZ·CX)    |
        +-------------------+----------------------------------------+
        | rxx               | :func:`append_ctrl_rxx` (H·c-RZZ·H)    |
        +-------------------+----------------------------------------+
        | ryy               | :func:`append_ctrl_ryy`                |
        +-------------------+----------------------------------------+
        """
        zero_alpha = np.zeros(self.model.d)
        H_fixed = self.model.hamiltonian(x, zero_alpha)

        for pauli_str, coeff in H_fixed.to_list():
            if abs(coeff) < 1e-12:
                continue
            theta = 2.0 * sign * dt * np.real(coeff)

            p_list = list(reversed(pauli_str))
            active_q = [(i, p) for i, p in enumerate(p_list) if p != 'I']

            if len(active_q) == 1:
                q0, p0 = active_q[0]
                if p0 == 'Z':
                    qc.crz(theta, ht, data[q0])
                elif p0 == 'X':
                    qc.crx(theta, ht, data[q0])
                elif p0 == 'Y':
                    qc.cry(theta, ht, data[q0])
            elif len(active_q) == 2:
                q0, p0 = active_q[0]
                q1, p1 = active_q[1]
                if p0 == 'Z' and p1 == 'Z':
                    append_ctrl_rzz(qc, ht, data[q0], data[q1], theta)
                elif p0 == 'X' and p1 == 'X':
                    append_ctrl_rxx(qc, ht, data[q0], data[q1], theta)
                elif p0 == 'Y' and p1 == 'Y':
                    append_ctrl_ryy(qc, ht, data[q0], data[q1], theta)

    def _append_ctrl_au_forward(self, qc, ht, data, anc, freqs, x, tau, r_steps) -> None:
        """Inline ``ht``-controlled :math:`\\mathcal{A}(U)` (forward).

        Direct twin of :meth:`_append_au_forward`: every elementary gate
        becomes its inline controlled form.  Synthesises into the
        ``CX / CRX / CRY / CRZ / CCX / MCX`` basis without invoking
        Quantum-Shannon decomposition.
        """
        dt = tau / r_steps
        for _ in range(r_steps):
            if self.trotter_order == 1:
                self._append_ctrl_fixed_hamiltonian(qc, ht, data, x, dt, 1.0)
                self._ctrl_dgd_forward(qc, ht, data, anc, freqs, x)
            elif self.trotter_order == 2:
                self._append_ctrl_fixed_hamiltonian(qc, ht, data, x, dt / 2.0, 1.0)
                self._ctrl_dgd_forward(qc, ht, data, anc, freqs, x)
                self._append_ctrl_fixed_hamiltonian(qc, ht, data, x, dt / 2.0, 1.0)

    def _append_ctrl_au_adjoint(self, qc, ht, data, anc, freqs, x, tau, r_steps) -> None:
        """Inline ``ht``-controlled :math:`\\mathcal{A}(U)^{\\dagger}` (adjoint)."""
        dt = tau / r_steps
        for _ in range(r_steps):
            if self.trotter_order == 1:
                self._ctrl_dgd_adjoint(qc, ht, data, anc, freqs, x)
                self._append_ctrl_fixed_hamiltonian(qc, ht, data, x, dt, -1.0)
            elif self.trotter_order == 2:
                self._append_ctrl_fixed_hamiltonian(qc, ht, data, x, dt / 2.0, -1.0)
                self._ctrl_dgd_adjoint(qc, ht, data, anc, freqs, x)
                self._append_ctrl_fixed_hamiltonian(qc, ht, data, x, dt / 2.0, -1.0)

    # ------------------------------------------------------------------
    # Public circuit factories
    # ------------------------------------------------------------------

    def build_au(self, num_qubits, x, tau, r_steps):
        """Construct :math:`\\mathcal{A}(U)^{\\dagger}\\,\\mathcal{A}(U)` on the data register.

        Returns
        -------
        QuantumCircuit
            Circuit on ``data + anc + freq_0 + ... + freq_{d-1}`` registers.
        list of QuantumRegister
            The ``d`` per-parameter frequency registers, index-ordered.
        """
        n_s = self.freq_register_size(num_qubits, r_steps)
        data  = QuantumRegister(num_qubits, "data")
        anc   = QuantumRegister(1, "anc")
        freqs = [QuantumRegister(n_s, f"freq_{q}") for q in range(self.model.d)]
        qc    = QuantumCircuit(data, anc, *freqs)

        self._append_au_forward(qc, data, anc, freqs, x, tau, r_steps)
        self._append_au_adjoint(qc, data, anc, freqs, x, tau, r_steps)
        return qc, freqs

    def build_aup(
        self,
        num_qubits,
        x,
        tau,
        r_steps,
        pauli,
        *,
        execution_mode: ExecutionMode = "statevector",
        target_freq: Optional[TargetFreq] = None,
    ):
        """Construct :math:`\\mathcal{A}(U, P)` for a single Pauli observable.

        Circuit structure (``"statevector"`` mode)
        ------------------------------------------
        ``A_forward → PauliGate(P) on data → A_adjoint``. The rank-:math:`d`
        Fourier tensor is read from the
        ``(data=0, anc=0, freq_0=l_0, ..., freq_{d-1}=l_{d-1})`` block of
        the statevector in a single shot.

        Circuit structure (``"hardware"`` mode)
        ---------------------------------------
        Rank-:math:`d` per-frequency Hadamard test (Corollary 2 / Figure 7)
        built via **inline controlled-gate assembly**:

        1. The HT-controlled :math:`\\mathcal{A}(U, P)` is constructed
           gate-by-gate on the outer circuit using the controlled
           primitives in :mod:`._controlled_ops` — c-RZZ / c-RXX / c-RYY
           via basis-conjugated CX·CRZ·CX, CCX (Toffoli) for the
           parity-cascade onto ``anc``, doubly-controlled cV± via MCX
           cascades, and c-PauliGate(P) as a sequence of
           ``cx``/``cy``/``cz``.  Mathematically identical to wrapping
           the inner block in ``.control(1)``, but **never invokes
           HighLevelSynthesis' controlled-unitary fallback**.
        2. Outer HT:
           ``H → c-A(U,P) (inline) → rank-d freq-selector CX ladder →
           H → measure(ht_control)``.  The selector shifts every
           ``freq_k`` simultaneously.

        Parameters
        ----------
        num_qubits : int
            Number of data qubits.
        x : InputX
            Classical input encoding the fixed part of :math:`H(x, \\alpha)`.
        tau, r_steps, pauli
            See :meth:`CircuitBuilder.build_aup`.
        execution_mode : {"statevector", "hardware_base", "hardware"}
            Execution-mode discriminant.
        target_freq : tuple of int, optional
            Required in ``"hardware"`` mode. Length-:math:`d` tuple of
            unsigned ints :math:`(u_0, \\ldots, u_{d-1})` with
            :math:`u_k \\in [0, 2^{n_s})`. Ignored otherwise.

        Returns
        -------
        QuantumCircuit
            Assembled circuit.
        list of QuantumRegister
            The ``d`` per-parameter frequency registers, index-ordered.
        """
        if execution_mode not in ("statevector", "hardware", "hardware_base"):
            raise ValueError(
                f"execution_mode must be 'statevector', 'hardware', or "
                f"'hardware_base', got {execution_mode!r}"
            )

        n_s = self.freq_register_size(num_qubits, r_steps)
        data  = QuantumRegister(num_qubits, "data")
        anc   = QuantumRegister(1, "anc")
        freqs = [QuantumRegister(n_s, f"freq_{q}") for q in range(self.model.d)]

        if execution_mode == "statevector":
            # Bare circuit; rank-d tensor read in one statevector shot.
            qc = QuantumCircuit(data, anc, *freqs)
            self._append_au_forward(qc, data, anc, freqs, x, tau, r_steps)
            qc.append(PauliGate(pauli), list(data))
            self._append_au_adjoint(qc, data, anc, freqs, x, tau, r_steps)
            return qc, freqs

        # ---- "hardware" / "hardware_base" inline-controlled assembly ----
        # Build c-A(U, P) gate-by-gate using the controlled-rotation,
        # controlled-Pauli, and doubly-controlled-cV± primitives in
        # :mod:`._controlled_ops`.  Mathematically identical to wrapping
        # the inner A(U, P) block in ``.control(1, annotated=True)`` but
        # **synthesised entirely in the CX / CRX / CRY / CRZ / CCX / MCX
        # basis** — no Quantum-Shannon decomposition fallback.  This is
        # the optimisation that turned the n=3 hardware-mode runs from
        # tens of minutes into seconds.

        ht_control = QuantumRegister(1, "ht_control")
        creg = ClassicalRegister(1, "creg")
        qc = QuantumCircuit(ht_control, data, anc, *freqs, creg)
        ht = ht_control[0]

        qc.h(ht)
        # c-A(U) (forward) ─ c-PauliGate(P) ─ c-A(U)† (adjoint)
        # — the entire HT-controlled inner block, inlined.
        self._append_ctrl_au_forward(qc, ht, data, anc, freqs, x, tau, r_steps)
        append_ctrl_pauli_string(qc, ht, list(data), pauli)
        self._append_ctrl_au_adjoint(qc, ht, data, anc, freqs, x, tau, r_steps)

        if execution_mode == "hardware_base":
            # Prefix only; caller appends the rank-d CX ladder + HT tail.
            return qc, freqs

        # ---- "hardware" mode: finalise HT for a single target_freq tuple.
        if target_freq is None:
            raise ValueError(
                "execution_mode='hardware' requires target_freq "
                f"(tuple of {self.model.d} unsigned ints u_k = l_k mod 2^n_s) "
                "for the per-frequency Hadamard test."
            )
        if isinstance(target_freq, int):
            raise ValueError(
                f"SeparateRegistersBuilder (d={self.model.d}) expects "
                f"target_freq as a tuple of length {self.model.d}, "
                f"got a scalar int {target_freq!r}"
            )
        target_tuple = tuple(int(u) for u in target_freq)
        if len(target_tuple) != self.model.d:
            raise ValueError(
                f"target_freq tuple length {len(target_tuple)} "
                f"does not match model.d={self.model.d}"
            )
        for k, u_k in enumerate(target_tuple):
            if not 0 <= u_k < (1 << n_s):
                raise ValueError(
                    f"target_freq[{k}]={u_k} out of range "
                    f"[0, 2^n_s={1 << n_s})"
                )

        # Rank-d frequency-selector CX ladder across every freq register.
        self._append_freq_selector(qc, ht_control, freqs, target_tuple)
        qc.h(ht_control[0])
        qc.measure(ht_control[0], creg[0])

        return qc, freqs
