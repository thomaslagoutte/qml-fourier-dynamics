"""Quantum overlap kernel circuit builder for Figure 8 of Barthe et al. (2025).

This module implements the True Quantum Overlap Kernel. Rather than extracting
the full Fourier tensor ``b(x)``, which scales as ``(4r + 1)^d`` per training
point in the ``d > 1`` regime, this builder evaluates the Gram matrix entry:

    K(x_1, x_2) = Re ⟨ψ(x_1) | ψ(x_2)⟩

via a single Hadamard-test circuit per pair. Here,
``|ψ(x)⟩ = A(U, P) |0⟩`` is the prepared state; the HT measurement on
``ht_control`` projects the real part of the full inner product across
the (data, ancilla, frequency) basis.

Inline controlled-gate assembly
-------------------------------
Earlier revisions of this builder wrapped the inner ``A(U, P)_{x_1} ∘
A(U, P)_{x_2}`` block in ``inner.to_gate().control(1, annotated=True)``.
Even with ``annotated=True``, the controlled gate is eventually
synthesised by Qiskit's ``HighLevelSynthesis`` pass via the default
``ControlModifier`` plugin, which falls back to Quantum-Shannon
decomposition for any dense unitary.  For a 16-17 qubit inner block
this dominated runtime — synthesis alone took tens of minutes per pair
on n=3, r=2.

The current implementation builds c-(A_x1 ∘ A_x2) **gate-by-gate**
using the singly- and doubly-controlled primitives in
:mod:`._controlled_ops`:

  * c-RZZ via ``CX → CRZ → CX``
  * c-RXX / c-RYY via unconditional basis change + c-RZZ
  * c-CX (parity cascade onto ``anc``) via Toffoli (``CCX``)
  * c-cV±  via MCX cascade with the additional ``ht`` control
  * c-PauliGate(P) via ``cx`` / ``cy`` / ``cz`` per non-I character

Mathematically identical to ``.control(1)``; computationally, it
synthesises in the standard CX / CR* / CCX / MCX basis without any
QSD fallback.  See :mod:`._controlled_ops` for the algebra justifying
the unconditional basis-change "sandwich trick".

Circuit Construction
--------------------
For a pair of inputs ``(x_1, x_2)`` and Pauli observables ``P_h``,
``P_{h'}`` (use ``pauli2 is None`` for the single-term case):

1. Outer registers: ``ht_control[1] | data[n] | anc[1] | freq_0[n_s] |
   ... | freq_{d-1}[n_s] | creg[1]``.

2. Apply ``H(ht_control)``.

3. Inline ``c-A(U, P_h)_{x_1}``: c-A(U) forward → c-PauliGate(P_h) →
   c-A(U)† adjoint, all on the outer circuit, every elementary gate
   replaced by its controlled twin.

4. Inline ``c-A(U, P_{h'})_{x_2}``: same, with ``x_2`` and ``P_{h'}``.
   Because ``A(U, P)`` is Hermitian, this is exactly the textbook
   Hadamard-test sequence ``A(x_2)† · A(x_1)``.

5. Apply ``H(ht_control)`` and ``measure(ht_control[0], creg[0])``.

6. Estimator: ``K(x_1, x_2) ≈ 2 · P(ht = 0) − 1``.
"""

from __future__ import annotations

from typing import Optional

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from .._types import InputX, PauliString
from ._controlled_ops import append_ctrl_pauli_string
from .separate_registers import SeparateRegistersBuilder


class KernelOverlapBuilder(SeparateRegistersBuilder):
    """Builds the quantum-overlap-kernel circuit (Figure 8 of Barthe et al. 2025).

    Inherits from :class:`SeparateRegistersBuilder` to reuse the
    inhomogeneous A(U) forward/adjoint primitives and the shared gate
    cache.  Functions correctly for both ``d = 1`` and ``d > 1`` regimes.

    Parameters
    ----------
    model : HamiltonianModel
        The Hamiltonian model defining the physical system, including the
        dimensionality ``d`` and upload Paulis.
    trotter_order : int, default=2
        The Trotter-Suzuki decomposition order. Must match the order used
        during classical label generation. Valid values are 1 or 2.
    """

    def build_overlap(
        self,
        num_qubits: int,
        x1: InputX,
        x2: InputX,
        tau: float,
        r_steps: int,
        pauli1: PauliString,
        pauli2: Optional[PauliString] = None,
    ) -> QuantumCircuit:
        """Constructs the Figure 8 overlap-kernel Hadamard-test circuit.

        Parameters
        ----------
        num_qubits : int
            Number of data qubits in the system.
        x1, x2 : InputX
            The two input samples whose overlap is being estimated.
        tau : float
            Total evolution time for the dynamics.
        r_steps : int
            Number of Trotter steps.
        pauli1 : PauliString
            Pauli observable term ``P_h`` applied to the ``x1`` sequence.
        pauli2 : PauliString, optional
            Pauli observable term ``P_{h'}`` applied to the ``x2``
            sequence.  Defaults to ``pauli1`` for the standard
            single-term overlap.  Pass a distinct Pauli to build the
            cross-term ``K_{h, h'}`` circuit needed by composite
            observables (see :class:`HardwareKernelEvaluator`).

        Returns
        -------
        QuantumCircuit
            Self-contained circuit ready for V2 Sampler submission.
            Only ``ht_control[0]`` is measured (into ``creg[0]``); the
            data / anc / freq registers are unmeasured because the HT
            projects the per-overlap statistic onto ``ht_control``'s
            ``⟨Z⟩`` directly.
        """
        if pauli2 is None:
            pauli2 = pauli1

        n_s = self.freq_register_size(num_qubits, r_steps)
        d = self.model.d

        # ---- Outer Hadamard-test circuit registers --------------------
        ht_control = QuantumRegister(1, "ht_control")
        data       = QuantumRegister(num_qubits, "data")
        anc        = QuantumRegister(1, "anc")
        freqs      = [QuantumRegister(n_s, f"freq_{q}") for q in range(d)]
        creg       = ClassicalRegister(1, "creg")
        qc         = QuantumCircuit(ht_control, data, anc, *freqs, creg)
        ht         = ht_control[0]

        # ---- Hadamard-test envelope -----------------------------------
        qc.h(ht)

        # Inline c-A(U, P_h)_{x_1}: forward → c-P_h → adjoint, every
        # elementary gate replaced by its controlled twin.  No
        # ``to_gate().control()`` ⇒ no Quantum-Shannon synthesis.
        self._append_ctrl_au_forward(qc, ht, data, anc, freqs, x1, tau, r_steps)
        append_ctrl_pauli_string(qc, ht, list(data), pauli1)
        self._append_ctrl_au_adjoint(qc, ht, data, anc, freqs, x1, tau, r_steps)

        # Inline c-A(U, P_{h'})_{x_2}: since A(U, P) is Hermitian, this
        # second forward block realises the textbook Hadamard-test
        # sequence A(x_2)† · A(x_1) on the ht=1 branch.
        self._append_ctrl_au_forward(qc, ht, data, anc, freqs, x2, tau, r_steps)
        append_ctrl_pauli_string(qc, ht, list(data), pauli2)
        self._append_ctrl_au_adjoint(qc, ht, data, anc, freqs, x2, tau, r_steps)

        qc.h(ht)
        qc.measure(ht, creg[0])

        return qc