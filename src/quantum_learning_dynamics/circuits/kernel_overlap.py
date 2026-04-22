"""Quantum overlap kernel circuit builder for Figure 8 of Barthe et al. (2025).

This module implements the True Quantum Overlap Kernel. Rather than extracting
the full Fourier tensor ``b(x)``, which scales as ``(4r + 1)^d`` per training
point in the ``d > 1`` regime, this builder evaluates the Gram matrix entry:

    K(x_1, x_2) = Re ⟨ψ(x_1) | ψ(x_2)⟩

This is evaluated directly via a single Hadamard-test circuit. Here,
``|ψ(x)⟩ = A(U, P) |0⟩`` represents the prepared state. The inner product
evaluates across the entire basis (data, ancilla, and frequency registers),
yielding a scalar kernel entry per pair.

Circuit Construction
--------------------
For a pair of inputs ``(x_1, x_2)`` and a single Pauli observable ``P``:

1. Inner Unitary Construction:
   The operator ``A(U, P)`` is Hermitian. Consequently, the forward sequence
   is equivalent to the standard Hadamard-test sequence:
       U_inner = A(U, P)_{x_1} ∘ A(U, P)_{x_2}

2. Controlled Execution:
   ``U_inner`` is promoted to a single-qubit-controlled gate using annotated
   synthesis to defer eager Quantum-Shannon decomposition.

3. Hadamard Test:
   The standard sequence is applied to the control qubit:
       H → controlled-U_inner → H → Measure

4. Estimator:
   K(x_1, x_2) ≈ 2 * P(ht=0) - 1
"""

from __future__ import annotations

from typing import Optional

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import PauliGate

from .._types import InputX, PauliString
from .separate_registers import SeparateRegistersBuilder


class KernelOverlapBuilder(SeparateRegistersBuilder):
    """Builds the quantum-overlap-kernel circuit (Figure 8 of Barthe et al. 2025).

    Inherits from :class:`SeparateRegistersBuilder` to utilize the inhomogeneous
    forward/adjoint primitives and the shared gate cache. Functions correctly
    for both ``d = 1`` and ``d > 1`` regimes.

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
        x1 : InputX
            The first input graph or structural sample.
        x2 : InputX
            The second input graph or structural sample.
        tau : float
            Total evolution time for the dynamics.
        r_steps : int
            Number of Trotter steps for the discretized evolution.
        pauli1 : PauliString
            The Pauli observable term applied to the ``x1`` sequence.
        pauli2 : PauliString, optional
            The Pauli observable term applied to the ``x2`` sequence. Defaults
            to ``pauli1`` to compute the standard single-term overlap. Cross-Pauli
            overlaps require explicit specification.

        Returns
        -------
        QuantumCircuit
            The assembled Hadamard-test circuit containing the control qubit,
            data register, ancilla, frequency registers, and classical register.
        """
        if pauli2 is None:
            pauli2 = pauli1

        n_s = self.freq_register_size(num_qubits, r_steps)
        d = self.model.d

        # 1. Construct the inner composite unitary: A(U, P)_x1 ∘ A(U, P)_x2
        inner_data = QuantumRegister(num_qubits, "data")
        inner_anc = QuantumRegister(1, "anc")
        inner_freqs = [QuantumRegister(n_s, f"freq_{q}") for q in range(d)]
        
        inner = QuantumCircuit(
            inner_data, inner_anc, *inner_freqs, name="K_overlap"
        )

        # Apply A(U, P)_x1
        self._append_au_forward(
            inner, inner_data, inner_anc, inner_freqs, x1, tau, r_steps
        )
        inner.append(PauliGate(pauli1), list(inner_data))
        self._append_au_adjoint(
            inner, inner_data, inner_anc, inner_freqs, x1, tau, r_steps
        )

        # Apply A(U, P)_x2
        self._append_au_forward(
            inner, inner_data, inner_anc, inner_freqs, x2, tau, r_steps
        )
        inner.append(PauliGate(pauli2), list(inner_data))
        self._append_au_adjoint(
            inner, inner_data, inner_anc, inner_freqs, x2, tau, r_steps
        )

        # 2. Promote to a single annotated controlled-composite gate
        # Annotated synthesis defers full matrix decomposition, preventing
        # memory exhaustion during transpilation of dense unitary blocks.
        controlled_overlap = inner.to_gate(label="K_overlap").control(
            1, annotated=True
        )

        # 3. Construct the outer Hadamard-test circuit
        ht_control = QuantumRegister(1, "ht_control")
        data = QuantumRegister(num_qubits, "data")
        anc = QuantumRegister(1, "anc")
        freqs = [QuantumRegister(n_s, f"freq_{q}") for q in range(d)]
        creg = ClassicalRegister(1, "creg")
        
        qc = QuantumCircuit(ht_control, data, anc, *freqs, creg)

        # Flatten frequency registers for the gate append
        freq_qubits_flat: list = []
        for fr in freqs:
            freq_qubits_flat.extend(list(fr))

        # 4. Apply the Hadamard-test sequence
        qc.h(ht_control[0])
        qc.append(
            controlled_overlap,
            [ht_control[0], *data, *anc, *freq_qubits_flat],
        )
        qc.h(ht_control[0])
        qc.measure(ht_control[0], creg[0])

        return qc