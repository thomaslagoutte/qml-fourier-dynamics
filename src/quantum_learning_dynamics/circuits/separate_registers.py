"""Separate-registers A(U, P) circuit builder — d > 1 regime (Method B).

One frequency register per unknown parameter.  Total statevector size is
2^(num_qubits + 1 + d * n_s); the d-dimensional Fourier tensor is extracted
from the statevector via :class:`MeshgridTensorExtractor`.

Concrete logic is ported EXCLUSIVELY from
``src/quantum_routines_kernel.py`` — specifically the ``KernelCircuitMixin``
primitives:

* ``_append_au_inhomogeneous_forward``
* ``_append_au_inhomogeneous_adjoint``
* ``build_au_inhomogeneous``
* ``build_aup_inhomogeneous``

The old ``KernelCircuitBuilder`` in ``src/quantum_routines.py`` (around
line 1448) is DEAD CODE and must NOT be reproduced in this package.
"""

from __future__ import annotations

from typing import List, Tuple

from qiskit import QuantumCircuit, QuantumRegister

from .._types import InputX, PauliString
from ..hamiltonians.base import HamiltonianModel
from .base import CircuitBuilder


class SeparateRegistersBuilder(CircuitBuilder):
    """A(U) / A(U, P) builder for the d > 1 regime."""

    def freq_register_size(self, r_steps: int) -> int:
        raise NotImplementedError(
            "SeparateRegistersBuilder.freq_register_size — concrete logic pending approval."
        )

    def build_au(
        self,
        model: HamiltonianModel,
        x: InputX,
        tau: float,
        r_steps: int,
    ) -> Tuple[QuantumCircuit, List[QuantumRegister]]:
        raise NotImplementedError(
            "SeparateRegistersBuilder.build_au — concrete logic pending approval."
        )

    def build_aup(
        self,
        model: HamiltonianModel,
        x: InputX,
        tau: float,
        r_steps: int,
        pauli: PauliString,
    ) -> Tuple[QuantumCircuit, List[QuantumRegister]]:
        raise NotImplementedError(
            "SeparateRegistersBuilder.build_aup — concrete logic pending approval."
        )
