"""Shared-register A(U, P) circuit builder — d = 1 regime (Method A).

Fourier features b_l(x) are encoded onto a SINGLE frequency register of
size n_s = ceil(log2(4r + 1)).  This is the original construction of
Figure 4 / Corollary 1 in Barthe et al. (2025).

Concrete logic is ported from the legacy ``CircuitBuilder`` in
``src/quantum_routines.py`` (the d = 1 parts only — the old
``KernelCircuitBuilder`` in that file is dead code and NOT used here).
"""

from __future__ import annotations

from typing import List, Tuple

from qiskit import QuantumCircuit, QuantumRegister

from .._types import InputX, PauliString
from ..hamiltonians.base import HamiltonianModel
from .base import CircuitBuilder


class SharedRegisterBuilder(CircuitBuilder):
    """A(U) / A(U, P) builder for the d = 1 regime."""

    def freq_register_size(self, r_steps: int) -> int:
        raise NotImplementedError(
            "SharedRegisterBuilder.freq_register_size — concrete logic pending approval."
        )

    def build_au(
        self,
        model: HamiltonianModel,
        x: InputX,
        tau: float,
        r_steps: int,
    ) -> Tuple[QuantumCircuit, List[QuantumRegister]]:
        raise NotImplementedError(
            "SharedRegisterBuilder.build_au — concrete logic pending approval."
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
            "SharedRegisterBuilder.build_aup — concrete logic pending approval."
        )
