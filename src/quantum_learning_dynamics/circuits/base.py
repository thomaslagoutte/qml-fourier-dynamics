"""Abstract base class for A(U) / A(U, P) Fourier-extraction circuit builders.

Two concrete implementations live in this subpackage:

* :class:`SharedRegisterBuilder`    — d = 1 regime (Method A, Fig. 4 of paper).
  Single frequency register shared across all A-tensor indices.
  Sourced from the d=1 parts of the legacy ``quantum_routines.py``.

* :class:`SeparateRegistersBuilder` — d > 1 regime (Method B, Fig. 8 adapted).
  One frequency register per unknown parameter.
  Sourced exclusively from ``quantum_routines_kernel.py``
  (``KernelCircuitMixin``); the old ``KernelCircuitBuilder`` in
  ``quantum_routines.py`` is dead code and MUST NOT be reproduced.

Both builders share the same cached V+/- gate infrastructure via
:class:`VGateCache`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

from qiskit import QuantumCircuit, QuantumRegister

from .._types import InputX, PauliString
from ..hamiltonians.base import HamiltonianModel
from .gate_cache import VGateCache


class CircuitBuilder(ABC):
    """Builds A(U) and A(U, P) Fourier-extraction circuits.

    The builder is stateless with respect to (x, tau, r_steps) — each call
    returns a fresh circuit.  The only mutable state is the
    :attr:`gate_cache`, which memoises V+/- gates across calls.
    """

    def __init__(self, gate_cache: VGateCache | None = None) -> None:
        self.gate_cache: VGateCache = gate_cache or VGateCache()

    # -- core primitives -------------------------------------------------

    @abstractmethod
    def freq_register_size(self, r_steps: int) -> int:
        """Number of qubits n_s needed per frequency register for r Trotter steps.

        The legacy convention is ``n_s = ceil(log2(4 * r_steps + 1))``.
        """

    @abstractmethod
    def build_au(
        self,
        model: HamiltonianModel,
        x: InputX,
        tau: float,
        r_steps: int,
    ) -> Tuple[QuantumCircuit, List[QuantumRegister]]:
        """Build the A(U) circuit (no observable applied).

        Returns
        -------
        qc : QuantumCircuit
            The A(U) circuit.
        freq_registers : list[QuantumRegister]
            * length 1 for :class:`SharedRegisterBuilder` (single shared register)
            * length d for :class:`SeparateRegistersBuilder` (one per unknown)
        """

    @abstractmethod
    def build_aup(
        self,
        model: HamiltonianModel,
        x: InputX,
        tau: float,
        r_steps: int,
        pauli: PauliString,
    ) -> Tuple[QuantumCircuit, List[QuantumRegister]]:
        """Build the A(U, P) circuit for a *single* Pauli observable ``P``.

        Called by :class:`FeatureExtractor` once per Pauli term when the
        observable is composite — see the linearity invariant in
        :mod:`quantum_learning_dynamics.features.base`.
        """

    # -- convenience -----------------------------------------------------

    def __repr__(self) -> str:
        return f"{type(self).__name__}(cache_size={len(self.gate_cache)})"
