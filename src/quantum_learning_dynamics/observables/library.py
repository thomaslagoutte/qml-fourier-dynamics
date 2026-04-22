"""Concrete physical observables for the TFIM and Schwinger models.

Provides declarative subclasses of :class:`Observable`. These classes 
define the measurement operators without dictating circuit construction.
"""

from __future__ import annotations

from typing import Iterator

from .._types import PauliString
from .base import Observable, PauliTerm


def _single_site(op: str, site: int, num_qubits: int) -> PauliString:
    """Constructs a single-site Pauli string utilizing Qiskit little-endian ordering."""
    chars = ["I"] * num_qubits
    chars[num_qubits - 1 - site] = op
    return "".join(chars)


def _two_site(op_i: str, site_i: int, op_j: str, site_j: int, num_qubits: int) -> PauliString:
    """Constructs a two-site Pauli string utilizing Qiskit little-endian ordering."""
    chars = ["I"] * num_qubits
    chars[num_qubits - 1 - site_i] = op_i
    chars[num_qubits - 1 - site_j] = op_j
    return "".join(chars)


class LocalMagnetization(Observable):
    """Single-site Pauli-Z magnetization: :math:`O = Z_i`."""

    def __init__(self, num_qubits: int, site: int = 0) -> None:
        if not (0 <= site < num_qubits):
            raise ValueError(f"site={site} out of range [0, {num_qubits})")
        self._num_qubits = num_qubits
        self.site = site

    def num_qubits(self) -> int:
        return self._num_qubits

    def terms(self) -> Iterator[PauliTerm]:
        yield PauliTerm(
            pauli=_single_site("Z", self.site, self._num_qubits),
            coefficient=1.0,
        )


class LocalPauli(Observable):
    """Arbitrary single-site Pauli observable: :math:`O = P_i`."""

    def __init__(self, num_qubits: int, op: str, site: int = 0) -> None:
        if op not in ("I", "X", "Y", "Z"):
            raise ValueError(f"op must be one of 'I', 'X', 'Y', 'Z', got {op}")
        if not (0 <= site < num_qubits):
            raise ValueError(f"site={site} out of range [0, {num_qubits})")
        self._num_qubits = num_qubits
        self.op = op
        self.site = site

    def num_qubits(self) -> int:
        return self._num_qubits

    def terms(self) -> Iterator[PauliTerm]:
        yield PauliTerm(
            pauli=_single_site(self.op, self.site, self._num_qubits),
            coefficient=1.0,
        )


class ElectricFlux(Observable):
    """Electric flux observable on a gauge link: :math:`O = X_l`.
    
    Specific to the 1D Z_2 Lattice Gauge Theory (Schwinger model).
    """

    def __init__(self, num_qubits: int, link_index: int = 0) -> None:
        self._num_qubits = num_qubits
        self.link_index = link_index
        self.link_qubit = 2 * link_index + 1
        
        if not (0 <= self.link_qubit < num_qubits):
            raise ValueError(f"link_qubit {self.link_qubit} out of range [0, {num_qubits})")

    def num_qubits(self) -> int:
        return self._num_qubits

    def terms(self) -> Iterator[PauliTerm]:
        yield PauliTerm(
            pauli=_single_site("X", self.link_qubit, self._num_qubits),
            coefficient=1.0,
        )


class StaggeredMagnetization(Observable):
    """Normalized staggered magnetization: :math:`O = \\frac{1}{N} \\sum_i (-1)^i Z_i`."""

    def __init__(self, num_qubits: int) -> None:
        self._num_qubits = num_qubits

    def num_qubits(self) -> int:
        return self._num_qubits

    def terms(self) -> Iterator[PauliTerm]:
        norm = 1.0 / self._num_qubits
        for i in range(self._num_qubits):
            sign = -1.0 if (i % 2) else 1.0
            yield PauliTerm(
                pauli=_single_site("Z", i, self._num_qubits),
                coefficient=norm * sign,
            )


class TwoPointZZCorrelator(Observable):
    """Two-site spatial correlator: :math:`O = Z_i Z_j`."""

    def __init__(self, num_qubits: int, site_i: int, site_j: int) -> None:
        if not (0 <= site_i < num_qubits) or not (0 <= site_j < num_qubits):
            raise ValueError(f"Sites ({site_i}, {site_j}) must be in [0, {num_qubits})")
        self._num_qubits = num_qubits
        self.site_i = site_i
        self.site_j = site_j

    def num_qubits(self) -> int:
        return self._num_qubits

    def terms(self) -> Iterator[PauliTerm]:
        yield PauliTerm(
            pauli=_two_site("Z", self.site_i, "Z", self.site_j, self._num_qubits),
            coefficient=1.0,
        )