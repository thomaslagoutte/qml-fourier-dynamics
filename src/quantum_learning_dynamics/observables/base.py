"""Abstract base class for Hermitian observables.

Observables are explicitly decomposed into a sum of Pauli operators:
:math:`O = \\sum_h c_h P_h`. This decomposition is mathematically required 
because the Fourier extraction circuit :math:`A(U, P_h)` can only encode a 
single Pauli operator per evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

from qiskit.quantum_info import SparsePauliOp

from .._types import PauliString


@dataclass(frozen=True)
class PauliTerm:
    """A single weighted Pauli operator in an observable's decomposition.
    
    Attributes
    ----------
    pauli : PauliString
        The string representation of the Pauli operator (e.g., "IXY").
    coefficient : float
        The real-valued coefficient weighting this term.
    """
    pauli: PauliString
    coefficient: float


class Observable(ABC):
    """Abstract base class for physical observables.

    Defines an observable strictly as a linear combination of Pauli strings.
    The `terms()` iterator allows execution engines to extract Fourier 
    features independently for each Pauli term, exploiting the linearity 
    of the amplitude encoding.
    """

    @abstractmethod
    def num_qubits(self) -> int:
        """The number of qubits the observable acts upon."""
        pass

    @abstractmethod
    def terms(self) -> Iterator[PauliTerm]:
        """Yields the Pauli terms comprising the observable.

        Yields
        ------
        PauliTerm
            A dataclass containing the Pauli string and its real coefficient.
        """
        pass

    def to_sparse_pauli_op(self) -> SparsePauliOp:
        """Constructs the dense Qiskit operator for exact simulation.

        Used exclusively for computing exact ground-truth labels during 
        classical emulation. This aggregate operator is never passed into 
        a quantum circuit extractor.

        Returns
        -------
        SparsePauliOp
            The combined sparse Pauli operator for the full observable.
        """
        return SparsePauliOp.from_list(
            [(term.pauli, term.coefficient) for term in self.terms()]
        )