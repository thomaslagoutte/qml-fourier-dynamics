"""Abstract base class for Hermitian observables O = sum_h beta_h P_h.

Observables are *explicitly* exposed as a list of (Pauli string, coefficient)
terms.  The :class:`FeatureExtractor` iterates over these terms and sums
the per-term feature tensors classically — never folds them into a single
``SparsePauliOp`` passed to Qiskit.  See
:mod:`quantum_learning_dynamics.features.base` for why.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Sequence

from qiskit.quantum_info import SparsePauliOp

from .._types import PauliString


@dataclass(frozen=True)
class PauliTerm:
    """One term in an observable: ``coefficient * pauli_string``.

    Attributes
    ----------
    pauli : PauliString
        Qiskit little-endian Pauli string (rightmost char acts on qubit 0).
    coefficient : float
        Real coefficient beta_h.  Imaginary parts are not supported because
        observables are Hermitian and we extract only real Fourier features.
    """

    pauli: PauliString
    coefficient: float


class Observable(ABC):
    """Hermitian observable O acting on ``num_qubits`` data qubits.

    Subclasses must override :meth:`terms`, which returns the Pauli
    decomposition.  Single-Pauli observables return a length-1 list;
    composite observables (staggered magnetization, electric flux, ...)
    return one entry per Pauli string.

    Immutable physics constraint
    ----------------------------
    Composite observables MUST NOT be pushed into Qiskit as a single
    ``SparsePauliOp`` within the ``A(U, P)`` extraction circuit.  The
    amplitude-encoding identity exploited by ``A(U, P)`` applies per-Pauli;
    summing first breaks it.  Linearity is enforced at the feature level
    (see :class:`FeatureExtractor`).
    """

    # -- subclass API ----------------------------------------------------

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """Number of data qubits this observable acts on."""

    @abstractmethod
    def terms(self) -> Sequence[PauliTerm]:
        """Return the Pauli decomposition.  Must be non-empty."""

    # -- convenience -----------------------------------------------------

    def __iter__(self) -> Iterator[PauliTerm]:
        return iter(self.terms())

    def __len__(self) -> int:
        return len(self.terms())

    @property
    def is_single_pauli(self) -> bool:
        """True iff the observable is a single Pauli string (|terms| == 1)."""
        return len(self.terms()) == 1

    def to_sparse_pauli_op(self) -> SparsePauliOp:
        """Assemble a Qiskit :class:`SparsePauliOp` for *reference* only.

        Used for exact label computation on the dense Hamiltonian matrix
        (where linearity of expectation values is fine).  It is NEVER
        passed to the A(U,P) builder.
        """
        paulis = [t.pauli for t in self.terms()]
        coeffs = [t.coefficient for t in self.terms()]
        return SparsePauliOp(paulis, coeffs)
