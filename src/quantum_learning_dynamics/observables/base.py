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
from typing import Iterator

from qiskit.quantum_info import SparsePauliOp

from .._types import PauliString


@dataclass(frozen=True)
class PauliTerm:
    """One real-coefficient Pauli term in an observable's linear decomposition."""
    pauli: PauliString
    coefficient: float


class Observable(ABC):
    """Observable expressed as O = Σ_h c_h · P_h (real coefficients).

    The `terms()` iterator EXPOSES the decomposition. This is load-bearing:
    FeatureExtractor.extract iterates the terms and sums
        feature(x) = Σ_h c_h · b_h(x)
    where each b_h is extracted from an A(U, P_h) circuit built from ONE Pauli
    string. Composite observables must NEVER be folded into a single
    SparsePauliOp for extraction — that would break the statevector-amplitude
    indexing that the extractors rely on.
    """

    @abstractmethod
    def num_qubits(self) -> int: ...

    @abstractmethod
    def terms(self) -> Iterator[PauliTerm]: ...

    # --- dense-evaluation helper (used only for ground-truth labels) ---------

    def to_sparse_pauli_op(self) -> SparsePauliOp:
        """For exact label computation ONLY. Do NOT pass this into an extractor."""
        return SparsePauliOp.from_list(
            [(t.pauli, t.coefficient) for t in self.terms()]
        )