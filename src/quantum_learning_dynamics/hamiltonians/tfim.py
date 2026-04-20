"""Transverse-field Ising models (d = 1 and d = n variants).

Both models share the ZZ interaction pattern
    H_zz(x) = sum_{(i,j) in x} Z_i Z_j
but differ in how they expose the unknown transverse field:

* :class:`TFIM`                — single homogeneous alpha (d = 1)
* :class:`InhomogeneousTFIM`   — per-qubit alpha_i (d = n)

These stubs preserve the external API of the legacy
``IsingTransverseFieldModel`` and ``InhomogeneousTFIM`` in
``src/models.py`` but expose ``d`` explicitly so that
:class:`Experiment` can auto-route on it.

Concrete implementations are populated in a later step, after the API
skeleton is approved.  Each method currently raises
``NotImplementedError`` so the import graph is wired correctly.
"""

from __future__ import annotations

import itertools
from typing import List, Tuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from .._types import AlphaVector, InputX, PauliString
from .base import HamiltonianModel


def _single_site(op: str, site: int, num_qubits: int) -> str:
    """Little-endian Pauli: rightmost char == qubit 0."""
    chars = ["I"] * num_qubits
    chars[num_qubits - 1 - site] = op
    return "".join(chars)


def _two_site(op_i: str, i: int, op_j: str, j: int, num_qubits: int) -> str:
    chars = ["I"] * num_qubits
    chars[num_qubits - 1 - i] = op_i
    chars[num_qubits - 1 - j] = op_j
    return "".join(chars)


class TFIM(HamiltonianModel):
    """Homogeneous transverse-field Ising model, d = 1.

        H(x, α) = Σ_{(i,j) ∈ x} Z_i Z_j  +  α · Σ_i X_i
    """

    def __init__(
        self,
        num_qubits: int,
        edge_prob: float = 0.5,
        alpha_range: Tuple[float, float] = (0.5, 1.5),
    ) -> None:
        if num_qubits < 2:
            raise ValueError(f"TFIM requires num_qubits >= 2, got {num_qubits}")
        self.num_qubits = num_qubits
        self.d = 1
        self.edge_prob = float(edge_prob)
        self.alpha_range = alpha_range

    @property
    def upload_paulis(self) -> List[PauliString]:
        # d = 1 but each of the n qubits carries the SAME α — the shared-register
        # builder iterates internally. We expose a canonical length-1 single-qubit
        # upload so multi-qubit-parity logic stays off for TFIM.
        return [_single_site("X", i, self.num_qubits) for i in range(self.num_qubits)]

    def hamiltonian(self, x: InputX, alpha: AlphaVector) -> SparsePauliOp:
        if len(alpha) != 1:
            raise ValueError(f"TFIM expects |α| == 1, got {len(alpha)}")
        a = float(alpha[0])
        n = self.num_qubits
        terms: List[Tuple[str, float]] = []
        for (i, j) in x:
            terms.append((_two_site("Z", i, "Z", j, n), 1.0))
        for i in range(n):
            terms.append((_single_site("X", i, n), a))
        return SparsePauliOp.from_list(terms)

    def sample_x(self, rng: np.random.Generator) -> InputX:
        return [
            (i, j)
            for (i, j) in itertools.combinations(range(self.num_qubits), 2)
            if rng.random() < self.edge_prob
        ]

    def sample_alpha(self, rng: np.random.Generator) -> AlphaVector:
        lo, hi = self.alpha_range
        return rng.uniform(lo, hi, size=1)


class InhomogeneousTFIM(HamiltonianModel):
    """Per-qubit transverse field, d = num_qubits.

        H(x, α) = Σ_{(i,j) ∈ x} Z_i Z_j  +  Σ_i α_i X_i
    """

    def __init__(
        self,
        num_qubits: int,
        edge_prob: float = 0.5,
        alpha_range: Tuple[float, float] = (0.5, 1.5),
    ) -> None:
        if num_qubits < 2:
            raise ValueError(f"InhomogeneousTFIM requires num_qubits >= 2, got {num_qubits}")
        self.num_qubits = num_qubits
        self.d = num_qubits
        self.edge_prob = float(edge_prob)
        self.alpha_range = alpha_range

    @property
    def upload_paulis(self) -> List[PauliString]:
        # One weight-1 X upload per qubit, ordered by qubit index.
        return [_single_site("X", i, self.num_qubits) for i in range(self.num_qubits)]

    def hamiltonian(self, x: InputX, alpha: AlphaVector) -> SparsePauliOp:
        if len(alpha) != self.num_qubits:
            raise ValueError(
                f"InhomogeneousTFIM expects |α| == {self.num_qubits}, got {len(alpha)}"
            )
        n = self.num_qubits
        terms: List[Tuple[str, float]] = []
        for (i, j) in x:
            terms.append((_two_site("Z", i, "Z", j, n), 1.0))
        for i in range(n):
            terms.append((_single_site("X", i, n), float(alpha[i])))
        return SparsePauliOp.from_list(terms)

    def sample_x(self, rng: np.random.Generator) -> InputX:
        return [
            (i, j)
            for (i, j) in itertools.combinations(range(self.num_qubits), 2)
            if rng.random() < self.edge_prob
        ]

    def sample_alpha(self, rng: np.random.Generator) -> AlphaVector:
        lo, hi = self.alpha_range
        return rng.uniform(lo, hi, size=self.num_qubits)