"""Transverse-Field Ising Models (TFIM).

Implements both the homogeneous (``d = 1``) and inhomogeneous (``d = n``)
variants of the Transverse-Field Ising Model.
"""

from __future__ import annotations

import itertools
from typing import List, Tuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from .._types import AlphaVector, InputX, PauliString
from .base import HamiltonianModel


def _single_site(op: str, site: int, num_qubits: int) -> str:
    """Constructs a single-site Pauli string using Qiskit little-endian ordering."""
    chars = ["I"] * num_qubits
    chars[num_qubits - 1 - site] = op
    return "".join(chars)


def _two_site(op_i: str, i: int, op_j: str, j: int, num_qubits: int) -> str:
    """Constructs a two-site Pauli string using Qiskit little-endian ordering."""
    chars = ["I"] * num_qubits
    chars[num_qubits - 1 - i] = op_i
    chars[num_qubits - 1 - j] = op_j
    return "".join(chars)


class TFIM(HamiltonianModel):
    """Homogeneous Transverse-Field Ising Model.

    The Hamiltonian is defined as:
    :math:`H(x, \\alpha) = \\sum_{(i,j) \\in x} Z_i Z_j + \\alpha \\sum_i X_i`

    Because the transverse field strength :math:`\\alpha` is uniform across
    all qubits, this model resides in the ``d = 1`` regime.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the system. Minimum 2.
    edge_prob : float, default=0.5
        Probability of an edge existing when sampling random graphs.
    alpha_range : Tuple[float, float], default=(0.5, 1.5)
        The uniform sampling domain for the global parameter :math:`\\alpha`.
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
        return [_single_site("X", i, self.num_qubits) for i in range(self.num_qubits)]

    def hamiltonian(self, x: InputX, alpha: AlphaVector) -> SparsePauliOp:
        if len(alpha) != 1:
            raise ValueError(f"TFIM expects |α| == 1, got {len(alpha)}")
        
        n = self.num_qubits
        terms: List[Tuple[str, float]] = []
        
        for (i, j) in x:
            terms.append((_two_site("Z", i, "Z", j, n), 1.0))
        for i in range(n):
            terms.append((_single_site("X", i, n), float(alpha[0])))
            
        return SparsePauliOp.from_list(terms)

    def sample_x(self, rng: np.random.Generator) -> InputX:
        edges = []
        for i, j in itertools.combinations(range(self.num_qubits), 2):
            if rng.random() < self.edge_prob:
                edges.append((i, j))
        return edges

    def sample_alpha(self, rng: np.random.Generator) -> AlphaVector:
        val = rng.uniform(self.alpha_range[0], self.alpha_range[1])
        return np.array([val], dtype=np.float64)


class InhomogeneousTFIM(HamiltonianModel):
    """Inhomogeneous Transverse-Field Ising Model.

    The Hamiltonian is defined as:
    :math:`H(x, \\alpha) = \\sum_{(i,j) \\in x} Z_i Z_j + \\sum_i \\alpha_i X_i`

    Because the transverse field strength varies per site, the parameter
    dimension scales with the system size, placing this model in the
    ``d = n`` regime.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the system. Minimum 2.
    edge_prob : float, default=0.5
        Probability of an edge existing when sampling random graphs.
    alpha_range : Tuple[float, float], default=(0.5, 1.5)
        The uniform sampling domain for each parameter :math:`\\alpha_i`.
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
        edges = []
        for i, j in itertools.combinations(range(self.num_qubits), 2):
            if rng.random() < self.edge_prob:
                edges.append((i, j))
        return edges

    def sample_alpha(self, rng: np.random.Generator) -> AlphaVector:
        return rng.uniform(self.alpha_range[0], self.alpha_range[1], size=self.d)