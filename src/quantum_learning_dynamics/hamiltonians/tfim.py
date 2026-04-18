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

from typing import List, Tuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from .base import HamiltonianModel

EdgeList = List[Tuple[int, int]]


class TFIM(HamiltonianModel):
    """TFIM with a single homogeneous transverse field alpha.

        H(x, alpha) = sum_{(i,j) in x} Z_i Z_j  +  alpha * sum_i X_i

    Known input ``x`` is a list of graph edges ``[(i, j), ...]`` with
    0 <= i < j < num_qubits.  The unknown parameter alpha is a scalar.

    This is the d = 1 regime of the paper (Section VI.A) — learned via
    the shared-register A(U, P) circuit and Lasso on b_l(x).
    """

    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits
        self.d = 1

    def generate_hamiltonian(self, x: EdgeList, alpha: float) -> SparsePauliOp:
        raise NotImplementedError("TFIM.generate_hamiltonian — concrete logic pending approval.")

    def exact_unitary(self, x: EdgeList, alpha: float, tau: float) -> np.ndarray:
        raise NotImplementedError("TFIM.exact_unitary — concrete logic pending approval.")

    def sample_x(self, rng: np.random.Generator) -> EdgeList:
        raise NotImplementedError("TFIM.sample_x — concrete logic pending approval.")

    def sample_alpha(self, rng: np.random.Generator) -> float:
        raise NotImplementedError("TFIM.sample_alpha — concrete logic pending approval.")


class InhomogeneousTFIM(HamiltonianModel):
    """TFIM with per-qubit unknown transverse fields.

        H(x, alpha) = sum_{(i,j) in x} Z_i Z_j  +  sum_i alpha_i X_i

    ``x`` is a graph edge list; ``alpha`` is a 1-D array of shape (n,)
    where n = num_qubits.  This places the model in the d = n regime
    (Section VI.B) — learned via the separate-registers circuit plus
    either the full Fourier tensor + Lasso or the quantum overlap kernel
    + Kernel Ridge Regression.
    """

    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits
        self.d = num_qubits  # one unknown alpha_i per qubit

    def generate_hamiltonian(self, x: EdgeList, alpha: np.ndarray) -> SparsePauliOp:
        raise NotImplementedError(
            "InhomogeneousTFIM.generate_hamiltonian — concrete logic pending approval."
        )

    def exact_unitary(self, x: EdgeList, alpha: np.ndarray, tau: float) -> np.ndarray:
        raise NotImplementedError(
            "InhomogeneousTFIM.exact_unitary — concrete logic pending approval."
        )

    def sample_x(self, rng: np.random.Generator) -> EdgeList:
        raise NotImplementedError(
            "InhomogeneousTFIM.sample_x — concrete logic pending approval."
        )

    def sample_alpha(self, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError(
            "InhomogeneousTFIM.sample_alpha — concrete logic pending approval."
        )
