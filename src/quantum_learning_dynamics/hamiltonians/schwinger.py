"""Z2 Schwinger / 1-D lattice gauge-theory Hamiltonian.

Model (interlaced matter / gauge-link layout, Qiskit little-endian):

    qubit 0: matter site 0
    qubit 1: gauge link between matter 0 and 1
    qubit 2: matter site 1
    qubit 3: gauge link between matter 1 and 2
    ...

    H(x, g) = m * sum_i (-1)^i Z_{m_i}                       (staggered mass, known)
            + h * sum_i X_{l_i}                              (electric field, known)
            + sum_{i: x_i == 1} g_i * X_{m_i} Z_{l_i} X_{m_{i+1}}   (XZX coupling, unknown)

* ``x`` is a gauge-link activation mask ``list[int]`` of length
  ``num_matter_sites - 1`` with entries in {0, 1}.  ``x_i = 1`` means link
  i is active and has a coupling g_i > 0 to be learned.
* ``alpha`` is the couplings vector ``g = (g_0, ..., g_{L-1})`` of shape
  ``(num_matter_sites - 1,)``.  (Concrete implementation may choose to
  treat only active links as unknowns; this is a detail resolved at
  implementation time.)

Placed in the d > 1 regime; routed through the separate-registers
circuit and either Lasso-on-tensor or the kernel pipeline depending on
the user's ``method=`` choice.
"""

from __future__ import annotations

from typing import List

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from .base import HamiltonianModel

GaugeMask = List[int]


class SchwingerZ2Model(HamiltonianModel):
    """See module docstring for the Hamiltonian and qubit layout."""

    def __init__(
        self,
        num_matter_sites: int,
        mass: float,
        electric_field: float,
    ) -> None:
        self.num_matter = num_matter_sites
        self.num_gauge = num_matter_sites - 1
        self.num_qubits = self.num_matter + self.num_gauge
        self.d = self.num_gauge  # one unknown g_i per gauge link
        self.mass = mass
        self.electric_field = electric_field

    def generate_hamiltonian(self, x: GaugeMask, alpha: np.ndarray) -> SparsePauliOp:
        raise NotImplementedError(
            "SchwingerZ2Model.generate_hamiltonian — concrete logic pending approval."
        )

    def exact_unitary(self, x: GaugeMask, alpha: np.ndarray, tau: float) -> np.ndarray:
        raise NotImplementedError(
            "SchwingerZ2Model.exact_unitary — concrete logic pending approval."
        )

    def sample_x(self, rng: np.random.Generator) -> GaugeMask:
        raise NotImplementedError(
            "SchwingerZ2Model.sample_x — concrete logic pending approval."
        )

    def sample_alpha(self, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError(
            "SchwingerZ2Model.sample_alpha — concrete logic pending approval."
        )
