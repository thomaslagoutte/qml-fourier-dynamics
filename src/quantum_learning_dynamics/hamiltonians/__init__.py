"""Parameterized Hamiltonian models for quantum dynamics simulation.

Provides the physical definitions of the systems whose dynamics are targeted
by the PAC-learning framework. Each model defines its interaction terms,
unknown parameter dimension (`d`), and the requisite Pauli operators for
Fourier feature extraction.
"""

from .base import HamiltonianModel
from .schwinger import SchwingerZ2Model
from .tfim import TFIM, InhomogeneousTFIM

__all__ = [
    "HamiltonianModel",
    "InhomogeneousTFIM",
    "SchwingerZ2Model",
    "TFIM",
]