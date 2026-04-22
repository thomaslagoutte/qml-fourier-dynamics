"""Qiskit circuit builders for the Fourier-coefficient extraction subroutine.

Exposes the three circuit factories corresponding to the two Hamiltonian-
dimensionality regimes of Barthe et al. (2025) plus the overlap-kernel
variant of Figure 8:

* :class:`SharedRegisterBuilder`     — ``d = 1`` Fourier extraction
  :math:`\\mathcal{A}(U,P)` with a single shared frequency register.
* :class:`SeparateRegistersBuilder`  — ``d > 1`` Fourier extraction
  :math:`\\mathcal{A}(U,P)` with one frequency register per
  :math:`\\alpha` component.
* :class:`KernelOverlapBuilder`      — ``d > 1`` quantum-overlap kernel
  :math:`K(x_1, x_2) = \\mathrm{Re}\\langle\\psi(x_1)|\\psi(x_2)\\rangle`.
"""

from .base import CircuitBuilder
from .gate_cache import VGateCache
from .kernel_overlap import KernelOverlapBuilder
from .separate_registers import SeparateRegistersBuilder
from .shared_register import SharedRegisterBuilder

__all__ = [
    "CircuitBuilder",
    "KernelOverlapBuilder",
    "SeparateRegistersBuilder",
    "SharedRegisterBuilder",
    "VGateCache",
]
