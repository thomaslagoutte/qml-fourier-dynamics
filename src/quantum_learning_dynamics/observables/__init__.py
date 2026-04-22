"""Hermitian observables for Fourier feature extraction.

Provides the abstract interface and concrete implementations for physical 
observables. Observables are strictly defined as linear combinations of 
Pauli terms to support individual feature extraction in the PAC-learning framework.
"""

from .base import Observable, PauliTerm
from .library import (
    ElectricFlux,
    LocalMagnetization,
    LocalPauli,
    StaggeredMagnetization,
    TwoPointZZCorrelator,
)

__all__ = [
    "ElectricFlux",
    "LocalMagnetization",
    "LocalPauli",
    "Observable",
    "PauliTerm",
    "StaggeredMagnetization",
    "TwoPointZZCorrelator",
]