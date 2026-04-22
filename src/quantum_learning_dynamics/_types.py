"""Core type aliases used throughout the package for static analysis."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np

# -----------------------------------------------------------------------------
# Quantum State & Operators
# -----------------------------------------------------------------------------

PauliString: TypeAlias = str
"""A Qiskit little-endian Pauli string. 

Convention: The rightmost character acts on qubit 0 (e.g., "IIIZ" applies Z on qubit 0).
"""

# -----------------------------------------------------------------------------
# Mathematical Inputs & Parameters
# -----------------------------------------------------------------------------

InputX: TypeAlias = Any
"""The known structural encoding of the Hamiltonian, :math:`x` in :math:`H(x, \\alpha)`.

The concrete data structure varies by :class:`HamiltonianModel`. For example:
- Transverse-Field Ising Models: ``List[Tuple[int, int]]`` representing graph edges.
- Schwinger Models: ``List[bool]`` representing gauge-link activation masks.
"""

AlphaVector: TypeAlias = "np.ndarray | float"
"""The continuous, unknown parameter vector :math:`\\alpha` of dimension :math:`d`.

When :math:`d = 1`, this is evaluated as a scalar float. When :math:`d > 1`, 
it is a 1-D ``np.ndarray`` of shape ``(d,)``.
"""

# -----------------------------------------------------------------------------
# Machine Learning Data Structures
# -----------------------------------------------------------------------------

LabelVector: TypeAlias = np.ndarray
"""1-D array of shape ``(num_samples,)`` containing real-valued target scalars."""

FeatureTensor: TypeAlias = np.ndarray
"""Rank-:math:`d` tensor of shape ``(4r+1, ..., 4r+1)`` containing Fourier coefficients."""

FeatureMatrix: TypeAlias = np.ndarray
"""2-D array of shape ``(num_samples, (4r+1)**d)`` containing flattened feature tensors."""

GramMatrix: TypeAlias = np.ndarray
"""2-D array containing pairwise overlap evaluations :math:`K(x_i, x_j)`."""