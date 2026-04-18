"""Shared type aliases used across the package.

Centralising these keeps the public surface consistent and makes
``mypy --strict`` happier.  Everything here is deliberately tiny — if a type
needs validation or invariants it should become a real class, not an alias.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np

# -----------------------------------------------------------------------------
# Pauli / observable
# -----------------------------------------------------------------------------

PauliString: TypeAlias = str
"""A Qiskit little-endian Pauli string.

Convention (matches Qiskit and the legacy code): the *rightmost* character
acts on qubit 0.  Example for 4 qubits: ``"IIIZ"`` is Z on qubit 0.
"""

# -----------------------------------------------------------------------------
# Input sample x
# -----------------------------------------------------------------------------

InputX: TypeAlias = Any
"""The 'known' part of the Hamiltonian, x in H(x, alpha).

Deliberately ``Any`` because the paper treats x as a generic
problem-dependent encoding.  Concrete :class:`HamiltonianModel` subclasses
spell out their own format in the class docstring, e.g.

* :class:`TFIM` / :class:`InhomogeneousTFIM` — ``list[tuple[int, int]]``
  of graph edges.
* :class:`SchwingerZ2Model` — ``list[int]`` gauge-link activation mask.
"""

# -----------------------------------------------------------------------------
# Parameter vector alpha (the unknown to be PAC-learned)
# -----------------------------------------------------------------------------

AlphaVector: TypeAlias = "np.ndarray | float"
"""The unknown parameter vector alpha of length d.

When ``d == 1`` this is a plain scalar; when ``d > 1`` a 1-D ``np.ndarray``
of shape ``(d,)``.  The :class:`HamiltonianModel.d` attribute is the
source of truth.
"""

# -----------------------------------------------------------------------------
# Labels and features
# -----------------------------------------------------------------------------

LabelVector: TypeAlias = np.ndarray
"""1-D array of shape ``(num_samples,)`` of real-valued target scalars
c_alpha(x_i) = <psi_0 | U(x_i, alpha)^dag O U(x_i, alpha) | psi_0>."""

FeatureTensor: TypeAlias = np.ndarray
"""Rank-d tensor of shape ``(4r+1,) * d`` of real Fourier coefficients
b_l(x) for a single sample x.  The learner chooses whether to flatten."""

FeatureMatrix: TypeAlias = np.ndarray
"""2-D array of shape ``(num_samples, (4r+1)**d)`` produced by
:meth:`FeatureExtractor.extract_batch` when ``flatten=True``."""

GramMatrix: TypeAlias = np.ndarray
"""Square ``(N, N)`` precomputed kernel K_ij = <b(x_i), b(x_j)>."""
