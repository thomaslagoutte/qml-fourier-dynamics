"""Hadamard-test Fourier coefficient extractor — d = 1 regime.

Extracts b_l(x) = Re <psi_0 | U(x, alpha)^dag P U(x, alpha) | psi_0>_{l}
for each signed frequency l in L = {-2r, ..., +2r} using the Hadamard test
on the shared-register A(U, P) circuit (Figure 7 / Corollary 2 of the
paper).  Returns a 1-D feature vector of length ``4r + 1``.

Concrete implementation pulled from the d = 1 logic of the legacy
``FourierDynamicsLearner._extract_b_l`` and
``FourierDynamicsLearner.extract_fourier_features``.
"""

from __future__ import annotations

import numpy as np

from .._types import FeatureTensor, InputX, PauliString
from ..circuits.base import CircuitBuilder
from ..hamiltonians.base import HamiltonianModel
from .base import FeatureExtractor


class HadamardBLExtractor(FeatureExtractor):
    """Single-register Hadamard-test extractor for d = 1.

    Parameters
    ----------
    builder : CircuitBuilder
        A :class:`SharedRegisterBuilder`.
    epsilon_b : float, default 0.0
        Simulated additive noise bound per b_l (Theorem 6 of the paper).
        Set to 0 for a noiseless statevector run.
    rng : np.random.Generator, optional
        Source of randomness for the epsilon_b noise draws.  When
        ``None``, a fresh ``np.random.default_rng()`` is created — but
        :class:`Experiment` always passes a seeded RNG derived from its
        master seed, so calls routed through the high-level API are
        deterministic.
    """

    def __init__(
        self,
        builder: CircuitBuilder,
        epsilon_b: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(builder)
        self.epsilon_b = epsilon_b
        self.rng = rng if rng is not None else np.random.default_rng()

    def extract_single_pauli(
        self,
        model: HamiltonianModel,
        x: InputX,
        tau: float,
        r_steps: int,
        pauli: PauliString,
    ) -> FeatureTensor:
        raise NotImplementedError(
            "HadamardBLExtractor.extract_single_pauli — concrete logic pending approval."
        )
