"""Lasso learner — fits sklearn.linear_model.Lasso on Fourier features."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Lasso

from .base import Learner


class LassoLearner(Learner):
    """Lasso regression on the flattened Fourier feature matrix.

    Used in both ``method="lasso-b_l"`` (shared register) and
    ``method="lasso-tensor"`` (separate registers, full tensor flattened).

    Parameters
    ----------
    alpha : float, default 1e-3
        L1 regularisation strength.  Tune via CV in the experiment layer.
    fit_intercept : bool, default False
        Paper's convention: no intercept (the zero-frequency term b_0(x)
        plays that role).
    max_iter : int, default 10_000
    random_state : int, optional
        Seed for the underlying sklearn.Lasso coordinate-descent
        initialisation.  :class:`Experiment` derives this from its
        master ``seed``.
    """

    def __init__(
        self,
        alpha: float = 1e-3,
        fit_intercept: bool = False,
        max_iter: int = 10_000,
        random_state: int | None = None,
    ) -> None:
        self._lasso = Lasso(
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            random_state=random_state,
        )
        self._fitted: bool = False

    def fit(self, features_or_gram: np.ndarray, y: np.ndarray) -> "LassoLearner":
        raise NotImplementedError("LassoLearner.fit — concrete logic pending approval.")

    def predict(self, features_or_gram: np.ndarray) -> np.ndarray:
        raise NotImplementedError("LassoLearner.predict — concrete logic pending approval.")

    @property
    def coef_(self) -> np.ndarray | None:
        if not self._fitted:
            return None
        return self._lasso.coef_
