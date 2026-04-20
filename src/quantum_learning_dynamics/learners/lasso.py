"""Lasso learner — fits sklearn.linear_model.Lasso on Fourier features."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import Lasso

from .._types import FeatureMatrix, LabelVector
from .base import Learner


class LassoLearner(Learner):
    """L1-regularized linear regression on the Fourier feature matrix.

    Thin wrapper around sklearn.linear_model.Lasso with fit_intercept=False
    — the Fourier-feature basis already spans the constant mode (l = 0), so
    forcing no intercept avoids double-counting it.
    """

    def __init__(
        self,
        alpha: float = 1e-3,
        fit_intercept: bool = False,
        max_iter: int = 10_000,
        random_state: Optional[int] = None,
    ) -> None:
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.max_iter = int(max_iter)
        self.random_state = random_state
        self._lasso = Lasso(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._fitted = False

    @property
    def coef_(self) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LassoLearner.coef_ called before fit")
        return self._lasso.coef_

    def fit(self, B: FeatureMatrix, y: LabelVector) -> "LassoLearner":
        self._lasso.fit(B, y)
        self._fitted = True
        return self

    def predict(self, B: FeatureMatrix) -> LabelVector:
        if not self._fitted:
            raise RuntimeError("LassoLearner.predict called before fit")
        return self._lasso.predict(B)
    
