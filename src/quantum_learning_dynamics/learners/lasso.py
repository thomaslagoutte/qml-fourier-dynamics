"""L1-regularized linear regression for explicit Fourier feature matrices."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import Lasso

from tqdm import tqdm

from .._types import FeatureMatrix, LabelVector
from .base import Learner


class LassoLearner(Learner):
    """L1-regularized linear regression on the Fourier feature space.

    Acts as a thin wrapper around :class:`sklearn.linear_model.Lasso`. By 
    default, ``fit_intercept`` is set to ``False`` because the Fourier feature 
    basis inherently spans the constant mode (:math:`l = 0`). Forcing the intercept 
    to zero prevents collinearity and double-counting of the zero-frequency term.

    Parameters
    ----------
    alpha : float, default=1e-3
        The L1 regularization penalty parameter.
    fit_intercept : bool, default=False
        Whether to calculate the intercept for this model.
    max_iter : int, default=10_000
        The maximum number of iterations for the solver.
    random_state : int, optional
        Seed for the random number generator used by the coordinate descent solver.
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