"""Kernel Ridge Regression for quantum-overlap Gram matrices."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.kernel_ridge import KernelRidge

from .._types import FeatureMatrix, GramMatrix, LabelVector
from .base import Learner


class KernelRidgeLearner(Learner):
    """Kernel Ridge Regression for quantum overlap features.

    Wraps :class:`sklearn.kernel_ridge.KernelRidge` utilizing a precomputed 
    linear kernel. This learner operates in two mathematical modes depending 
    on the computational backend:

    1. **Precomputed Mode** (``precomputed_kernel=True``):
       The learner accepts a directly evaluated Gram matrix :math:`K`. 
       This is utilized when deploying the True Quantum Overlap Kernel on 
       hardware, entirely bypassing the exponential feature space tensor.
    2. **Feature Matrix Mode** (``precomputed_kernel=False``):
       The learner accepts the explicit Fourier feature matrix :math:`B` 
       and internally evaluates the Gram matrix as :math:`K = B B^T`.

    Parameters
    ----------
    alpha : float, default=1.0
        The L2 regularization penalty parameter.
    precomputed_kernel : bool, default=False
        If True, the `fit` and `predict` methods expect a Gram matrix. 
        If False, they expect a feature matrix.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        precomputed_kernel: bool = False,
    ) -> None:
        self.alpha = float(alpha)
        self.precomputed_kernel = bool(precomputed_kernel)
        
        self._krr = KernelRidge(alpha=self.alpha, kernel="precomputed")
        self._B_train: Optional[FeatureMatrix] = None
        self._fitted = False

    @property
    def coef_(self) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("KernelRidgeLearner.coef_ called before fit")
        return self._krr.dual_coef_

    def fit(
        self, features_or_gram: np.ndarray, y: LabelVector
    ) -> "KernelRidgeLearner":
        features_or_gram = np.asarray(features_or_gram, dtype=np.float64)
        
        if self.precomputed_kernel:
            if features_or_gram.ndim != 2 or features_or_gram.shape[0] != features_or_gram.shape[1]:
                raise ValueError(
                    "precomputed_kernel=True requires a square (n_train, n_train) "
                    f"Gram matrix; got shape {features_or_gram.shape}"
                )
            K: GramMatrix = features_or_gram
        else:
            B: FeatureMatrix = features_or_gram
            K = B @ B.T
            self._B_train = B
            
        self._krr.fit(K, y)
        self._fitted = True
        return self

    def predict(self, features_or_gram: np.ndarray) -> LabelVector:
        if not self._fitted:
            raise RuntimeError("KernelRidgeLearner.predict called before fit")
            
        features_or_gram = np.asarray(features_or_gram, dtype=np.float64)
        
        if self.precomputed_kernel:
            if features_or_gram.ndim != 2:
                raise ValueError(
                    "precomputed_kernel=True requires a 2D (n_test, n_train) "
                    f"Gram matrix; got shape {features_or_gram.shape}"
                )
            K_test_train: GramMatrix = features_or_gram
        else:
            if self._B_train is None:
                raise RuntimeError(
                    "Training feature matrix was not retained. Ensure model was "
                    "fitted with precomputed_kernel=False."
                )
            B: FeatureMatrix = features_or_gram
            K_test_train = B @ self._B_train.T
            
        return self._krr.predict(K_test_train)