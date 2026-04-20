"""Kernel Ridge Regression learner on a precomputed quantum overlap kernel.

The Gram matrix K is assembled from the flattened Fourier feature matrix
B as K = B @ B.T.  This is mathematically identical to the Figure 8
Hadamard-test overlap kernel of Barthe et al. (2025), and is how the
legacy ``KernelDynamicsLearner`` in ``learning_kernel.py`` computed it.

Only instantiated by :class:`Experiment` when ``method='kernel'`` (and
therefore necessarily ``model.d > 1``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.kernel_ridge import KernelRidge

from .._types import FeatureMatrix, GramMatrix, LabelVector
from .base import Learner


class KernelRidgeLearner(Learner):
    """Kernel ridge regression on the quantum-overlap Gram matrix K = B @ B.T.

    sklearn's KernelRidge with kernel='precomputed' expects:
        fit(K_train, y)          -- K_train is (n_train, n_train)
        predict(K_test_train)    -- K_test_train is (n_test, n_train)
    We cache B_train at fit time and recompute the test-to-train Gram at
    predict time so downstream code can pass the raw feature matrix to both.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = float(alpha)
        self._krr = KernelRidge(alpha=self.alpha, kernel="precomputed")
        self._B_train: Optional[np.ndarray] = None
        self._fitted = False

    @property
    def coef_(self) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("KernelRidgeLearner.coef_ called before fit")
        return self._krr.dual_coef_

    def fit(self, B: FeatureMatrix, y: LabelVector) -> "KernelRidgeLearner":
        B = np.asarray(B, dtype=np.float64)
        K: GramMatrix = B @ B.T                  # inner-product kernel on Fourier features
        self._krr.fit(K, y)
        self._B_train = B
        self._fitted = True
        return self

    def predict(self, B: FeatureMatrix) -> LabelVector:
        if not self._fitted or self._B_train is None:
            raise RuntimeError("KernelRidgeLearner.predict called before fit")
        B = np.asarray(B, dtype=np.float64)
        K_test_train: GramMatrix = B @ self._B_train.T
        return self._krr.predict(K_test_train)
    
