"""Kernel Ridge Regression learner on a precomputed quantum overlap kernel.

The Gram matrix K is assembled from the flattened Fourier feature matrix
B as K = B @ B.T.  This is mathematically identical to the Figure 8
Hadamard-test overlap kernel of Barthe et al. (2025), and is how the
legacy ``KernelDynamicsLearner`` in ``learning_kernel.py`` computed it.

Only instantiated by :class:`Experiment` when ``method='kernel'`` (and
therefore necessarily ``model.d > 1``).
"""

from __future__ import annotations

import numpy as np
from sklearn.kernel_ridge import KernelRidge

from .base import Learner


class KernelRidgeLearner(Learner):
    """Kernel Ridge with ``kernel='precomputed'``.

    :meth:`fit` expects the training-by-training Gram matrix
    ``K_train`` of shape ``(T, T)``.  :meth:`predict` expects the
    cross-kernel matrix ``K_cross`` of shape ``(T_test, T_train)``.

    Parameters
    ----------
    alpha : float, default 0.1
        KRR regularisation lambda.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self._krr = KernelRidge(alpha=alpha, kernel="precomputed")
        self._fitted: bool = False

    def fit(self, features_or_gram: np.ndarray, y: np.ndarray) -> "KernelRidgeLearner":
        raise NotImplementedError(
            "KernelRidgeLearner.fit — concrete logic pending approval."
        )

    def predict(self, features_or_gram: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "KernelRidgeLearner.predict — concrete logic pending approval."
        )

    @property
    def coef_(self) -> np.ndarray | None:
        if not self._fitted:
            return None
        return self._krr.dual_coef_
