"""Abstract learner — fits a regressor on features or a precomputed Gram matrix."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Learner(ABC):
    """PAC regressor wrapping an sklearn estimator.

    Two concrete subclasses:

    * :class:`LassoLearner`       — fits :class:`sklearn.linear_model.Lasso`
      directly on the flattened Fourier feature matrix.
    * :class:`KernelRidgeLearner` — fits
      :class:`sklearn.kernel_ridge.KernelRidge` with ``kernel='precomputed'``
      on the Gram matrix K = B @ B.T.

    The methods take either a feature matrix or a Gram matrix depending on
    the concrete subclass — :class:`Experiment` wires them consistently.
    """

    @abstractmethod
    def fit(self, features_or_gram: np.ndarray, y: np.ndarray) -> "Learner": ...

    @abstractmethod
    def predict(self, features_or_gram: np.ndarray) -> np.ndarray: ...

    @property
    @abstractmethod
    def coef_(self) -> np.ndarray | None:
        """Learned weight vector (Lasso) or dual coefficients (KRR).

        Returns ``None`` before :meth:`fit` has been called.
        """
