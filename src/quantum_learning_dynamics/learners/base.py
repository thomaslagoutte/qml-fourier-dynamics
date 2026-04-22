"""Abstract base class for PAC-learning regressors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Learner(ABC):
    """Abstract PAC regressor wrapping an underlying ``scikit-learn`` estimator.

    Defines the standard interface for fitting and predicting quantum dynamics 
    data. Depending on the concrete subclass, the input array may represent 
    either an explicit feature matrix or a pairwise Gram matrix.
    """

    @abstractmethod
    def fit(self, features_or_gram: np.ndarray, y: np.ndarray) -> "Learner":
        """Fits the regression model to the training data.

        Parameters
        ----------
        features_or_gram : np.ndarray
            The training input data. May be a feature matrix of shape 
            ``(n_samples, n_features)`` or a Gram matrix of shape 
            ``(n_samples, n_samples)``.
        y : np.ndarray
            The target values (labels) of shape ``(n_samples,)``.

        Returns
        -------
        Learner
            The fitted learner instance.
        """
        pass

    @abstractmethod
    def predict(self, features_or_gram: np.ndarray) -> np.ndarray:
        """Predicts target values for the given test data.

        Parameters
        ----------
        features_or_gram : np.ndarray
            The test input data. May be a feature matrix of shape 
            ``(n_test, n_features)`` or a test-vs-train Gram matrix of 
            shape ``(n_test, n_train)``.

        Returns
        -------
        np.ndarray
            The predicted values of shape ``(n_test,)``.
        """
        pass

    @property
    @abstractmethod
    def coef_(self) -> np.ndarray | None:
        """Retrieves the learned coefficients.

        Returns
        -------
        np.ndarray or None
            The learned weight vector (for Lasso) or dual coefficients 
            (for Kernel Ridge). Returns ``None`` if the model is not yet fitted.
        """
        pass