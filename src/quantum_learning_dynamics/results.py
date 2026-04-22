"""Data structures for experiment results and metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ._types import InputX


@dataclass
class ExperimentResult:
    """Encapsulates the outputs, predictions, and metrics of a PAC-learning experiment.

    Attributes
    ----------
    y_true_exact : np.ndarray
        Exact ground-truth dynamics :math:`c_\\alpha(x_i)` computed via dense 
        unitary exponentiation. Shape: ``(num_test,)``.
    y_true_trotter : np.ndarray
        Trotterized ground-truth dynamics. Acts as the in-convention training 
        target for the learner. Shape: ``(num_test,)``.
    y_pred : np.ndarray
        Dynamics predicted by the fitted classical PAC learner. Shape: ``(num_test,)``.
    mse_exact : float
        Mean-squared error of ``y_pred`` evaluated against ``y_true_exact``.
    mse_trotter : float
        Mean-squared error of ``y_pred`` evaluated against ``y_true_trotter``.
    X_test : List[InputX], optional
        The held-out sample inputs utilized for testing.
    metadata : Dict[str, Any]
        Configuration dictionary recording model dimension, method, steps, 
        regularization constraints, and seeds.
    """

    y_true_exact: np.ndarray
    y_true_trotter: np.ndarray
    y_pred: np.ndarray
    mse_exact: float
    mse_trotter: float
    X_test: Optional[List[InputX]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ExperimentResult(mse_exact={self.mse_exact:.6e}, "
            f"mse_trotter={self.mse_trotter:.6e})"
        )