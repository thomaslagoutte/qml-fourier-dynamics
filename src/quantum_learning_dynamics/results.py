"""Result dataclass returned by :meth:`Experiment.run`.

Holds the three aligned curves — exact reference dynamics, Trotterised
reference dynamics, and the PAC-learned predictor — together with error
metrics and a metadata bag describing the run.  Naming uses standard ML
semantics: ``y_true_*`` for ground-truth curves, ``y_pred`` for learner
predictions.

The ``X_test`` slot is reserved for the held-out sample inputs (e.g. the
edge lists for TFIM or gauge masks for Schwinger).  It is optional so
the runner can omit it for large experiments where the inputs are
reconstructible from the seed.  Downstream notebooks that want to plot
predictions vs. a scalar x-axis should populate ``metadata`` with the
x-axis array instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ._types import InputX


@dataclass
class ExperimentResult:
    """Three aligned PAC curves plus errors, inputs, and metadata.

    Attributes
    ----------
    y_true_exact : np.ndarray
        Exact dynamics labels c_alpha(x_test) computed from the dense
        unitary ``exp(i tau H)``.  Shape ``(num_test,)``.
    y_true_trotter : np.ndarray
        A(U)-consistent Trotter labels — the in-convention training
        target used by the learner.  Shape ``(num_test,)``.
    y_pred : np.ndarray
        Predictions from the fitted PAC learner.  Shape ``(num_test,)``.
    mse_exact : float
        Mean-squared error of ``y_pred`` against ``y_true_exact`` —
        the physically meaningful generalisation error.
    mse_trotter : float
        Mean-squared error of ``y_pred`` against ``y_true_trotter`` —
        the in-convention PAC error the paper's bounds talk about.
    X_test : list, optional
        Held-out sample inputs (e.g. edge lists for TFIM, gauge masks
        for Schwinger).  ``None`` if the caller chose not to retain them.
    metadata : dict
        Run configuration and instrumentation: ``d``, ``method``,
        ``r_steps``, ``tau``, ``num_train``, ``num_test``,
        ``regularization``, ``seed``, timings, etc.  Informational only.
    """

    y_true_exact: np.ndarray
    y_true_trotter: np.ndarray
    y_pred: np.ndarray
    mse_exact: float
    mse_trotter: float
    X_test: Optional[List[InputX]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        n = len(self.y_pred)
        return (
            f"ExperimentResult(n_test={n}, "
            f"mse_exact={self.mse_exact:.3e}, "
            f"mse_trotter={self.mse_trotter:.3e})"
        )
