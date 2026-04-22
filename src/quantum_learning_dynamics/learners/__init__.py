"""Classical machine learning backends for PAC-learning quantum dynamics.

Provides regressors configured to consume either explicit Fourier feature 
matrices or precomputed quantum-overlap Gram matrices.
"""

from .base import Learner
from .kernel_ridge import KernelRidgeLearner
from .lasso import LassoLearner

__all__ = [
    "KernelRidgeLearner",
    "LassoLearner",
    "Learner",
]