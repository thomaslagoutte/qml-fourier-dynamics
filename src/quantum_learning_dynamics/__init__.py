"""Quantum Learning Dynamics — PAC learning of quantum dynamics via Fourier coefficients.

Implements the framework of
    Barthe et al., "Quantum Advantage in Learning Quantum Dynamics via
    Fourier Coefficient Extraction" (2025).

High-level entry point
----------------------
::

    from quantum_learning_dynamics import Experiment
    from quantum_learning_dynamics.hamiltonians import InhomogeneousTFIM
    from quantum_learning_dynamics.observables import StaggeredMagnetization

    exp = Experiment(
        model      = InhomogeneousTFIM(num_qubits=4),
        observable = StaggeredMagnetization(num_qubits=4),
        method     = "kernel",
        tau        = 0.5,
        r_steps    = 2,
    )
    result = exp.run(num_train=200, num_test=50)

See :mod:`quantum_learning_dynamics.experiment` for the auto-routing rules.
"""

from .experiment import Experiment
from .results import ExperimentResult

__all__ = ["Experiment", "ExperimentResult"]
__version__ = "0.1.0"
