"""Quantum Learning Dynamics — PAC learning of quantum dynamics via Fourier coefficients.

This package implements the framework detailed in Barthe et al. (2025): 
"Quantum Advantage in Learning Quantum Dynamics via Fourier Coefficient Extraction".

It provides a high-performance, hardware-aware execution layer for extracting 
exponential Fourier feature spaces (d=1) and evaluating True Quantum Overlap 
Kernels (d>1) to PAC-learn quantum dynamics.

Example
-------
::

    from quantum_learning_dynamics import Experiment
    from quantum_learning_dynamics.hamiltonians import InhomogeneousTFIM
    from quantum_learning_dynamics.observables import StaggeredMagnetization

    exp = Experiment(
        model=InhomogeneousTFIM(num_qubits=4),
        observable=StaggeredMagnetization(num_qubits=4),
        method="kernel",
        execution_mode="emulator",
        tau=0.5,
        r_steps=2,
    )
    result = exp.run(num_train=200, num_test=50)

"""

from .experiment import Experiment
from .results import ExperimentResult

__all__ = ["Experiment", "ExperimentResult"]
__version__ = "0.1.0"