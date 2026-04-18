"""Abstract base class for parameterised Hamiltonians H(x, alpha).

The paper's learning target is the concept class

    c_alpha(x) = <psi_0 | U(x, alpha)^dag O U(x, alpha) | psi_0>

where x is a *known* (observed) encoding of the Hamiltonian graph / mask and
alpha is the *unknown* parameter vector of length d.  This ABC centralises
the definition of (x, alpha, d, num_qubits) and provides hooks for
reference computations (exact unitary, random sampling).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from .._types import AlphaVector, InputX


class HamiltonianModel(ABC):
    """H(x, alpha) parameterized by graph structure x and unknown alpha.

    Subclass contract
    -----------------
    Every subclass MUST set in ``__init__``:

    * ``self.num_qubits`` — total number of data qubits.
    * ``self.d`` — number of unknown parameters in alpha.  This drives the
      auto-routing logic in :class:`Experiment` (d == 1 → shared-register
      circuit, d > 1 → separate-registers circuit).

    Subclasses MUST also override :meth:`generate_hamiltonian`,
    :meth:`exact_unitary`, :meth:`sample_x`, and :meth:`sample_alpha`.

    The ``x`` input is typed ``Any`` because the paper allows
    problem-dependent encodings; each subclass specifies its concrete format
    in its own docstring.
    """

    num_qubits: int
    d: int

    # -- core algebraic primitives --------------------------------------

    @abstractmethod
    def generate_hamiltonian(self, x: InputX, alpha: AlphaVector) -> SparsePauliOp:
        """Return the SparsePauliOp for H(x, alpha).

        Parameters
        ----------
        x : InputX
            Subclass-specific encoding of the known graph/mask.
        alpha : AlphaVector
            Scalar (d == 1) or 1-D array of shape (d,) (d > 1).
        """

    @abstractmethod
    def exact_unitary(self, x: InputX, alpha: AlphaVector, tau: float) -> np.ndarray:
        """Return the dense unitary e^{i tau H(x, alpha)} via matrix exponentiation.

        Used only for reference label generation and as a numerical
        sanity check.  Circuits go through the CircuitBuilder subclasses,
        not this method.
        """

    # -- sampling --------------------------------------------------------

    @abstractmethod
    def sample_x(self, rng: np.random.Generator) -> InputX:
        """Sample one training-input x from the subclass's natural distribution."""

    @abstractmethod
    def sample_alpha(self, rng: np.random.Generator) -> AlphaVector:
        """Sample the unknown parameter vector alpha* used to generate labels.

        In PAC learning this is drawn once per experiment, not per sample.
        """

    # -- convenience -----------------------------------------------------

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"{cls}(num_qubits={self.num_qubits}, d={self.d})"
