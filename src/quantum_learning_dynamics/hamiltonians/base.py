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
from typing import List

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm

from .._types import AlphaVector, InputX, PauliString


class HamiltonianModel(ABC):
    """Abstract base for physical Hamiltonian families H(x, α).

    Contract:
        num_qubits    : int                  -- size of the data register
        d             : int                  -- # of independent unknown parameters (|α|)
        upload_paulis : List[PauliString]    -- weight-k Pauli string per α_k, length == d.
                                                Tells the CircuitBuilder what the D block
                                                needs to parity-fold onto anc[0] before
                                                the frequency shift. For single-qubit
                                                uploads (TFIM) weight is 1; for Z₂
                                                Schwinger it's 3 (XZX).

    The `exact_unitary(x, α, τ) = exp(-i H(x, α) τ)` default implementation is
    dense-matrix; fine for small n, override for large n if needed.
    """

    # subclass must set these in __init__
    num_qubits: int
    d: int

    # --- abstract interface ------------------------------------------------

    @property
    @abstractmethod
    def upload_paulis(self) -> List[PauliString]:
        """One Pauli string per α component, in the order α is indexed."""

    @abstractmethod
    def hamiltonian(self, x: InputX, alpha: AlphaVector) -> SparsePauliOp:
        """Full H(x, α) as a SparsePauliOp on `num_qubits` qubits."""

    @abstractmethod
    def sample_x(self, rng: np.random.Generator) -> InputX:
        """Sample a fresh input x from the model's training distribution."""

    @abstractmethod
    def sample_alpha(self, rng: np.random.Generator) -> AlphaVector:
        """Sample a fresh α vector of length d."""

    # --- concrete helpers --------------------------------------------------

    def exact_unitary(
        self,
        x: InputX,
        alpha: AlphaVector,
        tau: float,
    ) -> np.ndarray:
        """Dense U(x, α) = exp(-i τ H(x, α))."""
        H = self.hamiltonian(x, alpha).to_matrix()
        return expm(-1j * tau * H)