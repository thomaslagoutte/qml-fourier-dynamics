"""Abstract base class for parameterized physical Hamiltonians."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm

from .._types import AlphaVector, InputX, PauliString


class HamiltonianModel(ABC):
    """Abstract base class defining physical Hamiltonian families :math:`H(x, \\alpha)`.

    This class serves as the interface between the physical system and the
    quantum circuit builders. It encapsulates the system size, the dimension
    of the unknown parameter space, and the specific Pauli terms required to
    encode the parameters into the quantum circuit.

    Attributes
    ----------
    num_qubits : int
        The total number of qubits required to simulate the data register.
    d : int
        The dimensionality of the unknown parameter vector :math:`\\alpha`.
    """

    num_qubits: int
    d: int

    @property
    @abstractmethod
    def upload_paulis(self) -> List[PauliString]:
        """The Pauli operators corresponding to the unknown parameters.

        Returns
        -------
        List[PauliString]
            A list of length ``d`` containing the Pauli string associated with
            each component of :math:`\\alpha`. For single-qubit uploads (e.g., TFIM),
            these are weight-1 operators. For gauge theories (e.g., Schwinger),
            these may be multi-qubit operators (e.g., XZX).
        """
        pass

    @abstractmethod
    def hamiltonian(self, x: InputX, alpha: AlphaVector) -> SparsePauliOp:
        """Constructs the full Hamiltonian observable.

        Parameters
        ----------
        x : InputX
            The structural input or graph configuration.
        alpha : AlphaVector
            The continuous parameter vector of length ``d``.

        Returns
        -------
        SparsePauliOp
            The parameterized Hamiltonian acting on ``num_qubits``.
        """
        pass

    @abstractmethod
    def sample_x(self, rng: np.random.Generator) -> InputX:
        """Samples a structural configuration from the model's distribution.

        Parameters
        ----------
        rng : np.random.Generator
            The random number generator instance.

        Returns
        -------
        InputX
            A valid structural input for this Hamiltonian.
        """
        pass

    @abstractmethod
    def sample_alpha(self, rng: np.random.Generator) -> AlphaVector:
        """Samples a parameter vector from the model's valid domain.

        Parameters
        ----------
        rng : np.random.Generator
            The random number generator instance.

        Returns
        -------
        AlphaVector
            An array of length ``d`` representing the continuous parameters.
        """
        pass

    def exact_unitary(
        self,
        x: InputX,
        alpha: AlphaVector,
        tau: float,
    ) -> np.ndarray:
        """Computes the exact dense unitary matrix for the time evolution.

        Computes :math:`U(x, \\alpha) = \\exp(-i \\tau H(x, \\alpha))`. This
        is utilized by the emulator for noiseless baseline comparisons.

        Parameters
        ----------
        x : InputX
            The structural input or graph configuration.
        alpha : AlphaVector
            The continuous parameter vector.
        tau : float
            The evolution time.

        Returns
        -------
        np.ndarray
            A dense complex unitary matrix of shape ``(2^n, 2^n)``.
        """
        H = self.hamiltonian(x, alpha)
        H_mat = H.to_matrix()
        return expm(-1j * tau * H_mat)