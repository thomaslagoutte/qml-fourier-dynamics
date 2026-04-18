from typing import List, Tuple
import numpy as np
from qiskit.quantum_info import SparsePauliOp


class IsingTransverseFieldModel:
    """
    Represents the Ising Hamiltonian on an arbitrary graph with a transverse field.
    H(x, alpha) = sum_{i,j} x_{i,j} Z_i Z_j + alpha * sum_i X_i
    """
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits

    def generate_hamiltonian(self, x_edges: List[Tuple[int, int]], alpha: float) -> SparsePauliOp:
        """
        Constructs the Hamiltonian for a given graph structure and transverse field strength.
        """
        pauli_terms = []
        coefficients = []

        for i, j in x_edges:
            if i >= self.num_qubits or j >= self.num_qubits:
                raise ValueError(f"Edge ({i}, {j}) exceeds qubit count {self.num_qubits}.")
            pauli_str = ['I'] * self.num_qubits
            pauli_str[self.num_qubits - 1 - i] = 'Z'
            pauli_str[self.num_qubits - 1 - j] = 'Z'
            pauli_terms.append("".join(pauli_str))
            coefficients.append(1.0)

        for i in range(self.num_qubits):
            pauli_str = ['I'] * self.num_qubits
            pauli_str[self.num_qubits - 1 - i] = 'X'
            pauli_terms.append("".join(pauli_str))
            coefficients.append(alpha)

        return SparsePauliOp(pauli_terms, coefficients)

    def exact_time_evolution_operator(self, x_edges: List[Tuple[int, int]], alpha: float, tau: float):
        """
        Computes the exact unitary time evolution matrix e^{i * tau * H}.
        """
        hamiltonian = self.generate_hamiltonian(x_edges, alpha)
        from scipy.linalg import expm
        H_matrix = hamiltonian.to_matrix()
        return expm(1j * tau * H_matrix)


class InhomogeneousTFIM:
    """Ising Hamiltonian with per-qubit independent transverse field strengths.

    This model has d = n unknown parameters (one per qubit), placing it
    in the d = poly(n) regime where explicit Fourier extraction is
    intractable and the quantum overlap kernel is required.

    Hamiltonian
    -----------
        H(x, alpha) = sum_{(i,j) in x} Z_i Z_j  +  sum_{i=0}^{n-1} alpha_i * X_i

    where alpha = (alpha_0, ..., alpha_{n-1}) is the unknown parameter vector
    and x encodes the graph topology (ZZ interaction edges).

    Concept class (Definition 4 of Barthe et al. 2025)
    ---------------------------------------------------
        c_alpha(x) = <psi_0 | U(x,alpha)^dag O U(x,alpha) | psi_0>

    with O a fixed local Pauli observable and psi_0 = |0>.

    Sparsity argument (Lieb-Robinson)
    ----------------------------------
    For a local observable O = Z_0 and short evolution times tau, the
    expectation value is dominated by the local neighbourhood of qubit 0.
    The Fourier coefficients b_l corresponding to distant qubit parameters
    alpha_i (large i) are exponentially suppressed.  The spectrum is
    *a priori* of size (4r+1)^n but *effectively* sparse, satisfying the
    conditions of Appendix G of the paper for efficient PAC learning.

    Parameters
    ----------
    num_qubits : int
        Number of data qubits n.  Also sets d = n (one unknown per qubit).
    """

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.d_params = num_qubits  # d = n: one unknown alpha_i per qubit

    def generate_hamiltonian(
        self,
        x_edges: List[Tuple[int, int]],
        alpha_vec: np.ndarray,
    ) -> SparsePauliOp:
        """Build SparsePauliOp for H(x, alpha).

        Parameters
        ----------
        x_edges : list of (int, int)
            Graph topology; edge (i,j) contributes Z_i Z_j.
        alpha_vec : np.ndarray, shape (n,)
            Per-qubit transverse field strengths alpha_i.

        Returns
        -------
        SparsePauliOp
        """
        if len(alpha_vec) != self.num_qubits:
            raise ValueError(
                f"alpha_vec length {len(alpha_vec)} != num_qubits {self.num_qubits}"
            )

        pauli_terms = []
        coefficients = []

        # ZZ interactions (graph-dependent, alpha-independent)
        for i, j in x_edges:
            if i >= self.num_qubits or j >= self.num_qubits:
                raise ValueError(f"Edge ({i},{j}) out of range for {self.num_qubits} qubits.")
            pauli_str = ['I'] * self.num_qubits
            pauli_str[self.num_qubits - 1 - i] = 'Z'
            pauli_str[self.num_qubits - 1 - j] = 'Z'
            pauli_terms.append("".join(pauli_str))
            coefficients.append(1.0)

        # Per-qubit transverse fields (alpha-dependent)
        for i in range(self.num_qubits):
            pauli_str = ['I'] * self.num_qubits
            pauli_str[self.num_qubits - 1 - i] = 'X'
            pauli_terms.append("".join(pauli_str))
            coefficients.append(float(alpha_vec[i]))

        return SparsePauliOp(pauli_terms, coefficients)

    def exact_time_evolution_operator(
        self,
        x_edges: List[Tuple[int, int]],
        alpha_vec: np.ndarray,
        tau: float,
    ) -> np.ndarray:
        """Compute exact unitary e^{i * tau * H(x, alpha)} via matrix exponentiation.

        Used only for reference / sanity checks.  Training labels for the
        kernel pipeline should use A(U)-consistent Trotter labels
        (see CircuitBuilder.compute_au_labels_inhomogeneous).

        Parameters
        ----------
        x_edges : list of (int, int)
        alpha_vec : np.ndarray, shape (n,)
        tau : float

        Returns
        -------
        np.ndarray, shape (2^n, 2^n), complex
        """
        from scipy.linalg import expm
        H = self.generate_hamiltonian(x_edges, alpha_vec).to_matrix()
        return expm(1j * tau * H)

    def generate_random_alpha(
        self,
        rng: np.random.Generator = None,
        low: float = 0.5,
        high: float = 1.5,
    ) -> np.ndarray:
        """Sample a random parameter vector alpha ~ Uniform([low, high])^n.

        Parameters
        ----------
        rng : np.random.Generator, optional
            Random generator for reproducibility.  If None, uses np.random.
        low, high : float
            Bounds for the uniform draw of each alpha_i.

        Returns
        -------
        np.ndarray, shape (n,)
        """
        if rng is None:
            return np.random.uniform(low, high, self.num_qubits)
        return rng.uniform(low, high, self.num_qubits)


class SchwingerZ2Model:
    """
    1D Z2 Lattice Gauge Theory (Schwinger Model) Hamiltonian.
    """
    def __init__(self, num_matter_sites: int):
        self.num_matter = num_matter_sites
        self.num_gauge = num_matter_sites - 1
        self.num_qubits = self.num_matter + self.num_gauge

    def generate_hamiltonian(self, mass: float, electric_field: float, interaction: float = 1.0) -> SparsePauliOp:
        paulis = []
        coeffs = []

        for i in range(self.num_matter):
            p_str = ['I'] * self.num_qubits
            p_str[self.num_qubits - 1 - (2*i)] = 'Z'
            paulis.append("".join(p_str))
            coeffs.append(mass)

        for i in range(self.num_gauge):
            p_str = ['I'] * self.num_qubits
            p_str[self.num_qubits - 1 - (2*i + 1)] = 'X'
            paulis.append("".join(p_str))
            coeffs.append(electric_field)

        for i in range(self.num_gauge):
            p_str = ['I'] * self.num_qubits
            p_str[self.num_qubits - 1 - (2*i)] = 'X'
            p_str[self.num_qubits - 1 - (2*i + 1)] = 'Z'
            p_str[self.num_qubits - 1 - (2*i + 2)] = 'X'
            paulis.append("".join(p_str))
            coeffs.append(interaction)

        return SparsePauliOp(paulis, coeffs)