from typing import List, Tuple
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
        
        Args:
            x_edges: A list of tuples (i, j) representing the edges of the graph where x_{i,j} = 1.
            alpha: The strength of the transverse field.
            
        Returns:
            A Qiskit SparsePauliOp representing the Hamiltonian.
        """
        pauli_terms = []
        coefficients = []

        # 1. ZZ Interactions (sum_{i,j} x_{i,j} Z_i Z_j)
        for i, j in x_edges:
            if i >= self.num_qubits or j >= self.num_qubits:
                raise ValueError(f"Edge ({i}, {j}) exceeds qubit count {self.num_qubits}.")
            
            # Create a string of 'I's, then replace the i-th and j-th positions with 'Z'
            pauli_str = ['I'] * self.num_qubits
            # Qiskit uses little-endian ordering (qubit 0 is the rightmost character)
            pauli_str[self.num_qubits - 1 - i] = 'Z'
            pauli_str[self.num_qubits - 1 - j] = 'Z'
            
            pauli_terms.append("".join(pauli_str))
            coefficients.append(1.0) # x_{i,j} is 1 for existing edges

        # 2. Transverse Field (alpha * sum_i X_i)
        for i in range(self.num_qubits):
            pauli_str = ['I'] * self.num_qubits
            pauli_str[self.num_qubits - 1 - i] = 'X'
            
            pauli_terms.append("".join(pauli_str))
            coefficients.append(alpha)

        return SparsePauliOp(pauli_terms, coefficients)

    def exact_time_evolution_operator(self, x_edges: List[Tuple[int, int]], alpha: float, tau: float):
        """
        Computes the exact unitary time evolution matrix e^{i * tau * H}.
        Useful for generating the exact labels to test our Trotterized approximation.
        """
        hamiltonian = self.generate_hamiltonian(x_edges, alpha)
        # We can extract the matrix and use scipy.linalg.expm for exact classical simulation
        from scipy.linalg import expm
        H_matrix = hamiltonian.to_matrix()
        return expm(1j * tau * H_matrix)    