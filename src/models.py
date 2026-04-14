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

class SchwingerZ2Model:
    """
    1D Z2 Lattice Gauge Theory (Schwinger Model) Hamiltonian.
    Matter sites: Even indices (0, 2, 4...) -> representing fermions/antifermions.
    Gauge links: Odd indices (1, 3, 5...) -> representing the electric field.
    """
    def __init__(self, num_matter_sites: int):
        self.num_matter = num_matter_sites
        self.num_gauge = num_matter_sites - 1
        self.num_qubits = self.num_matter + self.num_gauge

    def generate_hamiltonian(self, mass: float, electric_field: float, interaction: float = 1.0) -> SparsePauliOp:
        paulis = []
        coeffs = []

        # 1. Matter Mass (Z on even qubits)
        for i in range(self.num_matter):
            p_str = ['I'] * self.num_qubits
            p_str[self.num_qubits - 1 - (2*i)] = 'Z'
            paulis.append("".join(p_str))
            coeffs.append(mass)

        # 2. Electric Field (X on odd qubits)
        for i in range(self.num_gauge):
            p_str = ['I'] * self.num_qubits
            p_str[self.num_qubits - 1 - (2*i + 1)] = 'X'
            paulis.append("".join(p_str))
            coeffs.append(electric_field)

        # 3. Matter-Gauge Interaction (X_{2i} Z_{2i+1} X_{2i+2})
        for i in range(self.num_gauge):
            p_str = ['I'] * self.num_qubits
            p_str[self.num_qubits - 1 - (2*i)] = 'X'
            p_str[self.num_qubits - 1 - (2*i + 1)] = 'Z'
            p_str[self.num_qubits - 1 - (2*i + 2)] = 'X'
            paulis.append("".join(p_str))
            coeffs.append(interaction)

        return SparsePauliOp(paulis, coeffs)