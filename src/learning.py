import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from src.models import IsingTransverseFieldModel
from src.quantum_routines import CircuitBuilder

class FourierDynamicsLearner:
    """
    Implements the hybrid quantum-classical PAC learning pipeline 
    for Hamiltonian dynamics via Fourier coefficient extraction.
    """
    def __init__(self, num_qubits: int, tau: float, epsilon_b: float = 0.01):
        self.num_qubits = num_qubits
        self.tau = tau
        self.epsilon_b = epsilon_b
        self.model = IsingTransverseFieldModel(num_qubits)
        self.lasso = Lasso(alpha=0.01, fit_intercept=False) # L1 regularization

    def generate_random_graphs(self, num_samples: int) -> list:
        """Generates random graph structures (x) as bitstrings/edge lists."""
        datasets = []
        for _ in range(num_samples):
            edges = []
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    if np.random.rand() > 0.5:
                        edges.append((i, j))
            datasets.append(edges)
        return datasets

    def compute_exact_labels(self, datasets: list, true_alpha: float) -> np.ndarray:
        """
        Computes the exact expectation value y = <psi| Z_0 |psi> 
        after time evolution to serve as our training labels.
        """
        labels = []
        
        for edges in datasets:
            U_exact = self.model.exact_time_evolution_operator(edges, true_alpha, self.tau)
            
            # Initial state |0...0>
            psi_0 = np.zeros(2**self.num_qubits, dtype=complex)
            psi_0[0] = 1.0
            
            # Evolve state
            psi_t = U_exact @ psi_0
            
            # We measure the probability of the all zero state for simplicity in the toy model
            # as a proxy for an observable O
            prob_0 = np.abs(psi_t[0])**2
            labels.append(prob_0)
            
        return np.array(labels)

    def extract_fourier_features(self, datasets: list, r_steps: int = 1) -> np.ndarray:
        from qiskit.quantum_info import Statevector
        import math
        
        num_samples = len(datasets)
        max_freq = self.num_qubits * r_steps
        dim_freq = 2**math.ceil(math.log2(2 * max_freq + 1))
        
        # The true feature space for the expectation value has double the frequency range
        # because of the cross-terms (l - l')
        true_feature_dim = 2 * dim_freq - 1
        B_matrix = np.zeros((num_samples, true_feature_dim), dtype=float)
        
        builder = CircuitBuilder()
        state_stride = 2**(self.num_qubits + 1)
        
        for idx, edges in enumerate(datasets):
            qc, _ = builder.build_trotter_extraction_circuit(
                self.num_qubits, edges, self.tau, r_steps
            )
            sv = Statevector.from_instruction(qc)
            
            # 1. Extract raw state amplitudes a_{l, 0}
            a_l = np.zeros(dim_freq, dtype=complex)
            for freq_val in range(dim_freq):
                state_index = freq_val * state_stride
                a_l[freq_val] = sv.data[state_index]
                
            # 2. Compute the expectation value coefficients b_l via autocorrelation
            # b_m = sum_l (a_l * a_{l-m}^*)
            # We use numpy's correlate to handle the discrete cross-terms
            b_l_complex = np.correlate(a_l, a_l, mode='full')
            
            # The paper states the coefficients for a real observable are real/symmetric
            B_matrix[idx, :] = np.real(b_l_complex)
            
            # Add bounded noise 
            noise = np.random.uniform(-self.epsilon_b, self.epsilon_b, true_feature_dim)
            B_matrix[idx, :] += noise
            
        return B_matrix

    def train(self, B_train: np.ndarray, y_train: np.ndarray):
        """Fits the LASSO regression model to find the sparse weight vector w."""
        self.lasso.fit(B_train, y_train)
        return self.lasso.coef_

    def evaluate(self, B_test: np.ndarray, y_test: np.ndarray) -> float:
        """Evaluates the PAC learner's generalization error."""
        y_pred = self.lasso.predict(B_test)
        return mean_squared_error(y_test, y_pred)