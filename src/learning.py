import numpy as np
import math
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from src.models import IsingTransverseFieldModel
from src.quantum_routines import CircuitBuilder
from qiskit.quantum_info import Statevector

class FourierDynamicsLearner:
    """
    Implements the hybrid quantum-classical PAC learning pipeline.
    """
    def __init__(self, num_qubits: int, tau: float, epsilon_b: float = 0.01):
        self.num_qubits = num_qubits
        self.tau = tau
        self.epsilon_b = epsilon_b
        self.model = IsingTransverseFieldModel(num_qubits)
        self.lasso = Lasso(alpha=0.01, fit_intercept=False)

    def generate_random_graphs(self, num_samples: int) -> list:
        datasets = []
        for _ in range(num_samples):
            edges = [(i, j) for i in range(self.num_qubits) for j in range(i + 1, self.num_qubits) if np.random.rand() > 0.5]
            datasets.append(edges)
        return datasets

    def compute_exact_labels(self, datasets: list, true_alpha: float) -> np.ndarray:
        labels = []
        for edges in datasets:
            U_exact = self.model.exact_time_evolution_operator(edges, true_alpha, self.tau)
            psi_0 = np.zeros(2**self.num_qubits, dtype=complex)
            psi_0[0] = 1.0
            psi_t = U_exact @ psi_0
            labels.append(np.abs(psi_t[0])**2)
        return np.array(labels)

    def extract_fourier_features(self, datasets: list, r_steps: int = 1, d_params: int = 1) -> np.ndarray:
        num_samples = len(datasets)
        builder = CircuitBuilder()
        
        n_s = builder._freq_register_size(r_steps)
        freq_dim_per = 2**n_s
        total_freq_dim = freq_dim_per**d_params
        
        # Calculate theoretical feature size and resulting autocorrelation size
        m_theory = (4 * r_steps + 1)**d_params
        true_feature_dim = 2 * m_theory - 1
        
        B_matrix = np.zeros((num_samples, true_feature_dim), dtype=float)
        data_ancilla_dim = 2**(self.num_qubits + 1)
        
        for idx, edges in enumerate(datasets):
            qc, _ = builder.build_trotter_extraction_circuit(
                self.num_qubits, edges, self.tau, r_steps, d_params
            )
            sv = Statevector.from_instruction(qc).data
            
            # 1. Extract amplitudes corresponding to |data=0, ancilla=0>
            a_raw = np.array([sv[j * data_ancilla_dim] for j in range(total_freq_dim)], dtype=complex)
            
            # 2. Reshape to d-dimensional tensor
            a_nd = a_raw.reshape([freq_dim_per] * d_params)
            
            # 3. Valid indices for circular encoding in range [-2r, +2r]
            max_freq = 2 * r_steps
            valid_idx = list(range(0, max_freq + 1)) + list(range(freq_dim_per - max_freq, freq_dim_per))
            
            # Slice the valid states and flatten
            a_valid = a_nd[np.ix_(*([valid_idx] * d_params))].flatten()
            
            # 4. Compute expectation value coefficients b_l via autocorrelation
            b_l_complex = np.correlate(a_valid, a_valid, mode='full')
            B_matrix[idx, :] = np.real(b_l_complex)
            
            # Add noise
            noise = np.random.uniform(-self.epsilon_b, self.epsilon_b, true_feature_dim)
            B_matrix[idx, :] += noise
            
        return B_matrix

    def train(self, B_train: np.ndarray, y_train: np.ndarray):
        self.lasso.fit(B_train, y_train)
        return self.lasso.coef_

    def evaluate(self, B_test: np.ndarray, y_test: np.ndarray) -> float:
        y_pred = self.lasso.predict(B_test)
        return mean_squared_error(y_test, y_pred)