import numpy as np
import pytest
from src.learning import FourierDynamicsLearner

@pytest.fixture
def learner():
    """Fixture to provide a FourierDynamicsLearner instance."""
    # 3 qubits, tau = 1.0, epsilon_b = 0.05
    return FourierDynamicsLearner(num_qubits=3, tau=1.0, epsilon_b=0.05)

def test_data_generation_and_labels(learner):
    """
    Validates that the graph generation and exact label computation 
    produce the correct data structures and bounds.
    """
    num_samples = 10
    datasets = learner.generate_random_graphs(num_samples)
    
    assert len(datasets) == num_samples
    assert isinstance(datasets[0], list)
    
    # Compute labels using the exact time evolution (alpha = 0.5)
    labels = learner.compute_exact_labels(datasets, true_alpha=0.5)
    
    assert len(labels) == num_samples
    # Probabilities must be between 0 and 1
    assert np.all(labels >= 0.0) and np.all(labels <= 1.0)

def test_lasso_pipeline_with_dummy_linear_data(learner):
    """
    Verifies the classical LASSO pipeline by constructing a synthetic 
    dataset where y = B * w_true + noise. 
    The LASSO model should successfully fit this and yield a low MSE.
    """
    num_samples = 200
    num_freqs = 20
    
    # 1. Create a known, sparse weight vector (w_true)
    w_true = np.zeros(num_freqs)
    w_true[2] = 1.5
    w_true[7] = -0.8
    w_true[15] = 0.3
    
    # 2. Generate dummy Fourier features directly using NumPy (Bypassing Qiskit)
    # This prevents the 4-hour dense matrix simulation hang!
    B_train = np.random.randn(num_samples, num_freqs)
    
    # 3. Generate labels strictly following the linear model y = B * w
    y_train_clean = B_train @ w_true
    
    # Inject label noise bounded by epsilon_y
    epsilon_y = 0.02
    y_train = y_train_clean + np.random.uniform(-epsilon_y, epsilon_y, num_samples)
    
    # 4. Train the LASSO model
    learned_weights = learner.train(B_train, y_train)
    
    # 5. Evaluate the model on new validation data
    B_test = np.random.randn(50, num_freqs)
    y_test = B_test @ w_true # Clean labels for strict evaluation
    
    mse = learner.evaluate(B_test, y_test)
    
    # Assertions
    assert learned_weights.shape == (num_freqs,)
    assert mse < 0.05

def test_full_integration_pipeline(learner):
    """
    Verifies the end-to-end pipeline: builds the true Trotterized circuits, 
    extracts the Fourier features via the Statevector, and processes 
    them through the LASSO model without crashing.
    """
    num_samples = 5
    r_steps = 1
    
    # 1. Generate data
    datasets = learner.generate_random_graphs(num_samples)
    y_train = learner.compute_exact_labels(datasets, true_alpha=0.25)
    
    # 2. Extract TRUE quantum features using the CircuitBuilder integration
    B_train = learner.extract_fourier_features(datasets, r_steps=r_steps)
    
    # Assert the matrix shape matches the frequency register dimension
    import math
    expected_dim = 2**math.ceil(math.log2(2 * (learner.num_qubits * r_steps) + 1))
    assert B_train.shape == (num_samples, expected_dim)
    
    # 3. Train the model (Ensure LASSO accepts the real quantum features)
    weights = learner.train(B_train, y_train)
    assert len(weights) == expected_dim