import numpy as np
import pytest
from src.learning import FourierDynamicsLearner

@pytest.fixture
def learner():
    """Fixture to provide a FourierDynamicsLearner instance."""
    # 3 qubits, tau = 1.0, epsilon_b = 0.05
    return FourierDynamicsLearner(num_qubits=3, tau=1.0, epsilon_b=0.05)

def test_data_generation_and_labels(learner):
    num_samples = 10
    datasets = learner.generate_random_graphs(num_samples)
    
    assert len(datasets) == num_samples
    assert isinstance(datasets[0], list)
    
    labels = learner.compute_exact_labels(datasets, true_alpha=0.5)
    
    assert len(labels) == num_samples
    assert np.all(labels >= 0.0) and np.all(labels <= 1.0)

def test_lasso_pipeline_with_dummy_linear_data(learner):
    num_samples = 200
    num_freqs = 20
    
    w_true = np.zeros(num_freqs)
    w_true[2] = 1.5
    w_true[7] = -0.8
    w_true[15] = 0.3
    
    B_train = np.random.randn(num_samples, num_freqs)
    y_train_clean = B_train @ w_true
    
    epsilon_y = 0.02
    y_train = y_train_clean + np.random.uniform(-epsilon_y, epsilon_y, num_samples)
    
    learned_weights = learner.train(B_train, y_train)
    
    B_test = np.random.randn(50, num_freqs)
    y_test = B_test @ w_true 
    
    mse = learner.evaluate(B_test, y_test)
    
    assert learned_weights.shape == (num_freqs,)
    assert mse < 0.05

def test_full_integration_pipeline(learner):
    """
    Verifies the end-to-end pipeline using the CORRECTED d-dimensional 
    frequency architecture from the paper.
    """
    num_samples = 5
    r_steps = 1
    d_params = 1 # Default test case
    
    datasets = learner.generate_random_graphs(num_samples)
    y_train = learner.compute_exact_labels(datasets, true_alpha=0.25)
    
    # Extract TRUE quantum features using the fixed CircuitBuilder
    B_train = learner.extract_fourier_features(datasets, r_steps=r_steps, d_params=d_params)
    
    # NEW MATH FIX: The feature space must exactly equal 2*(4r+1)^d - 1
    m_theory = (4 * r_steps + 1)**d_params
    expected_dim = 2 * m_theory - 1
    
    assert B_train.shape == (num_samples, expected_dim), f"Expected shape {(num_samples, expected_dim)}, got {B_train.shape}"
    
    # Train the model
    weights = learner.train(B_train, y_train)
    assert len(weights) == expected_dim