"""Tests for FourierDynamicsLearner.

Tests are grouped in order of increasing integration depth:

  1. Data generation and label correctness.
  2. LASSO regression on synthetic linear data.
  3. A(U,P) circuit: b_l match DFT of <O>(alpha).
  4. Hadamard test: b_l from HT match A(U,P) statevector directly.
  5. Feature matrix shape and pipeline integration.
  6. Sim shortcut agrees with Hadamard-test extraction.
  7. Guard for unsupported d_params.
"""

import math

import numpy as np
import pytest
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector

from src.learning import FourierDynamicsLearner
from src.quantum_routines import CircuitBuilder


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def learner():
    """3-qubit sigma_z learner with small noise for integration tests."""
    return FourierDynamicsLearner(num_qubits=3, tau=1.0, epsilon_b=0.05)


@pytest.fixture
def noiseless_learner():
    """3-qubit sigma_z learner with zero noise for numerical comparisons."""
    return FourierDynamicsLearner(num_qubits=3, tau=1.0, epsilon_b=0.0)


@pytest.fixture
def builder():
    return CircuitBuilder()


# ---------------------------------------------------------------------------
# 1. Data generation
# ---------------------------------------------------------------------------

def test_data_generation_shape(learner):
    """Random graphs have the right count and type."""
    datasets = learner.generate_random_graphs(10)
    assert len(datasets) == 10
    assert isinstance(datasets[0], list)


def test_labels_are_bounded(learner):
    """Pauli expectation values lie in [-1, +1]."""
    datasets = learner.generate_random_graphs(8)
    labels = learner.compute_exact_labels(datasets, true_alpha=0.5)
    assert labels.shape == (8,)
    assert np.all(labels >= -1.0) and np.all(labels <= 1.0)


def test_labels_are_real_for_pauli_observable(learner):
    """Expectation values of a Hermitian observable are real."""
    datasets = learner.generate_random_graphs(5)
    labels = learner.compute_exact_labels(datasets, true_alpha=0.3)
    # compute_exact_labels already casts to float; just verify no NaN/Inf
    assert np.all(np.isfinite(labels))


# ---------------------------------------------------------------------------
# 2. LASSO pipeline on synthetic linear data
# ---------------------------------------------------------------------------

def test_lasso_recovers_sparse_weights(learner):
    """LASSO correctly recovers a sparse weight vector from noisy linear data."""
    num_freqs, num_train, num_test = 20, 200, 50

    w_true = np.zeros(num_freqs)
    w_true[2], w_true[7], w_true[15] = 1.5, -0.8, 0.3

    rng = np.random.default_rng(42)
    B_train = rng.standard_normal((num_train, num_freqs))
    y_train = B_train @ w_true + rng.uniform(-0.02, 0.02, num_train)

    B_test = rng.standard_normal((num_test, num_freqs))
    y_test = B_test @ w_true

    learner.train(B_train, y_train)
    mse = learner.evaluate(B_test, y_test)

    assert mse < 0.05


# ---------------------------------------------------------------------------
# 3. A(U,P) circuit correctness: b_l vs DFT of <O>(alpha)
# ---------------------------------------------------------------------------

def _build_exact_parametric_sigma_z(num_qubits, edges, tau, r_steps):
    """Build the parametric circuit U(alpha) that A(U) encodes.

    Convention: the A(U) circuit encodes e^{i*pi*alpha*X_0} on data qubit 0,
    which equals Rx(-2*pi*alpha) in Qiskit's convention.
    The DFT of <Z_0>(alpha) with this parametric circuit must match the
    b_l values produced by the A(U, Z_0) circuit.
    """
    alpha_p = Parameter("alpha")
    qc = QuantumCircuit(num_qubits)
    for _step in range(r_steps):
        for i, j in edges:
            qc.rzz(2.0 * tau / r_steps, i, j)
        # e^{i*pi*alpha*X_0} = Rx(-2*pi*alpha) in Qiskit convention
        qc.rx(-2.0 * np.pi * alpha_p, 0)
    return qc, alpha_p


@pytest.mark.parametrize("edges,r_steps", [
    ([], 1),
    ([(0, 1)], 1),
    ([(0, 1), (1, 2)], 1),
    ([(0, 1)], 2),
])
def test_aup_circuit_b_l_matches_dft(noiseless_learner, builder, edges, r_steps):
    """b_l from A(U,P) statevector matches DFT of <Z_0>(alpha).

    Verifies that the A(U, P) circuit from Corollary 1 / Figure 4 correctly
    amplitude-encodes the Fourier coefficients of the expectation value
    f(alpha; x) = <psi(alpha; x)|Z_0|psi(alpha; x)>.
    """
    num_qubits = noiseless_learner.num_qubits
    tau = noiseless_learner.tau
    pauli_obs = noiseless_learner.pauli_observable

    n_s = builder._freq_register_size(r_steps)
    freq_dim = 2 ** n_s
    data_ancilla_dim = 2 ** (num_qubits + 1)
    max_freq = 2 * r_steps

    # --- A(U,P) b_l values (circuit) ---
    aup_qc = builder.build_expectation_value_extraction_circuit(
        num_qubits, edges, tau, r_steps, pauli_obs
    )
    sv_aup = Statevector.from_instruction(aup_qc)

    b_circuit = {}
    for l in range(-max_freq, max_freq + 1):
        reg = l % freq_dim
        b_circuit[l] = sv_aup.data[reg * data_ancilla_dim]

    # --- DFT b_l from exact parametric sweep ---
    obs_matrix = SparsePauliOp(pauli_obs).to_matrix()
    qc_param, alpha_p = _build_exact_parametric_sigma_z(num_qubits, edges, tau, r_steps)

    alpha_vals = np.linspace(0.0, 2.0, 2048, endpoint=False)
    f_vals = []
    for a in alpha_vals:
        sv_a = Statevector.from_instruction(qc_param.assign_parameters({alpha_p: a}))
        f_vals.append(float(np.real(sv_a.data.conj() @ obs_matrix @ sv_a.data)))
    f_vals = np.array(f_vals)

    b_dft = {
        l: np.mean(f_vals * np.exp(-1j * np.pi * alpha_vals * l))
        for l in range(-max_freq, max_freq + 1)
    }

    # Spectral leakage from finite DFT grid can produce small non-zero values
    # at frequencies that are exactly zero; allow generous tolerance here.
    for l in range(-max_freq, max_freq + 1):
        assert np.isclose(b_circuit[l].real, b_dft[l].real, atol=0.02), (
            f"Re(b_{l}) mismatch (edges={edges}, r={r_steps}): "
            f"circuit={b_circuit[l].real:.5f}, DFT={b_dft[l].real:.5f}"
        )
        assert np.isclose(b_circuit[l].imag, b_dft[l].imag, atol=0.02), (
            f"Im(b_{l}) mismatch (edges={edges}, r={r_steps}): "
            f"circuit={b_circuit[l].imag:.5f}, DFT={b_dft[l].imag:.5f}"
        )


# ---------------------------------------------------------------------------
# 4. Hadamard test: b_l from HT matches A(U,P) statevector
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("edges", [
    [],
    [(0, 1)],
    [(0, 1), (1, 2), (0, 2)],
])
def test_hadamard_test_b_l_matches_aup_statevector(noiseless_learner, builder, edges):
    """_extract_b_l via HT matches direct A(U,P) statevector readout exactly.

    This is the most fundamental correctness check for the full pipeline.
    It validates:
      - build_expectation_value_hadamard_test (Corollary 2 / Figure 7)
      - _extract_b_l (the un-normalisation step)
    against the ground-truth amplitudes in the A(U,P) statevector.
    """
    num_qubits = noiseless_learner.num_qubits
    tau = noiseless_learner.tau
    r_steps = 1
    pauli_obs = noiseless_learner.pauli_observable

    n_s = builder._freq_register_size(r_steps)
    freq_dim = 2 ** n_s
    data_ancilla_dim = 2 ** (num_qubits + 1)
    max_freq = 2 * r_steps

    # Ground truth: direct A(U,P) statevector
    aup_qc = builder.build_expectation_value_extraction_circuit(
        num_qubits, edges, tau, r_steps, pauli_obs
    )
    sv_aup = Statevector.from_instruction(aup_qc)

    for l in range(-max_freq, max_freq + 1):
        reg = l % freq_dim
        b_direct = sv_aup.data[reg * data_ancilla_dim]

        b_ht = noiseless_learner._extract_b_l(builder, edges, r_steps, reg)

        assert np.isclose(b_direct.real, b_ht.real, atol=1e-9), (
            f"Re(b_{l}) mismatch (edges={edges}): "
            f"statevector={b_direct.real:.9f}, HT={b_ht.real:.9f}"
        )
        assert np.isclose(b_direct.imag, b_ht.imag, atol=1e-9), (
            f"Im(b_{l}) mismatch (edges={edges}): "
            f"statevector={b_direct.imag:.9f}, HT={b_ht.imag:.9f}"
        )


def test_b_l_hermitian_symmetry(noiseless_learner, builder):
    """b_l = conj(b_{-l}) for a real Hermitian observable.

    For O = sigma_z (Hermitian, real eigenvalues), the expectation value
    f(alpha; x) is real, so its Fourier series satisfies b_l = conj(b_{-l}).
    This is a necessary sanity check on the A(U,P) circuit output.
    """
    edges = [(0, 1)]
    r_steps = 1
    num_qubits = noiseless_learner.num_qubits
    tau = noiseless_learner.tau
    pauli_obs = noiseless_learner.pauli_observable

    n_s = builder._freq_register_size(r_steps)
    freq_dim = 2 ** n_s
    data_ancilla_dim = 2 ** (num_qubits + 1)
    max_freq = 2 * r_steps

    aup_qc = builder.build_expectation_value_extraction_circuit(
        num_qubits, edges, tau, r_steps, pauli_obs
    )
    sv = Statevector.from_instruction(aup_qc)

    b = {
        l: sv.data[(l % freq_dim) * data_ancilla_dim]
        for l in range(-max_freq, max_freq + 1)
    }

    for l in range(1, max_freq + 1):
        assert np.isclose(b[l], np.conj(b[-l]), atol=1e-9), (
            f"Hermitian symmetry broken at l={l}: "
            f"b_{{+l}}={b[l]:.6f}, conj(b_{{-l}})={np.conj(b[-l]):.6f}"
        )


# ---------------------------------------------------------------------------
# 5. Feature matrix shape and pipeline integration
# ---------------------------------------------------------------------------

def test_feature_matrix_shape(learner):
    """extract_fourier_features returns shape (n_samples, 4*r_steps + 1)."""
    datasets = learner.generate_random_graphs(4)
    r_steps = 1
    B = learner.extract_fourier_features(datasets, r_steps=r_steps)
    expected = (4, 4 * r_steps + 1)
    assert B.shape == expected, f"Expected {expected}, got {B.shape}"


def test_full_pipeline_train_evaluate(learner):
    """End-to-end: extract features, train, evaluate without error."""
    r_steps = 1
    datasets = learner.generate_random_graphs(6)
    y = learner.compute_exact_labels(datasets, true_alpha=0.4)
    B = learner.extract_fourier_features(datasets, r_steps=r_steps)
    w = learner.train(B, y)
    assert w.shape == (4 * r_steps + 1,)
    mse = learner.evaluate(B, y)
    assert np.isfinite(mse)


def test_d_params_gt_1_raises(learner):
    """d_params > 1 raises NotImplementedError."""
    datasets = learner.generate_random_graphs(2)
    with pytest.raises(NotImplementedError):
        learner.extract_fourier_features(datasets, r_steps=1, d_params=2)


# ---------------------------------------------------------------------------
# 6. Sim shortcut agrees with Hadamard-test extraction
# ---------------------------------------------------------------------------

def test_sim_shortcut_matches_hadamard_test(noiseless_learner, builder):
    """extract_fourier_features_sim matches extract_fourier_features (noise-free).

    Both methods should return identical b_l values because they read from
    the same A(U,P) circuit.  The sim shortcut reads amplitudes directly;
    the Hadamard-test method goes through three circuit evaluations per l.
    """
    datasets = [[], [(0, 1)], [(0, 1), (1, 2)]]
    r_steps = 1

    B_sim = noiseless_learner.extract_fourier_features_sim(datasets, r_steps=r_steps)
    B_ht  = noiseless_learner.extract_fourier_features(datasets, r_steps=r_steps)

    np.testing.assert_allclose(B_sim, B_ht, atol=1e-9,
        err_msg="Sim shortcut and Hadamard-test extraction disagree.")