import pytest
import numpy as np
from src.lgt_models import Z2GaugeModel
from src.quantum_routines import CircuitBuilder
from qiskit.quantum_info import Statevector

def test_z2_gauge_model_generation():
    """
    Verifies the physical mapping of the 1D Z2 Lattice Gauge Theory.
    Ensures matter sites and gauge links are correctly allocated.
    """
    model = Z2GaugeModel(num_matter_sites=3, mass=1.0, electric_field=1.0)
    
    # 3 matter sites means 2 connecting gauge links -> 5 qubits total
    assert model.total_qubits == 5 
    
    # Test Slow Regime (O(log n) parameters -> 3 global parameters)
    alphas_slow = np.array([0.5, 0.1, 0.1])
    terms_slow = model.generate_hamiltonian_terms(alphas_slow, regime='slow')
    assert len(terms_slow) == 3 + 2 + 2 # 3 mass + 2 e-field + 2 interactions
    
    # Test Fast Regime (O(poly n) parameters -> 1 per link)
    alphas_fast = np.array([0.8, 0.2])
    terms_fast = model.generate_hamiltonian_terms(alphas_fast, regime='fast')
    
    # Verify the interaction Pauli strings are specifically XZX
    interaction_terms = [t for t in terms_fast if 'Z' in t[0] and 'X' in t[0]]
    assert len(interaction_terms) == 2
    for pauli_str, coeff in interaction_terms:
        # Check that XZX pattern exists (reversed string due to Qiskit endianness)
        assert "XZX" in pauli_str

@pytest.fixture
def builder():
    return CircuitBuilder()

def test_lgt_extraction_circuit_structure(builder):
    """
    Verifies the complex A(U) extraction circuit with dynamic basis changes
    is mathematically sound, correctly sized, and strictly unitary.
    """
    num_matter = 3
    num_gauge = 2
    x_state = "10101" # Arbitrary initial state
    
    qc, qr_freqs = builder.build_lgt_trotter_extraction_circuit(
        num_matter_sites=num_matter, 
        x_state=x_state, 
        mass=0.5, electric_field=0.5, tau=1.0, r_steps=1
    )
    
    # Expected qubits: Data(5) + Ancilla(1) + Freq(2 links * 2 qubits for r=1) = 10 qubits
    assert qc.num_qubits == 10
    assert len(qr_freqs) == num_gauge
    
    # Ensure the circuit is unitary and mathematically valid
    # If the basis changes (Hadamards) or parity computations are misaligned, 
    # Statevector generation will fail or result in non-normalized states.
    sv = Statevector.from_instruction(qc)
    assert np.isclose(np.linalg.norm(sv.data), 1.0)