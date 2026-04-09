import numpy as np
import pytest
import math
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
from src.quantum_routines import CircuitBuilder

@pytest.fixture
def builder():
    """Fixture to provide a fresh CircuitBuilder instance for each test."""
    return CircuitBuilder()

def test_circular_shift_operators(builder):
    """
    Validates that V+ and V- operators are unitary and implement 
    the correct circular increment/decrement logic.
    """
    num_qubits = 2 # 4-dimensional frequency space: |0>, |1>, |2>, |3>
    dim = 2**num_qubits
    
    v_plus_matrix = builder._build_V_plus_gate(num_qubits).to_matrix()
    v_minus_matrix = builder._build_V_minus_gate(num_qubits).to_matrix()
    
    # 1. Unitarity Check: V V^dagger = I
    assert np.allclose(v_plus_matrix @ v_plus_matrix.conj().T, np.eye(dim))
    assert np.allclose(v_minus_matrix @ v_minus_matrix.conj().T, np.eye(dim))
    
    # 2. Circular Increment Check: V+ |3> -> |0>
    state_3 = np.array([0, 0, 0, 1])
    expected_state_0 = np.array([1, 0, 0, 0])
    assert np.allclose(v_plus_matrix @ state_3, expected_state_0)
    
    # 3. Circular Decrement Check: V- |0> -> |3>
    assert np.allclose(v_minus_matrix @ expected_state_0, state_3)
    
    # 4. Adjoint Relationship: V- should be the conjugate transpose of V+
    assert np.allclose(v_minus_matrix, v_plus_matrix.conj().T)

def test_parity_computation(builder):
    """
    Validates the D(b_l) operator correctly computes parity onto the ancilla.
    """
    # 2 data qubits, 1 ancilla
    qr = QuantumRegister(3)
    qc = QuantumCircuit(qr)
    
    # Initialize data qubits to |11> (Even parity)
    qc.x(0)
    qc.x(1)
    
    # Apply parity computation targeting ancilla (qubit 2)
    builder.apply_parity_computation(qc, data_qubits=[0, 1], ancilla_qubit=2)
    
    # Since parity is even, ancilla should be |0> (State |011> in Qiskit's little-endian)
    sv = Statevector.from_instruction(qc)
    expected_sv = Statevector.from_label('011')
    assert sv.equiv(expected_sv)
    
    # Reset and test odd parity |10>
    qc_odd = QuantumCircuit(qr)
    qc_odd.x(0)
    builder.apply_parity_computation(qc_odd, data_qubits=[0, 1], ancilla_qubit=2)
    
    # Ancilla should be |1> (State |101>)
    sv_odd = Statevector.from_instruction(qc_odd)
    expected_sv_odd = Statevector.from_label('101')
    assert sv_odd.equiv(expected_sv_odd)

def test_transform_data_upload_gate(builder):
    """
    Validates the full D(b_l) G(s_l) D(b_l) transformation logic.
    Ensures the frequency register is shifted appropriately based on data qubit parity
    and that the ancilla is successfully uncomputed back to |0>.
    """
    # 2 data qubits, 1 ancilla, 2 frequency qubits
    qr_data = QuantumRegister(2, 'data')
    qr_ancilla = QuantumRegister(1, 'ancilla')
    qr_freq = QuantumRegister(2, 'freq')
    qc = QuantumCircuit(qr_data, qr_ancilla, qr_freq)
    
    # Setup: Data = |01> (Odd parity -> should trigger V- decrement)
    # Freq = |01> (Integer 1)
    qc.x(qr_data[0])
    qc.x(qr_freq[0])
    
    # Apply the full transformation
    builder.transform_data_upload_gate(
        qc=qc, 
        data_qubits=[qr_data[0], qr_data[1]], 
        ancilla_qubit=qr_ancilla[0], 
        freq_register=qr_freq
    )
    
    sv = Statevector.from_instruction(qc)
    
    # Since parity is odd, frequency |01> (1) should decrement to |00> (0)
    # Ancilla must be uncomputed to |0>
    # Data remains |01>
    # Expected final state: Freq|00> Ancilla|0> Data|01> -> '00001'
    expected_sv = Statevector.from_label('00001')
    assert sv.equiv(expected_sv)

def test_hadamard_extraction_circuit(builder):
    """
    Validates that the physical Hadamard test circuit successfully extracts 
    the correct Fourier amplitude compared to the mathematical statevector.
    """
    num_qubits = 2
    x_edges = [(0, 1)]
    tau = 0.5
    r_steps = 1
    target_freq = 1
    
    # 1. Exact mathematical extraction via Statevector
    # FIX: Account for the list of frequency registers
    base_qc, qr_freqs = builder.build_trotter_extraction_circuit(num_qubits, x_edges, tau, r_steps)
    sv_base = Statevector.from_instruction(base_qc)
    
    # Amplitude of |freq=1>|ancilla=0>|data=00>
    # Little-endian stride for frequency register: 2**(num_qubits + 1)
    stride = 2**(num_qubits + 1)
    exact_b_1 = sv_base.data[target_freq * stride]
    
    # 2. Hardware-level extraction via Hadamard Test
    ht_qc = builder.build_fourier_hadamard_test(
        num_qubits, x_edges, tau, r_steps, target_freq, part='real'
    )
    
    # Remove measurement to allow Statevector probability extraction
    ht_qc.remove_final_measurements()
    sv_ht = Statevector.from_instruction(ht_qc)
    
    # qr_ht_ancilla is at qubit index 0 in Qiskit
    probs = sv_ht.probabilities([0])
    p_0 = probs[0]
    
    # Reconstruct the real part of the coefficient: Re(b_l) = 2 * P(0) - 1
    extracted_real_b_1 = 2 * p_0 - 1
    
    # Assert the hardware circuit measurement matches the exact mathematics
    assert np.isclose(extracted_real_b_1, np.real(exact_b_1), atol=1e-5)

def test_quantum_overlap_kernel_circuit(builder):
    """
    Validates that the quantum overlap circuit correctly computes the 
    dot product (kernel) between the feature maps of two different inputs.
    """
    num_qubits = 2
    tau = 0.5
    r_steps = 1
    
    # Define two different graph topologies (inputs x and x')
    x_edges_1 = [(0, 1)] 
    x_edges_2 = []       # Empty graph
    
    # 1. Exact mathematical extraction of the feature vectors
    qc_1, qr_freqs = builder.build_trotter_extraction_circuit(num_qubits, x_edges_1, tau, r_steps)
    qc_2, _ = builder.build_trotter_extraction_circuit(num_qubits, x_edges_2, tau, r_steps)
    
    # FIX: Dynamically get the dimension from the updated architecture instead of buggy math
    num_freq_qubits = qr_freqs[0].size
    dim_freq = 2**num_freq_qubits
    
    sv_1 = Statevector.from_instruction(qc_1)
    sv_2 = Statevector.from_instruction(qc_2)
    
    # Stride for base extraction circuit: freq starts at bit (num_qubits + 1)
    state_stride_base = 2**(num_qubits + 1)
    
    b_1 = np.zeros(dim_freq, dtype=complex)
    b_2 = np.zeros(dim_freq, dtype=complex)
    
    # Extract amplitudes projecting onto the |0...0> subspace for data/ancilla
    for freq_val in range(dim_freq):
        idx = freq_val * state_stride_base
        b_1[freq_val] = sv_1.data[idx]
        b_2[freq_val] = sv_2.data[idx]
        
    exact_overlap = np.real(np.vdot(b_1, b_2)) # b(x) \cdot b(x')
    
    # 2. Hardware-level extraction via the Kernel Overlap Circuit
    kernel_qc = builder.build_quantum_overlap_kernel_circuit(
        num_qubits, x_edges_1, x_edges_2, tau, r_steps
    )
    
    kernel_qc.remove_final_measurements()
    sv_kernel = Statevector.from_instruction(kernel_qc)
    
    # kernel_state_stride is shifted by 1 because of the k_ancilla at index 0
    kernel_state_stride = 2**(num_qubits + 2)
    
    prob_0_rest_0 = 0.0
    prob_1_rest_0 = 0.0
    
    # We must sum the overlap across ALL frequencies
    for freq_val in range(dim_freq):
        # Index where data+ancilla are 0
        idx_k_0 = freq_val * kernel_state_stride + 0 # k_ancilla = 0
        idx_k_1 = freq_val * kernel_state_stride + 1 # k_ancilla = 1
        
        prob_0_rest_0 += np.abs(sv_kernel.data[idx_k_0])**2
        prob_1_rest_0 += np.abs(sv_kernel.data[idx_k_1])**2
        
    # The Z observable expectation value projected on the |0><0| subspace
    extracted_overlap = prob_0_rest_0 - prob_1_rest_0
    
    assert np.isclose(extracted_overlap, exact_overlap, atol=1e-5)

def test_lcu_observable_extraction(builder):
    """
    Validates that the LCU extraction circuit correctly superimposes 
    the Fourier coefficients for a generic multi-term observable.
    """
    from qiskit.circuit.library import PauliGate
    
    num_qubits = 2
    x_edges = [(0, 1)]
    tau = 0.5
    r_steps = 1
    
    # Define O = 0.6 * (I \otimes Z) + 0.8 * (X \otimes X)
    observable_paulis = ['IZ', 'XX'] 
    observable_coeffs = [0.6, 0.8]
    
    # --- 1. Hardware-level LCU Extraction ---
    lcu_qc = builder.build_lcu_fourier_extraction_circuit(
        num_qubits, x_edges, tau, r_steps, observable_paulis, observable_coeffs
    )
    sv_lcu = Statevector.from_instruction(lcu_qc)
    
    # FIX: Dynamically read register sizes from the actual architecture
    base_qc, qr_freqs = builder.build_trotter_extraction_circuit(num_qubits, x_edges, tau, r_steps)
    num_freq_qubits = qr_freqs[0].size
    dim_freq = 2**num_freq_qubits
    num_lcu_qubits = max(1, math.ceil(math.log2(len(observable_coeffs))))
    
    # Stride to reach |freq=l> |ancilla=0> |data=0> |lcu=0>
    state_stride_lcu = 2**(num_lcu_qubits + num_qubits + 1)
    
    lcu_b_vector = np.zeros(dim_freq, dtype=complex)
    for freq_val in range(dim_freq):
        lcu_b_vector[freq_val] = sv_lcu.data[freq_val * state_stride_lcu]
        
    # Standard LCU algorithms scale the output amplitudes by 1 / ||beta||_1
    norm_1 = np.sum(np.abs(observable_coeffs))
    extracted_b = lcu_b_vector * norm_1
    
    # --- 2. Exact Mathematical Extraction ---
    exact_b = np.zeros(dim_freq, dtype=complex)
    
    qr_d = QuantumRegister(num_qubits)
    qr_a = QuantumRegister(1)
    qr_f = QuantumRegister(num_freq_qubits) # FIX: Use correct register size
    
    # Calculate A(U) -> P_h -> A(U)^dagger manually for each Pauli
    for pauli_str, coeff in zip(observable_paulis, observable_coeffs):
        test_qc = QuantumCircuit(qr_d, qr_a, qr_f)
        test_qc.append(base_qc.to_instruction(), test_qc.qubits)
        test_qc.append(PauliGate(pauli_str), qr_d)
        test_qc.append(base_qc.inverse().to_instruction(), test_qc.qubits)
        
        sv_test = Statevector.from_instruction(test_qc)
        stride_base = 2**(num_qubits + 1)
        
        for freq_val in range(dim_freq):
            exact_b[freq_val] += coeff * sv_test.data[freq_val * stride_base]
            
    # --- 3. Assertion ---
    assert np.allclose(extracted_b, exact_b, atol=1e-5)