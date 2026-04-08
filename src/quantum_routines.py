import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from typing import List

class CircuitBuilder:
    """
    Constructs and transforms quantum circuits for Fourier coefficient extraction.
    Implements the A(U) algorithm to map parameters to frequency registers.
    """
    def __init__(self):
        pass

    def _build_V_plus_gate(self, num_qubits: int) -> UnitaryGate:
        """
        Creates the circular increment operator V+ for a frequency register.
        Maps |k> to |(k+1) mod 2^n>.
        """
        dim = 2**num_qubits
        matrix = np.zeros((dim, dim), dtype=complex)
        for i in range(dim):
            matrix[(i + 1) % dim, i] = 1.0
            
        return UnitaryGate(matrix, label="V+")

    def _build_V_minus_gate(self, num_qubits: int) -> UnitaryGate:
        """
        Creates the circular decrement operator V- for a frequency register.
        Maps |k> to |(k-1) mod 2^n>.
        """
        dim = 2**num_qubits
        matrix = np.zeros((dim, dim), dtype=complex)
        for i in range(dim):
            matrix[(i - 1) % dim, i] = 1.0
            
        return UnitaryGate(matrix, label="V-")

    def apply_parity_computation(self, qc: QuantumCircuit, data_qubits: List[int], ancilla_qubit: int):
        """
        Applies D(b_l) = \prod C^{(c_k)} X^{(a)}.
        Computes the parity of the selected data_qubits onto the ancilla_qubit.
        """
        for qubit in data_qubits:
            qc.cx(qubit, ancilla_qubit)

    def apply_controlled_frequency_shift(self, qc: QuantumCircuit, ancilla_qubit: int, freq_register: QuantumRegister):
        """
        Applies G(s_l) = (C^{(a)}V_+^{(f_{s_l})})(\overline{C}^{(a)}V_-^{(f_{s_l})}).
        Increments frequency if parity is even (|0>), decrements if odd (|1>).
        """
        num_freq_qubits = freq_register.size
        
        # Build the base unitaries
        v_plus = self._build_V_plus_gate(num_freq_qubits)
        v_minus = self._build_V_minus_gate(num_freq_qubits)
        
        # Create controlled versions
        # V+ is controlled on ancilla being |0> (Even parity)
        ctrl_v_plus = v_plus.control(num_ctrl_qubits=1, ctrl_state='0')
        # V- is controlled on ancilla being |1> (Odd parity)
        ctrl_v_minus = v_minus.control(num_ctrl_qubits=1, ctrl_state='1')
        
        # Apply to circuit (ancilla is the control, freq_register are the targets)
        qc.append(ctrl_v_plus, [ancilla_qubit] + list(freq_register))
        qc.append(ctrl_v_minus, [ancilla_qubit] + list(freq_register))
        
    def transform_data_upload_gate(self, qc: QuantumCircuit, data_qubits: List[int], ancilla_qubit: int, freq_register: QuantumRegister):
        """
        Executes the full transformation for a single Z-basis data uploading gate:
        D(b_l) G(s_l) D(b_l)
        """
        # 1. Compute Parity
        self.apply_parity_computation(qc, data_qubits, ancilla_qubit)
        
        # 2. Shift Frequency based on Parity
        self.apply_controlled_frequency_shift(qc, ancilla_qubit, freq_register)
        
        # 3. Uncompute Parity (D(b_l) is its own inverse)
        self.apply_parity_computation(qc, data_qubits, ancilla_qubit)

    def build_trotter_extraction_circuit(self, num_qubits: int, x_edges: List[tuple], tau: float, r_steps: int) -> tuple:
        """
        Builds the A(U) transformed Trotter circuit for the Ising model.
        Returns the constructed QuantumCircuit and the frequency register.
        """
        import math
        
        # Calculate max frequency needed to avoid overflow. 
        # Each qubit has an X(alpha) gate per step. 
        max_freq = num_qubits * r_steps
        num_freq_qubits = math.ceil(math.log2(2 * max_freq + 1))
        
        qr_data = QuantumRegister(num_qubits, 'data')
        qr_ancilla = QuantumRegister(1, 'ancilla')
        qr_freq = QuantumRegister(num_freq_qubits, 'freq')
        
        # Order matters for Statevector parsing: Freq (MSB) -> Ancilla -> Data (LSB)
        qc = QuantumCircuit(qr_data, qr_ancilla, qr_freq)
        
        for step in range(r_steps):
            # 1. Fixed gates: ZZ interactions for edges
            for i, j in x_edges:
                # Qiskit's RZZ(theta) = exp(-i * theta/2 * ZZ). 
                # To get exp(-i * tau/r * ZZ), we use angle 2*tau/r
                qc.rzz(2 * tau / r_steps, qr_data[i], qr_data[j])
                
            # 2. Transformed parametrized gates: X_i(alpha * tau/r)
            for i in range(num_qubits):
                # Basis change to Z
                qc.h(qr_data[i])
                # Apply A(U) transformation
                self.transform_data_upload_gate(qc, [qr_data[i]], qr_ancilla[0], qr_freq)
                # Basis change back
                qc.h(qr_data[i])
                
        return qc, qr_freq
    
    def build_fourier_hadamard_test(self, num_qubits: int, x_edges: list, tau: float, r_steps: int, target_freq: int, part: str = 'real') -> QuantumCircuit:
        """
        Constructs the Hadamard test circuit to extract the real or imaginary part
        of a specific Fourier coefficient b_l natively on a QPU.
        """
        from qiskit import QuantumRegister, ClassicalRegister
        
        # 1. Build the base A(U) circuit and get its frequency register size
        base_qc, qr_freq_base = self.build_trotter_extraction_circuit(num_qubits, x_edges, tau, r_steps)
        num_freq_qubits = qr_freq_base.size
        
        # 2. Setup the full Hadamard Test registers
        qr_ht_ancilla = QuantumRegister(1, 'ht_ancilla')
        qr_data = QuantumRegister(num_qubits, 'data')
        qr_ancilla = QuantumRegister(1, 'ancilla')
        qr_freq = QuantumRegister(num_freq_qubits, 'freq')
        cr_measure = ClassicalRegister(1, 'meas')
        
        ht_qc = QuantumCircuit(qr_ht_ancilla, qr_data, qr_ancilla, qr_freq, cr_measure)
        
        # 3. Prepare the Hadamard control 
        ht_qc.h(qr_ht_ancilla)
        if part == 'imag':
            # S^dagger aligns the phase to extract the imaginary component 
            ht_qc.sdg(qr_ht_ancilla)
            
        # 4. Controlled A(U)
        # Convert the Trotterized extraction circuit into a controlled gate 
        controlled_a_u = base_qc.to_gate(label="A(U)").control(1)
        
        # Targets: data, ancilla, and freq (must perfectly match base_qc creation order)
        target_qubits = list(qr_data) + list(qr_ancilla) + list(qr_freq)
        ht_qc.append(controlled_a_u, [qr_ht_ancilla[0]] + target_qubits)
        
        # 5. Controlled V_l^dagger 
        # Since |l> is a computational basis state initialized from |0>, V_l^dagger is 
        # effectively uncomputing the binary representation of the integer l.
        binary_str = format(target_freq, f'0{num_freq_qubits}b')
        for i, bit in enumerate(reversed(binary_str)):
            if bit == '1':
                ht_qc.cx(qr_ht_ancilla[0], qr_freq[i])
                
        # 6. Finalize Hadamard Test 
        ht_qc.h(qr_ht_ancilla)
        ht_qc.measure(qr_ht_ancilla, cr_measure)
        
        return ht_qc
    
    def build_quantum_overlap_kernel_circuit(self, num_qubits: int, x_edges_1: list, x_edges_2: list, tau: float, r_steps: int) -> QuantumCircuit:
        """
        Constructs the Quantum Overlap Kernel circuit to evaluate k(x, x') = b(x) . b(x').
        Implements the interference circuit from Figure 8 of the paper.
        """
        from qiskit import QuantumRegister, ClassicalRegister
        
        # 1. Build the base extraction circuits for the two different inputs
        qc_1, qr_freq_base = self.build_trotter_extraction_circuit(num_qubits, x_edges_1, tau, r_steps)
        qc_2, _ = self.build_trotter_extraction_circuit(num_qubits, x_edges_2, tau, r_steps)
        
        num_freq_qubits = qr_freq_base.size
        
        # 2. Setup the global kernel registers
        qr_k_ancilla = QuantumRegister(1, 'kernel_ancilla')
        qr_data = QuantumRegister(num_qubits, 'data')
        qr_ancilla = QuantumRegister(1, 'ancilla')
        qr_freq = QuantumRegister(num_freq_qubits, 'freq')
        
        # We need classical registers to measure everything for the global |0><0| projection
        cr_k_ancilla = ClassicalRegister(1, 'meas_k_ancilla')
        cr_rest = ClassicalRegister(num_qubits + 1 + num_freq_qubits, 'meas_rest')
        
        kernel_qc = QuantumCircuit(qr_k_ancilla, qr_data, qr_ancilla, qr_freq, cr_k_ancilla, cr_rest)
        
        # 3. Create controlled versions of the extraction circuits
        # qc_1 is controlled on the kernel ancilla being |0>
        gate_1 = qc_1.to_gate(label="A(U_x)").control(1, ctrl_state='0')
        # qc_2 is controlled on the kernel ancilla being |1>
        gate_2 = qc_2.to_gate(label="A(U_x')").control(1, ctrl_state='1')
        
        # Target qubits list matching the base circuit architecture
        target_qubits = list(qr_data) + list(qr_ancilla) + list(qr_freq)
        
        # 4. Build the interference circuit
        kernel_qc.h(qr_k_ancilla) # Create superposition of paths
        
        kernel_qc.append(gate_1, [qr_k_ancilla[0]] + target_qubits)
        kernel_qc.append(gate_2, [qr_k_ancilla[0]] + target_qubits)
        
        kernel_qc.h(qr_k_ancilla) # Interfere the paths
        
        # 5. Measure all qubits to allow the Z \otimes |0><0| evaluation
        kernel_qc.measure(qr_k_ancilla, cr_k_ancilla)
        kernel_qc.measure(target_qubits, cr_rest)
        
        return kernel_qc
    
    def build_lcu_fourier_extraction_circuit(self, num_qubits: int, x_edges: list, tau: float, r_steps: int, observable_paulis: list, observable_coeffs: list):
        """
        Constructs the LCU circuit to extract Fourier coefficients for a generic 
        observable defined as a linear combination of Pauli strings (O = sum beta_h P_h).
        Implements the architecture from Figure 5 of the paper.
        """
        from qiskit import QuantumRegister, QuantumCircuit
        from qiskit.circuit.library import StatePreparation, PauliGate
        import math
        import numpy as np
        
        # 1. Prepare LCU state amplitudes (sqrt of coefficients for standard LCU)
        # We use complex type to gracefully handle negative coefficients via phase
        amplitudes = np.sqrt(np.array(observable_coeffs, dtype=complex))
        norm = np.linalg.norm(amplitudes)
        normalized_amps = amplitudes / norm
        
        num_lcu_qubits = max(1, math.ceil(math.log2(len(observable_coeffs))))
        padded_amps = np.zeros(2**num_lcu_qubits, dtype=complex)
        padded_amps[:len(normalized_amps)] = normalized_amps
        
        lcu_prep = StatePreparation(padded_amps)
        
        # 2. Build the base A(U) circuit
        base_qc, qr_freq_base = self.build_trotter_extraction_circuit(num_qubits, x_edges, tau, r_steps)
        num_freq_qubits = qr_freq_base.size
        
        # 3. Setup registers (LCU Ancilla -> Data -> Parity Ancilla -> Frequencies)
        qr_lcu = QuantumRegister(num_lcu_qubits, 'lcu')
        qr_data = QuantumRegister(num_qubits, 'data')
        qr_ancilla = QuantumRegister(1, 'ancilla')
        qr_freq = QuantumRegister(num_freq_qubits, 'freq')
        
        lcu_qc = QuantumCircuit(qr_lcu, qr_data, qr_ancilla, qr_freq)
        target_base_qubits = list(qr_data) + list(qr_ancilla) + list(qr_freq)
        
        # 4. Construct the LCU Sequence
        # Step A: V_beta (State Preparation)
        lcu_qc.append(lcu_prep, qr_lcu)
        
        # Step B: A(U) Base execution
        lcu_qc.append(base_qc.to_instruction(), target_base_qubits)
        
        # Step C: Controlled Paulis (c-P_h)
        for i, pauli_str in enumerate(observable_paulis):
            ctrl_state = format(i, f'0{num_lcu_qubits}b')
            p_gate = PauliGate(pauli_str).control(num_lcu_qubits, ctrl_state=ctrl_state)
            lcu_qc.append(p_gate, list(qr_lcu) + list(qr_data))
            
        # Step D: A(U)^\dagger (Inverse Base execution)
        lcu_qc.append(base_qc.inverse().to_instruction(), target_base_qubits)
        
        # Step E: V_beta^\dagger (Inverse State Preparation)
        lcu_qc.append(lcu_prep.inverse(), qr_lcu)
        
        return lcu_qc
    
    def build_lgt_trotter_extraction_circuit(self, num_matter_sites: int, x_state: str, mass: float, electric_field: float, tau: float, r_steps: int):
        """
        Constructs the A(U) Fourier extraction circuit specifically for the Z2 LGT.
        x_state: A binary string representing the initial computational basis state.
        """
        from qiskit import QuantumCircuit, QuantumRegister
        import math
        
        num_gauge_links = num_matter_sites - 1
        num_data_qubits = num_matter_sites + num_gauge_links
        num_params = num_gauge_links
        
        dim_freq = 2**math.ceil(math.log2(2 * r_steps + 1))
        num_freq_qubits = dim_freq.bit_length() - 1
        
        qr_data = QuantumRegister(num_data_qubits, 'data')
        qr_ancilla = QuantumRegister(1, 'ancilla')
        qr_freqs = [QuantumRegister(num_freq_qubits, f'freq_{i}') for i in range(num_params)]
        
        qc = QuantumCircuit(qr_data, qr_ancilla, *qr_freqs)
        
        # Qiskit is little-endian, so we reverse the string to match logical indices
        for i, bit in enumerate(reversed(x_state)):
            if bit == '1':
                qc.x(qr_data[i])

        # Helper to apply the BRGD25 subroutine with dynamic basis changes
        def apply_pauli_extraction(pauli_string, target_freq_reg):
            # 1. Basis change to Z
            for i, p in enumerate(reversed(pauli_string)):
                if p == 'X':
                    qc.h(qr_data[i])
            
            # 2. Compute Parity
            active_qubits = [qr_data[i] for i, p in enumerate(reversed(pauli_string)) if p in ['X', 'Z']]
            for q in active_qubits:
                qc.cx(q, qr_ancilla[0])
                
            # 3. Frequency Shift (V_+ / V_-)
            v_plus = self._build_V_plus_gate(num_freq_qubits)
            v_minus = self._build_V_minus_gate(num_freq_qubits)
            qc.append(v_plus.control(1, ctrl_state='0'), [qr_ancilla[0]] + list(target_freq_reg))
            qc.append(v_minus.control(1, ctrl_state='1'), [qr_ancilla[0]] + list(target_freq_reg))
            
            # 4. Uncompute Parity
            for q in reversed(active_qubits):
                qc.cx(q, qr_ancilla[0])
                
            # 5. Undo Basis change
            for i, p in enumerate(reversed(pauli_string)):
                if p == 'X':
                    qc.h(qr_data[i])

        dt = tau / r_steps
        
        for step in range(r_steps):
            # A. Fixed Term: Matter Mass (Z)
            for i in range(num_matter_sites):
                qc.rz(2 * mass * dt, qr_data[2*i])
            
            # B. Fixed Term: Electric Field (X)
            for i in range(num_gauge_links):
                qc.rx(2 * electric_field * dt, qr_data[2*i + 1])
                
            # C. Parametrized Terms: Matter-Gauge Interaction (X_i Z_{i,i+1} X_{i+1})
            for i in range(num_gauge_links):
                pauli_chars = ['I'] * num_data_qubits
                pauli_chars[2*i] = 'X'
                pauli_chars[2*i+1] = 'Z'
                pauli_chars[2*i+2] = 'X'
                pauli_str = "".join(pauli_chars)[::-1]
                
                apply_pauli_extraction(pauli_str, qr_freqs[i])
                
        return qc, qr_freqs
