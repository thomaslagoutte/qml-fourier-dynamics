import numpy as np
import math
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UnitaryGate, StatePreparation, PauliGate
from typing import List, Tuple

class CircuitBuilder:
    """
    Constructs and transforms quantum circuits for Fourier coefficient extraction.
    Implements the A(U) algorithm to map parameters to independent frequency registers.
    """
    def __init__(self):
        pass

    def _build_V_plus_gate(self, num_qubits: int) -> UnitaryGate:
        dim = 2**num_qubits
        matrix = np.zeros((dim, dim), dtype=complex)
        for i in range(dim):
            matrix[(i + 1) % dim, i] = 1.0
        return UnitaryGate(matrix, label="V+")

    def _build_V_minus_gate(self, num_qubits: int) -> UnitaryGate:
        dim = 2**num_qubits
        matrix = np.zeros((dim, dim), dtype=complex)
        for i in range(dim):
            matrix[(i - 1) % dim, i] = 1.0
        return UnitaryGate(matrix, label="V-")

    def apply_parity_computation(self, qc: QuantumCircuit, data_qubits: List[int], ancilla_qubit: int):
        for qubit in data_qubits:
            qc.cx(qubit, ancilla_qubit)

    def apply_controlled_frequency_shift(self, qc: QuantumCircuit, ancilla_qubit: int, freq_register: QuantumRegister):
        num_freq_qubits = freq_register.size
        v_plus = self._build_V_plus_gate(num_freq_qubits)
        v_minus = self._build_V_minus_gate(num_freq_qubits)
        
        ctrl_v_plus = v_plus.control(num_ctrl_qubits=1, ctrl_state='0')
        ctrl_v_minus = v_minus.control(num_ctrl_qubits=1, ctrl_state='1')
        
        qc.append(ctrl_v_plus, [ancilla_qubit] + list(freq_register))
        qc.append(ctrl_v_minus, [ancilla_qubit] + list(freq_register))
        
    def transform_data_upload_gate(self, qc: QuantumCircuit, data_qubits: List[int], ancilla_qubit: int, freq_register: QuantumRegister):
        self.apply_parity_computation(qc, data_qubits, ancilla_qubit)
        self.apply_controlled_frequency_shift(qc, ancilla_qubit, freq_register)
        self.apply_parity_computation(qc, data_qubits, ancilla_qubit)

    def _freq_register_size(self, r_steps: int) -> int:
        return max(3, math.ceil(math.log2(4 * r_steps + 2)))

    def build_trotter_extraction_circuit(self, num_qubits: int, x_edges: List[tuple], tau: float, r_steps: int, d_params: int = 1) -> Tuple[QuantumCircuit, List[QuantumRegister]]:
        n_s = self._freq_register_size(r_steps)
        qr_data = QuantumRegister(num_qubits, 'data')
        qr_ancilla = QuantumRegister(1, 'ancilla')
        # d independent registers
        qr_freqs = [QuantumRegister(n_s, f'freq_{s}') for s in range(d_params)]
        
        qc = QuantumCircuit(qr_data, qr_ancilla, *qr_freqs)
        
        for step in range(r_steps):
            for i, j in x_edges:
                qc.rzz(2 * tau / r_steps, qr_data[i], qr_data[j])
            for s in range(min(d_params, num_qubits)):
                qc.h(qr_data[s])
                self.transform_data_upload_gate(qc, [qr_data[s]], qr_ancilla[0], qr_freqs[s])
                qc.h(qr_data[s])
                
        return qc, qr_freqs

    def build_fourier_hadamard_test(self, num_qubits: int, x_edges: list, tau: float, r_steps: int, target_freq: int, part: str = 'real') -> QuantumCircuit:
        # Request d_params=1 to keep a single register for the base Hadamard Test
        base_qc, qr_freqs = self.build_trotter_extraction_circuit(num_qubits, x_edges, tau, r_steps, d_params=1)
        num_freq_qubits = qr_freqs[0].size
        
        qr_ht_ancilla = QuantumRegister(1, 'ht_ancilla')
        qr_data = QuantumRegister(num_qubits, 'data')
        qr_ancilla = QuantumRegister(1, 'ancilla')
        qr_freq = QuantumRegister(num_freq_qubits, 'freq_0')
        cr_measure = ClassicalRegister(1, 'meas')
        
        ht_qc = QuantumCircuit(qr_ht_ancilla, qr_data, qr_ancilla, qr_freq, cr_measure)
        
        ht_qc.h(qr_ht_ancilla)
        if part == 'imag':
            ht_qc.sdg(qr_ht_ancilla)
            
        controlled_a_u = base_qc.to_gate(label="A(U)").control(1)
        target_qubits = list(qr_data) + list(qr_ancilla) + list(qr_freq)
        ht_qc.append(controlled_a_u, [qr_ht_ancilla[0]] + target_qubits)
        
        binary_str = format(target_freq, f'0{num_freq_qubits}b')
        for i, bit in enumerate(reversed(binary_str)):
            if bit == '1':
                ht_qc.cx(qr_ht_ancilla[0], qr_freq[i])
                
        ht_qc.h(qr_ht_ancilla)
        ht_qc.measure(qr_ht_ancilla, cr_measure)
        
        return ht_qc

    def build_quantum_overlap_kernel_circuit(self, num_qubits: int, x_edges_1: list, x_edges_2: list, tau: float, r_steps: int) -> QuantumCircuit:
        qc_1, qr_freqs = self.build_trotter_extraction_circuit(num_qubits, x_edges_1, tau, r_steps, d_params=1)
        qc_2, _ = self.build_trotter_extraction_circuit(num_qubits, x_edges_2, tau, r_steps, d_params=1)
        num_freq_qubits = qr_freqs[0].size
        
        qr_k_ancilla = QuantumRegister(1, 'kernel_ancilla')
        qr_data = QuantumRegister(num_qubits, 'data')
        qr_ancilla = QuantumRegister(1, 'ancilla')
        qr_freq = QuantumRegister(num_freq_qubits, 'freq_0')
        
        cr_k_ancilla = ClassicalRegister(1, 'meas_k_ancilla')
        cr_rest = ClassicalRegister(num_qubits + 1 + num_freq_qubits, 'meas_rest')
        
        kernel_qc = QuantumCircuit(qr_k_ancilla, qr_data, qr_ancilla, qr_freq, cr_k_ancilla, cr_rest)
        
        gate_1 = qc_1.to_gate(label="A(U_x)").control(1, ctrl_state='0')
        gate_2 = qc_2.to_gate(label="A(U_x')").control(1, ctrl_state='1')
        
        target_qubits = list(qr_data) + list(qr_ancilla) + list(qr_freq)
        
        kernel_qc.h(qr_k_ancilla)
        kernel_qc.append(gate_1, [qr_k_ancilla[0]] + target_qubits)
        kernel_qc.append(gate_2, [qr_k_ancilla[0]] + target_qubits)
        kernel_qc.h(qr_k_ancilla)
        
        kernel_qc.measure(qr_k_ancilla, cr_k_ancilla)
        kernel_qc.measure(target_qubits, cr_rest)
        
        return kernel_qc

    def build_lcu_fourier_extraction_circuit(self, num_qubits: int, x_edges: list, tau: float, r_steps: int, observable_paulis: list, observable_coeffs: list):
        amplitudes = np.sqrt(np.array(observable_coeffs, dtype=complex))
        norm = np.linalg.norm(amplitudes)
        normalized_amps = amplitudes / norm
        
        num_lcu_qubits = max(1, math.ceil(math.log2(len(observable_coeffs))))
        padded_amps = np.zeros(2**num_lcu_qubits, dtype=complex)
        padded_amps[:len(normalized_amps)] = normalized_amps
        
        lcu_prep = StatePreparation(padded_amps)
        
        base_qc, qr_freqs = self.build_trotter_extraction_circuit(num_qubits, x_edges, tau, r_steps, d_params=1)
        num_freq_qubits = qr_freqs[0].size
        
        qr_lcu = QuantumRegister(num_lcu_qubits, 'lcu')
        qr_data = QuantumRegister(num_qubits, 'data')
        qr_ancilla = QuantumRegister(1, 'ancilla')
        qr_freq = QuantumRegister(num_freq_qubits, 'freq_0')
        
        lcu_qc = QuantumCircuit(qr_lcu, qr_data, qr_ancilla, qr_freq)
        target_base_qubits = list(qr_data) + list(qr_ancilla) + list(qr_freq)
        
        lcu_qc.append(lcu_prep, qr_lcu)
        lcu_qc.append(base_qc.to_instruction(), target_base_qubits)
        
        for i, pauli_str in enumerate(observable_paulis):
            ctrl_state = format(i, f'0{num_lcu_qubits}b')
            p_gate = PauliGate(pauli_str).control(num_lcu_qubits, ctrl_state=ctrl_state)
            lcu_qc.append(p_gate, list(qr_lcu) + list(qr_data))
            
        lcu_qc.append(base_qc.inverse().to_instruction(), target_base_qubits)
        lcu_qc.append(lcu_prep.inverse(), qr_lcu)
        
        return lcu_qc

    def build_lgt_trotter_extraction_circuit(self, num_matter_sites: int, x_state: str, mass: float, electric_field: float, tau: float, r_steps: int):
        num_gauge_links = num_matter_sites - 1
        num_data_qubits = num_matter_sites + num_gauge_links
        num_params = num_gauge_links
        
        n_s = self._freq_register_size(r_steps)
        
        qr_data = QuantumRegister(num_data_qubits, 'data')
        qr_ancilla = QuantumRegister(1, 'ancilla')
        qr_freqs = [QuantumRegister(n_s, f'freq_{i}') for i in range(num_params)]
        
        qc = QuantumCircuit(qr_data, qr_ancilla, *qr_freqs)
        
        for i, bit in enumerate(reversed(x_state)):
            if bit == '1':
                qc.x(qr_data[i])

        def apply_pauli_extraction(pauli_string, target_freq_reg):
            for i, p in enumerate(reversed(pauli_string)):
                if p == 'X':
                    qc.h(qr_data[i])
            
            active_qubits = [qr_data[i] for i, p in enumerate(reversed(pauli_string)) if p in ['X', 'Z']]
            for q in active_qubits:
                qc.cx(q, qr_ancilla[0])
                
            v_plus = self._build_V_plus_gate(n_s)
            v_minus = self._build_V_minus_gate(n_s)
            qc.append(v_plus.control(1, ctrl_state='0'), [qr_ancilla[0]] + list(target_freq_reg))
            qc.append(v_minus.control(1, ctrl_state='1'), [qr_ancilla[0]] + list(target_freq_reg))
            
            for q in reversed(active_qubits):
                qc.cx(q, qr_ancilla[0])
                
            for i, p in enumerate(reversed(pauli_string)):
                if p == 'X':
                    qc.h(qr_data[i])

        dt = tau / r_steps
        for step in range(r_steps):
            for i in range(num_matter_sites):
                qc.rz(2 * mass * dt, qr_data[2*i])
            for i in range(num_gauge_links):
                qc.rx(2 * electric_field * dt, qr_data[2*i + 1])
                
            for i in range(num_gauge_links):
                pauli_chars = ['I'] * num_data_qubits
                pauli_chars[2*i] = 'X'
                pauli_chars[2*i+1] = 'Z'
                pauli_chars[2*i+2] = 'X'
                pauli_str = "".join(pauli_chars)[::-1]
                apply_pauli_extraction(pauli_str, qr_freqs[i])
                
        return qc, qr_freqs