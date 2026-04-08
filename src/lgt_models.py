import numpy as np

class Z2GaugeModel:
    """
    A 1D Z_2 Lattice Gauge Theory model for testing PAC learning bounds.
    The lattice consists of alternating matter sites and gauge links.
    For N matter sites, there are N-1 gauge links.
    Total qubits required: 2N - 1.
    """
    def __init__(self, num_matter_sites: int, mass: float = 1.0, electric_field: float = 1.0):
        self.num_matter_sites = num_matter_sites
        self.num_gauge_links = num_matter_sites - 1
        self.total_qubits = num_matter_sites + self.num_gauge_links
        
        self.mass = mass
        self.electric_field = electric_field

    def _get_slow_coupling_profile(self, alphas: np.ndarray) -> np.ndarray:
        """
        O(log n) regime: The coupling strength g(x) changes smoothly across space.
        We model this as a truncated Fourier series.
        alphas = [DC_offset, cosine_amp, sine_amp]
        """
        assert len(alphas) == 3, "Slow profile requires exactly 3 alpha parameters."
        
        x_positions = np.arange(self.num_gauge_links)
        # Base spatial frequency
        k = 2 * np.pi / self.num_gauge_links 
        
        g_x = alphas[0] + alphas[1] * np.cos(k * x_positions) + alphas[2] * np.sin(k * x_positions)
        return g_x

    def _get_fast_coupling_profile(self, alphas: np.ndarray) -> np.ndarray:
        """
        O(poly n) regime: The coupling strength g(x) is completely independent 
        at every single spatial link.
        """
        assert len(alphas) == self.num_gauge_links, f"Fast profile requires {self.num_gauge_links} alpha parameters."
        return alphas

    def generate_hamiltonian_terms(self, alphas: np.ndarray, regime: str) -> list:
        """
        Generates the Pauli strings and their coefficients for the Trotterized simulation.
        Returns a list of tuples: (Pauli_String, Coefficient)
        
        Qubit mapping:
        Even indices (0, 2, 4...) are Matter sites.
        Odd indices (1, 3, 5...) are Gauge links.
        """
        if regime == 'slow':
            g_x = self._get_slow_coupling_profile(alphas)
        elif regime == 'fast':
            g_x = self._get_fast_coupling_profile(alphas)
        else:
            raise ValueError("Regime must be 'slow' or 'fast'.")

        terms = []

        # 1. Matter Mass Terms (Z on matter sites)
        for i in range(self.num_matter_sites):
            qubit_idx = 2 * i
            pauli_chars = ['I'] * self.total_qubits
            pauli_chars[qubit_idx] = 'Z'
            terms.append(("".join(pauli_chars)[::-1], self.mass)) # Qiskit string reversal

        # 2. Electric Field Terms (X on gauge links)
        for i in range(self.num_gauge_links):
            qubit_idx = 2 * i + 1
            pauli_chars = ['I'] * self.total_qubits
            pauli_chars[qubit_idx] = 'X'
            terms.append(("".join(pauli_chars)[::-1], self.electric_field))

        # 3. Matter-Gauge Interaction Terms (X_i Z_{i,i+1} X_{i+1})
        for i in range(self.num_gauge_links):
            matter_1 = 2 * i
            gauge_link = 2 * i + 1
            matter_2 = 2 * i + 2
            
            pauli_chars = ['I'] * self.total_qubits
            pauli_chars[matter_1] = 'X'
            pauli_chars[gauge_link] = 'Z'
            pauli_chars[matter_2] = 'X'
            
            # The coefficient is our unknown g(x) profile
            coeff = g_x[i]
            terms.append(("".join(pauli_chars)[::-1], coeff))

        return terms