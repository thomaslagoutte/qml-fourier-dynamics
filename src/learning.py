"""PAC learning pipeline for quantum Hamiltonian dynamics.

Implements the algorithm described in:
  Barthe et al., "Quantum Advantage in Learning Quantum Dynamics via
  Fourier Coefficient Extraction" (2025).
"""

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from qiskit.quantum_info import Statevector
from tqdm import tqdm

from src.models import IsingTransverseFieldModel
from src.quantum_routines import CircuitBuilder


class FourierDynamicsLearner:
    """Hybrid quantum-classical PAC learning pipeline.

    This class implements the full learning algorithm from Section IV.B of
    Barthe et al. (2025), including the paper's correct b_l extraction via
    the A(U, P) circuit (Corollary 1 / Figure 4) and the Hadamard test for
    individual coefficient recovery (Corollary 2 / Figure 7).

    Concept class
    -------------
    The learner targets the Hamiltonian dynamics concept class (Definition 4):

        c_alpha(x) = <psi_0 | U(x, alpha)^dag O U(x, alpha) | psi_0>

    where O is a fixed Pauli observable, x encodes the graph topology, and
    alpha is the unknown transverse-field strength.  For the 1D TFIM:

        H(x, alpha) = sum_{(i,j) in x} Z_i Z_j  +  alpha * X_0

    the notebook experiment targets the local magnetisation O = sigma_z on
    qubit 0.

    Pipeline (Section IV.B)
    -----------------------
    Training stage:
        1. For each training graph x_i and each frequency l in L = {-2r,...,+2r},
           extract the Fourier coefficient b_l(x_i) via the A(U,P) circuit and
           a Hadamard test (``_extract_b_l``).
        2. Assemble the feature matrix B in R^{T x |L|}.
        3. Fit LASSO: minimise ||y - Bw||^2 + lambda*||w||_1.

    Inference:
        Given a new graph x', extract its feature vector b(x') and return
        yhat = w_hat . b(x').

    Correctness of the linear model
    --------------------------------
    Because f(alpha; x) = sum_l b_l(x) e^{i*pi*alpha*l} is linear in
    w(alpha) = [e^{i*pi*alpha*l}]_{l in L}, we have:

        c_alpha(x) = b(x) . w(alpha)

    For the training labels y_t = c_{alpha*}(x_t) with the TRUE fixed alpha*,
    stacking gives B w(alpha*) = y, which is the linear system LASSO solves.

    Attributes
    ----------
    num_qubits : int
        Number of data qubits.
    tau : float
        Total evolution time tau.
    pauli_observable : str
        Pauli string for the observable O in Qiskit's little-endian convention,
        where the **rightmost** character acts on qubit 0.  To measure sigma_z
        on qubit 0 of an n-qubit system use ``'I'*(n-1)+'Z'``, e.g. ``'IIZ'``
        for n=3.  The default is set accordingly.  Note: ``'Z'+'I'*(n-1)`` would
        incorrectly target qubit n-1.
    epsilon_b : float
        Additive noise per extracted b_l, modelling finite-shot error
        (Theorem 6 of the paper).
    model : IsingTransverseFieldModel
    lasso : Lasso
    """

    def __init__(
        self,
        num_qubits: int,
        tau: float,
        pauli_observable: str = None,
        epsilon_b: float = 0.01,
    ):
        self.num_qubits = num_qubits
        self.tau = tau
        # Default observable: sigma_z on qubit 0 in the little-endian Qiskit convention.
        # 'Z' + 'I'*(n-1) means Z acts on qubit 0, identity on qubits 1..n-1.
        # Qiskit uses little-endian Pauli strings: the rightmost character acts on
        # qubit 0.  To target sigma_z on qubit 0 we need 'I'*(n-1)+'Z', e.g. 'IIZ'
        # for a 3-qubit system.  The incorrect default 'Z'+'I'*(n-1) = 'ZII' would
        # target qubit n-1 instead, making the observable insensitive to the alpha
        # upload (which acts only on qubit 0 when d_params=1).
        self.pauli_observable = pauli_observable or ("I" * (num_qubits - 1) + "Z")
        self.epsilon_b = epsilon_b
        self.model = IsingTransverseFieldModel(num_qubits)
        self.lasso = Lasso(alpha=0.01, fit_intercept=False)

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    def generate_random_graphs(self, num_samples: int) -> list:
        """Return a list of random edge sets over num_qubits vertices."""
        datasets = []
        for _ in range(num_samples):
            edges = [
                (i, j)
                for i in range(self.num_qubits)
                for j in range(i + 1, self.num_qubits)
                if np.random.rand() > 0.5
            ]
            datasets.append(edges)
        return datasets

    def compute_exact_labels(self, datasets: list, true_alpha: float) -> np.ndarray:
        """Compute the exact Pauli expectation value under time evolution.

        Evaluates c_{alpha}(x) = <psi_0|U(x,alpha)^dag O U(x,alpha)|psi_0>
        using exact (non-Trotterised) matrix exponentiation, where O is
        ``self.pauli_observable`` and psi_0 = |0>.

        Parameters
        ----------
        datasets : list of list[(int,int)]
            Training graphs x_i.
        true_alpha : float
            The fixed unknown parameter alpha* used to generate labels.

        Returns
        -------
        np.ndarray of shape (len(datasets),)
            Real-valued expectation values in [-1, +1].
        """
        from qiskit.quantum_info import SparsePauliOp

        # Build the Pauli matrix for the observable
        obs_matrix = SparsePauliOp(self.pauli_observable).to_matrix()

        labels = []
        for edges in datasets:
            U = self.model.exact_time_evolution_operator(edges, true_alpha, self.tau)
            psi_0 = np.zeros(2 ** self.num_qubits, dtype=complex)
            psi_0[0] = 1.0
            psi_t = U @ psi_0
            exp_val = float(np.real(psi_t.conj() @ obs_matrix @ psi_t))
            labels.append(exp_val)

        return np.array(labels)

    # ------------------------------------------------------------------
    # b_l extraction via A(U, P) and Hadamard test  (Corollary 2)
    # ------------------------------------------------------------------

    def _extract_b_l(
        self,
        builder: CircuitBuilder,
        edges: list,
        r_steps: int,
        register_index: int,
    ) -> complex:
        """Extract a single Fourier coefficient b_l via Hadamard test on A(U,P).

        Implements Corollary 2 of Barthe et al. (2025) using the A(U,P)
        circuit (Corollary 1 / Figure 4) as the controlled unitary inside the
        Hadamard test (Figure 7).

        The A(U, P) circuit A(U)† P A(U) is a unitary, so the ancilla always
        returns to |0⟩.  The b_l coefficients are encoded as amplitudes at
        |freq=l⟩|data=0⟩|anc=0⟩ in the output statevector, i.e.:

            sv[(l % 2^n_s) * da_stride] = b_l   (complex amplitude)

        where da_stride = 2^(num_qubits+1) accounts for data + ancilla in the
        low bits.  Summing their squared magnitudes gives:

            ||b||_2^2 = P(data=0 AND anc=0)
                      = sum_l |sv[(l%2^n_s) * da_stride]|^2

        This quantity is extracted by marginalising over the joint (data, anc)
        system, NOT over the ancilla alone — P(anc=0) = 1 for a unitary circuit
        and therefore gives the wrong (trivial) norm.

        The Hadamard test on A(U,P) returns:
            ⟨Z⟩_HT = Re(b_l) / ||b||_2   (real part)
            ⟨Z⟩_HT = Im(b_l) / ||b||_2   (imaginary part)

        Multiplying by ||b||_2 recovers b_l exactly.

        Parameters
        ----------
        builder : CircuitBuilder
        edges : list of (int, int)
            Graph topology x.
        r_steps : int
        register_index : int
            Unsigned frequency register index (l % 2^n_s).

        Returns
        -------
        complex
            The Fourier coefficient b_l(x).
        """
        # --- Build A(U,P) statevector (used only for the zero-norm guard) ----
        aup_qc = builder.build_expectation_value_extraction_circuit(
            self.num_qubits, edges, self.tau, r_steps, self.pauli_observable
        )
        sv_aup = Statevector.from_instruction(aup_qc)

        # Guard: if the b_l vector is all-zero, skip the Hadamard tests.
        # A(U,P) = A(U)† P A(U) is unitary, so P(anc=0) = 1 always and cannot
        # be used as a norm.  The correct ||b||_2^2 is P(data=0 AND anc=0):
        #   ||b||_2^2 = sum_l |b_l|^2 = sum_{l,k=0} |sv[(l%2^n_s)*da_stride]|^2
        # which equals the (0,0) entry of the marginal over (data, ancilla).
        data_and_ancilla = list(range(self.num_qubits + 1))   # qubits 0..n
        probs_aup = sv_aup.probabilities(data_and_ancilla)
        norm_sq = float(probs_aup[0])   # P(data=0 AND ancilla=0) = ||b||^2

        if norm_sq < 1e-12:
            return 0.0 + 0.0j

        # --- Step 2: Re(b_l) via Hadamard test ---------------------------
        # The A(U,P) circuit is a unitary: its statevector amplitudes are b_l
        # directly (not b_l/||b||_2 as in the abstract paper formulation where
        # A(U,P) applies a post-selection).  The Hadamard test on A(U,P) computes:
        #   <Z>_HT = Re(<l,0,0| A(U,P) |0>) = Re(b_l)
        # so z_exp_real is already Re(b_l).  No norm multiplication is needed.
        ht_real = builder.build_expectation_value_hadamard_test(
            self.num_qubits, edges, self.tau, r_steps,
            self.pauli_observable, register_index, part="real",
        )
        ht_real.remove_final_measurements()
        sv_real = Statevector.from_instruction(ht_real)
        probs_real = sv_real.probabilities([0])
        z_exp_real = 2.0 * float(probs_real[0]) - 1.0

        # --- Step 3: Im(b_l) via Hadamard test ---------------------------
        ht_imag = builder.build_expectation_value_hadamard_test(
            self.num_qubits, edges, self.tau, r_steps,
            self.pauli_observable, register_index, part="imag",
        )
        ht_imag.remove_final_measurements()
        sv_imag = Statevector.from_instruction(ht_imag)
        probs_imag = sv_imag.probabilities([0])
        z_exp_imag = 2.0 * float(probs_imag[0]) - 1.0

        # --- Step 4: Return b_l ------------------------------------------
        # z_exp_real = Re(b_l), z_exp_imag = Im(b_l) — return directly.
        return z_exp_real + 1j * z_exp_imag

    def extract_fourier_features(
        self,
        datasets: list,
        r_steps: int = 1,
        d_params: int = 1,
    ) -> np.ndarray:
        """Build the feature matrix B from b_l(x) extracted via A(U,P).

        For each training sample x_i and each l in L = {-2r,...,+2r}, the
        Fourier coefficient b_l(x_i) is obtained independently via
        ``_extract_b_l``, which implements Corollary 2 of the paper using the
        full A(U,P) circuit from Corollary 1 / Figure 4.

        The feature vector for sample x is:

            phi(x) = [Re(b_{-2r}(x)), ..., Re(b_0(x)), ..., Re(b_{+2r}(x))]

        Because the observable is Hermitian and real-valued labels are used,
        the weight vector w(alpha*) is real, so only Re(b_l) matters for
        regression.  Additive noise in [-epsilon_b, +epsilon_b] per entry
        simulates finite-shot estimation error (Theorem 6).

        Parameters
        ----------
        datasets : list of list[(int,int)]
        r_steps : int
        d_params : int
            Currently only d_params=1 is supported.

        Returns
        -------
        np.ndarray of shape (len(datasets), 4*r_steps + 1)
            Entry [i, j] is the noisy real part of b_{l_j}(x_i).

        Raises
        ------
        NotImplementedError
            If d_params != 1.
        """
        if d_params != 1:
            raise NotImplementedError(
                "Hadamard-test b_l extraction is implemented for d_params=1 "
                "only.  For d > 1 use the kernel approach (Section VI.B)."
            )

        builder = CircuitBuilder()
        num_samples = len(datasets)

        # Signed frequency range L = {-2r, ..., +2r}, |L| = 4r+1
        max_freq = 2 * r_steps
        freq_indices = list(range(-max_freq, max_freq + 1))

        # Circular register: negative l stored at (2^n_s + l) % 2^n_s
        n_s = builder._freq_register_size(r_steps)
        freq_dim = 2 ** n_s

        B_matrix = np.zeros((num_samples, len(freq_indices)), dtype=float)

        for sample_idx, edges in enumerate(datasets):
            for col_idx, l in enumerate(freq_indices):
                register_idx = l % freq_dim
                b_l = self._extract_b_l(builder, edges, r_steps, register_idx)
                noise = np.random.uniform(-self.epsilon_b, self.epsilon_b)
                B_matrix[sample_idx, col_idx] = float(np.real(b_l)) + noise

        return B_matrix

    # ------------------------------------------------------------------
    # Training and evaluation
    # ------------------------------------------------------------------

    def train(self, B_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Fit the LASSO regressor and return the learned weight vector w-hat."""
        self.lasso.fit(B_train, y_train)
        return self.lasso.coef_

    def evaluate(self, B_test: np.ndarray, y_test: np.ndarray) -> float:
        """Return MSE of the learned model on held-out data."""
        y_pred = self.lasso.predict(B_test)
        return mean_squared_error(y_test, y_pred)

    def extract_gram_matrix_sim(
        self, 
        builder: CircuitBuilder, 
        datasets: list, 
        r_steps: int
    ) -> np.ndarray:
        """
        Builds the Gram matrix K where K_ij = k(x_i, x_j) = Re(b(x_i) . b(x_j))
        using exact statevector simulation of the Figure 8 overlap circuit.
        """
        num_samples = len(datasets)
        K = np.zeros((num_samples, num_samples))
        
        # Outer loop – progress bar over the first index i
        for i in tqdm(range(num_samples), desc="Computing Gram matrix", unit="row"):
            # Optional inner bar – shows progress over j (only the upper‑triangular part)
            for j in tqdm(range(i, num_samples), desc="inner", leave=False, unit="col"):
                qc_kernel = builder.build_quantum_overlap_kernel_circuit(
                    self.num_qubits, datasets[i], datasets[j], self.tau, r_steps, d_params=self.num_qubits
                    )
                
                # Remove measurements for exact statevector simulation
                qc_kernel.remove_final_measurements()
                sv = Statevector(qc_kernel)
                
                # In Qiskit little-endian, the kernel ancilla is the last qubit 
                # (since it was added first in the register list).
                # The state |rest=0, k_anc=0> is index 0.
                # The state |rest=0, k_anc=1> is index 2**(total_qubits - 1).
                
                total_qubits = qc_kernel.num_qubits
                ancilla_1_idx = 2 ** (total_qubits - 1)
                
                prob_0_0 = np.abs(sv.data[0])**2
                prob_1_0 = np.abs(sv.data[ancilla_1_idx])**2
                
                # Re(b(x).b(x')) = P(anc=0, rest=0) - P(anc=1, rest=0)
                overlap = prob_0_0 - prob_1_0
                
                K[i, j] = overlap
                K[j, i] = overlap
                
        return K

    def train_kernel(self, K_train: np.ndarray, y_train: np.ndarray, alpha_reg: float = 0.01):
        """Fit the Kernel Ridge Regressor using the precomputed quantum Gram matrix."""
        self.krr = KernelRidge(alpha=alpha_reg, kernel='precomputed')
        self.krr.fit(K_train, y_train)
        return self.krr
    
import numpy as np
import itertools
from sklearn.kernel_ridge import KernelRidge
from qiskit.quantum_info import Statevector
from src.quantum_routines import KernelCircuitBuilder

class KernelDynamicsLearner:
    def __init__(self, num_qubits: int, tau: float, pauli_observable: str):
        self.num_qubits = num_qubits
        self.tau = tau
        self.pauli_observable = pauli_observable
        self.builder = KernelCircuitBuilder()
        self.krr = None

    def _extract_b_vector_sim(self, edges: list, r_steps: int) -> np.ndarray:
        """Extracts the exact flattened feature vector b(x) utilizing fast vectorization."""
        from qiskit.quantum_info import Statevector
        
        aup_qc = self.builder.build_inhomogeneous_aup_circuit(
            self.num_qubits, edges, self.tau, r_steps, self.pauli_observable
        )
        
        # Execute fast statevector simulation
        sv = Statevector(aup_qc).data

        da_stride = 2 ** (self.num_qubits + 1)
        n_s = self.builder._freq_register_size(r_steps)
        freq_dim = 2 ** n_s
        
        # Generate the 1D frequency spectrum [-2r, ..., +2r]
        freq_1d = np.arange(-2 * r_steps, 2 * r_steps + 1)
        
        # Create an n-dimensional meshgrid for all qubits simultaneously
        mesh = np.meshgrid(*[freq_1d] * self.num_qubits, indexing='ij')
        
        # Vectorized calculation of all statevector indices
        sv_indices = np.zeros_like(mesh[0])
        for dim in range(self.num_qubits):
            reg_stride = da_stride * (freq_dim ** dim)
            sv_indices += (mesh[dim] % freq_dim) * reg_stride
            
        # Flatten the indices and extract directly from the complex array
        return sv[sv_indices.flatten()].real

    def extract_feature_matrix(self, datasets: list, r_steps: int, desc: str = "Extracting Features") -> np.ndarray:
        """Extracts B matrix of shape (N_samples, (4r+1)^n)."""
        total_modes = (4 * r_steps + 1) ** self.num_qubits
        B = np.zeros((len(datasets), total_modes))
        
        # Wrap the enumeration with tqdm for a progress bar
        for i, edges in enumerate(tqdm(datasets, desc=desc, leave=False)):
            B[i, :] = self._extract_b_vector_sim(edges, r_steps)
            
        return B

    def compute_gram_matrix(self, B_left: np.ndarray, B_right: np.ndarray = None) -> np.ndarray:
        """Computes k(x, x') = B_left @ B_right.T mathematically identical to Figure 8."""
        if B_right is None:
            return B_left @ B_left.T
        return B_left @ B_right.T

    def train(self, K_train: np.ndarray, y_train: np.ndarray, alpha_reg: float = 0.1):
        self.krr = KernelRidge(alpha=alpha_reg, kernel='precomputed')
        self.krr.fit(K_train, y_train)
        return self.krr
    
    def predict(self, K_test: np.ndarray):
        return self.krr.predict(K_test)