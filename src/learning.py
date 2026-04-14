"""PAC learning pipeline for quantum Hamiltonian dynamics.

Implements the algorithm described in:
  Barthe et al., "Quantum Advantage in Learning Quantum Dynamics via
  Fourier Coefficient Extraction" (2025).
"""

import math

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from qiskit.quantum_info import Statevector

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
        epsilon_b: float = 0.01
    ):
        self.num_qubits = num_qubits
        self.tau = tau
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

        The A(U, P) circuit prepares (Corollary 1, Eq. 9):

            A(U,P)|0> = (1/||b||_2) sum_l b_l |l>_freq |0>_anc  +  |trash>|1>_anc

        Step 1 recovers ||b||_2 = sqrt(P(anc=0)) from the bare A(U,P) statevector.
        Steps 2-3 run real and imaginary Hadamard tests to get
        Re(b_l)/||b||_2 and Im(b_l)/||b||_2 respectively.
        Step 4 multiplies back by ||b||_2 to recover b_l.

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
        # --- Step 1: ||b||_2^2 = P(ancilla=0) from A(U,P) ---------------
        aup_qc = builder.build_expectation_value_extraction_circuit(
            self.num_qubits, edges, self.tau, r_steps, self.pauli_observable
        )
        sv_aup = Statevector.from_instruction(aup_qc)

        # Qubit layout of A(U,P): [data_0..data_{n-1}, ancilla, freq_0..freq_{n_s-1}]
        ancilla_qubit = self.num_qubits
        probs_aup = sv_aup.probabilities([ancilla_qubit])
        norm_sq = float(probs_aup[0])  # P(ancilla = |0>)

        if norm_sq < 1e-12:
            return 0.0 + 0.0j

        norm = math.sqrt(norm_sq)

        # --- Step 2: Re(b_l) via Hadamard test ---------------------------
        # ht_ancilla is qubit 0; <Z> = 2*P(|0>) - 1 = Re(b_l)/||b||_2
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

        # --- Step 4: Recover b_l -----------------------------------------
        # b_l = ||b||_2 * (Re(b_l)/||b||_2 + i * Im(b_l)/||b||_2)
        return norm * (z_exp_real + 1j * z_exp_imag)

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
    # Simulation-optimised shortcut (classical statevector only)
    # ------------------------------------------------------------------

    def extract_fourier_features_sim(
        self,
        datasets: list,
        r_steps: int = 1,
    ) -> np.ndarray:
        """Build B by reading b_l directly from the A(U,P) statevector.

        On a classical statevector simulator, the amplitude at state
        |freq=l>|data=0>|anc=0> in the A(U,P) output equals b_l up to
        the post-selection normalisation ||b||_2.  This avoids constructing
        three separate Hadamard-test circuits per frequency, giving an
        order-of-magnitude speedup suitable for notebook-level experiments.

        .. note::
            This is a **simulation shortcut**.  On real quantum hardware,
            use ``extract_fourier_features`` (Hadamard-test circuits).

        Parameters
        ----------
        datasets : list of list[(int,int)]
        r_steps : int

        Returns
        -------
        np.ndarray of shape (len(datasets), 4*r_steps + 1)
            Same semantics as ``extract_fourier_features``.
        """
        builder = CircuitBuilder()
        num_samples = len(datasets)

        n_s = builder._freq_register_size(r_steps)
        freq_dim = 2 ** n_s
        # State index for |freq=reg>|anc=0>|data=0> (little-endian)
        data_ancilla_dim = 2 ** (self.num_qubits + 1)

        max_freq = 2 * r_steps
        freq_indices = list(range(-max_freq, max_freq + 1))

        B_matrix = np.zeros((num_samples, len(freq_indices)), dtype=float)

        for sample_idx, edges in enumerate(datasets):
            aup_qc = builder.build_expectation_value_extraction_circuit(
                self.num_qubits, edges, self.tau, r_steps, self.pauli_observable
            )
            sv = Statevector.from_instruction(aup_qc).data

            # b_l is stored at state index reg * data_ancilla_dim (data=0, anc=0)
            for col_idx, l in enumerate(freq_indices):
                reg = l % freq_dim
                b_l = sv[reg * data_ancilla_dim]
                noise = np.random.uniform(-self.epsilon_b, self.epsilon_b)
                B_matrix[sample_idx, col_idx] = float(b_l.real) + noise

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