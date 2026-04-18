"""Kernel-based PAC learning pipeline for the d = poly(n) regime.

Implements the Quantum Overlap Kernel method from Section VI.B of:
  Barthe et al., "Quantum Advantage in Learning Quantum Dynamics via
  Fourier Coefficient Extraction" (2025).

This module targets the inhomogeneous TFIM toy model where d = n
(one unknown transverse field strength per qubit).  The explicit
Fourier feature extraction used by FourierDynamicsLearner is
intractable when d = poly(n) because the total number of Fourier
modes m = (4r+1)^n grows exponentially.  The kernel method sidesteps
this by evaluating k(x,x') = Re<b(x)|b(x')> directly via the
Hadamard-test kernel circuit (Figure 8), without ever constructing
the exponentially large feature vectors.

Pipeline summary (Section VI.B)
--------------------------------
1. For each pair (x_i, x_j) in training+test data, compute
   k(x_i, x_j) = Re<b(x_i)|b(x_j)> via quantum kernel circuit.

2. Assemble the T×T Gram matrix K_train and the (T_test × T_train)
   cross-kernel matrix K_cross.

3. Fit Kernel Ridge Regression (KRR):
      a_opt = (K_train + lambda*I)^{-1} y_train

4. Predict on test data:
      y_pred = K_cross @ a_opt

5. Report train/test MSE.

Classes
-------
KernelDynamicsLearner
    Full pipeline: data generation, label computation, Gram matrix,
    KRR fitting, and evaluation.
"""

import time
from typing import List, Tuple

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error

# Import the mixin — we instantiate it alongside CircuitBuilder
# by creating a combined class at runtime (see __init__)
from src.quantum_routines_kernel import KernelCircuitMixin
from src.models import InhomogeneousTFIM


class _KernelBuilder(KernelCircuitMixin):
    """Minimal concrete class that provides the KernelCircuitMixin API.

    The mixin assumes self._freq_register_size and self._get_cached_v_gates
    are available.  We pull these in from the existing CircuitBuilder.

    Rather than modifying the original CircuitBuilder (to keep the existing
    codebase untouched), we compose: _KernelBuilder inherits the mixin and
    delegates cache methods to a CircuitBuilder instance.

    Falls back to the stub CircuitBuilder if the full quantum_routines is not
    yet present in the project (e.g. during isolated testing of the kernel
    extension).  In a full project setup, quantum_routines.py must be present
    in src/ alongside this file.
    """

    def __init__(self):
        try:
            from src.quantum_routines import CircuitBuilder
        except ModuleNotFoundError:
            # Fallback: use stub that provides only the cache infrastructure.
            # This occurs when quantum_routines.py has not yet been copied into
            # the extension's src/ directory.  In a complete project, the full
            # CircuitBuilder from quantum_routines.py is always available.
            from src.quantum_routines_stub import CircuitBuilder

        self._cb = CircuitBuilder()
        # Share the V-gate cache with the parent CircuitBuilder instance
        self._v_gate_cache = self._cb._v_gate_cache

    def _freq_register_size(self, r_steps: int) -> int:
        return self._cb._freq_register_size(r_steps)

    def _get_cached_v_gates(self, n_s: int) -> dict:
        return self._cb._get_cached_v_gates(n_s)


class KernelDynamicsLearner:
    """Quantum Overlap Kernel PAC learning pipeline for d = poly(n) dynamics.

    Targets the inhomogeneous TFIM concept class:

        c_{alpha}(x) = <0| U(x,alpha)^dag O U(x,alpha) |0>

    with H(x,alpha) = sum_{(i,j) in x} ZiZj + sum_i alpha_i Xi,
    d = n unknown parameters, and O a fixed local Pauli observable.

    The kernel k(x,x') = Re<b(x)|b(x')> is computed by simulating the
    Hadamard-test circuit from Figure 8 of the paper (adapted for d=n
    separate frequency registers).  Kernel Ridge Regression then fits
    a predictor in the (implicitly exponentially large) feature space.

    Parameters
    ----------
    num_qubits : int
        Number of data qubits n.  Also sets d = n.
    tau : float
        Total evolution time tau.
    r_steps : int
        Number of first-order Trotter steps r.
    pauli_observable : str
        Pauli string for the observable O.  Qiskit little-endian convention:
        rightmost character acts on qubit 0.
        Default: 'I'*(n-1)+'Z'  (= Z_0, local magnetisation on qubit 0).
    lambda_reg : float
        KRR regularisation strength lambda.  Fixed during validation;
        tune via cross-validation once the pipeline is confirmed correct.
    random_seed : int, optional
        Seed for graph generation reproducibility.
    """

    def __init__(
        self,
        num_qubits: int,
        tau: float,
        r_steps: int = 2,
        pauli_observable: str = None,
        lambda_reg: float = 0.1,
        random_seed: int = 42,
    ):
        self.num_qubits = num_qubits
        self.tau = tau
        self.r_steps = r_steps
        self.pauli_observable = pauli_observable or ("I" * (num_qubits - 1) + "Z")
        self.lambda_reg = lambda_reg
        self.random_seed = random_seed

        self.model = InhomogeneousTFIM(num_qubits)
        self._builder = _KernelBuilder()

        # KRR with precomputed kernel
        self.krr = KernelRidge(
            alpha=lambda_reg,
            kernel="precomputed",
        )

        # Stored after fitting
        self._alpha_opt = None      # KRR dual coefficients
        self._X_train = None        # training graphs (for cross-kernel)
        self._K_train = None        # training Gram matrix

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    def generate_random_graphs(self, num_samples: int) -> List[List[Tuple[int, int]]]:
        """Sample random graphs (edge sets) over num_qubits vertices.

        Each possible edge (i,j) with i<j is included independently with
        probability 0.5.  The graph encodes the ZZ interaction topology x.

        Parameters
        ----------
        num_samples : int

        Returns
        -------
        list of list of (int, int)  — length num_samples
        """
        rng = np.random.default_rng(self.random_seed)
        datasets = []
        for _ in range(num_samples):
            edges = [
                (i, j)
                for i in range(self.num_qubits)
                for j in range(i + 1, self.num_qubits)
                if rng.random() > 0.5
            ]
            datasets.append(edges)
        return datasets

    # ------------------------------------------------------------------
    # Label generation
    # ------------------------------------------------------------------

    def compute_labels(
        self,
        graphs: List[List[Tuple[int, int]]],
        alpha_vec: np.ndarray,
    ) -> np.ndarray:
        """Compute A(U)-consistent Trotter labels c_{alpha*}(x_i).

        Uses the per-qubit generalisation of compute_au_labels, ensuring
        the label convention matches the A(U) circuit's upload convention.

        Parameters
        ----------
        graphs : list of edge lists
        alpha_vec : np.ndarray, shape (n,)
            The true unknown parameter vector alpha*.

        Returns
        -------
        np.ndarray, shape (len(graphs),)
        """
        return self._builder.compute_au_labels_inhomogeneous(
            graphs=graphs,
            num_qubits=self.num_qubits,
            alpha_vec=alpha_vec,
            tau=self.tau,
            r_steps=self.r_steps,
            pauli_observable=self.pauli_observable,
        )

    # ------------------------------------------------------------------
    # Gram matrix computation
    # ------------------------------------------------------------------

    def compute_gram_matrix(
        self,
        graphs: List[List[Tuple[int, int]]],
        verbose: bool = True,
    ) -> np.ndarray:
        """Compute the T×T quantum kernel Gram matrix.

        Delegates to KernelCircuitMixin.compute_gram_matrix, which
        evaluates each k(x_i, x_j) = Re<b(x_i)|b(x_j)> via exact
        statevector simulation of the Hadamard-test kernel circuit.

        Parameters
        ----------
        graphs : list of edge lists  — T samples
        verbose : bool

        Returns
        -------
        np.ndarray, shape (T, T)  — symmetric Gram matrix
        """
        return self._builder.compute_gram_matrix(
            num_qubits=self.num_qubits,
            datasets=graphs,
            tau=self.tau,
            r_steps=self.r_steps,
            pauli_observable=self.pauli_observable,
            verbose=verbose,
        )

    def compute_cross_kernel(
        self,
        test_graphs: List[List[Tuple[int, int]]],
        train_graphs: List[List[Tuple[int, int]]],
        verbose: bool = True,
    ) -> np.ndarray:
        """Compute the T_test × T_train cross-kernel matrix K_cross[i,j] = k(x_i^test, x_j^train).

        Used for prediction:  y_pred = K_cross @ alpha_opt.

        Parameters
        ----------
        test_graphs : list of edge lists  — T_test samples
        train_graphs : list of edge lists  — T_train samples
        verbose : bool

        Returns
        -------
        np.ndarray, shape (T_test, T_train)
        """
        T_test  = len(test_graphs)
        T_train = len(train_graphs)
        K_cross = np.zeros((T_test, T_train), dtype=float)

        total = T_test * T_train
        computed = 0

        for i, x_test in enumerate(test_graphs):
            for j, x_train in enumerate(train_graphs):
                k_val = self._builder.compute_kernel_entry(
                    num_qubits=self.num_qubits,
                    x_edges_1=x_test,
                    x_edges_2=x_train,
                    tau=self.tau,
                    r_steps=self.r_steps,
                    pauli_observable=self.pauli_observable,
                )
                K_cross[i, j] = k_val
                computed += 1
                if verbose and computed % 10 == 0:
                    pct = 100.0 * computed / total
                    print(f"  Cross-kernel: {computed}/{total} ({pct:.1f}%) "
                          f"— K_cross[{i},{j}] = {k_val:.4f}")

        return K_cross

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        K_train: np.ndarray,
        y_train: np.ndarray,
        train_graphs: List[List[Tuple[int, int]]],
    ) -> np.ndarray:
        """Fit KRR on the precomputed Gram matrix.

        Solves:  a_opt = (K_train + lambda*I)^{-1} y_train
        Stores the dual coefficients for prediction.

        Parameters
        ----------
        K_train : np.ndarray, shape (T, T)
        y_train : np.ndarray, shape (T,)
        train_graphs : list of edge lists  — needed for cross-kernel at test time

        Returns
        -------
        np.ndarray  — dual coefficients a_opt, shape (T,)
        """
        self._X_train = train_graphs
        self._K_train = K_train
        self.krr.fit(K_train, y_train)
        self._alpha_opt = self.krr.dual_coef_
        return self._alpha_opt

    # ------------------------------------------------------------------
    # Prediction and evaluation
    # ------------------------------------------------------------------

    def predict(
        self,
        K_cross: np.ndarray,
    ) -> np.ndarray:
        """Predict labels on test data using precomputed cross-kernel.

        y_pred = K_cross @ a_opt

        Parameters
        ----------
        K_cross : np.ndarray, shape (T_test, T_train)

        Returns
        -------
        np.ndarray, shape (T_test,)
        """
        return self.krr.predict(K_cross)

    def evaluate(
        self,
        K_train: np.ndarray,
        y_train: np.ndarray,
        K_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """Compute train and test MSE.

        Parameters
        ----------
        K_train : np.ndarray, shape (T_train, T_train)
        y_train : np.ndarray, shape (T_train,)
        K_test  : np.ndarray, shape (T_test, T_train)
        y_test  : np.ndarray, shape (T_test,)

        Returns
        -------
        dict with keys 'train_mse', 'test_mse', 'train_pred', 'test_pred'
        """
        y_train_pred = self.krr.predict(K_train)
        y_test_pred  = self.krr.predict(K_test)
        return {
            "train_mse":  mean_squared_error(y_train, y_train_pred),
            "test_mse":   mean_squared_error(y_test,  y_test_pred),
            "train_pred": y_train_pred,
            "test_pred":  y_test_pred,
        }

    # ------------------------------------------------------------------
    # Convenience: full pipeline in one call
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        alpha_vec: np.ndarray,
        n_train: int = 20,
        n_test: int = 10,
        verbose: bool = True,
    ) -> dict:
        """Execute the complete kernel PAC learning pipeline.

        Steps
        -----
        1. Generate random training and test graphs.
        2. Compute A(U)-consistent Trotter labels.
        3. Compute training Gram matrix K_train (T_train × T_train).
        4. Compute cross-kernel matrix K_cross (T_test × T_train).
        5. Fit KRR on K_train, y_train.
        6. Predict and evaluate on test set.

        Parameters
        ----------
        alpha_vec : np.ndarray, shape (n,)
            Fixed true unknown parameter vector alpha*.
        n_train : int
            Number of training samples.
        n_test : int
            Number of test samples.
        verbose : bool

        Returns
        -------
        dict with keys:
            'train_graphs', 'test_graphs',
            'y_train', 'y_test',
            'K_train', 'K_cross',
            'alpha_opt',
            'train_mse', 'test_mse',
            'train_pred', 'test_pred',
            'timing'  (sub-dict with wall-clock seconds per stage)
        """
        timing = {}

        # Step 1: Generate graphs
        t0 = time.time()
        all_graphs = self.generate_random_graphs(n_train + n_test)
        train_graphs = all_graphs[:n_train]
        test_graphs  = all_graphs[n_train:]
        timing["graph_generation"] = time.time() - t0

        if verbose:
            print(f"Generated {n_train} training + {n_test} test graphs.")
            print(f"  num_qubits={self.num_qubits}, tau={self.tau}, "
                  f"r_steps={self.r_steps}, d_params={self.num_qubits}")
            print(f"  Observable: {self.pauli_observable}")
            print(f"  True alpha*: {alpha_vec}")
            print(f"  lambda_reg: {self.lambda_reg}\n")

        # Step 2: Compute labels
        t0 = time.time()
        if verbose:
            print("Computing training labels...")
        y_train = self.compute_labels(train_graphs, alpha_vec)
        if verbose:
            print("Computing test labels...")
        y_test  = self.compute_labels(test_graphs, alpha_vec)
        timing["label_computation"] = time.time() - t0

        if verbose:
            print(f"  y_train: mean={y_train.mean():.4f}, std={y_train.std():.4f}, "
                  f"range=[{y_train.min():.4f}, {y_train.max():.4f}]")
            print(f"  y_test:  mean={y_test.mean():.4f},  std={y_test.std():.4f}, "
                  f"range=[{y_test.min():.4f}, {y_test.max():.4f}]\n")

        # Step 3: Training Gram matrix
        t0 = time.time()
        if verbose:
            print(f"Computing training Gram matrix ({n_train}×{n_train}, "
                  f"{n_train*(n_train+1)//2} kernel evaluations)...")
        K_train = self.compute_gram_matrix(train_graphs, verbose=verbose)
        timing["gram_train"] = time.time() - t0

        if verbose:
            print(f"  Training Gram matrix done in {timing['gram_train']:.1f}s\n")

        # Step 4: Cross-kernel matrix
        t0 = time.time()
        if verbose:
            print(f"Computing cross-kernel matrix ({n_test}×{n_train}, "
                  f"{n_test*n_train} kernel evaluations)...")
        K_cross = self.compute_cross_kernel(test_graphs, train_graphs, verbose=verbose)
        timing["gram_cross"] = time.time() - t0

        if verbose:
            print(f"  Cross-kernel done in {timing['gram_cross']:.1f}s\n")

        # Step 5: Fit KRR
        t0 = time.time()
        if verbose:
            print(f"Fitting KRR (lambda={self.lambda_reg})...")
        alpha_opt = self.fit(K_train, y_train, train_graphs)
        timing["krr_fit"] = time.time() - t0

        # Step 6: Evaluate
        results = self.evaluate(K_train, y_train, K_cross, y_test)
        timing["total"] = sum(timing.values())

        if verbose:
            print(f"\n{'='*50}")
            print(f"  Train MSE : {results['train_mse']:.6f}")
            print(f"  Test  MSE : {results['test_mse']:.6f}")
            print(f"  Total wall-clock: {timing['total']:.1f}s")
            print(f"{'='*50}")

        return {
            "train_graphs": train_graphs,
            "test_graphs":  test_graphs,
            "y_train":      y_train,
            "y_test":       y_test,
            "K_train":      K_train,
            "K_cross":      K_cross,
            "alpha_opt":    alpha_opt,
            "timing":       timing,
            **results,
        }


    # ------------------------------------------------------------------
    # Cached Gram / cross-kernel  (statevector reuse within a tau value)
    # ------------------------------------------------------------------

    def compute_gram_matrix_cached(
        self,
        graphs,
        tau: float,
        cache,
        verbose: bool = False,
    ):
        """Gram matrix using the statevector cache (call per tau value)."""
        return self._builder.compute_gram_matrix_cached(
            num_qubits=self.num_qubits,
            datasets=graphs,
            tau=tau,
            r_steps=self.r_steps,
            pauli_observable=self.pauli_observable,
            cache=cache,
            verbose=verbose,
        )

    def compute_cross_kernel_cached(
        self,
        test_graphs,
        train_graphs,
        tau: float,
        cache,
        verbose: bool = False,
    ):
        """Cross-kernel matrix using the statevector cache."""
        return self._builder.compute_cross_kernel_cached(
            num_qubits=self.num_qubits,
            test_datasets=test_graphs,
            train_datasets=train_graphs,
            tau=tau,
            r_steps=self.r_steps,
            pauli_observable=self.pauli_observable,
            cache=cache,
            verbose=verbose,
        )

    def compute_kernel_row_cached(
        self,
        query_graph,
        train_graphs,
        tau: float,
        cache,
    ):
        """Compute one row of the cross-kernel for a single query graph.

        Used to get the kernel prediction for the held-out graph at each tau
        without computing the full cross-kernel matrix again.

        Parameters
        ----------
        query_graph : list of (int, int)  — single held-out graph
        train_graphs : list of edge lists  — T_train training graphs

        Returns
        -------
        np.ndarray, shape (T_train,)  — k(query, x_j) for j in train
        """
        return np.array([
            self._builder.compute_kernel_entry_cached(
                num_qubits=self.num_qubits,
                x_edges_1=query_graph,
                x_edges_2=x_train,
                tau=tau,
                r_steps=self.r_steps,
                pauli_observable=self.pauli_observable,
                cache=cache,
            )
            for x_train in train_graphs
        ])