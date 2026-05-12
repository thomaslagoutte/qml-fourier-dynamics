#!/usr/bin/env python
# coding: utf-8

"""
PAC‑Learning Hardware Experiment: Kernel Method Scaling for Inhomogeneous TFIM.
Simplified plotting: only a semi‑log runtime vs. d (no fitting).
"""

import time
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

# Qiskit imports
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

# Custom quantum learning dynamics imports
from quantum_learning_dynamics import Experiment
from quantum_learning_dynamics.observables.library import LocalMagnetization
from quantum_learning_dynamics.hamiltonians.tfim import InhomogeneousTFIM

# ----------------------------------------------------------------------
# Plot style (matches the example you provided)
# ----------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "lines.linewidth": 2,
    "lines.markersize": 8,
})

# ----------------------------------------------------------------------
# 1. Experiment Parameters
# ----------------------------------------------------------------------
N_SPINS_VALS = [2, 3, 4, 5, 6]          # number of spins
TAU = 1.5
R_STEPS = 2
SHOTS = 5000
SEED = 42
rng = np.random.default_rng(SEED)

# Data containers
d_vals, qubits_vals, times_kernel = [], [], []
pac_results, trotter_results, exact_results = [], [], []

# ----------------------------------------------------------------------
# 2. Backend Setup (CPU AerSimulator)
# ----------------------------------------------------------------------
num_cores = multiprocessing.cpu_count()
print(f"Starting TFIM Kernel Scaling Experiment (Tau={TAU}, R={R_STEPS})...")
print(f"Initializing CPU Simulator across {num_cores} CPU threads...\n")

optimized_cpu_sim = AerSimulator(
    method="statevector",
    device="CPU",
    precision="single",
    max_parallel_threads=0,      # use all cores for matrix math
    max_parallel_experiments=1   # avoid memory overlap
)
sampler = AerSamplerV2.from_backend(optimized_cpu_sim)

# ----------------------------------------------------------------------
# 3. Scaling Sweep Loop
# ----------------------------------------------------------------------
for n in N_SPINS_VALS:
    print(f"{'-'*40}")

    # Model & observable
    model = InhomogeneousTFIM(num_qubits=n, edge_prob=0.5, alpha_range=(0.6, 1.4))
    obs = LocalMagnetization(num_qubits=model.num_qubits, site=0)

    d = model.d
    d_vals.append(d)
    qubits_vals.append(model.num_qubits)
    print(f"System: n={n} spins | Qubits={model.num_qubits} | Dimension d={d}")

    # ---- PAC dataset construction (unchanged) ----
    test_state = [(i, i + 1) for i in range(n - 1)]
    test_state_tuple = tuple(sorted(test_state))

    max_possible_edges = n * (n - 1) // 2
    max_unique_graphs = 2 ** max_possible_edges
    target_n = 5 * d
    N_TRAIN = min(max_unique_graphs - 1, target_n)
    print(f"PAC Constraint: N_train selected = {N_TRAIN}")

    X_train_set = set()
    while len(X_train_set) < N_TRAIN:
        candidate = tuple(sorted(model.sample_x(rng)))
        if candidate != test_state_tuple:
            X_train_set.add(candidate)

    X_train = [list(x) for x in X_train_set]
    X_test = [list(test_state_tuple)]

    # Ground‑truth parameters
    alpha_star = model.sample_alpha(rng)

    # ---- Exact physics (unchanged) ----
    O_mat = obs.to_sparse_pauli_op().to_matrix()
    U_exact = model.exact_unitary(test_state, alpha_star, float(TAU))
    psi = U_exact[:, 0]
    exact_val = float(np.real(np.conj(psi) @ O_mat @ psi))
    exact_results.append(exact_val)

    # ---- Experiment engine ----
    exp = Experiment(
        model=model, observable=obs, method="kernel", execution_mode="hardware",
        sampler=sampler, shots=SHOTS, tau=float(TAU), r_steps=R_STEPS, seed=SEED
    )

    # ---- Trotter reference (unchanged) ----
    y_train = exp.compute_trotter_labels(X_train, alpha_star, float(TAU), R_STEPS)
    trotter_val = exp.compute_trotter_labels(X_test, alpha_star, float(TAU), R_STEPS)[0]
    trotter_results.append(float(trotter_val))

    # ---- Kernel PAC workload (timed) ----
    t0 = time.time()
    K_train = exp.engine.compute_gram(X_train, None, float(TAU), R_STEPS, obs)
    exp.learner.fit(K_train, y_train)

    K_test = exp.engine.compute_gram(X_test, X_train, float(TAU), R_STEPS, obs)
    pac_val = exp.learner.predict(K_test)[0]

    elapsed = time.time() - t0
    times_kernel.append(elapsed)
    pac_results.append(float(pac_val))

    print(f"Results   -> Exact: {exact_val:+.4f} | Trotter: {trotter_val:+.4f} | Kernel: {pac_val:+.4f}")
    print(f"Exec Time -> {elapsed:.2f} seconds")

# ----------------------------------------------------------------------
# 4. Simple Semi‑Log Plot (runtime vs. d)
# ----------------------------------------------------------------------
print("\nGenerating semi‑log runtime plot...")

d_arr = np.array(d_vals, dtype=float)
t_arr = np.array(times_kernel, dtype=float)

fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogy(d_arr, t_arr, 'ko', label='Measurements', markersize=8)

ax.set_xlabel(r'Dimension $d$', fontsize=12)
ax.set_ylabel('Execution Time (seconds)', fontsize=12)
ax.set_title('Kernel‑Method Runtime Scaling', fontsize=14, pad=15)
ax.grid(True, which='both', ls=':', alpha=0.5)
ax.legend(frameon=True)

plt.tight_layout()
plt.savefig("kernel_scaling.pdf", dpi=300, bbox_inches='tight')
plt.show()