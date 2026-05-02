#!/usr/bin/env python
# coding: utf-8

"""
Hardware-mode validation of the Schwinger ℤ₂ lattice gauge theory.
Kernel version: True quantum overlap computation via Figure 8 circuit.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Quantum-Learning-Dynamics imports
# ----------------------------------------------------------------------
from quantum_learning_dynamics import Experiment
from quantum_learning_dynamics.hamiltonians.schwinger import SchwingerZ2Model
from quantum_learning_dynamics.observables.library import LocalMagnetization

# ----------------------------------------------------------------------
# Qiskit Aer imports (shot-based hardware simulation)
# ----------------------------------------------------------------------
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

# ----------------------------------------------------------------------
# Plot style
# ----------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
})

# ----------------------------------------------------------------------
# 1. Model, observable, and random seed
# ----------------------------------------------------------------------
SEED = 7
rng = np.random.default_rng(SEED)

model = SchwingerZ2Model(
    num_matter=4,
    mass=0.5,
    electric_field=1.0,
    g_range=(0.6, 1.4)
)
obs = LocalMagnetization(num_qubits=model.num_qubits, site=0)

# ----------------------------------------------------------------------
# 2. Training / test topology split
# ----------------------------------------------------------------------
X_train = (
    [[True,  True,  True ]] * 1 +
    [[False, True,  True ]] * 1 +
    [[True,  False, True ]] * 1 +
    [[True,  True,  False]] * 1
)
rng.shuffle(X_train)

test_state = [True, False, False]

# ----------------------------------------------------------------------
# 3. Exact reference trajectories
# ----------------------------------------------------------------------
T_DENSE = np.linspace(0.0, 3.0, 100)
T_VALS  = np.linspace(0.01, 3.0, 13)

alpha_star = model.sample_alpha(rng)

O_mat = obs.to_sparse_pauli_op().to_matrix()
n_qubits = model.num_qubits

exact_dense = np.empty(len(T_DENSE))
for i, t in enumerate(T_DENSE):
    if t == 0.0:
        psi = np.zeros(2**n_qubits, dtype=complex)
        psi[0] = 1.0
    else:
        U = model.exact_unitary(test_state, alpha_star, float(t))
        psi = U[:, 0]
    exact_dense[i] = float(np.real(np.conj(psi) @ O_mat @ psi))

exact_vals = np.empty(len(T_VALS))
for i, t in enumerate(T_VALS):
    U = model.exact_unitary(test_state, alpha_star, float(t))
    psi = U[:, 0]
    exact_vals[i] = float(np.real(np.conj(psi) @ O_mat @ psi))

# ----------------------------------------------------------------------
# 4. Hardware backend (Optimized GPU Mode)
# ----------------------------------------------------------------------
optimized_gpu_sim = AerSimulator(
    method="statevector",
    device="GPU",
    precision="single",  # CRITICAL: Use single precision to save VRAM on mega-batches
    batched_shots_gpu=True # Tells Aer to aggressively batch shots on the GPU
)

sampler = AerSamplerV2.from_backend(optimized_gpu_sim)

# ----------------------------------------------------------------------
# 5. Experiment loop (Kernel)
# ----------------------------------------------------------------------
R_STEPS = 5
KERNEL_ALPHA = 1e-3  # Ridge regression regularization
SHOTS = 5000

trotter_pts = np.empty(len(T_VALS))
pac_pts     = np.empty(len(T_VALS))

# Define the union ONCE outside the loop for the mega-batch
X_all = list(X_train) + [test_state]
N_train = len(X_train)

print("\nRunning hardware-mode validation (Kernel)…\n")
for idx, tau in enumerate(T_VALS):
    t0 = time.time()
    
    # Instantiate the experiment for the Kernel method
    exp = Experiment(
        model=model, observable=obs, method="kernel",
        execution_mode="hardware", sampler=sampler, shots=SHOTS,
        tau=float(tau), r_steps=R_STEPS, trotter_order=1,
        kernel_alpha=KERNEL_ALPHA, seed=SEED
    )
    
    # MEGA-BATCH: Compute the full symmetric Gram matrix (Train + Test) in one pass
    K_all = exp.engine.compute_gram(X_all, None, float(tau), R_STEPS, obs)
    
    # Extract the Train-vs-Train block and the Test-vs-Train block
    K_train = K_all[:N_train, :N_train]
    K_test  = K_all[N_train:, :N_train]
    
    # Train and Predict using the precomputed Gram matrices
    y_train = exp.compute_trotter_labels(X_train, alpha_star, float(tau), R_STEPS)
    exp.learner.fit(K_train, y_train)
    pac_pts[idx] = float(exp.learner.predict(K_test)[0])
    
    # Exact Trotter reference
    trotter_pts[idx] = float(
        exp.compute_trotter_labels([test_state], alpha_star, float(tau), R_STEPS)[0]
    )
    
    # Logging
    print(f"t={tau:5.2f} | Exact: {exact_vals[idx]:+.4f} | "
          f"Trotter: {trotter_pts[idx]:+.4f} | Kernel: {pac_pts[idx]:+.4f} | "
          f"Time: {time.time() - t0:.1f}s")

# ----------------------------------------------------------------------
# 6. Metrics
# ----------------------------------------------------------------------
mse_trotter = np.mean((pac_pts - trotter_pts) ** 2)
mse_exact   = np.mean((pac_pts - exact_vals) ** 2)

print("\n--- Summary ---")
print(f"Trotter-generalization MSE: {mse_trotter:.6f}")
print(f"Exact-physics MSE:          {mse_exact:.6f}")

# ----------------------------------------------------------------------
# 7. Plot
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    T_DENSE, exact_dense,
    label="Exact (statevector)",
    color="#a8cce3",
    linewidth=4,
    alpha=0.8
)

ax.plot(
    T_VALS, trotter_pts,
    label=f"Trotter (r={R_STEPS})",
    color="#1f78b4",
    linewidth=2,
    linestyle="--"
)

ax.scatter(
    T_VALS, pac_pts,
    label="Kernel (hardware)",
    color="black",
    marker="o",
    s=40,
    zorder=5
)

ax.set_xlabel("Evolution time $t$")
ax.set_ylabel(r"$\langle O(t) \rangle$")
ax.set_title(
    "Schwinger $\\mathbb{Z}_2$ LGT – Kernel Feature Learning\n"
    "Hardware-mode validation",
    fontsize=13,
    pad=15
)

ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="best", frameon=True, borderpad=1)

fig.tight_layout()
plt.savefig("hardware_validation_kernel.png", dpi=300)
plt.show()