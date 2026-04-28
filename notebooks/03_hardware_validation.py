import time
import numpy as np
import matplotlib.pyplot as plt
from qiskit.primitives import StatevectorSampler

from quantum_learning_dynamics import Experiment
from quantum_learning_dynamics.hamiltonians.tfim import TFIM, InhomogeneousTFIM
from quantum_learning_dynamics.observables.library import LocalMagnetization, StaggeredMagnetization
from quantum_learning_dynamics.features.engines import FeatureEngine
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

plt.rcParams.update({
    'figure.dpi': 120, 
    'axes.grid': True, 
    'grid.alpha': 0.3, 
    'font.family': 'sans-serif'
})


# # Setup a simple d=1 model to verify the Hadamard Test mechanics
# model_1d = TFIM(num_qubits=3)
# observable_1d = LocalMagnetization(num_qubits=3, site=0) 
# pauli = "IIZ"
# x_graph = [(0, 1), (1, 2)] 
# tau = 1.0
# r_steps = 1
# max_freq = 2 * r_steps * model_1d.num_qubits
# rng = np.random.default_rng(42)

# # 1. Exact Statevector Baseline (shots=None)
# engine_exact = FeatureEngine(
#     model=model_1d, trotter_order=1, execution_mode="emulator", 
#     shots=None, sampler=None, rng=rng
# )
# b_sv = engine_exact._extract_emulator(x_graph, tau, r_steps, pauli, max_freq)

# # 2. Shot-based Hardware Emulation (shots=8000)
# shots = (500)
# engine_hw = FeatureEngine(
#     model=model_1d, trotter_order=1, execution_mode="hardware", 
#     shots=shots, sampler=StatevectorSampler(), rng=rng  
# )
# b_hw = engine_hw._extract_emulator(x_graph, tau, r_steps, pauli, max_freq)

# # 3. Compare Results
# freqs = np.arange(-max_freq, max_freq + 1)

# print(f"Comparing Extraction Methods (Shots = {shots})")
# print(f"Expected RMS Error: {1/np.sqrt(shots):.4f}\n")
# print(f"{'Freq':>5} | {'Statevector':>11} | {'Hardware':>11} | {'Diff':>6}")
# print("-" * 43)

# for f, sv, hw in zip(freqs, b_sv, b_hw):
#     if abs(sv) > 1e-3 or abs(hw) > 1.5/np.sqrt(shots):
#         print(f"{f:5d} | {sv:11.4f} | {hw:11.4f} | {abs(sv-hw):6.4f}")

# rms_error = np.sqrt(np.mean((b_sv - b_hw)**2))
# print("-" * 43)
# print(f"Measured RMS Error: {rms_error:.4f}")

# # Run the full ML pipeline purely on shot-based features
# exp_hw = Experiment(
#     model=model_1d,
#     observable=observable_1d,
#     method="lasso",               
#     execution_mode="hardware",    
#     shots=500,
#     sampler=AerSamplerV2.from_backend(AerSimulator(method="statevector")),              
#     tau=1.5,
#     r_steps=2,
#     lasso_alpha=0.001,  # Threshold the noise
#     seed=42
# )

# result_hw = exp_hw.run(num_train=30, num_test=10)

# print(f"--- Hardware PAC-Learning Results ({exp_hw.shots} shots/circuit) ---")
# print(f"Trotter Generalization MSE: {result_hw.mse_trotter:.6f}")
# print(f"Exact Physics MSE:          {result_hw.mse_exact:.6f}\n")

# print("Predictions on Unseen Topologies:")
# for i in range(5):
#     print(f"Test {i}: True = {result_hw.y_true_trotter[i]:+.4f} | Pred = {result_hw.y_pred[i]:+.4f}")

from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

# The highly-optimized V100 GPU Engine
optimized_gpu_sim = AerSimulator(
    method="statevector",
    device="GPU",
    precision="single",          # 1. Double the math speed, halve the memory
    max_parallel_experiments=0,  # 2. Automatically saturate the 32GB VRAM
    batched_shots_gpu=True       # 3. Offload measurement RNG to the GPU
)

optimized_cpu_sim = AerSimulator(
    method="statevector",
    device="CPU",                 # <--- Back to CPU
    max_parallel_threads=110,     # <--- Weaponize 110 of your 112 cores
    max_parallel_experiments=0    # <--- Run as many circuits at once as possible
)

# Setup a d>1 model
model_nd = InhomogeneousTFIM(num_qubits=3)
obs_nd = StaggeredMagnetization(num_qubits=3)

exp_kernel = Experiment(
    model=model_nd,
    observable=obs_nd,
    method="kernel",              # Overlap Circuit (Bypasses Tensor)
    execution_mode="hardware",    # Fast O(M) Statevector caching
    sampler=AerSamplerV2.from_backend(optimized_cpu_sim),
    shots=5000,                   # Inject Binomial Hardware Noise
    tau=1.0,
    r_steps=2,
    kernel_alpha=0.05,            # Higher Ridge penalty for noisy Gram matrix
    seed=42
)

# This would take ~1 hour on Qiskit Sampler, but completes in < 1 second here!
t0 = time.time()
result_kernel = exp_kernel.run(num_train=20, num_test=10, show_progress=True)
exec_time = time.time() - t0

print(f"--- d>1 True Quantum Kernel Results ({exp_kernel.shots} shots) ---")
print(f"Execution Time:             {exec_time:.2f} seconds")
print(f"Trotter Generalization MSE: {result_kernel.mse_trotter:.6f}")
print(f"Exact Physics MSE:          {result_kernel.mse_exact:.6f}")


