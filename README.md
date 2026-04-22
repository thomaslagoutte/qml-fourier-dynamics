# Quantum Learning Dynamics

[![arXiv](https://img.shields.io/badge/arXiv-2501.XXXXX-b31b1b.svg)](https://arxiv.org/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit 1.0+](https://img.shields.io/badge/qiskit-1.0+-6929c4.svg)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Core Thesis:** A scalable, hardware-aware framework for PAC-learning quantum dynamics via Fourier coefficient extraction and True Quantum Overlap Kernels.

This repository contains the official implementation of the framework described in **"Quantum Advantage in Learning Quantum Dynamics via Fourier Coefficient Extraction"** (Barthe et al., 2025). It provides a high-performance, object-oriented API for evaluating Hamiltonian dynamics, extracting exponential feature spaces ($d=1$), and natively executing $\mathcal{O}(M^2)$ Quantum Overlap Kernels on noisy hardware ($d>1$).

---

## 📐 Mathematical Framework

The library is built around the extraction of Fourier components from parameterized quantum circuits. For a given Hamiltonian $H(x, \alpha)$ and observable $O$, the expected value can be decomposed as:

$$
\langle O(t) \rangle = \sum_{l} b_l(x) e^{-i l \alpha t}
$$

Instead of relying on classical CPU simulation bottlenecks, this architecture leverages exact statevector caching and hardware-native V2 Sampler batching to evaluate the Gram matrix

$$
K(x_i, x_j) = \mathrm{Re}\langle \psi(x_i) | \psi(x_j) \rangle
$$

directly on QPUs.

---

## 🏗️ Architecture

```
src/quantum_learning_dynamics/
├── experiment.py
├── features/
│   └── engines.py
├── circuits/
│   ├── base.py
│   ├── kernel_overlap.py
│   └── shared_register.py
├── hamiltonians/
└── learners/
```

---

## 🚀 Quick Start

```python
import numpy as np
from quantum_learning_dynamics import Experiment
from quantum_learning_dynamics.hamiltonians import InhomogeneousTFIM
from quantum_learning_dynamics.observables import StaggeredMagnetization

model = InhomogeneousTFIM(num_qubits=4)
observable = StaggeredMagnetization(num_qubits=4)

exp = Experiment(
    model=model,
    observable=observable,
    method="kernel",
    execution_mode="emulator",
    shots=4000,
    tau=1.0,
    r_steps=2,
    kernel_alpha=0.05,
    seed=42
)

result = exp.run(num_train=30, num_test=10)

print(f"Trotter Generalization MSE: {result.mse_trotter:.6f}")
print(f"Exact Physics MSE:          {result.mse_exact:.6f}")
```

---

## ⚙️ Execution Modes

- emulator (exact or noisy)
- hardware (Qiskit SamplerV2)

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/quantum-learning-dynamics.git
cd quantum-learning-dynamics
pip install -e .
```

---

## 📖 Citation

```bibtex
@article{barthe2025quantum,
  title={Quantum Advantage in Learning Quantum Dynamics via Fourier Coefficient Extraction},
  author={Barthe, [Firstname] and [Co-authors]},
  journal={arXiv preprint arXiv:XXXX.YYYYY},
  year={2025}
}
```
