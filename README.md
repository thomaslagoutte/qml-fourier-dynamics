# quantum_learning_dynamics

**PAC learning of quantum dynamics via Fourier coefficient extraction — a modular Qiskit implementation of Barthe et al., 2025.**

[![python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![qiskit](https://img.shields.io/badge/qiskit-%E2%89%A51.0-6929c4.svg)](https://qiskit.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![status](https://img.shields.io/badge/status-alpha-orange.svg)]()

---

## Overview

`quantum_learning_dynamics` is a research-grade Python package that implements the PAC-learning protocol introduced in

> J.-G. Barthe, B. Hansen, A. Anshu, and J. Preskill.
> *Quantum Advantage in Learning Quantum Dynamics via Fourier Coefficient Extraction* (2025).

The paper shows that a learner holding coherent access to a parameterised quantum process `U(x, α*)` can recover the conditional expectation
`⟨ψ₀| U†(x, α*) O U(x, α*) |ψ₀⟩` for any held-out input `x` — *without ever learning the hidden parameter* `α*`. The mechanism is a controlled quantum circuit (the **A(U)** / **A(U, P)** construction of Figure 4 / Corollary 1) whose statevector on the frequency register directly encodes the Fourier coefficients `b_ℓ(x)` of the analytic map `α ↦ y(x, α)`. A classical learner (Lasso or Kernel Ridge) then fits `y` from those Fourier features.

This package provides a clean, modular, fully typed reimplementation of that pipeline. It is designed for reproducibility rather than raw throughput — every object has a narrow interface, every conversion point is documented, and the feature extractor is bit-for-bit consistent with the Trotter labels used to train the learner (the *A(U)-consistent* convention introduced by the legacy `compute_au_labels_inhomogeneous` routine).

### What this package gives you

- A one-call `Experiment` API that runs the complete pipeline (sample → build circuit → extract Fourier features → fit → predict → score) and returns an `ExperimentResult` with both the exact target, the Trotterised target, and the learner's prediction — so generalisation error and Trotter error can be decomposed cleanly.
- Three concrete methods corresponding to the two regimes of the paper:
  - `"hadamard_lasso"` — shared-register `A(U, P)` circuit + Hadamard-test coefficient recovery + Lasso (required for `d = 1`).
  - `"meshgrid_lasso"` / `"meshgrid_kernel"` — separate-registers circuit + meshgrid Fourier-tensor extraction + Lasso or Kernel Ridge with precomputed Gram (required for `d > 1`).
- Three built-in Hamiltonian models spanning the regimes in the paper: `TFIM` (homogeneous, `d = 1`), `InhomogeneousTFIM` (`d = n`), and `SchwingerZ₂Model` (lattice gauge theory with multi-qubit `XZX` upload strings and `d = N − 1`).
- A native ripple-carry `V±` implementation of the frequency-shift gates, with controlled and inverse variants cached through `VGateCache.get(n_s)` — so every circuit shares the exact same counter gates.
- Two validation notebooks (see below) reproducing the central experimental stories: learning-curve scaling and the zero-knowledge paradox on the Z₂ Schwinger model.

---

## Repository structure

```
qml-fourier-dynamics/
├── pyproject.toml
├── requirements.txt
├── README.md
├── src/
│   └── quantum_learning_dynamics/
│       ├── __init__.py              # top-level: Experiment, ExperimentResult
│       ├── _types.py                # InputX, AlphaVector, FeatureMatrix, LabelVector
│       ├── experiment.py            # end-to-end pipeline + A(U)-consistent labels
│       ├── results.py               # ExperimentResult dataclass
│       ├── circuits/
│       │   ├── base.py              # CircuitBuilder ABC
│       │   ├── gate_cache.py        # native ripple-carry V± and controlled variants
│       │   ├── shared_register.py   # d = 1 : A(U, P), first-order Trotter
│       │   └── separate_registers.py# d > 1 : A(U), Strang / Suzuki-2 Trotter
│       ├── features/
│       │   ├── base.py              # FeatureExtractor ABC
│       │   ├── hadamard_b_l.py      # single-coefficient Hadamard test (d = 1)
│       │   └── meshgrid_tensor.py   # strict np.meshgrid Fourier tensor (d > 1)
│       ├── hamiltonians/
│       │   ├── base.py              # HamiltonianModel ABC + exact_unitary
│       │   ├── tfim.py              # TFIM, InhomogeneousTFIM
│       │   └── schwinger.py         # SchwingerZ2Model (2N − 1 qubits)
│       ├── observables/
│       │   ├── base.py              # Observable ABC, PauliTerm
│       │   └── library.py           # LocalZ, StaggeredMagnetization, ElectricFlux, ...
│       └── learners/
│           ├── base.py              # Learner ABC
│           ├── lasso.py             # sklearn Lasso wrapper (fit_intercept=False)
│           └── kernel_ridge.py      # sklearn KernelRidge with precomputed Gram
├── notebooks/
│   ├── 01_tfim_validation.ipynb     # TFIM d = 1 + Inhomogeneous TFIM d = 3, trajectory plots
│   └── 02_schwinger_validation.ipynb# Schwinger Z₂ zero-knowledge paradox
└── tests/
    └── qld/                         # pytest suite (gate cache, extractors, models, API)
```

---

## Installation

The package targets **Python 3.12+** and is built with `hatchling`. From a fresh virtual environment at the repository root:

```bash
python -m venv .venv
source .venv/bin/activate            # or .venv\Scripts\activate on Windows
pip install -U pip

# Runtime only
pip install -r requirements.txt

# Editable install of the package itself
pip install -e .

# Optional — notebooks and/or dev tooling
pip install -e ".[notebooks]"
pip install -e ".[dev]"
```

The `pyproject.toml` also exposes two extras, `[dev]` (pytest, ruff, black, mypy) and `[notebooks]` (jupyter, matplotlib, ipykernel), for tooling parity with CI.

### Verifying the install

```bash
pytest -q
```

The test suite covers the ripple-carry `V±` gate cache, the `d = 1` Hadamard extractor versus the legacy reference, the `d > 1` meshgrid tensor, the two Hamiltonian families, the observable algebra, and the `Experiment` routing rules. The suite runs in under a minute on a laptop; tests that build the full Trotter stack are marked `slow`.

---

## Quickstart

A minimal Inhomogeneous TFIM experiment at `d = 3`, predicting staggered magnetisation from coherent queries:

```python
import numpy as np
from quantum_learning_dynamics import Experiment
from quantum_learning_dynamics.hamiltonians import InhomogeneousTFIM
from quantum_learning_dynamics.observables import StaggeredMagnetization

model      = InhomogeneousTFIM(num_qubits=3, edge_prob=0.6)
observable = StaggeredMagnetization(num_qubits=3)

exp = Experiment(
    model       = model,
    observable  = observable,
    method      = "meshgrid_kernel",   # d > 1 ⇒ tensor pipeline
    tau         = 1.0,
    r_steps     = 2,
    kernel_alpha= 0.1,
    seed        = 42,
)

result = exp.run(num_train=200, num_test=50)

print(f"MSE vs exact   : {result.mse_exact:.3e}")
print(f"MSE vs Trotter : {result.mse_trotter:.3e}")
```

`result.y_true_exact`, `result.y_true_trotter`, and `result.y_pred` decompose the error into Trotter discretisation error and learner generalisation error — the two quantities the paper bounds separately.

### Choosing a method

The method string is the pipeline; it is never inferred from `model.d`. The routing rules are strict and will raise at `__init__` with a hint if mismatched:

| Method            | Circuit                  | Extractor        | Learner      | Required `d` |
| ----------------- | ------------------------ | ---------------- | ------------ | ------------ |
| `hadamard_lasso`  | `SharedRegisterBuilder`  | `HadamardBL`     | Lasso        | `= 1`        |
| `meshgrid_lasso`  | `SeparateRegistersBuilder` | `MeshgridTensor` | Lasso        | `> 1`        |
| `meshgrid_kernel` | `SeparateRegistersBuilder` | `MeshgridTensor` | Kernel Ridge | `> 1`        |

---

## The validation notebooks

Both notebooks are written around a single, physically meaningful diagnostic: the **time-evolution trajectory** of an observable at a held-out test input, with the PAC prediction compared against the exact Schrödinger evolution and against the same-order Trotterisation used to train the learner.

### `notebooks/01_tfim_validation.ipynb`

Two parts, both on the same plotting template:

- **Part 1 — TFIM, `d = 1`, `hadamard_lasso`.** `n = 3` qubits, `r = 3` Trotter steps, staggered magnetisation. Demonstrates that a single shared frequency register plus a Hadamard-test readout recovers the learning-curve-correct Fourier coefficients for the homogeneous transverse field.
- **Part 2 — Inhomogeneous TFIM, `d = 3`, `meshgrid_kernel`.** `n = 3` qubits, `r = 2`, local magnetisation. Demonstrates that the `np.meshgrid` tensor construction is the only extraction consistent with the full joint-frequency structure when the transverse field is site-dependent — no marginalisation, no per-axis factorisation.

For each test input, the notebook sweeps `τ ∈ [0, 3]` on a dense 100-point grid for the exact curve and a 15-point grid for the Trotter / PAC comparison, retraining a fresh learner per `τ` to honour the paper's per-`τ` feature construction.

### `notebooks/02_schwinger_validation.ipynb`

The **zero-knowledge paradox** experiment on the Z₂ Schwinger model (`N = 3` matter sites, `N − 1 = 2` links, `2N − 1 = 5` qubits):

- Training inputs are biased toward active-link configurations (`[T, T]`, `[T, F]`, `[F, T]`), while the held-out test input is `[F, F]` — a link configuration the learner has *never seen active*, for which the dynamics at the test site are α-independent (only the DC Fourier mode survives).
- The learner still produces a correct prediction, because the Fourier features extracted via `meshgrid_kernel` encode exactly the analytic structure the observable sees — the paradox is resolved by the A(U) circuit, not by the training distribution.
- A second log-scale panel tracks `|PAC − Trotter|` per sample point, confirming that the residual is a pure Trotter-discretisation artefact rather than a learning failure.

The notebook is intentionally short — the story is the two plots, not the wall of code.

---

## Engineering highlights

A few conventions worth calling out because they are load-bearing and easy to break on first contact:

- **A(U)-consistent Trotter labels.** Training labels are computed by *the same Trotter decomposition* that the A(U) circuit implements, via the dense decomposition `H(x, α) = H(x, 0) + [H(x, α) − H(x, 0)]`. This is fully model-agnostic — the `Experiment` never touches `upload_paulis` or any model-specific upload structure for label generation. Trotter order (1 for shared-register, 2 / Strang for separate-registers) is read off `self.builder.trotter_order` so the label convention always matches the extractor's circuit.
- **D·G·D replaces physical `RX`.** The α-rotation is implemented as `H → controlled-V₊ (ctrl=0) → controlled-V₋ (ctrl=1) → H` on each data qubit, mapping computational-basis states to `±1` frequency shifts on the dedicated register. There is **no** physical `RX` on data qubits — the rotation is encoded, not performed.
- **Ancilla stays in |0⟩ inside the builders.** No `qc.h(anc[0])` inside `build_au` / `build_aup`; the controlled `V±` gates rely on the ancilla being strictly `|0⟩` at the point of application. The Hadamard-test readout in `HadamardBLExtractor` is where superposition on the ancilla is introduced (Corollary 2 / Figure 7).
- **Multi-qubit upload Paulis are parity-folded.** For the Schwinger model each `XZX` upload string spans three qubits. The D-block computes the parity into a single register qubit via CNOTs, applies the controlled-`V±` pair, then undoes the parity fold — so the frequency-shift interpretation is preserved even for non-local upload terms.
- **RZZ sign convention: `+2·dt` (shared) / `+1·dt` halves (separate).** Both builders use a positive-forward convention for the `ZZ` layer; the factor-of-two difference reflects the full-step vs. half-step Trotter split.
- **Strict `np.meshgrid(..., indexing='ij')` for `d > 1`.** The Fourier tensor is indexed by `Σ_k (mesh[k] % freq_dim) · (da_stride · freq_dim**k)` and returned flattened to `(4r + 1)**d` — no marginalisation, no axis reordering. `MeshgridTensorExtractor` is the only object in the package with this responsibility.
- **Observable linearity is enforced at the extractor boundary.** Composite observables `O = Σ_h β_h P_h` are handled *per Pauli term* inside the extractor — the coefficients never fold into a single `SparsePauliOp` before feature extraction, because the A(U, P) circuit is defined for a single Pauli at a time.
- **Seeded reproducibility.** A single `seed` argument on `Experiment` is spawned into three independent streams via `np.random.SeedSequence(seed).spawn(3)` for sampling, extractor noise, and learner state — so every figure in the notebooks is byte-reproducible.
- **Native ripple-carry `V±` with cached controlled variants.** `VGateCache.get(n_s)` returns a single dict `{"cVp", "cVm", "cVp_dag", "cVm_dag"}` built from `.to_gate().control(1, ctrl_state=...)`, so the controlled counter gates are constructed once per frequency-register width and reused across every circuit build.

---

## Citing

If this implementation helps your work, please cite the original paper:

```bibtex
@article{barthe2025quantum,
  title   = {Quantum Advantage in Learning Quantum Dynamics via Fourier Coefficient Extraction},
  author  = {Barthe, J.-G. and Hansen, B. and Anshu, A. and Preskill, J.},
  journal = {(preprint)},
  year    = {2025},
}
```

---

## License

MIT — see `pyproject.toml` for the authoritative metadata.