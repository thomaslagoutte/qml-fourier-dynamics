#!/usr/bin/env python
"""
fig2_verify.py
==============

Reproduces Figure 2 of Barthe et al. (2025) using our stack (CircuitBuilder).

Draws three circuits side‑by‑side:
  LEFT   — U_r(x, α)  Eq. (17):  plain Trotter circuit
  MIDDLE — A(U) d_params=1  (current implementation):  only qubit 0 encoded
  RIGHT  — A(U) d_params=n  (paper Fig. 2):  all n qubits encoded

Run from the repo root:
    python fig2_verify.py
Output: fig2_verify.png
"""

import sys
import os
sys.path.append(os.path.abspath('..'))

import matplotlib
matplotlib.use('Agg')               # non‑interactive backend for script use
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer

from src.quantum_routines import CircuitBuilder

# ── Parameters matching Figure 2 ──────────────────────────────────────────────
NUM_QUBITS = 3
X_EDGES    = [(0, 1), (1, 2), (0, 2)]   # complete graph (all edges, x_ij=1)
TAU        = 1.0
R_STEPS    = 1                           # Fig 2 shows one step, "repeated r times"
ALPHA      = 1.0

cb = CircuitBuilder()

# ── Circuit 1: U_r(x, α) — left panel of Fig. 2, Eq. (17) ────────────────
def build_Ur(num_qubits, x_edges, tau, r_steps, alpha):
    """
    U_r(x, α) from Eq. (17):
        [ prod_{(i,j)∈E} RZZ(2*tau/r)  *  prod_i RX(2*α*tau/r) ]^r
    This is the plain Trotter circuit BEFORE the A‑transform.
    """
    qc = QuantumCircuit(num_qubits, name='Ur(x,α)')
    for step in range(r_steps):
        for (i, j) in x_edges:
            qc.rzz(2.0 * tau / r_steps, i, j)     # ZZ gate: angle 2*x_ij*tau/r
        for q in range(num_qubits):
            qc.rx(2.0 * alpha * tau / r_steps, q)  # X gate: angle 2*α*tau/r
        if step < r_steps - 1:
            qc.barrier()
    return qc

qc_ur = build_Ur(NUM_QUBITS, X_EDGES, TAU, R_STEPS, ALPHA)

# ── Circuit 2: A(U) d_params=1 — our current implementation ──────────────────
# Only qubit 0 gets the D·G·D upload block; qubits 1 and 2 are ignored.
qc_au1, freqs1 = cb.build_trotter_extraction_circuit(
    num_qubits = NUM_QUBITS,
    x_edges    = X_EDGES,
    tau        = TAU,
    r_steps    = R_STEPS,
    d_params   = 1,           # current implementation: only qubit 0 encoded
)

# ── Circuit 3: A(U) d_params=n — paper Figure 2 right panel ──────────────────
# All n qubits get their own D·G·D block and their own freq register.
qc_au3, freqs3 = cb.build_trotter_extraction_circuit(
    num_qubits = NUM_QUBITS,
    x_edges    = X_EDGES,
    tau        = TAU,
    r_steps    = R_STEPS,
    d_params   = NUM_QUBITS,  # paper Fig. 2: all qubits encoded
)

# ── Plot ───────────────────────────────────────────────────────────────────────
STYLE = dict(
    backgroundcolor = '#1a1f2e',
    linecolor       = '#90caf9',
    textcolor       = '#e8eaf6',
    gatetextcolor   = '#0f1118',
    gatefacecolor   = '#42a5f5',
    barrierfacecolor = '#2d3748',
    fold            = -1,
)

# *** Fixed line: use plt.subplots to obtain both Figure and Axes ***
fig, ax = plt.subplots(figsize=(24, 14))
fig.patch.set_facecolor('#0f1118')   # dark figure background

# Panel 3 — A(U) d_params=n (paper)
circuit_drawer(qc_au3, output='mpl', ax=ax, style=STYLE)

ax.set_title(
    "Figure 2 verification  —  Barthe et al. (2025)\n"
    r"Circuits produced by  src/quantum_routines.py :: CircuitBuilder"
    "\n"
    r"$H(x,\alpha)=\sum_{i,j}x_{ij}Z_iZ_j + \alpha\sum_i X_i$  (Eqs. 16–17)",
    color='#e8eaf6', fontsize=13, y=1.01, fontweight='bold')

plt.savefig('fig2_verify.png', dpi=150, bbox_inches='tight', facecolor='#0f1118')
print("Saved: fig2_verify.png")