"""Shared-register A(U, P) circuit builder — d = 1 regime (Method A).

Fourier features b_l(x) are encoded onto a SINGLE frequency register of
size n_s = ceil(log2(4r + 1)).  This is the original construction of
Figure 4 / Corollary 1 in Barthe et al. (2025).

Concrete logic is ported from the legacy ``CircuitBuilder`` in
``src/quantum_routines.py`` (the d = 1 parts only — the old
``KernelCircuitBuilder`` in that file is dead code and NOT used here).
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import PauliGate

from .base import CircuitBuilder


class SharedRegisterBuilder(CircuitBuilder):
    """d=1 encoding: one shared frequency register of size n_s.

    First-order Trotter step (per step of r_steps):
        1. RZZ(+2 dt) on every edge in `edges`                  -- physical ZZ evolution
        2. D · G · D on each qubit, writing into `freq`         -- A-transformed X rotation
           D = H on data[q]
           G = cVp (control=0) then cVm (control=1)             -- data[q] is the control
    No physical RX on data qubits. The alpha dependence is entirely encoded as
    ±1 shifts in the shared freq register; alpha does not appear in the circuit.
    """
    # ---- initialization --------------------------------------------------
    def __init__(self, model, trotter_order: int = 2):
        super().__init__()
        self.model = model
        if trotter_order not in (1, 2):
            raise ValueError("trotter_order must be 1 or 2")
        self.trotter_order = trotter_order

    # ---- register sizing -------------------------------------------------

    def freq_register_size(self, num_qubits: int, r_steps: int) -> int:
        # REVERTED: Must scale with num_qubits for d=1, not d!
        return max(3, math.ceil(math.log2(4 * num_qubits * r_steps + 2)))

    # ---- D·G·D helpers ---------------------------------------------------

    def _dgd_forward(self, qc, data, freq, cache) -> None:
        """H · cVp · cVm · H on each qubit, targeting the shared freq register."""
        for q in range(len(data)):
            qc.h(data[q])
            qc.append(cache["cVp"], [data[q], *freq])
            qc.append(cache["cVm"], [data[q], *freq])
            qc.h(data[q])

    def _dgd_adjoint(self, qc, data, freq, cache) -> None:
        """Exact inverse of _dgd_forward: reverse qubit order, reverse V± order, dagger V±."""
        for q in reversed(range(len(data))):
            qc.h(data[q])
            qc.append(cache["cVm_dag"], [data[q], *freq])
            qc.append(cache["cVp_dag"], [data[q], *freq])
            qc.h(data[q])

    # ---- forward / adjoint evolutions -----------------------------------
    def _append_au_forward(self, qc, data, freq, edges, tau, r_steps) -> None:
        dt = tau / r_steps
        cache = self.gate_cache.get(len(freq))
        for _ in range(r_steps):
            if self.trotter_order == 1:
                # REVERTED: Pure native Qiskit gates. Zero sparse matrices!
                for (i, j) in edges:
                    qc.rzz(+2.0 * dt, data[i], data[j])
                self._dgd_forward(qc, data, freq, cache)
            elif self.trotter_order == 2:
                for (i, j) in edges:
                    qc.rzz(+1.0 * dt, data[i], data[j])
                self._dgd_forward(qc, data, freq, cache)
                for (i, j) in edges:
                    qc.rzz(+1.0 * dt, data[i], data[j])

    def _append_au_adjoint(self, qc, data, freq, edges, tau, r_steps) -> None:
        dt = tau / r_steps
        cache = self.gate_cache.get(len(freq))
        rev_edges = list(reversed(list(edges)))
        for _ in range(r_steps):
            if self.trotter_order == 1:
                self._dgd_adjoint(qc, data, freq, cache)
                for (i, j) in rev_edges:
                    qc.rzz(-2.0 * dt, data[i], data[j])
            elif self.trotter_order == 2:
                for (i, j) in rev_edges:
                    qc.rzz(-1.0 * dt, data[i], data[j])
                self._dgd_adjoint(qc, data, freq, cache)
                for (i, j) in rev_edges:
                    qc.rzz(-1.0 * dt, data[i], data[j])

    # ---- public circuit factories ---------------------------------------

    def build_au(
        self,
        num_qubits: int,
        x: Sequence[Tuple[int, int]],
        tau: float,
        r_steps: int,
    ) -> Tuple[QuantumCircuit, List[QuantumRegister]]:
        edges = x
        n_s = self.freq_register_size(num_qubits, r_steps)
        data = QuantumRegister(num_qubits, "data")
        anc  = QuantumRegister(1, "anc")        # starts and stays in |0> inside the builder
        freq = QuantumRegister(n_s, "freq")
        qc   = QuantumCircuit(data, anc, freq)

        self._append_au_forward(qc, data, freq, edges, tau, r_steps)
        self._append_au_adjoint(qc, data, freq, edges, tau, r_steps)
        return qc, [freq]

    def build_aup(
        self,
        num_qubits: int,
        x: Sequence[Tuple[int, int]],
        tau: float,
        r_steps: int,
        pauli: str,
    ) -> Tuple[QuantumCircuit, List[QuantumRegister]]:
        edges = x
        n_s = self.freq_register_size(num_qubits, r_steps)
        data = QuantumRegister(num_qubits, "data")
        anc  = QuantumRegister(1, "anc")
        freq = QuantumRegister(n_s, "freq")
        qc   = QuantumCircuit(data, anc, freq)

        self._append_au_forward(qc, data, freq, edges, tau, r_steps)
        qc.append(PauliGate(pauli), list(data))
        self._append_au_adjoint(qc, data, freq, edges, tau, r_steps)
        return qc, [freq]