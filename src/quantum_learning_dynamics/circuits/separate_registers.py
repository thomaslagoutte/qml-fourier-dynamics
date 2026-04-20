"""Separate-registers A(U, P) circuit builder — d > 1 regime (Method B).

One frequency register per unknown parameter.  Total statevector size is
2^(num_qubits + 1 + d * n_s); the d-dimensional Fourier tensor is extracted
from the statevector via :class:`MeshgridTensorExtractor`.

Concrete logic is ported EXCLUSIVELY from
``src/quantum_routines_kernel.py`` — specifically the ``KernelCircuitMixin``
primitives:

* ``_append_au_inhomogeneous_forward``
* ``_append_au_inhomogeneous_adjoint``
* ``build_au_inhomogeneous``
* ``build_aup_inhomogeneous``

The old ``KernelCircuitBuilder`` in ``src/quantum_routines.py`` (around
line 1448) is DEAD CODE and must NOT be reproduced in this package.
"""

from __future__ import annotations

import math
import numpy as np
from typing import List, Sequence, Tuple

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import PauliGate

from .base import CircuitBuilder


class SeparateRegistersBuilder(CircuitBuilder):
    """d>1 encoding: one frequency register per qubit."""
    # ---- initialization --------------------------------------------------
    def __init__(self, model, trotter_order: int = 2):
        super().__init__()
        self.model = model
        if trotter_order not in (1, 2):
            raise ValueError("trotter_order must be 1 or 2")
        self.trotter_order = trotter_order

    def freq_register_size(self, num_qubits: int, r_steps: int) -> int:
        return max(3, math.ceil(math.log2(4 * r_steps + 2)))

    # ---- D·G·D helpers ---------------------------------------------------

    def _dgd_forward(self, qc, data, anc, freqs, cache, x) -> None:
        """Dynamically routes H · cVp · cVm · H ONLY to active graph parameters."""
        ones_alpha = np.ones(self.model.d)
        zero_alpha = np.zeros(self.model.d)
        H_up = self.model.hamiltonian(x, ones_alpha) - self.model.hamiltonian(x, zero_alpha)
        
        # Identify which Paulis are actually present in the graph x
        active_paulis = {p for p, c in H_up.to_list() if abs(c) > 1e-12}

        for k, pauli_str in enumerate(self.model.upload_paulis):
            if pauli_str not in active_paulis:
                continue  # SUPER IMPORTANT: Skip severed gauge links!

            active = []
            for i, p in enumerate(reversed(pauli_str)):
                if p == 'X':
                    qc.h(data[i])
                    active.append(data[i])
                elif p == 'Y':
                    qc.sdg(data[i])
                    qc.h(data[i])
                    active.append(data[i])
                elif p == 'Z':
                    active.append(data[i])
            
            for q in active: qc.cx(q, anc[0])
            qc.append(cache["cVp"], [anc[0], *freqs[k]])
            qc.append(cache["cVm"], [anc[0], *freqs[k]])
            for q in reversed(active): qc.cx(q, anc[0])
                
            for i, p in enumerate(reversed(pauli_str)):
                if p == 'X':
                    qc.h(data[i])
                elif p == 'Y':
                    qc.h(data[i])
                    qc.s(data[i])

    def _dgd_adjoint(self, qc, data, anc, freqs, cache, x) -> None:
        ones_alpha = np.ones(self.model.d)
        zero_alpha = np.zeros(self.model.d)
        H_up = self.model.hamiltonian(x, ones_alpha) - self.model.hamiltonian(x, zero_alpha)
        active_paulis = {p for p, c in H_up.to_list() if abs(c) > 1e-12}

        for k, pauli_str in reversed(list(enumerate(self.model.upload_paulis))):
            if pauli_str not in active_paulis:
                continue

            active = []
            for i, p in enumerate(reversed(pauli_str)):
                if p == 'X':
                    qc.h(data[i])
                    active.append(data[i])
                elif p == 'Y':
                    qc.sdg(data[i])
                    qc.h(data[i])
                    active.append(data[i])
                elif p == 'Z':
                    active.append(data[i])
            
            for q in active: qc.cx(q, anc[0])
            qc.append(cache["cVm_dag"], [anc[0], *freqs[k]])
            qc.append(cache["cVp_dag"], [anc[0], *freqs[k]])
            for q in reversed(active): qc.cx(q, anc[0])
                
            for i, p in enumerate(reversed(pauli_str)):
                if p == 'X':
                    qc.h(data[i])
                elif p == 'Y':
                    qc.h(data[i])
                    qc.s(data[i])

    # ---- forward / adjoint evolutions -----------------------------------

    def _append_fixed_hamiltonian(self, qc, data, x, dt, sign=1.0):
        """Native compiler: guarantees zero sparse matrix fallbacks for insane speed."""
        zero_alpha = np.zeros(self.model.d)
        H_fixed = self.model.hamiltonian(x, zero_alpha)
        
        for pauli_str, coeff in H_fixed.to_list():
            if abs(coeff) < 1e-12: continue
            theta = 2.0 * sign * dt * np.real(coeff)
            
            p_list = list(reversed(pauli_str))
            active_q = [(i, p) for i, p in enumerate(p_list) if p != 'I']
            
            if len(active_q) == 1:
                q0, p0 = active_q[0]
                if p0 == 'Z': qc.rz(theta, data[q0])
                elif p0 == 'X': qc.rx(theta, data[q0])
                elif p0 == 'Y': qc.ry(theta, data[q0])
            elif len(active_q) == 2:
                q0, p0 = active_q[0]
                q1, p1 = active_q[1]
                if p0 == 'Z' and p1 == 'Z': qc.rzz(theta, data[q0], data[q1])
                elif p0 == 'X' and p1 == 'X': qc.rxx(theta, data[q0], data[q1])
                elif p0 == 'Y' and p1 == 'Y': qc.ryy(theta, data[q0], data[q1])

    def _append_au_forward(self, qc, data, anc, freqs, x, tau, r_steps) -> None:
        dt = tau / r_steps
        cache = self.gate_cache.get(len(freqs[0]))
        for _ in range(r_steps):
            if self.trotter_order == 1:
                self._append_fixed_hamiltonian(qc, data, x, dt, 1.0)
                self._dgd_forward(qc, data, anc, freqs, cache, x)
            elif self.trotter_order == 2:
                self._append_fixed_hamiltonian(qc, data, x, dt/2.0, 1.0)
                self._dgd_forward(qc, data, anc, freqs, cache, x)
                self._append_fixed_hamiltonian(qc, data, x, dt/2.0, 1.0)

    def _append_au_adjoint(self, qc, data, anc, freqs, x, tau, r_steps) -> None:
        dt = tau / r_steps
        cache = self.gate_cache.get(len(freqs[0]))
        for _ in range(r_steps):
            if self.trotter_order == 1:
                self._dgd_adjoint(qc, data, anc, freqs, cache, x)
                self._append_fixed_hamiltonian(qc, data, x, dt, -1.0)
            elif self.trotter_order == 2:
                self._append_fixed_hamiltonian(qc, data, x, dt/2.0, -1.0)
                self._dgd_adjoint(qc, data, anc, freqs, cache, x)
                self._append_fixed_hamiltonian(qc, data, x, dt/2.0, -1.0)

    # ---- public circuit factories ---------------------------------------

    def build_au(self, num_qubits, x, tau, r_steps):
        n_s = self.freq_register_size(num_qubits, r_steps)
        data  = QuantumRegister(num_qubits, "data")
        anc   = QuantumRegister(1, "anc")
        freqs = [QuantumRegister(n_s, f"freq_{q}") for q in range(self.model.d)]
        qc    = QuantumCircuit(data, anc, *freqs)

        self._append_au_forward(qc, data, anc, freqs, x, tau, r_steps)
        self._append_au_adjoint(qc, data, anc, freqs, x, tau, r_steps)
        return qc, freqs

    def build_aup(self, num_qubits, x, tau, r_steps, pauli):
        n_s = self.freq_register_size(num_qubits, r_steps)
        data  = QuantumRegister(num_qubits, "data")
        anc   = QuantumRegister(1, "anc")
        freqs = [QuantumRegister(n_s, f"freq_{q}") for q in range(self.model.d)]
        qc    = QuantumCircuit(data, anc, *freqs)

        self._append_au_forward(qc, data, anc, freqs, x, tau, r_steps)
        qc.append(PauliGate(pauli), list(data))
        self._append_au_adjoint(qc, data, anc, freqs, x, tau, r_steps)
        return qc, freqs