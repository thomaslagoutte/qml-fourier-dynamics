"""Controlled frequency-shift (V+/-) gate cache.

IMMUTABLE PHYSICS CONSTRAINT
----------------------------
The controlled V+ and V- frequency-shift gates are expensive to compile
(they expand into a QFT + phase + inverse-QFT on the frequency register).
The legacy code relied on building them *once per register size* and
reusing them across all samples.  Rebuilding inline makes extractor
throughput collapse on the Schwinger runs.

Any :class:`CircuitBuilder` subclass must route its V+/- access through
this object — never construct the gates inline.
"""
from __future__ import annotations
from typing import Dict
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator

# Global cache to persist across ALL Experiment instantiations
_GLOBAL_V_CACHE: Dict[int, Dict[str, UnitaryGate]] = {}

def _build_native_cvp(n_s: int) -> QuantumCircuit:
    """Native controlled V+ (ancilla is qubit 0, target is 1..n_s)."""
    qc = QuantumCircuit(1 + n_s)
    qc.x(0)  # ctrl_state = "0"
    for i in range(n_s - 1, 0, -1):
        qc.mcx([0] + list(range(1, i + 1)), i + 1)
    qc.cx(0, 1)
    qc.x(0)
    return qc

def _build_native_cvm(n_s: int) -> QuantumCircuit:
    """Native controlled V- (ancilla is qubit 0, target is 1..n_s)."""
    qc = QuantumCircuit(1 + n_s)
    qc.cx(0, 1) # ctrl_state = "1" (no X needed)
    for i in range(1, n_s):
        qc.mcx([0] + list(range(1, i + 1)), i + 1)
    return qc

class VGateCache:
    """Per-register-size cache of controlled V+ / V- gates and their adjoints.

    Key schema:
      "cVp"     : V+ controlled on ancilla == 0
      "cVm"     : V- controlled on ancilla == 1
      "cVp_dag" : (V+)† controlled on ancilla == 0
      "cVm_dag" : (V-)† controlled on ancilla == 1

    The ctrl_state split (0 for +, 1 for -) is what creates the two evolution
    branches inside the Hadamard test sandwich.
    """
    def get(self, n_s: int) -> Dict[str, UnitaryGate]:
        if n_s not in _GLOBAL_V_CACHE:
            _GLOBAL_V_CACHE[n_s] = self._build(n_s)
        return _GLOBAL_V_CACHE[n_s]

    @staticmethod
    def _build(n_s: int) -> Dict[str, UnitaryGate]:
        if n_s < 1:
            raise ValueError(f"VGateCache requires n_s >= 1, got {n_s}")
        
        cvp_qc = _build_native_cvp(n_s)
        cvm_qc = _build_native_cvm(n_s)
        
        # Convert directly to dense unitary matrices! 
        # This makes Qiskit Statevector simulation infinitely faster.
        vp_mat = Operator(cvp_qc).data
        vm_mat = Operator(cvm_qc).data
        
        return {
            "cVp":     UnitaryGate(vp_mat, label="cV+"),
            "cVm":     UnitaryGate(vm_mat, label="cV-"),
            "cVp_dag": UnitaryGate(vp_mat.conj().T, label="cV+†"),
            "cVm_dag": UnitaryGate(vm_mat.conj().T, label="cV-†"),
        }
    