"""Cache of the controlled frequency-shift gates :math:`cV^{+}` and :math:`cV^{-}`.

Physics
-------
The Fourier-extraction subroutine :math:`\\mathcal{A}(U)` of Barthe et al.
(2025) replaces each continuous Pauli rotation :math:`e^{i\\alpha Z}` with a
pair of controlled shift operators :math:`V^{\\pm}` acting on a frequency
register:

* :math:`V^{+}` increments the frequency register by one (applied on the
  even-parity branch, ``ctrl_state = 0``);
* :math:`V^{-}` decrements the frequency register by one (applied on the
  odd-parity branch, ``ctrl_state = 1``).

Performance
-----------
Synthesising the controlled :math:`V^{\\pm}` gates expands into a
QFT / phase / inverse-QFT block whose cost grows with :math:`n_s`. Because
the register size is fixed for a given ``(num_qubits, r_steps)``
configuration, the same four :class:`~qiskit.circuit.library.UnitaryGate`
objects can be reused across every sample of an experiment. This module
exposes a process-wide cache keyed on :math:`n_s` so the expensive
compilation runs exactly once per register width.

Any subclass of :class:`CircuitBuilder` MUST go through :class:`VGateCache`
rather than instantiating :math:`V^{\\pm}` inline.
"""

from __future__ import annotations

from typing import Dict

from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator

# Process-wide cache; keyed on n_s (width of the frequency register).
_GLOBAL_V_CACHE: Dict[int, Dict[str, UnitaryGate]] = {}


def _build_native_cvp(n_s: int) -> QuantumCircuit:
    """Controlled-:math:`V^{+}` increment gate on an :math:`n_s`-qubit register.

    Parameters
    ----------
    n_s : int
        Width of the frequency register.

    Returns
    -------
    QuantumCircuit
        Circuit on ``1 + n_s`` qubits. Qubit ``0`` is the control (active
        on :math:`|0\\rangle`); qubits ``1..n_s`` carry the frequency
        register in LSB-first order.
    """
    qc = QuantumCircuit(1 + n_s)
    qc.x(0)  # implements ctrl_state = "0"
    for i in range(n_s - 1, 0, -1):
        qc.mcx([0] + list(range(1, i + 1)), i + 1)
    qc.cx(0, 1)
    qc.x(0)
    return qc


def _build_native_cvm(n_s: int) -> QuantumCircuit:
    """Controlled-:math:`V^{-}` decrement gate on an :math:`n_s`-qubit register.

    Parameters
    ----------
    n_s : int
        Width of the frequency register.

    Returns
    -------
    QuantumCircuit
        Circuit on ``1 + n_s`` qubits. Qubit ``0`` is the control (active
        on :math:`|1\\rangle`); qubits ``1..n_s`` carry the frequency
        register in LSB-first order.
    """
    qc = QuantumCircuit(1 + n_s)
    qc.cx(0, 1)  # ctrl_state = "1": no X wrapping required
    for i in range(1, n_s):
        qc.mcx([0] + list(range(1, i + 1)), i + 1)
    return qc


class VGateCache:
    """Process-wide cache of controlled :math:`V^{\\pm}` gates.

    The cache is keyed on the frequency-register width :math:`n_s` and
    stores, for each value, the forward and adjoint controlled gates in
    both polarities:

    +-------------+----------------------------------------------------+
    | Key         | Gate                                               |
    +=============+====================================================+
    | ``"cVp"``     | :math:`cV^{+}` controlled on ancilla = :math:`|0\\rangle`    |
    +-------------+----------------------------------------------------+
    | ``"cVm"``     | :math:`cV^{-}` controlled on ancilla = :math:`|1\\rangle`    |
    +-------------+----------------------------------------------------+
    | ``"cVp_dag"`` | :math:`(cV^{+})^{\\dagger}`                                  |
    +-------------+----------------------------------------------------+
    | ``"cVm_dag"`` | :math:`(cV^{-})^{\\dagger}`                                  |
    +-------------+----------------------------------------------------+

    The opposite control polarities of :math:`cV^{+}` and :math:`cV^{-}`
    are what yield the two interfering branches inside the Hadamard-test
    sandwich of the :math:`\\mathcal{A}(U)` subroutine.
    """

    def get(self, n_s: int) -> Dict[str, UnitaryGate]:
        """Return the cached gate set for register width :math:`n_s`.

        Parameters
        ----------
        n_s : int
            Width of the frequency register. Must be :math:`\\ge 1`.

        Returns
        -------
        dict[str, UnitaryGate]
            Dictionary with keys ``"cVp"``, ``"cVm"``, ``"cVp_dag"``,
            ``"cVm_dag"``.
        """
        if n_s not in _GLOBAL_V_CACHE:
            _GLOBAL_V_CACHE[n_s] = self._build(n_s)
        return _GLOBAL_V_CACHE[n_s]

    @staticmethod
    def _build(n_s: int) -> Dict[str, UnitaryGate]:
        """Compile the four gates for a given register width.

        The gates are materialised as dense unitary matrices wrapped in
        :class:`~qiskit.circuit.library.UnitaryGate`. This bypasses
        repeated Quantum-Shannon decomposition inside the statevector
        simulator and gives the extraction loop a constant-cost append.
        """
        if n_s < 1:
            raise ValueError(f"VGateCache requires n_s >= 1, got {n_s}")

        cvp_qc = _build_native_cvp(n_s)
        cvm_qc = _build_native_cvm(n_s)

        vp_mat = Operator(cvp_qc).data
        vm_mat = Operator(cvm_qc).data

        return {
            "cVp":     UnitaryGate(vp_mat, label="cV+"),
            "cVm":     UnitaryGate(vm_mat, label="cV-"),
            "cVp_dag": UnitaryGate(vp_mat.conj().T, label="cV+†"),
            "cVm_dag": UnitaryGate(vm_mat.conj().T, label="cV-†"),
        }
