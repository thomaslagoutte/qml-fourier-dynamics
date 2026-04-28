"""Controlled-gate primitives for the inline Hadamard-test refactor.

Background
----------
The Figure 7 (per-frequency) and Figure 8 (overlap-kernel) Hadamard tests
both require a *singly-controlled* version of the A(U, P) sequence:

    H(ht) · c-A(U, P) · ... · H(ht) · measure(ht)

The naive implementation wraps the entire A(U, P) sequence as a Qiskit
``Gate`` and calls ``.control(1, annotated=True)``.  Even with
``annotated=True`` the controlled gate is eventually synthesised by the
``HighLevelSynthesis`` pass through the default
``MCXSynthesisDefault`` / ``ControlModifier`` plugins, which fall back
to Quantum-Shannon decomposition for any dense unitary.  For a 16-17
qubit inner block this is catastrophic — both in transpilation time
(minutes) and in the depth of the synthesised circuit (10⁵+ CX gates).

This module replaces that black-box wrapper with **gate-by-gate inline
controlled assembly**.  We exploit two structural facts:

  (1) c-(U_K · ... · U_1) = c-U_K · ... · c-U_1.  Controlling each
      elementary gate is mathematically identical to controlling the
      entire product.

  (2) For a self-inverse "basis change" Φ (Hadamard, S†·H, ...) that
      bracets a target gate M:
              c-(Φ⁻¹ · M · Φ)  =  Φ⁻¹ · c-M · Φ
      with Φ applied **unconditionally**.  The Φ and Φ⁻¹ cancel on the
      ht = 0 branch; on the ht = 1 branch they realise exactly the
      conjugated operation.  This lets us leave every basis change in
      the D·G·D block unconditional and only control the inner
      Toffoli / cV± / RZ rotations — saving us the cost of a c-H
      decomposition on every D·G·D fold.

Every primitive below is mathematically exact (no Trotter / Shannon
approximations) and synthesises into the standard CX/CRZ/CCX/MCX basis.

Conventions
-----------
* All Pauli strings follow Qiskit's little-endian: the *rightmost*
  character of the string acts on qubit 0.  This matches the rest of
  the repository.
* Every helper is a free function that takes a ``QuantumCircuit`` and
  appends to it in place.  No helper allocates new registers or
  classical bits.
* ``ht`` is always the Hadamard-test control qubit (a single
  ``Qubit``, not a register).  ``ctrl`` arguments inside ``cV±`` are
  the per-block parity controls (data[q] for shared, anc[0] for
  separate).
"""

from __future__ import annotations

from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.circuit import QuantumRegister


# ---------------------------------------------------------------------
# Singly-controlled rotations and basis-conjugated rotations.
# ---------------------------------------------------------------------

def append_ctrl_rzz(
    qc: QuantumCircuit,
    ht: Qubit,
    q0: Qubit,
    q1: Qubit,
    theta: float,
) -> None:
    """Append c-RZZ(θ) on (q0, q1) controlled by ``ht``.

    Decomposition (depth 5, 2 CX + 1 CRZ, no synthesis fallback)::

        RZZ(θ)  =  CX(q0, q1) · RZ(θ, q1) · CX(q0, q1)
        c-RZZ   =  CX(q0, q1) · CRZ(θ, ht, q1) · CX(q0, q1)

    The outer CX gates are applied **unconditionally**: they are
    self-inverse, so on the ht = 0 branch the sandwich CX·I·CX collapses
    to identity, and on ht = 1 it realises the full RZZ(θ).
    """
    qc.cx(q0, q1)
    qc.crz(theta, ht, q1)
    qc.cx(q0, q1)


def append_ctrl_rxx(
    qc: QuantumCircuit,
    ht: Qubit,
    q0: Qubit,
    q1: Qubit,
    theta: float,
) -> None:
    """Append c-RXX(θ) on (q0, q1) controlled by ``ht``.

    Uses ``RXX(θ) = (H ⊗ H) · RZZ(θ) · (H ⊗ H)`` with the unconditional
    basis change exploiting H·H = I::

        c-RXX  =  (H ⊗ H) · c-RZZ(θ) · (H ⊗ H)
    """
    qc.h(q0)
    qc.h(q1)
    append_ctrl_rzz(qc, ht, q0, q1, theta)
    qc.h(q0)
    qc.h(q1)


def append_ctrl_ryy(
    qc: QuantumCircuit,
    ht: Qubit,
    q0: Qubit,
    q1: Qubit,
    theta: float,
) -> None:
    """Append c-RYY(θ) on (q0, q1) controlled by ``ht``.

    Uses ``RYY(θ) = (S·H ⊗ S·H) · RZZ(θ) · (H·S† ⊗ H·S†)`` with the
    unconditional basis change exploiting (S·H) · (H·S†) = I::

        c-RYY  =  (S·H ⊗ S·H) · c-RZZ(θ) · (H·S† ⊗ H·S†)

    In Qiskit (gate order is left-to-right in time), the front
    basis change is ``S†, H`` per qubit and the back is ``H, S``.
    """
    # Pre-conjugation: H · S†  on each qubit.
    qc.sdg(q0); qc.h(q0)
    qc.sdg(q1); qc.h(q1)
    append_ctrl_rzz(qc, ht, q0, q1, theta)
    # Post-conjugation: S · H  on each qubit.
    qc.h(q0); qc.s(q0)
    qc.h(q1); qc.s(q1)


# ---------------------------------------------------------------------
# Controlled Pauli-string observable (the P inside A(U,P)).
# ---------------------------------------------------------------------

def append_ctrl_pauli_string(
    qc: QuantumCircuit,
    ht: Qubit,
    qs: Sequence[Qubit],
    pauli: str,
) -> None:
    """Append c-(P_n ⊗ ... ⊗ P_0) controlled by ``ht``.

    Pauli operators on different qubits commute, so

        c-(P_0 ⊗ P_1 ⊗ ... ⊗ P_{n-1})  =  ∏_i c-P_i

    where c-X = CX, c-Y = CY, c-Z = CZ are all native single-controlled
    gates.  No synthesis fallback.

    Parameters
    ----------
    pauli : str
        Pauli string in Qiskit little-endian convention; the rightmost
        character acts on ``qs[0]``.
    """
    for i, p in enumerate(reversed(pauli)):
        if p == "I":
            continue
        if p == "X":
            qc.cx(ht, qs[i])
        elif p == "Y":
            qc.cy(ht, qs[i])
        elif p == "Z":
            qc.cz(ht, qs[i])
        else:
            raise ValueError(
                f"append_ctrl_pauli_string: unknown Pauli char {p!r} in "
                f"{pauli!r}.  Allowed: 'I', 'X', 'Y', 'Z'."
            )


# ---------------------------------------------------------------------
# Doubly-controlled cV± gates: ht-controlled wrappers around the
# legacy parity-controlled increment / decrement registers.
# ---------------------------------------------------------------------
#
# Mathematical contract — these are bit-for-bit equivalent to wrapping
# the cV± gates from VGateCache with one extra ``.control(1)`` call,
# but they avoid synthesising the dense UnitaryGate matrix into a
# controlled unitary (which would otherwise trigger QSD).
#
# The legacy gate_cache.cVp implements the circuit (Qiskit indexing,
# qubit 0 is the inner control, qubits 1..n_s are the freq register
# in LSB-first order):
#
#     X(0)
#     for i in [n_s-1, n_s-2, ..., 1]:
#         MCX([0, 1, 2, ..., i], i+1)
#     CX(0, 1)
#     X(0)
#
# i.e. "increment freq if inner control is 0".  We add the ht control
# to every MCX/CX in the cascade.  The wrapping X gates are applied
# UNCONDITIONALLY: they cancel on the ht=0 branch (X·I·X = I), and on
# ht=1 they correctly flip the inner control polarity so the cascade
# fires on the legacy (inner control = 0) condition.
#
# Verification table for ``append_dctrl_cVp(ht, ctrl, freq)``:
#
#     | ht  | ctrl_orig | After 1st X | Cascade fires? | After 2nd X | Net effect           |
#     |-----|-----------|-------------|----------------|-------------|----------------------|
#     |  0  |     0     |   ctrl=1    |  No (ht=0)     |   ctrl=0    | identity             |
#     |  0  |     1     |   ctrl=0    |  No (ht=0)     |   ctrl=1    | identity             |
#     |  1  |     0     |   ctrl=1    |  Yes           |   ctrl=0    | freq incremented     |
#     |  1  |     1     |   ctrl=0    |  No (ctrl=0)   |   ctrl=1    | identity             |
#
# This is exactly the legacy semantics for cV+ wrapped with one
# additional control on ht.

def _ladder_qubits(
    ht: Qubit,
    ctrl: Qubit,
    freq: QuantumRegister,
    upto: int,
) -> list:
    """[ht, ctrl, freq[0], freq[1], ..., freq[upto-1]] — the per-step
    MCX control list used by every cV± cascade variant below."""
    return [ht, ctrl, *list(freq[:upto])]


def append_dctrl_cVp(
    qc: QuantumCircuit,
    ht: Qubit,
    ctrl: Qubit,
    freq: QuantumRegister,
) -> None:
    """Doubly-controlled cV+: increment ``freq`` iff ``ht=1`` AND ``ctrl=0``.

    Bit-for-bit equivalent to ``cV+`` from
    :class:`VGateCache` wrapped with one additional ``.control(1)`` on
    ``ht``, but expanded inline as a chain of MCX gates so no
    Quantum-Shannon decomposition is invoked.
    """
    n_s = len(freq)
    qc.x(ctrl)                                          # neg-polarity flip
    for i in range(n_s - 1, 0, -1):
        qc.mcx(_ladder_qubits(ht, ctrl, freq, i), freq[i])
    qc.ccx(ht, ctrl, freq[0])                           # final LSB toggle
    qc.x(ctrl)                                          # restore ctrl


def append_dctrl_cVm(
    qc: QuantumCircuit,
    ht: Qubit,
    ctrl: Qubit,
    freq: QuantumRegister,
) -> None:
    """Doubly-controlled cV-: decrement ``freq`` iff ``ht=1`` AND ``ctrl=1``.

    Bit-for-bit equivalent to ``cV-`` from :class:`VGateCache` with one
    additional control on ``ht``.  No X wrappers are needed because
    cV- is already positive-polarity.
    """
    n_s = len(freq)
    qc.ccx(ht, ctrl, freq[0])                           # initial LSB toggle
    for i in range(1, n_s):
        qc.mcx(_ladder_qubits(ht, ctrl, freq, i), freq[i])


def append_dctrl_cVp_dag(
    qc: QuantumCircuit,
    ht: Qubit,
    ctrl: Qubit,
    freq: QuantumRegister,
) -> None:
    """Doubly-controlled (cV+)†: decrement ``freq`` iff ``ht=1`` AND ``ctrl=0``.

    Inverse of :func:`append_dctrl_cVp` — gate sequence reversed; the
    outer X(ctrl) wrappers stay in place because they are self-inverse
    and frame the negative-polarity flip identically in both
    directions.
    """
    n_s = len(freq)
    qc.x(ctrl)
    qc.ccx(ht, ctrl, freq[0])
    for i in range(1, n_s):
        qc.mcx(_ladder_qubits(ht, ctrl, freq, i), freq[i])
    qc.x(ctrl)


def append_dctrl_cVm_dag(
    qc: QuantumCircuit,
    ht: Qubit,
    ctrl: Qubit,
    freq: QuantumRegister,
) -> None:
    """Doubly-controlled (cV-)†: increment ``freq`` iff ``ht=1`` AND ``ctrl=1``.

    Inverse of :func:`append_dctrl_cVm` — gate sequence reversed.
    """
    n_s = len(freq)
    for i in range(n_s - 1, 0, -1):
        qc.mcx(_ladder_qubits(ht, ctrl, freq, i), freq[i])
    qc.ccx(ht, ctrl, freq[0])
