"""Abstract base classes for Fourier-extraction circuit builders.

Defines the architectural contract between the circuit-construction layer
(builders) and the execution layer (engines in
:mod:`quantum_learning_dynamics.features.engines`).

Separation of concerns
----------------------
* A :class:`CircuitBuilder` knows *how* to instantiate the Fourier-extraction
  subroutine :math:`\\mathcal{A}(U, P)` of Barthe et al. (2025) for a given
  ``(x, tau, r_steps, P)`` — including the hardware-friendly per-frequency
  Hadamard test of Corollary 2 / Figure 7.
* A :class:`KernelBuilder` knows how to instantiate the quantum-overlap
  kernel circuit of Figure 8.
* Execution engines choose *when* to invoke a builder, coordinate the
  V2 Sampler, and convert measurement outcomes into real-valued features
  or Gram-matrix entries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Sequence, Tuple, Union

from qiskit import QuantumCircuit, QuantumRegister

from .._types import InputX
from ..hamiltonians.base import HamiltonianModel
from .gate_cache import VGateCache

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ExecutionMode = Literal["statevector", "hardware_base", "hardware"]
"""Three execution modes a builder may synthesise:

* ``"statevector"``   — bare :math:`\\mathcal{A}(U,P)` circuit; all Fourier
  amplitudes are read simultaneously from the full statevector.
* ``"hardware_base"`` — heavy prefix up to and including the controlled
  :math:`\\mathcal{A}(U,P)` gate; the caller appends the cheap
  per-frequency tail for each target frequency.
* ``"hardware"``      — complete per-frequency Hadamard-test circuit
  (Corollary 2 / Figure 7 of Barthe et al., 2025).
"""

TargetFreq = Union[int, Tuple[int, ...]]
"""Target frequency index :math:`u` for the hardware Hadamard test.

* ``d == 1`` : an unsigned ``int`` :math:`u = l \\bmod 2^{n_s}`.
* ``d > 1``  : a tuple ``(u_0, ..., u_{d-1})`` with
  :math:`u_k = l_k \\bmod 2^{n_s}` per frequency register.
"""


class CircuitBuilder(ABC):
    """Abstract factory for the :math:`\\mathcal{A}(U, P)` subroutine.

    Subclasses correspond to the two Hamiltonian-dimensionality regimes of
    the paper (Barthe et al., 2025):

    * :class:`SharedRegisterBuilder`    (``d = 1``) — one frequency
      register shared across all parameter uploads.
    * :class:`SeparateRegistersBuilder` (``d > 1``) — one frequency
      register per component of :math:`\\alpha`.

    Parameters
    ----------
    model : HamiltonianModel
        Parameterised Hamiltonian :math:`H(x, \\alpha)` defining the
        upload Pauli strings that drive the parity-fold /
        frequency-shift (``D·G·D``) block.
    trotter_order : int, default 1
        Order of the product-formula decomposition used to compile
        :math:`e^{-i\\tau H}` into the discrete PQC.
    """

    def __init__(self, model: HamiltonianModel, trotter_order: int = 1) -> None:
        self.model = model
        self.trotter_order = trotter_order
        self.gate_cache = VGateCache()

    @abstractmethod
    def freq_register_size(self, num_qubits: int, r_steps: int) -> int:
        """Return the qubit count :math:`n_s` of one frequency register.

        The bandwidth must accommodate the signed Fourier indices the
        subroutine can shift through — i.e.
        :math:`2^{n_s} \\ge 2\\,l_{\\max} + 1` for the largest representable
        frequency :math:`l_{\\max}` of the target regime.
        """

    @abstractmethod
    def build_aup(
        self,
        num_qubits: int,
        x: InputX,
        tau: float,
        r_steps: int,
        pauli: str,
        *,
        execution_mode: ExecutionMode = "statevector",
        target_freq: Optional[TargetFreq] = None,
    ) -> Tuple[QuantumCircuit, List[QuantumRegister]]:
        """Instantiate :math:`\\mathcal{A}(U, P)` for a single Pauli observable.

        Parameters
        ----------
        num_qubits : int
            Number of data qubits (``self.model.num_qubits``).
        x : InputX
            Classical input :math:`x` encoding the fixed, known part of
            the Hamiltonian (e.g. TFIM edge list, Schwinger gauge mask).
        tau : float
            Total evolution time :math:`\\tau` — same value the learner uses.
        r_steps : int
            Number of Trotter steps :math:`r`.
        pauli : str
            Single Pauli string :math:`P_h` (one term of the observable's
            linear decomposition :math:`O = \\sum_h \\beta_h P_h`).
        execution_mode : {"statevector", "hardware_base", "hardware"}
            See :data:`ExecutionMode`.
        target_freq : TargetFreq, optional
            Required in ``"hardware"`` mode; ignored otherwise.

        Returns
        -------
        QuantumCircuit
            The assembled circuit.
        list of QuantumRegister
            The frequency register(s) (one in the ``d = 1`` regime, ``d``
            in the ``d > 1`` regime), in index order.
        """

    @staticmethod
    def _append_freq_selector(
        qc: QuantumCircuit,
        ht_control: QuantumRegister,
        freq_registers: Sequence[QuantumRegister],
        target_freq: TargetFreq,
    ) -> None:
        """Append the CX ladder that selects the target Fourier index.

        Implements the coherent shift :math:`|f\\rangle \\mapsto |f \\oplus l\\rangle`
        on the branch where ``ht_control == 1``. This moves the interference
        slot from :math:`f = 0` to :math:`f = l`, so that the final Hadamard
        on ``ht_control`` projects :math:`\\mathrm{Re}(b_l)` rather than
        :math:`\\mathrm{Re}(b_0)` (Corollary 2 / Figure 7).

        Parameters
        ----------
        qc : QuantumCircuit
            Circuit being built; mutated in place.
        ht_control : QuantumRegister
            Single-qubit Hadamard-test control register.
        freq_registers : sequence of QuantumRegister
            Frequency registers of length :math:`d` (``d = 1`` collapses
            to a single register).
        target_freq : TargetFreq
            Unsigned target index per frequency register.
        """
        n_s = len(freq_registers[0])

        # Normalise to a tuple of length d for uniform iteration below.
        if isinstance(target_freq, int):
            target_tuple: Tuple[int, ...] = (target_freq,)
        else:
            target_tuple = tuple(int(u) for u in target_freq)

        for fr, u_k in zip(freq_registers, target_tuple):
            # LSB-first binary pattern: freq[0] is the least-significant bit.
            for j in range(n_s):
                if (u_k >> j) & 1:
                    qc.cx(ht_control[0], fr[j])


class KernelBuilder(ABC):
    """Abstract factory for the quantum-overlap-kernel circuit (Figure 8).

    Subclasses evaluate the real part of the inner product

    .. math::

        K(x_1, x_2) = \\mathrm{Re}\\,
            \\langle \\psi(x_1) \\,|\\, \\psi(x_2) \\rangle,
        \\qquad
        |\\psi(x)\\rangle = \\mathcal{A}(U, P)\\,|0\\rangle,

    directly on a single Hadamard-test circuit — avoiding explicit
    extraction of the exponentially large Fourier tensor :math:`b(x)` in
    the :math:`d > 1` regime.
    """

    def __init__(self, model: HamiltonianModel, trotter_order: int = 1) -> None:
        self.model = model
        self.trotter_order = trotter_order
        self.gate_cache = VGateCache()

    @abstractmethod
    def build_overlap(
        self,
        num_qubits: int,
        x1: InputX,
        x2: InputX,
        tau: float,
        r_steps: int,
        pauli1: str,
        pauli2: str,
    ) -> QuantumCircuit:
        """Assemble the Hadamard-test circuit for
        :math:`\\mathcal{A}(U, P_1)_{x_1}\\,\\mathcal{A}(U, P_2)_{x_2}^{\\dagger}`.
        """
