"""Concrete observables used across the TFIM and Schwinger experiments.

Each class here is a thin, declarative subclass of :class:`Observable`.
No circuit construction happens — that is strictly the
:class:`FeatureExtractor`'s job.
"""

from __future__ import annotations

from typing import Sequence

from .base import Observable, PauliTerm


class LocalPauli(Observable):
    """A single-Pauli observable on ``num_qubits``, e.g. ``Z_0``.

    Parameters
    ----------
    num_qubits : int
        Number of data qubits in the system.
    pauli : str
        Full Qiskit little-endian Pauli string of length ``num_qubits``,
        e.g. ``"IIIZ"`` for Z on qubit 0 of a 4-qubit system.
    coefficient : float, default 1.0
    """

    def __init__(self, num_qubits: int, pauli: str, coefficient: float = 1.0) -> None:
        if len(pauli) != num_qubits:
            raise ValueError(
                f"pauli must have length num_qubits={num_qubits}, got {len(pauli)}."
            )
        self._num_qubits = num_qubits
        self._term = PauliTerm(pauli=pauli, coefficient=coefficient)

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    def terms(self) -> Sequence[PauliTerm]:
        return (self._term,)


class LocalZ(LocalPauli):
    """Z_q observable — single Pauli Z on qubit ``q``."""

    def __init__(self, num_qubits: int, qubit: int = 0, coefficient: float = 1.0) -> None:
        if not 0 <= qubit < num_qubits:
            raise ValueError(f"qubit index {qubit} out of range for {num_qubits} qubits.")
        pauli = ["I"] * num_qubits
        pauli[num_qubits - 1 - qubit] = "Z"
        super().__init__(num_qubits, "".join(pauli), coefficient)


class StaggeredMagnetization(Observable):
    """Staggered magnetization M_s = kappa * sum_i (-1)^i Z_i.

    Composite observable — exercises the observable-linearity path in
    :class:`FeatureExtractor`.  Used in notebooks 02 / 03 (TFIM) and in
    the Schwinger validation for the fermion-sector staggered mass.

    NORMALISATION IS USER-CONTROLLED.  The ``1/N`` prefactor is a
    cosmetic choice — it changes the absolute magnitude of every b_l(x)
    and therefore the effective signal-to-noise seen by the regressor.
    Pick the convention that matches the reference you're comparing
    against:

    * ``normalize=True``  → kappa = 1/N  (intensive magnetization,
                            matches notebooks 02 / 03)
    * ``normalize=False`` → kappa = 1    (extensive sum, matches the
                            paper's raw definition)

    Parameters
    ----------
    num_qubits : int
        System size N.  The ``(-1)^i`` sign is applied to qubit i.
    normalize : bool, default True
        See above.  The default matches the legacy notebooks.
    """

    def __init__(self, num_qubits: int, normalize: bool = True) -> None:
        self._num_qubits = num_qubits
        self._normalize = normalize
        prefactor = 1.0 / num_qubits if normalize else 1.0
        self._terms: tuple[PauliTerm, ...] = tuple(
            PauliTerm(pauli=self._z_at(num_qubits, i), coefficient=prefactor * ((-1) ** i))
            for i in range(num_qubits)
        )

    @property
    def normalize(self) -> bool:
        """Whether the ``1/N`` intensive prefactor is applied."""
        return self._normalize

    @staticmethod
    def _z_at(num_qubits: int, qubit: int) -> str:
        pauli = ["I"] * num_qubits
        pauli[num_qubits - 1 - qubit] = "Z"
        return "".join(pauli)

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    def terms(self) -> Sequence[PauliTerm]:
        return self._terms


class ElectricFlux(Observable):
    """Gauge-link electric flux for the Schwinger Z2 model.

    On the interlaced (matter, link, matter, link, …) qubit layout the
    gauge links live on odd qubits.  The electric flux observable is
    ``kappa * sum_i X_{l_i}`` where ``l_i`` is the i-th gauge-link qubit.

    NORMALISATION IS USER-CONTROLLED (same rationale as
    :class:`StaggeredMagnetization`):

    * ``normalize=True``  → kappa = 1 / L   (intensive, matches notebook 04)
    * ``normalize=False`` → kappa = 1       (extensive)

    Parameters
    ----------
    num_matter_sites : int
        N matter sites ⇒ N-1 gauge links ⇒ 2N-1 total data qubits.
    normalize : bool, default True
    """

    def __init__(self, num_matter_sites: int, normalize: bool = True) -> None:
        num_gauge = num_matter_sites - 1
        total = num_matter_sites + num_gauge
        self._normalize = normalize
        prefactor = 1.0 / num_gauge if (normalize and num_gauge > 0) else 1.0
        self._num_qubits = total
        self._terms = tuple(
            PauliTerm(
                pauli=self._x_at(total, 2 * i + 1),
                coefficient=prefactor,
            )
            for i in range(num_gauge)
        )

    @property
    def normalize(self) -> bool:
        """Whether the ``1/L`` intensive prefactor is applied."""
        return self._normalize

    @staticmethod
    def _x_at(num_qubits: int, qubit: int) -> str:
        pauli = ["I"] * num_qubits
        pauli[num_qubits - 1 - qubit] = "X"
        return "".join(pauli)

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    def terms(self) -> Sequence[PauliTerm]:
        return self._terms
