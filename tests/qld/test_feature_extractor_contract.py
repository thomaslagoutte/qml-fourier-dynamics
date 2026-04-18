"""Tests for the observable-linearity contract on FeatureExtractor.

The base class marks ``extract`` and ``extract_batch`` as final via
``@final`` for type checkers and via ``__init_subclass__`` at runtime.
This file asserts both:

1. A well-behaved subclass (only overriding ``extract_single_pauli``)
   instantiates fine, and the inherited ``extract`` runs the linearity
   loop — summing per-Pauli contributions with their coefficients.
2. A misbehaving subclass that tries to override ``extract`` raises
   ``TypeError`` at class creation time.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pytest

from quantum_learning_dynamics.circuits.base import CircuitBuilder
from quantum_learning_dynamics.features.base import FeatureExtractor
from quantum_learning_dynamics.hamiltonians.base import HamiltonianModel
from quantum_learning_dynamics.observables.base import Observable, PauliTerm


# ---- Fakes --------------------------------------------------------------


class _FakeBuilder(CircuitBuilder):
    """Bare-minimum builder — contracts require the attribute to exist; no ops called."""

    def freq_register_size(self, r_steps: int) -> int:  # pragma: no cover — unused
        return 0

    def build_au(self, model, x, tau, r_steps):  # pragma: no cover — unused
        raise NotImplementedError

    def build_aup(self, model, x, tau, r_steps, pauli):  # pragma: no cover — unused
        raise NotImplementedError


class _FakeObservable(Observable):
    """Two-term observable: 0.5 * 'II' + (-1.0) * 'XX'.

    Abuses the Pauli-string field as an opaque key so the
    ``extract_single_pauli`` stub can return a distinct tensor per term.
    """

    @property
    def num_qubits(self) -> int:
        return 2

    def terms(self) -> Sequence[PauliTerm]:
        return (
            PauliTerm(pauli="II", coefficient=0.5),
            PauliTerm(pauli="XX", coefficient=-1.0),
        )


class _StubExtractor(FeatureExtractor):
    """Returns a per-Pauli vector encoded by the Pauli string identity."""

    def extract_single_pauli(self, model, x, tau, r_steps, pauli):
        if pauli == "II":
            return np.array([1.0, 2.0, 3.0])
        if pauli == "XX":
            return np.array([10.0, 20.0, 30.0])
        raise AssertionError(f"unexpected pauli {pauli!r}")


# ---- Tests --------------------------------------------------------------


def test_linearity_loop_sums_per_pauli_with_coefficients() -> None:
    builder = _FakeBuilder()
    extractor = _StubExtractor(builder)
    obs = _FakeObservable()
    out = extractor.extract(model=None, x=None, tau=0.0, r_steps=0, observable=obs)
    # 0.5 * [1, 2, 3] + (-1.0) * [10, 20, 30] = [-9.5, -19.0, -28.5]
    np.testing.assert_allclose(out, np.array([-9.5, -19.0, -28.5]))


def test_overriding_extract_is_rejected_at_class_creation() -> None:
    with pytest.raises(TypeError, match="final"):

        class _BadExtractor(FeatureExtractor):  # noqa: D401 — test artefact
            def extract_single_pauli(self, model, x, tau, r_steps, pauli):
                return np.zeros(1)

            def extract(self, model, x, tau, r_steps, observable):  # type: ignore[override]
                return np.zeros(1)


def test_overriding_extract_batch_is_rejected_at_class_creation() -> None:
    with pytest.raises(TypeError, match="final"):

        class _BadBatchExtractor(FeatureExtractor):  # noqa: D401 — test artefact
            def extract_single_pauli(self, model, x, tau, r_steps, pauli):
                return np.zeros(1)

            def extract_batch(self, model, xs, tau, r_steps, observable, flatten=True):  # type: ignore[override]
                return np.zeros((1, 1))


def test_extract_batch_flattens_by_default() -> None:
    builder = _FakeBuilder()
    extractor = _StubExtractor(builder)
    obs = _FakeObservable()
    B = extractor.extract_batch(
        model=None,
        xs=["x1", "x2"],
        tau=0.0,
        r_steps=0,
        observable=obs,
    )
    # Each row is the summed feature vector, shape (2, 3)
    assert B.shape == (2, 3)
    np.testing.assert_allclose(B[0], np.array([-9.5, -19.0, -28.5]))
    np.testing.assert_allclose(B[1], np.array([-9.5, -19.0, -28.5]))
