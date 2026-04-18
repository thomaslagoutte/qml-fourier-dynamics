"""API-level tests for the observables subpackage.

These tests will be populated once the concrete implementations are in
place.  For now, they assert that the public API imports cleanly and
that the Observable ABC enforces its contract.
"""

import pytest

from quantum_learning_dynamics.observables import (
    ElectricFlux,
    LocalPauli,
    LocalZ,
    Observable,
    PauliTerm,
    StaggeredMagnetization,
)


def test_public_api_is_importable() -> None:
    # Smoke test: the names are re-exported.
    assert Observable is not None
    assert PauliTerm is not None
    assert LocalPauli is not None
    assert LocalZ is not None
    assert StaggeredMagnetization is not None
    assert ElectricFlux is not None


def test_observable_abc_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        Observable()  # type: ignore[abstract]
