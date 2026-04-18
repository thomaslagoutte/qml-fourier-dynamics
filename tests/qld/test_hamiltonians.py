"""Smoke tests for the hamiltonians subpackage.

Concrete correctness tests (against the legacy models in src/models.py)
land once the implementations are populated.
"""

import pytest

from quantum_learning_dynamics.hamiltonians import (
    HamiltonianModel,
    InhomogeneousTFIM,
    SchwingerZ2Model,
    TFIM,
)


def test_public_api_is_importable() -> None:
    assert HamiltonianModel is not None
    assert TFIM is not None
    assert InhomogeneousTFIM is not None
    assert SchwingerZ2Model is not None


def test_d_is_exposed_on_each_subclass() -> None:
    assert TFIM(num_qubits=3).d == 1
    assert InhomogeneousTFIM(num_qubits=3).d == 3
    assert SchwingerZ2Model(num_matter_sites=3, mass=1.0, electric_field=0.5).d == 2


def test_hamiltonian_model_abc_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        HamiltonianModel()  # type: ignore[abstract]
