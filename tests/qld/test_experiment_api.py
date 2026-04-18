"""End-to-end routing and validation tests for the Experiment API."""

import numpy as np
import pytest

from quantum_learning_dynamics import Experiment
from quantum_learning_dynamics.hamiltonians import InhomogeneousTFIM, TFIM
from quantum_learning_dynamics.observables import LocalZ


def test_lasso_b_l_wires_shared_register_and_hadamard() -> None:
    from quantum_learning_dynamics.circuits.shared_register import SharedRegisterBuilder
    from quantum_learning_dynamics.features.hadamard_b_l import HadamardBLExtractor
    from quantum_learning_dynamics.learners.lasso import LassoLearner

    exp = Experiment(
        model=TFIM(num_qubits=3),
        observable=LocalZ(num_qubits=3, qubit=0),
        method="lasso-b_l",
        tau=0.3,
        r_steps=2,
    )
    assert isinstance(exp.builder, SharedRegisterBuilder)
    assert isinstance(exp.extractor, HadamardBLExtractor)
    assert isinstance(exp.learner, LassoLearner)


def test_lasso_tensor_wires_separate_registers_and_meshgrid() -> None:
    from quantum_learning_dynamics.circuits.separate_registers import SeparateRegistersBuilder
    from quantum_learning_dynamics.features.meshgrid_tensor import MeshgridTensorExtractor
    from quantum_learning_dynamics.learners.lasso import LassoLearner

    exp = Experiment(
        model=InhomogeneousTFIM(num_qubits=3),
        observable=LocalZ(num_qubits=3, qubit=0),
        method="lasso-tensor",
        tau=0.3,
        r_steps=2,
    )
    assert isinstance(exp.builder, SeparateRegistersBuilder)
    assert isinstance(exp.extractor, MeshgridTensorExtractor)
    assert isinstance(exp.learner, LassoLearner)


def test_kernel_tensor_wires_separate_registers_and_kernel_ridge() -> None:
    from quantum_learning_dynamics.circuits.separate_registers import SeparateRegistersBuilder
    from quantum_learning_dynamics.features.meshgrid_tensor import MeshgridTensorExtractor
    from quantum_learning_dynamics.learners.kernel_ridge import KernelRidgeLearner

    exp = Experiment(
        model=InhomogeneousTFIM(num_qubits=3),
        observable=LocalZ(num_qubits=3, qubit=0),
        method="kernel-tensor",
        tau=0.3,
        r_steps=2,
    )
    assert isinstance(exp.builder, SeparateRegistersBuilder)
    assert isinstance(exp.extractor, MeshgridTensorExtractor)
    assert isinstance(exp.learner, KernelRidgeLearner)


# --- Validation ----------------------------------------------------------


@pytest.mark.parametrize("method", ["lasso-tensor", "kernel-tensor"])
def test_d_gt_1_methods_reject_d1_models(method: str) -> None:
    with pytest.raises(ValueError, match="d=1"):
        Experiment(
            model=TFIM(num_qubits=3),
            observable=LocalZ(num_qubits=3, qubit=0),
            method=method,  # type: ignore[arg-type]
            tau=0.3,
            r_steps=2,
        )


def test_lasso_b_l_rejects_dN_models() -> None:
    with pytest.raises(ValueError, match="d=3"):
        Experiment(
            model=InhomogeneousTFIM(num_qubits=3),
            observable=LocalZ(num_qubits=3, qubit=0),
            method="lasso-b_l",
            tau=0.3,
            r_steps=2,
        )


def test_observable_num_qubits_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="num_qubits"):
        Experiment(
            model=TFIM(num_qubits=3),
            observable=LocalZ(num_qubits=4, qubit=0),
            method="lasso-b_l",
            tau=0.3,
            r_steps=2,
        )


def test_unknown_method_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        Experiment(
            model=TFIM(num_qubits=3),
            observable=LocalZ(num_qubits=3, qubit=0),
            method="lasso",  # type: ignore[arg-type]  — old string, now invalid
            tau=0.3,
            r_steps=2,
        )


# --- Reproducibility / seed propagation ---------------------------------


def test_seed_propagates_to_extractor_and_learner() -> None:
    exp_a = Experiment(
        model=TFIM(num_qubits=3),
        observable=LocalZ(num_qubits=3, qubit=0),
        method="lasso-b_l",
        tau=0.3,
        r_steps=2,
        seed=123,
    )
    exp_b = Experiment(
        model=TFIM(num_qubits=3),
        observable=LocalZ(num_qubits=3, qubit=0),
        method="lasso-b_l",
        tau=0.3,
        r_steps=2,
        seed=123,
    )
    # Extractor rng produces the same stream:
    assert np.array_equal(exp_a.extractor.rng.random(5), exp_b.extractor.rng.random(5))  # type: ignore[attr-defined]
    # Learner has an explicit, matching random_state:
    assert exp_a.learner._lasso.random_state == exp_b.learner._lasso.random_state  # type: ignore[attr-defined]
    # Master rng also matches:
    assert np.array_equal(exp_a.rng.random(5), exp_b.rng.random(5))


def test_seed_zero_still_produces_deterministic_rng() -> None:
    exp_a = Experiment(
        model=TFIM(num_qubits=3),
        observable=LocalZ(num_qubits=3, qubit=0),
        method="lasso-b_l",
        tau=0.3,
        r_steps=2,
        seed=0,
    )
    exp_b = Experiment(
        model=TFIM(num_qubits=3),
        observable=LocalZ(num_qubits=3, qubit=0),
        method="lasso-b_l",
        tau=0.3,
        r_steps=2,
        seed=0,
    )
    assert np.array_equal(exp_a.rng.random(10), exp_b.rng.random(10))


# --- describe() smoke ---------------------------------------------------


def test_describe_returns_config_dict() -> None:
    exp = Experiment(
        model=InhomogeneousTFIM(num_qubits=3),
        observable=LocalZ(num_qubits=3, qubit=0),
        method="kernel-tensor",
        tau=0.3,
        r_steps=2,
        seed=7,
    )
    desc = exp.describe()
    assert desc["method"] == "kernel-tensor"
    assert desc["d"] == 3
    assert desc["num_qubits"] == 3
    assert desc["seed"] == 7
