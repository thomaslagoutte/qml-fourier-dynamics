"""Regression: MeshgridTensorExtractor output matches the legacy d>1 pipeline.

Compares the new separate-registers + MeshgridTensorExtractor feature
tensor against the legacy ``KernelDynamicsLearner._extract_b_vector_sim``
in ``src/learning.py`` / ``learning_kernel.py`` on the checked-in
``precomputed_schwinger_*.npz`` and ``precomputed_model4_*.npz`` fixtures.

Skeleton only — populated after concrete logic lands.
"""

import pytest

pytestmark = pytest.mark.skip(reason="Populated after concrete d>1 logic is implemented.")


def test_meshgrid_tensor_matches_legacy_schwinger_fixture() -> None:
    raise NotImplementedError


def test_meshgrid_tensor_matches_legacy_model4_fixture() -> None:
    raise NotImplementedError
