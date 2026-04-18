"""Regression: HadamardBLExtractor output matches the legacy d=1 pipeline.

Compares the new shared-register + HadamardBLExtractor feature matrix
against ``src/learning.py``'s ``FourierDynamicsLearner.extract_fourier_features``
on the same (edges, tau, r_steps, observable) inputs using the
checked-in ``precomputed_tfim_validation.npz`` fixture.

Skeleton only — populated after concrete logic lands.
"""

import pytest

pytestmark = pytest.mark.skip(reason="Populated after concrete d=1 logic is implemented.")


def test_d1_feature_matrix_matches_legacy_on_precomputed_fixture() -> None:
    raise NotImplementedError
