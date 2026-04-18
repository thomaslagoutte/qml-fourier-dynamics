"""Tests for the VGateCache invariant.

Concrete tests (asserting that a controlled V+/- is built only once per
register size across many circuit construction calls) go here once
concrete logic is populated.  For now, a smoke test that the cache
memoises calls.
"""

from unittest.mock import patch

from quantum_learning_dynamics.circuits import VGateCache


def test_cache_is_empty_on_construction() -> None:
    c = VGateCache()
    assert len(c) == 0


def test_cache_memoises_build_calls() -> None:
    c = VGateCache()
    sentinel = {"cVp": object(), "cVm": object()}
    with patch.object(VGateCache, "_build", return_value=sentinel) as m:
        first = c.get(3)
        second = c.get(3)
    assert first is second
    assert m.call_count == 1
    assert 3 in c
