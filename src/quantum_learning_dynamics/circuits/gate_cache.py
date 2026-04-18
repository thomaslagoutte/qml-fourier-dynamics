"""Controlled frequency-shift (V+/-) gate cache.

IMMUTABLE PHYSICS CONSTRAINT
----------------------------
The controlled V+ and V- frequency-shift gates are expensive to compile
(they expand into a QFT + phase + inverse-QFT on the frequency register).
The legacy code relied on building them *once per register size* and
reusing them across all samples.  Rebuilding inline makes extractor
throughput collapse on the Schwinger runs.

Any :class:`CircuitBuilder` subclass must route its V+/- access through
this object — never construct the gates inline.
"""

from __future__ import annotations

from typing import Dict


class VGateCache:
    """Cache of controlled V+/- gates keyed by frequency-register size n_s.

    Notes
    -----
    * The cache is populated lazily: first call to :meth:`get` for a given
      n_s builds both ``cVp`` and ``cVm`` and stores them.
    * The cache is *instance-local* so that tests can instantiate isolated
      builders; a shared global cache is easy to add later by passing the
      same :class:`VGateCache` into multiple builders.
    * Construction logic is delegated to :meth:`_build` which subclasses /
      concrete implementations override or fill in.
    """

    def __init__(self) -> None:
        self._cache: Dict[int, Dict[str, object]] = {}

    def get(self, n_s: int) -> Dict[str, object]:
        """Return ``{"cVp": <Gate>, "cVm": <Gate>}`` for register size ``n_s``.

        Builds and caches on first access.
        """
        cached = self._cache.get(n_s)
        if cached is not None:
            return cached
        built = self._build(n_s)
        self._cache[n_s] = built
        return built

    def clear(self) -> None:
        """Drop all cached gates (used in tests)."""
        self._cache.clear()

    def __contains__(self, n_s: int) -> bool:
        return n_s in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    # ------------------------------------------------------------------
    # Subclass / concrete-implementation hook
    # ------------------------------------------------------------------

    def _build(self, n_s: int) -> Dict[str, object]:
        """Construct ``{"cVp", "cVm"}`` for register size ``n_s``.

        Concrete implementation is filled in once the API skeleton is
        approved; the logic is lifted verbatim from the legacy
        ``CircuitBuilder._get_cached_v_gates`` in
        ``src/quantum_routines.py``.
        """
        raise NotImplementedError(
            "VGateCache._build — concrete V+/- construction pending approval."
        )
