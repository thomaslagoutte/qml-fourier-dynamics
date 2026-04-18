"""Abstract feature extractor — turns circuits into real Fourier tensors.

==============================================================================
IMMUTABLE PHYSICS CONTRACT — OBSERVABLE LINEARITY
==============================================================================
For a composite observable O = sum_h beta_h P_h we MUST NOT pass O to
Qiskit as a single :class:`SparsePauliOp` inside the A(U, P) extraction
circuit.  ``PauliGate`` accepts only one Pauli string, and folding a
linear combination into a single operator breaks the amplitude-encoding
identity exploited by A(U, P).

The correct procedure, enforced here as a CONTRACT rather than a
convention:

    for each Pauli term (P_h, beta_h) in observable.terms():
        b_h(x) = extract_single_pauli(model, x, tau, r_steps, P_h)
    b(x) = sum_h beta_h * b_h(x)

This classical linear combination MUST happen in :meth:`FeatureExtractor.extract`
AFTER per-Pauli extraction.

Enforcement mechanism
---------------------
Subclasses MUST override :meth:`extract_single_pauli` (abstract) and MUST
NOT override :meth:`extract` or :meth:`extract_batch` (marked ``@final``
at the type-check level and guarded at class-creation time via
:meth:`__init_subclass__`, which raises :class:`TypeError` if a subclass
declares its own ``extract`` or ``extract_batch``).  The sole
legitimate way to customise feature extraction is to change how a
*single* Pauli is extracted — the linearity loop is off-limits.

Regression tests cross-validate this against the legacy pipeline on
``precomputed_*.npz`` fixtures; see ``tests/qld/test_features_*.py``.
==============================================================================
Tensor shape convention
-----------------------
For ``d == 1`` the feature "tensor" is a 1-D array of length ``4r + 1``.
For ``d >  1`` it is a rank-d array of shape ``(4r + 1,) * d``.
The learner decides when to flatten.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, final

import numpy as np

from .._types import FeatureMatrix, FeatureTensor, InputX, PauliString
from ..circuits.base import CircuitBuilder
from ..hamiltonians.base import HamiltonianModel
from ..observables.base import Observable

# Methods that subclasses are forbidden to override.  Kept as a module
# constant so the guard is easy to audit.
_FINAL_METHODS: frozenset[str] = frozenset({"extract", "extract_batch"})


class FeatureExtractor(ABC):
    """Extract the real Fourier feature tensor b(x) for each training input x.

    See the module docstring for the observable-linearity contract
    (enforced via ``@final`` + ``__init_subclass__``).
    """

    # ------------------------------------------------------------------
    # Contract enforcement: forbid subclasses from overriding the
    # linearity loop.
    # ------------------------------------------------------------------

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        for name in _FINAL_METHODS:
            # Allow the base class's own definition to pass through
            # (``cls.__dict__`` contains only what the subclass itself defined).
            if name in cls.__dict__:
                raise TypeError(
                    f"{cls.__module__}.{cls.__name__} overrides FeatureExtractor."
                    f"{name!r}, which is final.  Subclasses must implement "
                    f"`extract_single_pauli` only; the observable-linearity "
                    f"loop in `extract` / `extract_batch` is immutable.  "
                    f"See features/base.py for the physics rationale."
                )

    def __init__(self, builder: CircuitBuilder) -> None:
        self.builder = builder

    # ------------------------------------------------------------------
    # Subclass hook — the ONLY method a subclass may override.
    # ------------------------------------------------------------------

    @abstractmethod
    def extract_single_pauli(
        self,
        model: HamiltonianModel,
        x: InputX,
        tau: float,
        r_steps: int,
        pauli: PauliString,
    ) -> FeatureTensor:
        """Extract b_h(x) for a SINGLE Pauli observable term.

        Concrete subclasses implement this differently:

        * :class:`HadamardBLExtractor`     (d = 1) — Hadamard test on A(U, P),
          marginalising over (data, ancilla) and reading the frequency
          register amplitude.
        * :class:`MeshgridTensorExtractor` (d > 1) — full statevector
          simulation of A(U, P), then :func:`numpy.meshgrid` indexing to
          pull out the rank-d tensor.

        The observable-linearity loop in :meth:`extract` drives this
        method once per Pauli term; DO NOT fold a linear combination
        into a single Qiskit operator inside this implementation.
        """

    # ------------------------------------------------------------------
    # Final (non-overridable) linearity loop.
    # ------------------------------------------------------------------

    @final
    def extract(
        self,
        model: HamiltonianModel,
        x: InputX,
        tau: float,
        r_steps: int,
        observable: Observable,
    ) -> FeatureTensor:
        """Real Fourier feature tensor b(x) = sum_h beta_h * b_h(x).

        This method is FINAL.  The linearity loop is the contract:
        every concrete extractor inherits the correct behaviour; no
        subclass may override it.
        """
        tensor: FeatureTensor | None = None
        for term in observable.terms():
            b_h = self.extract_single_pauli(model, x, tau, r_steps, term.pauli)
            contribution = term.coefficient * b_h
            tensor = contribution if tensor is None else tensor + contribution
        if tensor is None:  # pragma: no cover — Observable.terms() non-empty by contract
            raise ValueError("Observable returned no Pauli terms.")
        return tensor

    @final
    def extract_batch(
        self,
        model: HamiltonianModel,
        xs: Sequence[InputX],
        tau: float,
        r_steps: int,
        observable: Observable,
        flatten: bool = True,
    ) -> FeatureMatrix:
        """Stack features for many samples.  FINAL — see :meth:`extract`.

        Parameters
        ----------
        flatten : bool, default True
            If True, each per-sample tensor is flattened to shape
            ``((4r+1) ** d,)`` before stacking — the convention expected
            by sklearn's Lasso / KernelRidge.  If False, returns a rank
            ``1 + d`` array with sample index on axis 0.
        """
        tensors = [self.extract(model, x, tau, r_steps, observable) for x in xs]
        if flatten:
            return np.stack([t.ravel() for t in tensors])
        return np.stack(tensors)
