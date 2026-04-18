"""High-level experiment API — the one-call interface to the paper's pipeline.

Typical usage (the "wow factor"):

    from quantum_learning_dynamics import Experiment
    from quantum_learning_dynamics.hamiltonians import InhomogeneousTFIM
    from quantum_learning_dynamics.observables import StaggeredMagnetization

    exp = Experiment(
        model       = InhomogeneousTFIM(num_qubits=4),
        observable  = StaggeredMagnetization(num_qubits=4, normalize=True),
        method      = "kernel-tensor",     # explicit — not inferred from d
        tau         = 0.5,
        r_steps     = 2,
        seed        = 42,
    )
    result = exp.run(num_train=200, num_test=50)
    result.y_true_exact, result.y_true_trotter, result.y_pred

Method names
------------
The method string IS the pipeline.  No inference from ``model.d``:

* ``"lasso-b_l"``     — shared-register circuit + Hadamard-test b_l
                        extractor + Lasso.  Requires ``model.d == 1``.
* ``"lasso-tensor"``  — separate-registers circuit + meshgrid tensor
                        extractor + Lasso on the flattened tensor.
                        Requires ``model.d > 1``.
* ``"kernel-tensor"`` — separate-registers circuit + meshgrid tensor
                        extractor + Kernel Ridge on Gram = B @ B.T.
                        Requires ``model.d > 1``.

If a chosen method is incompatible with the model's ``d``, ``__init__``
raises ``ValueError`` with a precise hint.

Label generation
----------------
Exact and Trotterised labels are generated inside :class:`Experiment`,
not in a separate dynamics module.  Exact labels come from
``model.exact_unitary`` + dense matmul; Trotterised labels are produced
through the same A(U) circuit the extractor uses, preserving the
A(U)-consistent convention that the legacy
``compute_au_labels_inhomogeneous`` established.

Seed propagation
----------------
The optional ``seed`` parameter drives the ONLY source of randomness in
the experiment.  It is passed to:

* :meth:`HamiltonianModel.sample_x` / :meth:`sample_alpha` via ``self.rng``
* the feature extractor (e.g. HadamardBLExtractor's epsilon_b noise)
* sklearn learners that accept ``random_state`` (LassoLearner)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Literal, Tuple

import numpy as np

from ._types import AlphaVector, InputX
from .circuits.base import CircuitBuilder
from .features.base import FeatureExtractor
from .hamiltonians.base import HamiltonianModel
from .learners.base import Learner
from .observables.base import Observable
from .results import ExperimentResult

Method = Literal["lasso-b_l", "lasso-tensor", "kernel-tensor"]

# ``required_d`` encodes the one piece of validation the router needs.
# Use a callable so we can say "any d > 1" compactly, rather than ==.
_METHOD_REQUIRES_D: Dict[str, Callable[[int], bool]] = {
    "lasso-b_l":     lambda d: d == 1,
    "lasso-tensor":  lambda d: d > 1,
    "kernel-tensor": lambda d: d > 1,
}

_METHOD_D_HINT: Dict[str, str] = {
    "lasso-b_l":     "model.d must equal 1 (shared-register Hadamard-test regime).",
    "lasso-tensor":  "model.d must be > 1 (separate-registers tensor regime).",
    "kernel-tensor": "model.d must be > 1 (separate-registers kernel regime).",
}


class Experiment:
    """High-level PAC-learning experiment orchestrator.

    Parameters
    ----------
    model : HamiltonianModel
        Exposes ``num_qubits`` and ``d``.
    observable : Observable
        Target observable O = sum_h beta_h P_h.  ``observable.num_qubits``
        must equal ``model.num_qubits``.
    method : {"lasso-b_l", "lasso-tensor", "kernel-tensor"}
        Explicit pipeline selection — no inference from d.  See module
        docstring for what each method wires.
    tau : float
        Total evolution time.
    r_steps : int
        First-order Trotter step count.
    regularization : float, default 0.1
        L1 strength for Lasso, lambda for KRR.
    epsilon_b : float, default 0.0
        Simulated per-coefficient noise for the HadamardBLExtractor.
    seed : int, default 0
        Master seed for all stochastic components.  See module docstring.
    """

    def __init__(
        self,
        model: HamiltonianModel,
        observable: Observable,
        method: Method,
        tau: float,
        r_steps: int,
        regularization: float = 0.1,
        epsilon_b: float = 0.0,
        seed: int = 0,
    ) -> None:
        # ---- qubit-count sanity ----
        if observable.num_qubits != model.num_qubits:
            raise ValueError(
                f"observable.num_qubits ({observable.num_qubits}) != "
                f"model.num_qubits ({model.num_qubits})."
            )

        # ---- method sanity ----
        if method not in _METHOD_REQUIRES_D:
            valid = sorted(_METHOD_REQUIRES_D.keys())
            raise ValueError(f"Unknown method: {method!r}. Valid methods: {valid}.")
        if not _METHOD_REQUIRES_D[method](model.d):
            raise ValueError(
                f"method={method!r} is incompatible with model.d={model.d}. "
                f"{_METHOD_D_HINT[method]}"
            )

        self.model = model
        self.observable = observable
        self.method: Method = method
        self.tau = tau
        self.r_steps = r_steps
        self.regularization = regularization
        self.epsilon_b = epsilon_b
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Derive independent child seeds from the master rng so the
        # extractor noise RNG and the sklearn random_state don't
        # inadvertently couple with sampling draws in self.rng.
        ss = np.random.SeedSequence(seed)
        self._extractor_seed, self._learner_seed = (
            int(s) for s in ss.generate_state(2, dtype=np.uint32)
        )

        self.builder, self.extractor, self.learner = self._route()

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route(self) -> Tuple[CircuitBuilder, FeatureExtractor, Learner]:
        """Direct dispatch from ``self.method`` to the concrete triple.

        Concrete class imports are lazy so that tests can stub subsystems
        without eagerly importing Qiskit.  Each factory below receives
        no extra information beyond ``self`` — the routing logic is
        purely a method-string → factory lookup.
        """
        factories: Dict[str, Callable[[], Tuple[CircuitBuilder, FeatureExtractor, Learner]]] = {
            "lasso-b_l":     self._build_lasso_b_l,
            "lasso-tensor":  self._build_lasso_tensor,
            "kernel-tensor": self._build_kernel_tensor,
        }
        return factories[self.method]()

    # --- Factory methods ------------------------------------------------

    def _build_lasso_b_l(self) -> Tuple[CircuitBuilder, FeatureExtractor, Learner]:
        from .circuits.shared_register import SharedRegisterBuilder
        from .features.hadamard_b_l import HadamardBLExtractor
        from .learners.lasso import LassoLearner

        builder = SharedRegisterBuilder()
        extractor = HadamardBLExtractor(
            builder,
            epsilon_b=self.epsilon_b,
            rng=np.random.default_rng(self._extractor_seed),
        )
        learner = LassoLearner(alpha=self.regularization, random_state=self._learner_seed)
        return builder, extractor, learner

    def _build_lasso_tensor(self) -> Tuple[CircuitBuilder, FeatureExtractor, Learner]:
        from .circuits.separate_registers import SeparateRegistersBuilder
        from .features.meshgrid_tensor import MeshgridTensorExtractor
        from .learners.lasso import LassoLearner

        builder = SeparateRegistersBuilder()
        extractor = MeshgridTensorExtractor(builder)
        learner = LassoLearner(alpha=self.regularization, random_state=self._learner_seed)
        return builder, extractor, learner

    def _build_kernel_tensor(self) -> Tuple[CircuitBuilder, FeatureExtractor, Learner]:
        from .circuits.separate_registers import SeparateRegistersBuilder
        from .features.meshgrid_tensor import MeshgridTensorExtractor
        from .learners.kernel_ridge import KernelRidgeLearner

        builder = SeparateRegistersBuilder()
        extractor = MeshgridTensorExtractor(builder)
        learner = KernelRidgeLearner(alpha=self.regularization)
        return builder, extractor, learner

    # ------------------------------------------------------------------
    # Label generation (lives on Experiment per design decision B)
    # ------------------------------------------------------------------

    def compute_exact_labels(
        self,
        xs: list,
        alpha_star: AlphaVector,
    ) -> np.ndarray:
        """Exact c_alpha(x) = <0| U^dag O U |0> via dense matrix exponentiation.

        Used as the reference curve ``y_true_exact`` for evaluation.
        """
        raise NotImplementedError(
            "Experiment.compute_exact_labels — concrete logic pending approval."
        )

    def compute_trotter_labels(
        self,
        xs: list,
        alpha_star: AlphaVector,
    ) -> np.ndarray:
        """A(U)-consistent Trotter labels — the in-convention training target.

        Generated through the same A(U) circuit the extractor uses so
        that training labels match the extractor's internal representation
        exactly (the invariant flagged in the legacy
        ``compute_au_labels_inhomogeneous`` docstring).
        """
        raise NotImplementedError(
            "Experiment.compute_trotter_labels — concrete logic pending approval."
        )

    # ------------------------------------------------------------------
    # End-to-end run
    # ------------------------------------------------------------------

    def run(self, num_train: int, num_test: int) -> ExperimentResult:
        """Full PAC pipeline: sample, extract, fit, predict, score.

        Pseudocode (fully expanded in the concrete impl):

        1. alpha_star   = model.sample_alpha(self.rng)
        2. xs_train     = [model.sample_x(self.rng) for _ in range(num_train)]
           xs_test      = [model.sample_x(self.rng) for _ in range(num_test)]
        3. y_train          = compute_trotter_labels(xs_train, alpha_star)
           y_true_trotter   = compute_trotter_labels(xs_test,  alpha_star)
           y_true_exact     = compute_exact_labels  (xs_test,  alpha_star)
        4. B_train = extractor.extract_batch(model, xs_train, tau, r_steps, obs, flatten=True)
           B_test  = extractor.extract_batch(model, xs_test,  tau, r_steps, obs, flatten=True)
        5. if method == "kernel-tensor":
               K_train = B_train @ B_train.T
               K_cross = B_test  @ B_train.T
               learner.fit(K_train, y_train); y_pred = learner.predict(K_cross)
           else:  # lasso-b_l or lasso-tensor
               learner.fit(B_train, y_train); y_pred = learner.predict(B_test)
        6. Score vs y_true_* and return ExperimentResult.
        """
        raise NotImplementedError(
            "Experiment.run — concrete logic pending approval."
        )

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        """Configuration bag mirroring what gets attached to ExperimentResult.metadata."""
        return {
            "model": repr(self.model),
            "d": self.model.d,
            "num_qubits": self.model.num_qubits,
            "observable": type(self.observable).__name__,
            "num_pauli_terms": len(self.observable),
            "method": self.method,
            "tau": self.tau,
            "r_steps": self.r_steps,
            "regularization": self.regularization,
            "epsilon_b": self.epsilon_b,
            "seed": self.seed,
        }
