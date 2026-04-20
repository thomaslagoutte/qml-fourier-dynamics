"""Z2 Schwinger / 1-D lattice gauge-theory Hamiltonian.

Model (interlaced matter / gauge-link layout, Qiskit little-endian):

    qubit 0: matter site 0
    qubit 1: gauge link between matter 0 and 1
    qubit 2: matter site 1
    qubit 3: gauge link between matter 1 and 2
    ...

    H(x, g) = m * sum_i (-1)^i Z_{m_i}                       (staggered mass, known)
            + h * sum_i X_{l_i}                              (electric field, known)
            + sum_{i: x_i == 1} g_i * X_{m_i} Z_{l_i} X_{m_{i+1}}   (XZX coupling, unknown)

* ``x`` is a gauge-link activation mask ``list[int]`` of length
  ``num_matter_sites - 1`` with entries in {0, 1}.  ``x_i = 1`` means link
  i is active and has a coupling g_i > 0 to be learned.
* ``alpha`` is the couplings vector ``g = (g_0, ..., g_{L-1})`` of shape
  ``(num_matter_sites - 1,)``.  (Concrete implementation may choose to
  treat only active links as unknowns; this is a detail resolved at
  implementation time.)

Placed in the d > 1 regime; routed through the separate-registers
circuit and either Lasso-on-tensor or the kernel pipeline depending on
the user's ``method=`` choice.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from .._types import AlphaVector, InputX, PauliString
from .base import HamiltonianModel


class SchwingerZ2Model(HamiltonianModel):
    """1-D Z₂ lattice-gauge Schwinger model (Kogut–Susskind, staggered).

    Layout:
        N matter sites + (N - 1) gauge links, interleaved
        qubit index:  0        1        2        3        ...       2N-2
                      matter_0 link_0   matter_1 link_1   ...       matter_{N-1}
        num_qubits = 2 N - 1
        num_links  = N - 1

    Hamiltonian:
        H(x, g) = -m Σ_i (-1)^i Z_{matter_i}              # staggered mass
                 + ε  Σ_l X_{link_l}                       # electric field
                 +    Σ_{l ∈ active(x)} g_l · X_{m_l} Z_{link_l} X_{m_{l+1}}
                                                          # hopping, gauge-invariant

    Unknowns: g_l for each of the N - 1 gauge links, so d = N - 1.
    Input x: boolean indicator per link — which hopping terms are active this sample.

    `upload_paulis[l] = XZX string on (m_l, link_l, m_{l+1})` is weight-3, so the
    CircuitBuilder's D block must parity-fold these three qubits onto anc[0] before
    applying the controlled V± shift into freqs[l].
    """

    def __init__(
        self,
        num_matter: int,
        mass: float = 0.5,
        electric_field: float = 1.0,
        link_prob: float = 0.8,
        g_range: Tuple[float, float] = (0.5, 1.5),
    ) -> None:
        if num_matter < 2:
            raise ValueError(f"Schwinger requires num_matter >= 2, got {num_matter}")
        self.num_matter = num_matter
        self.num_links = num_matter - 1
        self.num_qubits = 2 * num_matter - 1
        self.d = self.num_links
        self.mass = float(mass)
        self.electric_field = float(electric_field)
        self.link_prob = float(link_prob)
        self.g_range = g_range

    # --- qubit-index helpers -------------------------------------------------

    def matter_qubit(self, i: int) -> int:
        """Qubit index of matter site i, i ∈ [0, num_matter)."""
        return 2 * i

    def link_qubit(self, l: int) -> int:
        """Qubit index of gauge link l (between matter sites l and l+1)."""
        return 2 * l + 1

    # --- Pauli-string builders (little-endian) -------------------------------

    def _single(self, op: str, q: int) -> str:
        chars = ["I"] * self.num_qubits
        chars[self.num_qubits - 1 - q] = op
        return "".join(chars)

    def _xzx(self, l: int) -> str:
        """XZX on (matter_l, link_l, matter_{l+1})."""
        chars = ["I"] * self.num_qubits
        chars[self.num_qubits - 1 - self.matter_qubit(l)]     = "X"
        chars[self.num_qubits - 1 - self.link_qubit(l)]       = "Z"
        chars[self.num_qubits - 1 - self.matter_qubit(l + 1)] = "X"
        return "".join(chars)

    # --- abstract-method implementations -------------------------------------

    @property
    def upload_paulis(self) -> List[PauliString]:
        # d = N - 1 weight-3 XZX uploads, one per gauge link.
        return [self._xzx(l) for l in range(self.num_links)]

    def hamiltonian(self, x: InputX, alpha: AlphaVector) -> SparsePauliOp:
        if len(alpha) != self.d:
            raise ValueError(f"Schwinger expects |α| == {self.d}, got {len(alpha)}")
        active = list(x)
        if len(active) != self.num_links:
            raise ValueError(
                f"x must have length num_links={self.num_links}, got {len(active)}"
            )

        terms: List[Tuple[str, float]] = []

        # Staggered mass on matter sites:  -m Σ_i (-1)^i Z_{m_i}
        for i in range(self.num_matter):
            sign = -1.0 if (i % 2) else 1.0
            terms.append((self._single("Z", self.matter_qubit(i)), -self.mass * sign))

        # Electric field on gauge links:  +ε Σ_l X_{link_l}
        for l in range(self.num_links):
            terms.append((self._single("X", self.link_qubit(l)), self.electric_field))

        # Gauge-invariant hopping:  Σ_{l active} g_l · XZX
        for l in range(self.num_links):
            if active[l]:
                terms.append((self._xzx(l), float(alpha[l])))

        return SparsePauliOp.from_list(terms)

    def sample_x(self, rng: np.random.Generator) -> InputX:
        # Bernoulli(link_prob) per link.
        return [bool(rng.random() < self.link_prob) for _ in range(self.num_links)]

    def sample_alpha(self, rng: np.random.Generator) -> AlphaVector:
        lo, hi = self.g_range
        return rng.uniform(lo, hi, size=self.d)