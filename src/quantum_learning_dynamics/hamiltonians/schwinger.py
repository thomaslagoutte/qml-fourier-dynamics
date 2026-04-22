"""1D Z_2 Lattice Gauge Theory (Staggered Schwinger Model)."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from .._types import AlphaVector, InputX, PauliString
from .base import HamiltonianModel


class SchwingerZ2Model(HamiltonianModel):
    """1D Z_2 staggered Schwinger model with dynamical gauge links.

    The physical system is modeled via interlaced matter and gauge-link qubits.
    The Hamiltonian features a staggered fermion mass, a background electric
    field, and structurally variable gauge-invariant hopping terms:

    .. math::

        H(x, g) = m \\sum_i (-1)^i Z_{m_i} + \\varepsilon \\sum_l X_{l_i} 
                  + \\sum_{l \\in x} g_l X_{m_l} Z_{link_l} X_{m_{l+1}}

    The structural input ``x`` acts as a binary mask activating gauge links, 
    and the unknown parameters ``g`` represent the coupling strengths of 
    the active links (``d > 1`` regime).

    Parameters
    ----------
    num_matter : int
        The number of matter sites. The total qubit count will be
        ``2 * num_matter - 1``.
    mass : float, default=0.5
        The staggered fermion mass :math:`m`.
    electric_field : float, default=1.0
        The background electric field :math:`\\varepsilon`.
    g_range : Tuple[float, float], default=(0.5, 1.5)
        The uniform sampling domain for the unknown gauge couplings :math:`g_l`.
    """

    def __init__(
        self,
        num_matter: int,
        mass: float = 0.5,
        electric_field: float = 1.0,
        g_range: Tuple[float, float] = (0.5, 1.5),
    ) -> None:
        if num_matter < 2:
            raise ValueError(f"Schwinger requires num_matter >= 2, got {num_matter}")
        
        self.num_matter = num_matter
        self.num_links = num_matter - 1
        
        self.num_qubits = self.num_matter + self.num_links
        self.d = self.num_links
        
        self.mass = float(mass)
        self.electric_field = float(electric_field)
        self.g_range = g_range

    def matter_qubit(self, i: int) -> int:
        """Returns the logical qubit index for matter site i."""
        return 2 * i

    def link_qubit(self, l: int) -> int:
        """Returns the logical qubit index for gauge link l."""
        return 2 * l + 1

    def _single(self, op: str, target: int) -> str:
        """Constructs a single-qubit Pauli operator (little-endian)."""
        chars = ["I"] * self.num_qubits
        chars[self.num_qubits - 1 - target] = op
        return "".join(chars)

    def _xzx(self, l: int) -> str:
        """Constructs the XZX gauge-invariant hopping operator for link l."""
        m_left = self.matter_qubit(l)
        link = self.link_qubit(l)
        m_right = self.matter_qubit(l + 1)
        
        chars = ["I"] * self.num_qubits
        chars[self.num_qubits - 1 - m_left] = "X"
        chars[self.num_qubits - 1 - link] = "Z"
        chars[self.num_qubits - 1 - m_right] = "X"
        
        return "".join(chars)

    @property
    def upload_paulis(self) -> List[PauliString]:
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

        # Staggered mass on matter sites: -m Σ_i (-1)^i Z_{m_i}
        for i in range(self.num_matter):
            sign = -1.0 if (i % 2) else 1.0
            terms.append((self._single("Z", self.matter_qubit(i)), -self.mass * sign))

        # Electric field on gauge links: +ε Σ_l X_{link_l}
        for l in range(self.num_links):
            terms.append((self._single("X", self.link_qubit(l)), self.electric_field))

        # Gauge-invariant hopping: Σ_{l active} g_l * XZX
        for l in range(self.num_links):
            if active[l]:
                terms.append((self._xzx(l), float(alpha[l])))

        return SparsePauliOp.from_list(terms)

    def sample_x(self, rng: np.random.Generator) -> InputX:
        # Returns a binary mask where each link has a 50% probability of being active
        return [bool(rng.integers(0, 2)) for _ in range(self.num_links)]

    def sample_alpha(self, rng: np.random.Generator) -> AlphaVector:
        return rng.uniform(self.g_range[0], self.g_range[1], size=self.d)