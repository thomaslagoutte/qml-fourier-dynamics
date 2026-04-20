"""Hadamard-test Fourier coefficient extractor — d = 1 regime.

Extracts b_l(x) = Re <psi_0 | U(x, alpha)^dag P U(x, alpha) | psi_0>_{l}
for each signed frequency l in L = {-2r, ..., +2r} using the Hadamard test
on the shared-register A(U, P) circuit (Figure 7 / Corollary 2 of the
paper).  Returns a 1-D feature vector of length ``4r + 1``.

Concrete implementation pulled from the d = 1 logic of the legacy
``FourierDynamicsLearner._extract_b_l`` and
``FourierDynamicsLearner.extract_fourier_features``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from qiskit.quantum_info import Statevector

from .._types import FeatureTensor, InputX, PauliString
from .base import FeatureExtractor


class HadamardBLExtractor(FeatureExtractor):
    """d=1 extraction (shared frequency register).

    For a single Pauli term P_h, returns the real Fourier vector
        b_h(x)[l] = Re <psi_0| U(x,alpha)† P_h U(x,alpha) |psi_0>_l
    for l in {-2 n r, ..., +2 n r}, so the output has length 4 n r + 1.

    The A(U, P_h) circuit places the per-l coefficients as computational-basis
    amplitudes on the (data = 0, anc = 0) sector of the shared freq register.
    Statevector index layout (Qiskit little-endian, registers declared as
    data → anc → freq):
        idx = data + anc * 2^n + freq_value * 2^(n+1)
    So for data = 0, anc = 0:
        idx(l) = (l mod 2^n_s) * da_stride        with da_stride = 2^(n+1)

    Optional shot-noise model: uniform noise in [-epsilon_b, +epsilon_b] is
    added per coefficient, sampled from self.rng (seeded by Experiment).
    """

    def __init__(
        self,
        builder,
        epsilon_b: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(builder)
        if epsilon_b < 0.0:
            raise ValueError(f"epsilon_b must be non-negative, got {epsilon_b}")
        self.epsilon_b = float(epsilon_b)
        self.rng = rng if rng is not None else np.random.default_rng()

    def extract_single_pauli(
        self,
        x: InputX,
        tau: float,
        r_steps: int,
        pauli: PauliString,
    ) -> FeatureTensor:
        num_qubits = len(pauli)
        n_s = self.builder.freq_register_size(num_qubits, r_steps)

        qc, _freq_regs = self.builder.build_aup(
            num_qubits=num_qubits,
            x=x,
            tau=tau,
            r_steps=r_steps,
            pauli=pauli,
        )
        sv = Statevector(qc).data

        da_stride = 2 ** (num_qubits + 1)      # data(n) + anc(1)
        freq_dim  = 2 ** n_s
        max_freq  = 2 * num_qubits * r_steps   # full shared-register band

        freqs   = np.arange(-max_freq, max_freq + 1)
        indices = (freqs % freq_dim) * da_stride
        b_h     = np.ascontiguousarray(sv[indices].real, dtype=np.float64)

        if self.epsilon_b > 0.0:
            b_h = b_h + self.rng.uniform(-self.epsilon_b, self.epsilon_b, size=b_h.shape)

        return b_h