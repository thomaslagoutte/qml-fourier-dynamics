"""Full-tensor Fourier coefficient extractor — d > 1 regime.

IMMUTABLE PHYSICS CONSTRAINT — MESHGRID TENSOR EXTRACTION
---------------------------------------------------------
When extracting features for separate frequency registers (d > 1) you
CANNOT marginalise or concatenate the statevector amplitudes across
registers.  You must use :func:`numpy.meshgrid` to carve out the full
rank-d tensor of shape ``(4r + 1,) * d`` directly from the Qiskit
statevector.

The correct indexing procedure (verbatim from the legacy
``KernelDynamicsLearner._extract_b_vector_sim``):

    da_stride = 2 ** (num_qubits + 1)        # data + ancilla
    freq_dim  = 2 ** n_s                     # one frequency register
    freq_1d   = arange(-2r, 2r + 1)          # signed freq spectrum

    mesh = np.meshgrid(*[freq_1d] * d, indexing='ij')
    sv_indices = sum_k (mesh[k] % freq_dim) * (da_stride * freq_dim ** k)
    b(x) = sv[sv_indices].real

Flattening or marginalising earlier than this throws away the
joint-frequency information that the kernel ``k(x, x') = <b(x), b(x')>``
depends on.

Concrete logic sourced EXCLUSIVELY from ``quantum_routines_kernel.py`` +
``learning_kernel.py``.  The d > 1 logic in the old
``quantum_routines.py`` / ``learning.py`` is dead code.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from qiskit.quantum_info import Statevector

from .._types import FeatureTensor, InputX, PauliString
from .base import FeatureExtractor


class MeshgridTensorExtractor(FeatureExtractor):
    """d>1 extraction (one frequency register per qubit).

    For a single Pauli term P_h, returns the flattened d-dimensional real
    Fourier tensor
        b_h(x)[l_1, ..., l_d] = Re <psi_0| U(x, α)† P_h U(x, α) |psi_0>_{l_1,...,l_d}
    with each l_k in {-2r, ..., +2r}. Flattened length = (4r + 1) ** d.

    STRICT np.meshgrid logic — no marginalization, no per-dim concatenation,
    no partial flattening over individual axes. The full d-dimensional
    entanglement between frequency registers is preserved and read out in one
    statevector lookup.

    Statevector index layout (Qiskit little-endian, declaration order
    data → anc → freq_0 → freq_1 → ... → freq_{d-1}):
        idx = data + anc*2^n + freq_0*da_stride + freq_1*(da_stride*freq_dim)
                                + ... + freq_{d-1}*(da_stride*freq_dim^(d-1))
    Reading the (data = 0, anc = 0) sector and signed frequencies l_k:
        idx = Σ_k  (l_k mod freq_dim) * (da_stride * freq_dim^k)
    """

    def __init__(
        self,
        builder,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(builder)
        # rng retained for downstream noise modelling / API symmetry with HadamardBLExtractor.
        self.rng = rng if rng is not None else np.random.default_rng()

    def extract_single_pauli(
        self,
        x: InputX,
        tau: float,
        r_steps: int,
        pauli: PauliString,
    ) -> FeatureTensor: 
        num_qubits = self.builder.model.num_qubits
        d = self.builder.model.d                         # dynamically route the tensor rank
        n_s = self.builder.freq_register_size(num_qubits, r_steps)

        qc, _freq_regs = self.builder.build_aup(
            num_qubits=num_qubits,
            x=x,
            tau=tau,
            r_steps=r_steps,
            pauli=pauli,
        )
        sv = Statevector(qc).data

        da_stride = 2 ** (num_qubits + 1)                # data(n) + anc(1)
        freq_dim  = 2 ** n_s
        freq_1d   = np.arange(-2 * r_steps, 2 * r_steps + 1)

        # STRICT np.meshgrid — indexing='ij' so axis k corresponds to qubit k.
        # Do NOT replace with np.ix_, np.outer, or per-dim flattening; those
        # shortcuts silently marginalize off-diagonal Fourier structure.
        mesh = np.meshgrid(*[freq_1d] * d, indexing="ij")

        sv_indices = np.zeros_like(mesh[0], dtype=np.int64)
        for k in range(d):
            reg_stride = da_stride * (freq_dim ** k)
            sv_indices = sv_indices + (mesh[k] % freq_dim) * reg_stride

        # One lookup on the flattened index grid; returns a 1D feature vector
        # of length (4r + 1) ** d (sklearn-compatible).
        b_h = np.ascontiguousarray(sv[sv_indices.ravel()].real, dtype=np.float64)
        return b_h