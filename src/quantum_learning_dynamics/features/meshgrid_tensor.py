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

from .._types import FeatureTensor, InputX, PauliString
from ..hamiltonians.base import HamiltonianModel
from .base import FeatureExtractor


class MeshgridTensorExtractor(FeatureExtractor):
    """Full-tensor Fourier extractor for d > 1 using np.meshgrid indexing."""

    def extract_single_pauli(
        self,
        model: HamiltonianModel,
        x: InputX,
        tau: float,
        r_steps: int,
        pauli: PauliString,
    ) -> FeatureTensor:
        raise NotImplementedError(
            "MeshgridTensorExtractor.extract_single_pauli — concrete logic pending approval."
        )
