"""Feature extractors — turn A(U, P) circuits into real Fourier tensors."""

from .base import FeatureExtractor
from .hadamard_b_l import HadamardBLExtractor
from .meshgrid_tensor import MeshgridTensorExtractor

__all__ = [
    "FeatureExtractor",
    "HadamardBLExtractor",
    "MeshgridTensorExtractor",
]
