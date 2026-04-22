"""Quantum execution engines for feature extraction and kernel evaluation.

This module exposes the unified execution layer for the PAC-learning framework.
It handles the translation of high-level feature or kernel requests into
quantum circuit evaluations across exact statevectors, noisy emulators, and
physical QPUs.
"""

from .engines import FeatureEngine, KernelEngine

__all__ = [
    "FeatureEngine",
    "KernelEngine",
]