"""Helpers for normalized plane-wave wavefunctions."""

import numpy as np


def normalize_coefficients(coefficients: np.ndarray) -> np.ndarray:
    """Normalize one wavefunction or a batch along its coefficient axis."""
    coeff = np.asarray(coefficients, dtype=np.complex128)
    norms = np.linalg.norm(coeff, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("coefficients: zero-norm wavefunction")
    return coeff / norms


def overlap_matrix(coefficients: np.ndarray) -> np.ndarray:
    """Return the band-by-band inner-product matrix."""
    coeff = np.asarray(coefficients, dtype=np.complex128)
    return coeff.conj() @ coeff.T


def orthonormality_error(coefficients: np.ndarray) -> float:
    """Return the largest absolute deviation of the overlap matrix from identity."""
    overlap = overlap_matrix(coefficients)
    identity = np.eye(overlap.shape[0], dtype=overlap.dtype)
    return float(np.max(np.abs(overlap - identity)))
