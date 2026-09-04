"""Lowest-state solvers for matrix-free plane-wave Hamiltonians."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from .hamiltonian import Hamiltonian
from .wavefunction import normalize_coefficients


@dataclass(frozen=True)
class EigenResult:
    """Eigenpairs and their Euclidean residual norms."""

    eigenvalues: np.ndarray
    coefficients: np.ndarray
    residual_norms: np.ndarray


def solve_lowest(
    hamiltonian: Hamiltonian,
    n_bands: int,
    tolerance: float,
    max_iterations: int,
) -> EigenResult:
    """Solve for the requested number of lowest Hamiltonian eigenstates."""
    npw = hamiltonian.basis.npw
    if (
        not isinstance(n_bands, (int, np.integer))
        or isinstance(n_bands, bool)
        or not 1 <= n_bands <= npw
    ):
        raise ValueError(f"n_bands: expected an integer in [1, {npw}]")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance: expected a positive finite number")
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations: expected a positive integer")

    if n_bands < npw - 1:
        spectral_scale = max(
            1.0,
            float(np.max(np.abs(hamiltonian.basis.kinetic_energies))),
            float(np.max(np.abs(hamiltonian.local_potential))),
        )
        spectral_shift = spectral_scale / np.pi
        shifted_operator = LinearOperator(
            shape=(npw, npw),
            matvec=lambda vector: hamiltonian.apply(vector) + spectral_shift * vector,
            dtype=np.complex128,
        )
        eigenvalues, eigenvectors = eigsh(
            shifted_operator,
            k=n_bands,
            which="SA",
            tol=tolerance,
            maxiter=max_iterations,
        )
        eigenvalues -= spectral_shift
    else:
        identity = np.eye(npw, dtype=np.complex128)
        dense_hamiltonian = hamiltonian.apply_many(identity).T
        eigenvalues, eigenvectors = np.linalg.eigh(dense_hamiltonian)
        eigenvalues = eigenvalues[:n_bands]
        eigenvectors = eigenvectors[:, :n_bands]

    order = np.argsort(eigenvalues)
    eigenvalues = np.asarray(eigenvalues[order], dtype=float)
    coefficients = _orthonormalize_degenerate_states(
        eigenvalues, eigenvectors[:, order].T, tolerance
    )
    residual_norms = np.linalg.norm(
        hamiltonian.apply_many(coefficients) - eigenvalues[:, np.newaxis] * coefficients,
        axis=1,
    )
    return EigenResult(
        eigenvalues=eigenvalues,
        coefficients=coefficients,
        residual_norms=np.asarray(residual_norms, dtype=float),
    )


def _orthonormalize_degenerate_states(
    eigenvalues: np.ndarray, coefficients: np.ndarray, tolerance: float
) -> np.ndarray:
    """Orthonormalize only eigensolver vectors sharing an eigenvalue."""
    normalized = normalize_coefficients(coefficients)
    start = 0
    while start < len(eigenvalues):
        stop = start + 1
        threshold = max(10.0 * tolerance, 100.0 * np.finfo(float).eps)
        while stop < len(eigenvalues) and abs(eigenvalues[stop] - eigenvalues[start]) <= threshold:
            stop += 1
        if stop - start > 1:
            orthonormal_columns, _ = np.linalg.qr(normalized[start:stop].T)
            normalized[start:stop] = orthonormal_columns.T
        start = stop
    return normalized
