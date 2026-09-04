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
        spectral_shift = np.finfo(float).eps
        shifted_operator = LinearOperator(
            shape=(npw, npw),
            matvec=lambda vector: hamiltonian.apply(vector) + spectral_shift * vector,
            dtype=np.complex128,
        )
        _, eigenvectors = eigsh(
            shifted_operator,
            k=n_bands,
            which="SA",
            tol=tolerance,
            maxiter=max_iterations,
        )
    else:
        identity = np.eye(npw, dtype=np.complex128)
        dense_hamiltonian = hamiltonian.apply_many(identity).T
        _, eigenvectors = np.linalg.eigh(dense_hamiltonian)
        eigenvectors = eigenvectors[:, :n_bands]

    eigenvalues, coefficients = _rayleigh_ritz_refinement(hamiltonian, eigenvectors)
    residual_norms = np.linalg.norm(
        hamiltonian.apply_many(coefficients) - eigenvalues[:, np.newaxis] * coefficients,
        axis=1,
    )
    _validate_unshifted_residuals(residual_norms, tolerance, n_bands)
    return EigenResult(
        eigenvalues=np.asarray(eigenvalues, dtype=float),
        coefficients=coefficients,
        residual_norms=np.asarray(residual_norms, dtype=float),
    )


def _rayleigh_ritz_refinement(
    hamiltonian: Hamiltonian, candidate_vectors: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize the original Hamiltonian in a rank-checked candidate subspace."""
    candidates = np.asarray(candidate_vectors, dtype=np.complex128)
    singular_values = np.linalg.svd(candidates, compute_uv=False)
    rank_threshold = (
        np.finfo(float).eps * max(candidates.shape) * singular_values[0]
    )
    if singular_values[-1] <= rank_threshold:
        raise RuntimeError("eigensolver: candidate subspace is rank deficient")

    orthonormal_basis, _ = np.linalg.qr(candidates)
    applied_basis = hamiltonian.apply_many(orthonormal_basis.T).T
    projected = orthonormal_basis.conj().T @ applied_basis
    projected = 0.5 * (projected + projected.conj().T)
    eigenvalues, rotations = np.linalg.eigh(projected)
    coefficients = normalize_coefficients((orthonormal_basis @ rotations).T)
    return np.asarray(eigenvalues, dtype=float), coefficients


def _validate_unshifted_residuals(
    residual_norms: np.ndarray, tolerance: float, n_bands: int
) -> None:
    """Reject eigenpairs that do not meet an absolute residual-quality bound."""
    residual_limit = max(tolerance, 100.0 * np.finfo(float).eps)
    maximum_residual = float(np.max(residual_norms))
    if not np.isfinite(maximum_residual) or maximum_residual > residual_limit:
        raise RuntimeError(
            "eigensolver: unshifted residual "
            f"{maximum_residual:.3e} exceeds {residual_limit:.3e} "
            f"for {n_bands} requested bands"
        )
