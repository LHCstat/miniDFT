import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator
from types import SimpleNamespace

from mini_dft.basis import PlaneWaveBasis
from mini_dft.eigensolver import solve_lowest
from mini_dft.fft_grid import FFTGrid
from mini_dft.hamiltonian import Hamiltonian
from mini_dft.lattice import Lattice
from mini_dft.wavefunction import orthonormality_error


def make_hamiltonian(constant: float = 0.0) -> Hamiltonian:
    """Build a basis large enough to exercise the sparse solver path."""
    lattice = Lattice.from_row_vectors(np.eye(3) * 8.0)
    basis = PlaneWaveBasis.from_cutoff(lattice, encut=1.0)
    grid = FFTGrid.from_basis(basis)
    return Hamiltonian(basis, grid, np.full(grid.shape, constant))


class DiagonalHamiltonian:
    """Small matrix-free diagonal operator for solver numerical regressions."""

    def __init__(self, diagonal: np.ndarray) -> None:
        self.diagonal = np.asarray(diagonal, dtype=float)
        self.basis = SimpleNamespace(
            npw=self.diagonal.size, kinetic_energies=self.diagonal
        )
        self.local_potential = np.zeros(1)

    def apply(self, coefficients: np.ndarray) -> np.ndarray:
        return self.diagonal * np.asarray(coefficients, dtype=np.complex128)

    def apply_many(self, coefficients: np.ndarray) -> np.ndarray:
        return self.diagonal * np.asarray(coefficients, dtype=np.complex128)

    def as_linear_operator(self) -> LinearOperator:
        return LinearOperator(
            (self.basis.npw, self.basis.npw), matvec=self.apply, dtype=np.complex128
        )


def test_solver_returns_lowest_free_electron_states_with_small_residuals():
    """Catch wrong sparse ordering, band orientation, or residual calculation."""
    hamiltonian = make_hamiltonian()
    n_bands = 3

    result = solve_lowest(
        hamiltonian, n_bands=n_bands, tolerance=1e-11, max_iterations=1000
    )

    assert np.allclose(
        result.eigenvalues,
        np.sort(hamiltonian.basis.kinetic_energies)[:n_bands],
        atol=1e-10,
    )
    assert result.coefficients.shape == (n_bands, hamiltonian.basis.npw)
    assert orthonormality_error(result.coefficients) < 1e-10
    assert np.max(result.residual_norms) < 1e-8


def test_constant_potential_shifts_every_requested_eigenvalue():
    """Catch a local-potential term omitted from the iterative eigenproblem."""
    n_bands = 3
    zero = solve_lowest(make_hamiltonian(), n_bands, 1e-11, 1000)
    shift = -0.23
    shifted = solve_lowest(make_hamiltonian(shift), n_bands, 1e-11, 1000)

    assert np.allclose(shifted.eigenvalues, zero.eigenvalues + shift, atol=1e-10)
    assert np.max(shifted.residual_norms) < 1e-8


def test_dense_fallback_returns_the_complete_small_spectrum():
    """Catch a transposed dense matrix or an omitted final eigensolver state."""
    hamiltonian = make_hamiltonian()

    result = solve_lowest(hamiltonian, hamiltonian.basis.npw, 1e-11, 1000)

    assert np.allclose(
        result.eigenvalues, np.sort(hamiltonian.basis.kinetic_energies), atol=1e-12
    )
    assert orthonormality_error(result.coefficients) < 1e-10
    assert np.max(result.residual_norms) < 1e-8


def test_solver_rejects_noninteger_or_out_of_range_band_counts():
    """Catch invalid requests before passing an unusable count to SciPy."""
    hamiltonian = make_hamiltonian()

    with pytest.raises(ValueError, match="n_bands"):
        solve_lowest(hamiltonian, 0, 1e-11, 1000)
    with pytest.raises(ValueError, match="n_bands"):
        solve_lowest(hamiltonian, hamiltonian.basis.npw + 1, 1e-11, 1000)
    with pytest.raises(ValueError, match="n_bands"):
        solve_lowest(hamiltonian, 1.5, 1e-11, 1000)


def test_sparse_solver_rejects_large_scale_candidates_with_poor_residuals():
    """Catch cancellation that hides inaccurate unshifted large-scale states."""
    hamiltonian = DiagonalHamiltonian(1.0e16 + 4.0 * np.arange(8))

    with pytest.raises(RuntimeError, match="unshifted residual"):
        solve_lowest(hamiltonian, n_bands=2, tolerance=1e-12, max_iterations=1000)


def test_sparse_solver_refines_close_nondegenerate_candidate_subspace(monkeypatch):
    """Catch QR rotations that retain stale eigenvalues for close distinct states."""
    import mini_dft.eigensolver as eigensolver

    separation = 5.0e-8
    hamiltonian = DiagonalHamiltonian(np.array([0.0, separation, 1.0, 2.0]))

    def close_nonorthogonal_candidates(*_args, **_kwargs):
        candidates = np.zeros((hamiltonian.basis.npw, 2), dtype=np.complex128)
        candidates[:2, 0] = (1.0, 1.0)
        candidates[:2, 1] = (1.0, -1.0)
        return np.array([0.0, separation]), candidates / np.sqrt(2.0)

    monkeypatch.setattr(eigensolver, "eigsh", close_nonorthogonal_candidates)

    result = solve_lowest(hamiltonian, n_bands=2, tolerance=1e-8, max_iterations=100)

    assert np.allclose(result.eigenvalues, [0.0, separation], atol=1e-14)
    assert orthonormality_error(result.coefficients) < 1e-12
    assert np.max(result.residual_norms) < 1e-12
