import numpy as np
import pytest

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
