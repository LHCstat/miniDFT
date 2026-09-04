import numpy as np
import pytest

from mini_dft.basis import PlaneWaveBasis
from mini_dft.fft_grid import FFTGrid
from mini_dft.lattice import Lattice
from mini_dft.wavefunction import (
    normalize_coefficients,
    orthonormality_error,
    overlap_matrix,
)


def make_small_basis() -> PlaneWaveBasis:
    """Build a nontrivial cubic basis for Fourier-transform tests."""
    lattice = Lattice.from_row_vectors(np.eye(3) * 8.0)
    return PlaneWaveBasis.from_cutoff(lattice, encut=1.0)


def test_wavefunction_round_trip_and_parseval():
    """Catch incorrect plane-wave FFT scaling or reciprocal-index mapping."""
    basis = make_small_basis()
    grid = FFTGrid.from_basis(basis)
    rng = np.random.default_rng(42)
    coeff = rng.normal(size=basis.npw) + 1j * rng.normal(size=basis.npw)
    coeff /= np.linalg.norm(coeff)

    psi = grid.coefficients_to_real(coeff)

    assert grid.integrate(np.abs(psi) ** 2) == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(grid.real_to_coefficients(psi), coeff, atol=1e-12)


def test_scalar_field_round_trip_uses_its_own_fourier_normalization():
    """Catch accidental reuse of normalized-wavefunction scaling for fields."""
    grid = FFTGrid.from_basis(make_small_basis())
    rng = np.random.default_rng(7)
    field = rng.normal(size=grid.shape)

    fourier = grid.field_to_fourier(field)

    assert np.allclose(grid.fourier_to_field(fourier), field, atol=1e-12)
    assert fourier[(0, 0, 0)] == pytest.approx(np.mean(field), abs=1e-12)


def test_batched_wavefunction_transform_preserves_leading_band_dimensions():
    """Catch transforms that flatten bands or apply FFTs along band axes."""
    basis = make_small_basis()
    grid = FFTGrid.from_basis(basis)
    rng = np.random.default_rng(9)
    coefficients = rng.normal(size=(2, 3, basis.npw)) + 1j * rng.normal(
        size=(2, 3, basis.npw)
    )

    real_space = grid.coefficients_to_real(coefficients)

    assert real_space.shape == (2, 3, *grid.shape)
    assert np.allclose(
        grid.real_to_coefficients(real_space), coefficients, atol=1e-12
    )


def test_grid_dimensions_oversample_every_basis_index_extent():
    """Catch undersized grids that alias products of represented plane waves."""
    basis = make_small_basis()
    grid = FFTGrid.from_basis(basis)

    assert all(
        size >= 4 * maximum + 1
        for size, maximum in zip(grid.shape, basis.max_abs_indices, strict=True)
    )


def test_grid_geometry_helpers_follow_fractional_and_reciprocal_conventions():
    """Catch grid-coordinate helpers that use Cartesian spacing or omit G=0."""
    basis = make_small_basis()
    grid = FFTGrid.from_basis(basis)

    fractional = grid.fractional_mesh()
    reciprocal_squared = grid.reciprocal_squared()

    assert fractional.shape == (*grid.shape, 3)
    assert np.array_equal(fractional[0, 0, 0], np.zeros(3))
    assert reciprocal_squared.shape == grid.shape
    assert reciprocal_squared[0, 0, 0] == pytest.approx(0.0)
    assert reciprocal_squared[1, 0, 0] == pytest.approx(
        np.dot(basis.lattice.B[:, 0], basis.lattice.B[:, 0])
    )


def test_wavefunction_helpers_normalize_and_measure_orthonormality():
    """Catch normalization over the wrong axis or an incorrect overlap metric."""
    coefficients = np.array([[3.0 + 4.0j, 0.0], [0.0, 2.0j]])

    normalized = normalize_coefficients(coefficients)

    assert np.allclose(np.linalg.norm(normalized, axis=-1), 1.0)
    assert np.allclose(overlap_matrix(normalized), np.eye(2))
    assert orthonormality_error(normalized) == pytest.approx(0.0, abs=1e-12)


def test_normalize_coefficients_rejects_zero_norm_vectors():
    """Catch silent propagation of invalid all-zero wavefunctions."""
    with pytest.raises(ValueError, match="coefficients: zero-norm wavefunction"):
        normalize_coefficients(np.zeros((2, 3)))
