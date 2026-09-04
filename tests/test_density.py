import numpy as np
import pytest

from mini_dft.basis import PlaneWaveBasis
from mini_dft.density import density_from_coefficients, density_integral
from mini_dft.fft_grid import FFTGrid
from mini_dft.lattice import Lattice
from mini_dft.occupations import integer_occupations


def make_grid() -> FFTGrid:
    lattice = Lattice.from_row_vectors(np.eye(3) * 8.0)
    return FFTGrid.from_basis(PlaneWaveBasis.from_cutoff(lattice, encut=1.0))


def test_integer_occupations_fill_lowest_spatial_orbitals():
    """Catch occupations that use flags instead of two electrons per orbital."""
    assert np.array_equal(integer_occupations(4, 4), np.array([2.0, 2.0, 0.0, 0.0]))


@pytest.mark.parametrize("electrons, bands", [(3, 2), (-2, 2), (6, 2)])
def test_integer_occupations_reject_invalid_electron_requests(electrons, bands):
    """Catch odd, negative, or overfilled integer-occupation requests."""
    with pytest.raises(ValueError, match="electron|band"):
        integer_occupations(electrons, bands)


def test_density_from_orthonormal_coefficients_preserves_charge():
    """Catch a density transform that drops bands, occupations, or FFT scaling."""
    grid = make_grid()
    coefficients = np.eye(grid.basis.npw, dtype=np.complex128)[:2]
    occupations = np.array([2.0, 2.0])

    density = density_from_coefficients(coefficients, occupations, grid)

    assert np.isrealobj(density)
    assert np.all(density >= 0.0)
    assert density_integral(density, grid) == pytest.approx(4.0, abs=1e-12)


@pytest.mark.parametrize("occupations", [np.array([1.0]), np.array([1.5])])
def test_density_rejects_noninteger_spatial_orbital_occupations(occupations):
    """Catch fractional occupations accepted outside the v0.1 2-or-0 domain."""
    grid = make_grid()
    coefficients = np.eye(grid.basis.npw, dtype=np.complex128)[:1]

    with pytest.raises(ValueError, match="occupations"):
        density_from_coefficients(coefficients, occupations, grid)


def test_density_rejects_occupations_with_nonzero_imaginary_parts():
    """Catch float coercion that discards an invalid occupation's imaginary part."""
    grid = make_grid()
    coefficients = np.eye(grid.basis.npw, dtype=np.complex128)[:1]

    with pytest.raises(ValueError, match="occupations"):
        density_from_coefficients(coefficients, np.array([2.0 + 1.0j]), grid)


@pytest.mark.parametrize(
    "density",
    [
        lambda grid: np.zeros((1, *grid.shape)),
        lambda grid: np.full(grid.shape, "not-a-density", dtype=object),
    ],
)
def test_density_integral_rejects_non_scalar_or_unconvertible_fields(density):
    """Catch batched fields and raw NumPy conversion failures at the public boundary."""
    grid = make_grid()

    with pytest.raises(ValueError, match="density"):
        density_integral(density(grid), grid)
