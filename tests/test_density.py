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
