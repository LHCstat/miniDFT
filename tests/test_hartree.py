import numpy as np
import pytest

from mini_dft.basis import PlaneWaveBasis
from mini_dft.fft_grid import FFTGrid
from mini_dft.hartree import hartree_from_density
from mini_dft.lattice import Lattice


def make_grid() -> FFTGrid:
    """Build a cubic grid with an exactly represented first cosine mode."""
    lattice = Lattice.from_row_vectors(np.eye(3) * 8.0)
    basis = PlaneWaveBasis.from_cutoff(lattice, encut=1.0)
    return FFTGrid(basis=basis, shape=(8, 8, 8))


def test_uniform_density_has_zero_hartree_potential():
    """Catch a periodic Poisson solve that retains the undefined G=0 mode."""
    grid = make_grid()
    density = np.full(grid.shape, 2.0 / grid.volume)

    result = hartree_from_density(density, grid)

    assert np.max(np.abs(result.potential)) < 1e-13
    assert abs(result.energy) < 1e-13


def test_cosine_density_matches_periodic_poisson_solution_and_energy():
    """Catch incorrect Fourier normalization or a missing 4*pi/G**2 factor."""
    grid = make_grid()
    amplitude = 0.02
    background = 2.0 / grid.volume
    x = grid.fractional_mesh()[..., 0]
    density = background + amplitude * np.cos(2.0 * np.pi * x)

    result = hartree_from_density(density, grid)
    g_squared = float(
        np.dot(grid.basis.lattice.B[:, 0], grid.basis.lattice.B[:, 0])
    )
    expected_amplitude = 4.0 * np.pi * amplitude / g_squared
    measured_amplitude = 2.0 * np.mean(result.potential * np.cos(2.0 * np.pi * x))

    assert measured_amplitude == pytest.approx(expected_amplitude, rel=1e-12)
    assert result.energy == pytest.approx(
        0.5 * np.real(grid.integrate(density * result.potential)), rel=1e-12
    )
