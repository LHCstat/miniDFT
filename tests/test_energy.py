import numpy as np
import pytest

from mini_dft.basis import PlaneWaveBasis
from mini_dft.density import density_from_coefficients
from mini_dft.energy import calculate_energy, eigenvalue_energy
from mini_dft.fft_grid import FFTGrid
from mini_dft.hartree import hartree_from_density
from mini_dft.lattice import Lattice
from mini_dft.xc import lda_pz81


def make_grid() -> FFTGrid:
    lattice = Lattice.from_row_vectors(np.eye(3) * 8.0)
    return FFTGrid.from_basis(PlaneWaveBasis.from_cutoff(lattice, encut=1.0))


def test_calculate_energy_uses_occupied_coefficient_kinetic_energy():
    """Catch kinetic energy taken from a band index or unoccupied state."""
    grid = make_grid()
    coefficients = np.eye(grid.basis.npw, dtype=np.complex128)[:2]
    occupations = np.array([2.0, 0.0])
    density = density_from_coefficients(coefficients, occupations, grid)
    hartree = hartree_from_density(density, grid)
    xc = lda_pz81(density, grid)

    energy = calculate_energy(
        coefficients,
        occupations,
        density,
        np.zeros(grid.shape),
        hartree,
        xc,
        grid,
    )

    assert energy.kinetic == pytest.approx(
        2.0 * grid.basis.kinetic_energies[0], abs=1e-12
    )


def test_calculate_energy_integrates_constant_external_potential():
    """Catch an external-energy term that omits the density or volume factor."""
    grid = make_grid()
    coefficients = np.eye(grid.basis.npw, dtype=np.complex128)[:1]
    occupations = np.array([2.0])
    density = density_from_coefficients(coefficients, occupations, grid)
    hartree = hartree_from_density(density, grid)
    xc = lda_pz81(density, grid)
    constant = -0.23

    energy = calculate_energy(
        coefficients,
        occupations,
        density,
        np.full(grid.shape, constant),
        hartree,
        xc,
        grid,
    )

    assert energy.external == pytest.approx(2.0 * constant, abs=1e-12)


def test_direct_energy_matches_eigenvalue_double_counting_diagnostic():
    """Catch a missing Hartree/XC double-counting correction in either energy form."""
    grid = make_grid()
    coefficients = np.eye(grid.basis.npw, dtype=np.complex128)[:1]
    occupations = np.array([2.0])
    density = density_from_coefficients(coefficients, occupations, grid)
    hartree = hartree_from_density(density, grid)
    xc = lda_pz81(density, grid)
    external = np.full(grid.shape, -0.17)
    direct = calculate_energy(
        coefficients, occupations, density, external, hartree, xc, grid
    )
    eigenvalues = np.array([-0.17 + xc.potential[0, 0, 0]])

    diagnostic = eigenvalue_energy(
        eigenvalues,
        occupations,
        density,
        hartree.potential,
        xc.potential,
        xc.energy,
        grid,
    )

    assert direct.total == pytest.approx(diagnostic, abs=1e-12)
