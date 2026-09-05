import numpy as np
import pytest

from mini_dft.basis import PlaneWaveBasis
from mini_dft.fft_grid import FFTGrid
from mini_dft.ionic_potential import build_ionic_potential
from mini_dft.lattice import Lattice
from mini_dft.potentials import PotentialSet
from mini_dft.system import (
    CosinePotentialConfig,
    GaussianAtom,
    GaussianAtomsPotentialConfig,
)


def make_grid(
    lattice: Lattice | None = None, shape: tuple[int, int, int] = (8, 8, 8)
) -> FFTGrid:
    lattice = lattice or Lattice.from_row_vectors(np.eye(3) * 8.0)
    basis = PlaneWaveBasis.from_cutoff(lattice, encut=1.0)
    return FFTGrid(basis=basis, shape=shape)


def test_cosine_ionic_potential_matches_fractional_mode_formula():
    """Catch cosine modes evaluated in Cartesian coordinates or without phase."""
    grid = make_grid()
    config = CosinePotentialConfig(
        amplitudes=np.array([-0.7]),
        modes=np.array([[1, 0, 0]]),
        phases=np.array([0.2]),
    )

    potential = build_ionic_potential(config, grid)

    x = grid.fractional_mesh()[..., 0]
    assert np.allclose(potential, -0.7 * np.cos(2.0 * np.pi * x + 0.2), atol=1e-12)


def test_gaussian_ionic_potential_is_periodic_and_deepest_at_center():
    """Catch unwrapped atomic positions or a Gaussian with the wrong distance scale."""
    grid = make_grid()
    atom = GaussianAtom(position=np.array([0.5, 0.5, 0.5]), depth=-2.0, width=0.4)
    config = GaussianAtomsPotentialConfig(atoms=(atom,))
    translated = GaussianAtomsPotentialConfig(
        atoms=(
            GaussianAtom(
                position=atom.position + np.array([1.0, -2.0, 3.0]),
                depth=-2.0,
                width=0.4,
            ),
        )
    )

    potential = build_ionic_potential(config, grid)

    assert np.all(np.isfinite(potential))
    assert potential[4, 4, 4] == pytest.approx(np.min(potential))
    assert np.allclose(potential, build_ionic_potential(translated, grid), atol=1e-12)


def test_gaussian_ionic_potential_uses_cartesian_minimum_image_in_skew_cell():
    """Catch component-wise wrapping, which is not a shortest image in skew cells."""
    lattice = Lattice.from_row_vectors(
        [[1.0, 0.0, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    grid = make_grid(lattice, shape=(7, 6, 5))
    atom = GaussianAtom(position=np.array([0.93, 0.48, 0.17]), depth=-1.3, width=0.18)

    potential = build_ionic_potential(GaussianAtomsPotentialConfig(atoms=(atom,)), grid)

    delta = grid.fractional_mesh() - np.mod(atom.position, 1.0)
    translations = np.array(
        [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=float,
    )
    images = delta[..., None, :] + translations
    cartesian = images @ lattice.A.T
    distance_squared = np.min(np.sum(cartesian**2, axis=-1), axis=-1)
    expected = atom.depth * np.exp(-distance_squared / (2.0 * atom.width**2))

    assert np.allclose(potential, expected, atol=1e-12)


def test_potential_set_composes_matching_fields_and_rejects_shape_mismatches():
    """Catch a silent broadcast while constructing the effective local potential."""
    components = PotentialSet(
        ionic=np.full((2, 3, 4), -1.0),
        hartree=np.full((2, 3, 4), 0.2),
        xc=np.full((2, 3, 4), -0.1),
    )

    assert np.array_equal(components.effective, np.full((2, 3, 4), -0.9))
    with pytest.raises(ValueError, match="matching shapes"):
        PotentialSet(
            ionic=np.zeros((2, 3, 4)),
            hartree=np.zeros((2, 3, 1)),
            xc=np.zeros((2, 3, 4)),
        ).effective
