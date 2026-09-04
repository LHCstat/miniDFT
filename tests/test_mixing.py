import numpy as np
import pytest

from mini_dft.basis import PlaneWaveBasis
from mini_dft.density import density_integral
from mini_dft.fft_grid import FFTGrid
from mini_dft.lattice import Lattice
from mini_dft.mixing import density_rms_residual, mix_density


def make_grid() -> FFTGrid:
    lattice = Lattice.from_row_vectors(np.eye(3) * 4.0)
    basis = PlaneWaveBasis.from_cutoff(lattice, encut=1.0)
    return FFTGrid(basis=basis, shape=(4, 4, 4))


def test_mix_density_is_linear_when_inputs_already_have_target_charge():
    """Catch reversed mixing weights or normalization applied before mixing."""
    grid = make_grid()
    electrons = 2.0
    input_density = np.full(grid.shape, electrons / grid.volume)
    modulation = 0.02 * np.cos(2.0 * np.pi * grid.fractional_mesh()[..., 0])
    output_density = input_density + modulation

    mixed = mix_density(input_density, output_density, 0.25, electrons, grid)

    assert np.allclose(mixed, 0.75 * input_density + 0.25 * output_density)


def test_mix_density_renormalizes_the_final_charge():
    """Catch a mixer that returns a density with the wrong electron count."""
    grid = make_grid()
    input_density = np.full(grid.shape, 0.3)
    output_density = np.full(grid.shape, 0.7)

    mixed = mix_density(input_density, output_density, 0.4, 2.0, grid)

    assert density_integral(mixed, grid) == pytest.approx(2.0, abs=1e-12)


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.1])
def test_mix_density_rejects_alphas_outside_open_closed_interval(alpha):
    """Catch invalid damping factors before they silently change SCF behavior."""
    grid = make_grid()
    density = np.full(grid.shape, 2.0 / grid.volume)

    with pytest.raises(ValueError, match="alpha"):
        mix_density(density, density, alpha, 2.0, grid)


def test_mix_density_rejects_materially_negative_values():
    """Catch clipping that hides physically meaningful negative density values."""
    grid = make_grid()
    density = np.full(grid.shape, 2.0 / grid.volume)
    output_density = density.copy()
    output_density[0, 0, 0] = -1.0e-6

    with pytest.raises(ValueError, match="non-negative"):
        mix_density(density, output_density, 1.0, 2.0, grid)


@pytest.mark.parametrize("electrons", [1, 1.5, 3])
def test_mix_density_rejects_odd_or_fractional_target_electron_counts(electrons):
    """Catch target charges outside the v0.1 even-electron occupation domain."""
    grid = make_grid()
    density = np.full(grid.shape, 2.0 / grid.volume)

    with pytest.raises(ValueError, match="electrons"):
        mix_density(density, density, 1.0, electrons, grid)


def test_density_rms_residual_matches_direct_grid_mean():
    """Catch residuals that are normalized by charge or cell volume instead of points."""
    input_density = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
    output_density = np.array([[[2.0, 0.0]], [[5.0, 1.0]]])

    residual = density_rms_residual(input_density, output_density)

    assert residual == pytest.approx(np.sqrt(np.mean((output_density - input_density) ** 2)))
