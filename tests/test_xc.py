import numpy as np
import pytest

from mini_dft.basis import PlaneWaveBasis
from mini_dft.fft_grid import FFTGrid
from mini_dft.lattice import Lattice
from mini_dft.xc import lda_pz81


def make_grid() -> FFTGrid:
    lattice = Lattice.from_row_vectors(np.eye(3) * 8.0)
    basis = PlaneWaveBasis.from_cutoff(lattice, encut=1.0)
    return FFTGrid(basis=basis, shape=(4, 4, 4))


def _pz81_reference(density: float) -> tuple[float, float]:
    """Evaluate the published unpolarized PZ81 branches independently."""
    rs = (3.0 / (4.0 * np.pi * density)) ** (1.0 / 3.0)
    eps_x = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0) * density ** (1.0 / 3.0)
    v_x = (4.0 / 3.0) * eps_x
    if rs < 1.0:
        a, b, c, d = 0.0311, -0.048, 0.0020, -0.0116
        eps_c = a * np.log(rs) + b + c * rs * np.log(rs) + d * rs
        deps_c_drs = a / rs + c * (np.log(rs) + 1.0) + d
    else:
        gamma, beta1, beta2 = -0.1423, 1.0529, 0.3334
        denominator = 1.0 + beta1 * np.sqrt(rs) + beta2 * rs
        eps_c = gamma / denominator
        deps_c_drs = -gamma * (beta1 / (2.0 * np.sqrt(rs)) + beta2) / denominator**2
    return eps_x + eps_c, v_x + eps_c - rs * deps_c_drs / 3.0


def test_zero_density_has_exact_finite_lda_outputs():
    """Catch zero-density divisions that leak NaN or an arbitrary offset."""
    grid = make_grid()

    result = lda_pz81(np.zeros(grid.shape), grid)

    assert np.array_equal(result.potential, np.zeros(grid.shape))
    assert np.array_equal(result.energy_per_particle, np.zeros(grid.shape))
    assert result.energy == 0.0
    assert np.all(np.isfinite(result.potential))
    assert np.all(np.isfinite(result.energy_per_particle))


@pytest.mark.parametrize("density", [1.0, 3.0 / (32.0 * np.pi)])
def test_pz81_matches_independent_reference_on_both_rs_branches(density: float):
    """Catch incorrect PZ81 branch constants or correlation-potential derivative."""
    grid = make_grid()
    expected_energy_per_particle, expected_potential = _pz81_reference(density)

    result = lda_pz81(np.full(grid.shape, density), grid)

    assert result.energy_per_particle == pytest.approx(
        expected_energy_per_particle, rel=1e-12
    )
    assert result.potential == pytest.approx(expected_potential, rel=1e-12)
    assert result.energy == pytest.approx(
        grid.volume * density * expected_energy_per_particle, rel=1e-12
    )
