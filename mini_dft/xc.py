"""Unpolarized Dirac exchange and Perdew-Zunger 1981 LDA correlation."""

from dataclasses import dataclass

import numpy as np

from .fft_grid import FFTGrid


@dataclass(frozen=True)
class XCResult:
    """Local exchange-correlation potential, energy density, and total energy."""

    potential: np.ndarray
    energy_per_particle: np.ndarray
    energy: float


def lda_pz81(density: np.ndarray, grid: FFTGrid) -> XCResult:
    """Evaluate unpolarized PZ81 LDA, using its analytic zero-density limit."""
    density_array = np.asarray(density)
    grid.integrate(density_array)
    if not np.all(np.isfinite(density_array)):
        raise ValueError("density: expected finite values")
    if np.iscomplexobj(density_array) and np.max(np.abs(density_array.imag)) > 1e-11:
        raise ValueError("density: expected a real field")
    density_real = np.real(density_array)
    if np.any(density_real < 0.0):
        raise ValueError("density: expected non-negative values")

    energy_per_particle = np.zeros(grid.shape, dtype=float)
    potential = np.zeros(grid.shape, dtype=float)
    positive = density_real > 0.0
    n = density_real[positive]
    rs = (3.0 / (4.0 * np.pi * n)) ** (1.0 / 3.0)

    eps_x = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    v_x = (4.0 / 3.0) * eps_x
    eps_c = np.empty_like(rs)
    deps_c_drs = np.empty_like(rs)
    high_density = rs < 1.0
    low_density = ~high_density
    if np.any(high_density):
        r = rs[high_density]
        eps_c[high_density] = (
            0.0311 * np.log(r) - 0.048 + 0.0020 * r * np.log(r) - 0.0116 * r
        )
        deps_c_drs[high_density] = (
            0.0311 / r + 0.0020 * (np.log(r) + 1.0) - 0.0116
        )
    if np.any(low_density):
        r = rs[low_density]
        denominator = 1.0 + 1.0529 * np.sqrt(r) + 0.3334 * r
        eps_c[low_density] = -0.1423 / denominator
        deps_c_drs[low_density] = 0.1423 * (
            1.0529 / (2.0 * np.sqrt(r)) + 0.3334
        ) / denominator**2

    energy_per_particle[positive] = eps_x + eps_c
    potential[positive] = v_x + eps_c - rs * deps_c_drs / 3.0
    if not np.all(np.isfinite(energy_per_particle)) or not np.all(np.isfinite(potential)):
        raise ValueError("LDA PZ81: expected finite outputs")
    energy = float(grid.integrate(density_real * energy_per_particle))
    return XCResult(
        potential=potential,
        energy_per_particle=energy_per_particle,
        energy=energy,
    )
