"""Electron-density construction and charge integration."""

import numpy as np

from .fft_grid import FFTGrid


def density_from_coefficients(
    coefficients: np.ndarray, occupations: np.ndarray, grid: FFTGrid
) -> np.ndarray:
    """Construct a real-space density from occupied plane-wave bands."""
    coefficient_array = np.asarray(coefficients, dtype=np.complex128)
    if coefficient_array.ndim != 2 or coefficient_array.shape[1] != grid.basis.npw:
        raise ValueError(
            "coefficients: expected shape "
            f"(bands, {grid.basis.npw}), received {coefficient_array.shape}"
        )
    if not np.all(np.isfinite(coefficient_array)):
        raise ValueError("coefficients: expected finite values")

    occupation_array = _occupations(occupations, coefficient_array.shape[0])
    wavefunctions = grid.coefficients_to_real(coefficient_array)
    density = np.einsum("b,bxyz->xyz", occupation_array, np.abs(wavefunctions) ** 2)
    density_real = np.asarray(np.real_if_close(density, tol=1000).real, dtype=float)
    if not np.all(np.isfinite(density_real)):
        raise ValueError("density: expected finite values")
    if np.any(density_real < 0.0):
        raise ValueError("density: expected non-negative values")
    return density_real


def density_integral(density: np.ndarray, grid: FFTGrid) -> float:
    """Return the number of electrons represented by a density field."""
    density_real = _real_field(density, grid, "density")
    integral = float(grid.integrate(density_real))
    if not np.isfinite(integral):
        raise ValueError("density: expected a finite integral")
    return integral


def _occupations(occupations: np.ndarray, n_bands: int) -> np.ndarray:
    occupation_array = np.asarray(occupations, dtype=float)
    if occupation_array.ndim != 1 or occupation_array.shape[0] != n_bands:
        raise ValueError(
            "occupations: expected shape "
            f"({n_bands},), received {occupation_array.shape}"
        )
    if not np.all(np.isfinite(occupation_array)) or np.any(occupation_array < 0.0):
        raise ValueError("occupations: expected finite non-negative values")
    return occupation_array


def _real_field(field: np.ndarray, grid: FFTGrid, name: str) -> np.ndarray:
    field_array = np.asarray(field)
    grid.integrate(field_array)
    if not np.all(np.isfinite(field_array)):
        raise ValueError(f"{name}: expected finite values")
    if np.iscomplexobj(field_array) and np.max(np.abs(field_array.imag)) > 1e-11:
        raise ValueError(f"{name}: expected a real field")
    return np.asarray(np.real(field_array), dtype=float)
