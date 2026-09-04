"""Charge-preserving linear density mixing."""

import numpy as np

from .density import _real_field, density_integral
from .fft_grid import FFTGrid


_NEGATIVE_DENSITY_TOLERANCE = 1.0e-14


def mix_density(
    input_density: np.ndarray,
    output_density: np.ndarray,
    alpha: float,
    electrons: float,
    grid: FFTGrid,
) -> np.ndarray:
    """Linearly mix two densities and rescale the result to the target charge."""
    mixing_alpha = _alpha(alpha)
    target_charge = _electron_count(electrons)
    input_real = _real_field(input_density, grid, "input_density")
    output_real = _real_field(output_density, grid, "output_density")
    mixed = (1.0 - mixing_alpha) * input_real + mixing_alpha * output_real
    if np.any(mixed < -_NEGATIVE_DENSITY_TOLERANCE):
        raise ValueError("mixed density: expected non-negative values")
    mixed = np.maximum(mixed, 0.0)

    charge = density_integral(mixed, grid)
    if target_charge > 0.0 and charge <= 0.0:
        raise ValueError("mixed density: zero charge for a positive-electron system")
    if charge == 0.0:
        return mixed
    return mixed * (target_charge / charge)


def density_rms_residual(input_density: np.ndarray, output_density: np.ndarray) -> float:
    """Return the grid-point RMS change between input and output densities."""
    input_array = _real_array(input_density, "input_density")
    output_array = _real_array(output_density, "output_density")
    if input_array.shape != output_array.shape:
        raise ValueError(
            "density residual: expected input and output densities with matching shapes"
        )
    return float(np.sqrt(np.mean((output_array - input_array) ** 2)))


def _alpha(alpha: float) -> float:
    if isinstance(alpha, bool):
        raise ValueError("alpha: expected a finite number in (0, 1]")
    try:
        value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha: expected a finite number in (0, 1]") from exc
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("alpha: expected a finite number in (0, 1]")
    return value


def _electron_count(electrons: float) -> float:
    if isinstance(electrons, bool):
        raise ValueError("electrons: expected a non-negative finite number")
    try:
        value = float(electrons)
    except (TypeError, ValueError) as exc:
        raise ValueError("electrons: expected a non-negative finite number") from exc
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("electrons: expected a non-negative finite number")
    return value


def _real_array(array: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(array)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name}: expected finite values")
    if np.iscomplexobj(value) and np.max(np.abs(value.imag)) > 1e-11:
        raise ValueError(f"{name}: expected a real field")
    return np.asarray(np.real(value), dtype=float)
