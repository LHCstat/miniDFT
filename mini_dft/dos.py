"""Gamma-point density-of-states broadening."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


@dataclass(frozen=True)
class DOSResult:
    """A uniformly sampled, Gaussian-broadened discrete spectrum in Hartree."""

    energies: np.ndarray
    values: np.ndarray


def gaussian_dos(
    eigenvalues: np.ndarray,
    occupations: np.ndarray,
    points: int,
    width: float,
    margin_sigmas: float = 5.0,
) -> DOSResult:
    """Broaden Gamma-point eigenvalues with normalized Gaussian functions.

    ``values`` is expressed per Hartree.  This is a discrete Gamma-point
    spectrum visualization, rather than a Brillouin-zone sampled DOS.
    """
    eigenvalue_array = _eigenvalues(eigenvalues)
    occupation_array = _occupations(occupations, eigenvalue_array.shape[0])
    point_count = _positive_integer(points, "points")
    gaussian_width = _positive_scalar(width, "width")
    margin = _positive_scalar(margin_sigmas, "margin_sigmas")

    lower = float(np.min(eigenvalue_array) - margin * gaussian_width)
    upper = float(np.max(eigenvalue_array) + margin * gaussian_width)
    energies = np.linspace(lower, upper, point_count, dtype=float)
    offsets = (energies[:, np.newaxis] - eigenvalue_array[np.newaxis, :]) / gaussian_width
    gaussians = np.exp(-0.5 * offsets**2) / (gaussian_width * np.sqrt(2.0 * np.pi))
    values = gaussians @ occupation_array
    energies.setflags(write=False)
    values.setflags(write=False)
    return DOSResult(energies=energies, values=values)


def _eigenvalues(values: np.ndarray) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("eigenvalues: expected a finite one-dimensional array") from exc
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("eigenvalues: expected a non-empty finite one-dimensional array")
    return array


def _occupations(values: np.ndarray, bands: int) -> np.ndarray:
    try:
        complex_values = np.asarray(values, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise ValueError("occupations: expected finite non-negative values") from exc
    if np.any(complex_values.imag != 0.0):
        raise ValueError("occupations: expected real-valued finite non-negative values")
    array = complex_values.real
    if (
        array.ndim != 1
        or array.shape != (bands,)
        or not np.all(np.isfinite(array))
        or np.any(array < 0.0)
    ):
        raise ValueError(f"occupations: expected shape ({bands},) with finite non-negative values")
    return array


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name}: expected an integer of at least 1")
    return int(value)


def _positive_scalar(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name}: expected a positive finite number")
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name}: expected a positive finite number")
    return scalar
