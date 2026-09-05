"""Model Kohn-Sham energy components and an eigenvalue diagnostic."""

from dataclasses import dataclass

import numpy as np

from .density import _occupations, _real_field
from .fft_grid import FFTGrid
from .hartree import HartreeResult
from .xc import XCResult


@dataclass(frozen=True)
class EnergyComponents:
    """Directly evaluated model electronic energy contributions."""

    kinetic: float
    external: float
    hartree: float
    xc: float

    @property
    def total(self) -> float:
        """Return ``T_s + E_ext + E_H + E_xc``."""
        return self.kinetic + self.external + self.hartree + self.xc


def calculate_energy(
    coefficients: np.ndarray,
    occupations: np.ndarray,
    density: np.ndarray,
    external_potential: np.ndarray,
    hartree: HartreeResult,
    xc: XCResult,
    grid: FFTGrid,
) -> EnergyComponents:
    """Evaluate the direct model functional from a consistent density state."""
    coefficient_array = np.asarray(coefficients, dtype=np.complex128)
    if coefficient_array.ndim != 2 or coefficient_array.shape[1] != grid.basis.npw:
        raise ValueError(
            "coefficients: expected shape "
            f"(bands, {grid.basis.npw}), received {coefficient_array.shape}"
        )
    if not np.all(np.isfinite(coefficient_array)):
        raise ValueError("coefficients: expected finite values")
    occupation_array = _occupations(occupations, coefficient_array.shape[0])
    density_real = _real_field(density, grid, "density")
    external_real = _real_field(external_potential, grid, "external_potential")
    hartree_potential = _real_field(hartree.potential, grid, "hartree.potential")

    kinetic = float(
        np.einsum(
            "b,bg,g->",
            occupation_array,
            np.abs(coefficient_array) ** 2,
            grid.basis.kinetic_energies,
        )
    )
    external = float(grid.integrate(density_real * external_real))
    hartree_energy = float(0.5 * grid.integrate(density_real * hartree_potential))
    xc_energy = _finite_scalar(xc.energy, "xc.energy")
    return EnergyComponents(kinetic, external, hartree_energy, xc_energy)


def eigenvalue_energy(
    eigenvalues: np.ndarray,
    occupations: np.ndarray,
    density: np.ndarray,
    v_h: np.ndarray,
    v_xc: np.ndarray,
    xc_energy: float,
    grid: FFTGrid,
) -> float:
    """Return the double-counting-corrected eigenvalue energy diagnostic."""
    eigenvalue_array = np.asarray(eigenvalues, dtype=float)
    if eigenvalue_array.ndim != 1 or not np.all(np.isfinite(eigenvalue_array)):
        raise ValueError("eigenvalues: expected a finite one-dimensional array")
    occupation_array = _occupations(occupations, eigenvalue_array.shape[0])
    density_real = _real_field(density, grid, "density")
    hartree_potential = _real_field(v_h, grid, "v_h")
    xc_potential = _real_field(v_xc, grid, "v_xc")
    corrected = (
        float(np.dot(occupation_array, eigenvalue_array))
        - 0.5 * float(grid.integrate(density_real * hartree_potential))
        - float(grid.integrate(density_real * xc_potential))
        + _finite_scalar(xc_energy, "xc_energy")
    )
    if not np.isfinite(corrected):
        raise ValueError("eigenvalue energy: expected a finite result")
    return corrected


def _finite_scalar(value: float, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: expected a finite number") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name}: expected a finite number")
    return scalar
