"""Local educational ionic model potentials on periodic FFT grids."""

import numpy as np

from .fft_grid import FFTGrid
from .system import CosinePotentialConfig, GaussianAtomsPotentialConfig


def build_ionic_potential(
    config: CosinePotentialConfig | GaussianAtomsPotentialConfig, grid: FFTGrid
) -> np.ndarray:
    """Build a finite periodic local ionic potential on ``grid``."""
    if isinstance(config, CosinePotentialConfig):
        potential = _cosine_potential(config, grid)
    elif isinstance(config, GaussianAtomsPotentialConfig):
        potential = _gaussian_atoms_potential(config, grid)
    else:
        raise ValueError(f"unsupported ionic potential configuration: {type(config).__name__}")
    if not np.all(np.isfinite(potential)):
        raise ValueError("ionic potential: expected finite values")
    return potential


def _cosine_potential(config: CosinePotentialConfig, grid: FFTGrid) -> np.ndarray:
    amplitudes = np.asarray(config.amplitudes, dtype=float)
    modes = np.asarray(config.modes, dtype=float)
    phases = np.asarray(config.phases, dtype=float)
    if (
        amplitudes.ndim != 1
        or phases.shape != amplitudes.shape
        or modes.shape != (amplitudes.size, 3)
    ):
        raise ValueError("cosine potential: incompatible amplitudes, modes, and phases")
    if not np.all(np.isfinite(amplitudes)) or not np.all(np.isfinite(phases)):
        raise ValueError("cosine potential: expected finite amplitudes and phases")
    arguments = 2.0 * np.pi * np.einsum(
        "...i,mi->...m", grid.fractional_mesh(), modes
    ) + phases
    return np.sum(amplitudes * np.cos(arguments), axis=-1)


def _gaussian_atoms_potential(
    config: GaussianAtomsPotentialConfig, grid: FFTGrid
) -> np.ndarray:
    fractional = grid.fractional_mesh()
    potential = np.zeros(grid.shape, dtype=float)
    translations = np.stack(
        np.meshgrid(
            (-1.0, 0.0, 1.0),
            (-1.0, 0.0, 1.0),
            (-1.0, 0.0, 1.0),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    for atom in config.atoms:
        position = np.asarray(atom.fractional_position, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("gaussian atom position: expected three finite coordinates")
        if not np.isfinite(atom.depth) or not np.isfinite(atom.width) or atom.width <= 0:
            raise ValueError("gaussian atom: expected finite depth and positive finite width")
        delta = fractional - np.mod(position, 1.0)
        delta -= np.rint(delta)
        images = delta[..., None, :] + translations
        cartesian = images @ grid.basis.lattice.A.T
        distance_squared = np.min(np.sum(cartesian * cartesian, axis=-1), axis=-1)
        potential += atom.depth * np.exp(-distance_squared / (2.0 * atom.width**2))
    return potential
