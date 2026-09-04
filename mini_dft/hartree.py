"""Reciprocal-space periodic Hartree potential with a zero-average gauge."""

from dataclasses import dataclass

import numpy as np

from .fft_grid import FFTGrid


@dataclass(frozen=True)
class HartreeResult:
    """Hartree potential and its double-counting-corrected energy."""

    potential: np.ndarray
    energy: float


def hartree_from_density(density: np.ndarray, grid: FFTGrid) -> HartreeResult:
    """Solve periodic Poisson's equation after setting the G=0 mode to zero."""
    density_array = np.asarray(density)
    grid.field_to_fourier(density_array)
    if not np.all(np.isfinite(density_array)):
        raise ValueError("density: expected finite values")
    if np.iscomplexobj(density_array) and np.max(np.abs(density_array.imag)) > 1e-11:
        raise ValueError("density: expected a real field")
    density_real = np.real(density_array)

    density_fourier = grid.field_to_fourier(density_real)
    reciprocal_axes = [np.fft.fftfreq(size) * size for size in grid.shape]
    indices = np.stack(np.meshgrid(*reciprocal_axes, indexing="ij"), axis=-1)
    vectors = indices @ grid.basis.lattice.B.T
    raw_squared = np.sum(vectors * vectors, axis=-1)
    conjugate = np.ix_(*[(-np.arange(size)) % size for size in grid.shape])
    # Nyquist indices are self-negative in an even FFT axis.  Averaging the
    # aliased conjugate pair keeps this real reciprocal-space kernel Hermitian
    # in skew cells, where their raw Cartesian norms can otherwise differ.
    squared = 0.5 * (raw_squared + raw_squared[conjugate])
    potential_fourier = np.zeros(grid.shape, dtype=np.complex128)
    nonzero = squared > 0.0
    potential_fourier[nonzero] = (
        4.0 * np.pi * density_fourier[nonzero] / squared[nonzero]
    )
    potential_complex = grid.fourier_to_field(potential_fourier)
    imaginary_residual = float(np.max(np.abs(potential_complex.imag)))
    if imaginary_residual > 1e-11:
        raise ValueError("Hartree potential: unexpected imaginary residual")
    potential = potential_complex.real
    if not np.all(np.isfinite(potential)):
        raise ValueError("Hartree potential: expected finite values")
    energy = float(0.5 * np.real(grid.integrate(density_real * potential)))
    return HartreeResult(potential=potential, energy=energy)
