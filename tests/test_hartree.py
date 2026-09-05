import numpy as np
import pytest

from mini_dft.basis import PlaneWaveBasis
from mini_dft.fft_grid import FFTGrid
from mini_dft.hartree import hartree_from_density
from mini_dft.lattice import Lattice


def make_grid() -> FFTGrid:
    """Build a cubic grid with an exactly represented first cosine mode."""
    lattice = Lattice.from_row_vectors(np.eye(3) * 8.0)
    basis = PlaneWaveBasis.from_cutoff(lattice, encut=1.0)
    return FFTGrid(basis=basis, shape=(8, 8, 8))


def test_uniform_density_has_zero_hartree_potential():
    """Catch a periodic Poisson solve that retains the undefined G=0 mode."""
    grid = make_grid()
    density = np.full(grid.shape, 2.0 / grid.volume)

    result = hartree_from_density(density, grid)

    assert np.max(np.abs(result.potential)) < 1e-13
    assert abs(result.energy) < 1e-13


def test_hartree_rejects_batched_density_with_contextual_exact_shape_error():
    """Catch leading density dimensions reaching FFT and reciprocal indexing."""
    grid = make_grid()
    density = np.zeros((2, *grid.shape))

    with pytest.raises(ValueError) as error:
        hartree_from_density(density, grid)

    assert str(error.value) == (
        f"density: expected shape {grid.shape}, received {density.shape}"
    )


def test_cosine_density_matches_periodic_poisson_solution_and_energy():
    """Catch incorrect Fourier normalization or a missing 4*pi/G**2 factor."""
    grid = make_grid()
    amplitude = 0.02
    background = 2.0 / grid.volume
    x = grid.fractional_mesh()[..., 0]
    density = background + amplitude * np.cos(2.0 * np.pi * x)

    result = hartree_from_density(density, grid)
    g_squared = float(
        np.dot(grid.basis.lattice.B[:, 0], grid.basis.lattice.B[:, 0])
    )
    expected_amplitude = 4.0 * np.pi * amplitude / g_squared
    measured_amplitude = 2.0 * np.mean(result.potential * np.cos(2.0 * np.pi * x))

    assert measured_amplitude == pytest.approx(expected_amplitude, rel=1e-12)
    assert result.energy == pytest.approx(
        0.5 * np.real(grid.integrate(density * result.potential)), rel=1e-12
    )


def test_skew_even_grid_nyquist_density_uses_a_hermitian_hartree_kernel():
    """Catch skew-cell Nyquist aliases whose unequal kernels make real fields complex."""
    lattice = Lattice.from_row_vectors(
        [[1.0, 0.0, 0.0], [0.8, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    basis = PlaneWaveBasis.from_cutoff(lattice, encut=1.0)
    grid = FFTGrid(basis=basis, shape=(6, 5, 5))
    indices = np.indices(grid.shape)
    density = 0.1 + 0.01 * (-1.0) ** indices[0] * np.cos(
        2.0 * np.pi * indices[1] / grid.shape[1]
    )

    result = hartree_from_density(density, grid)

    reciprocal_axes = [np.fft.fftfreq(size) * size for size in grid.shape]
    reciprocal_indices = np.stack(
        np.meshgrid(*reciprocal_axes, indexing="ij"), axis=-1
    )
    vectors = reciprocal_indices @ lattice.B.T
    raw_squared = np.sum(vectors * vectors, axis=-1)
    conjugate = np.ix_(*[(-np.arange(size)) % size for size in grid.shape])
    symmetric_squared = 0.5 * (raw_squared + raw_squared[conjugate])
    expected_fourier = np.zeros(grid.shape, dtype=complex)
    nonzero = symmetric_squared > 0.0
    expected_fourier[nonzero] = (
        4.0 * np.pi * grid.field_to_fourier(density)[nonzero]
        / symmetric_squared[nonzero]
    )
    expected_potential = grid.fourier_to_field(expected_fourier)

    assert np.max(np.abs(expected_potential.imag)) < 1e-13
    assert np.all(np.isfinite(result.potential))
    assert np.allclose(result.potential, expected_potential.real, atol=1e-12)
    assert result.energy == pytest.approx(
        0.5 * grid.integrate(density * result.potential), rel=1e-12
    )
