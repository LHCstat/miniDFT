import numpy as np
import pytest

from mini_dft.basis import PlaneWaveBasis
from mini_dft.fft_grid import FFTGrid
from mini_dft.hamiltonian import Hamiltonian
from mini_dft.lattice import Lattice


def make_hamiltonian(potential: np.ndarray | None = None) -> Hamiltonian:
    """Build a representative periodic Hamiltonian for operator tests."""
    lattice = Lattice.from_row_vectors(np.eye(3) * 8.0)
    basis = PlaneWaveBasis.from_cutoff(lattice, encut=1.0)
    grid = FFTGrid.from_basis(basis)
    if potential is None:
        potential = np.zeros(grid.shape)
    return Hamiltonian(basis=basis, grid=grid, local_potential=potential)


def test_free_electron_action_is_diagonal_in_the_plane_wave_basis():
    """Catch an FFT contribution when the local potential is zero."""
    hamiltonian = make_hamiltonian()
    rng = np.random.default_rng(12)
    coefficients = rng.normal(size=hamiltonian.basis.npw) + 1j * rng.normal(
        size=hamiltonian.basis.npw
    )

    actual = hamiltonian.apply(coefficients)

    assert np.allclose(
        actual, hamiltonian.basis.kinetic_energies * coefficients, atol=1e-12
    )
    assert actual.dtype == np.complex128


def test_constant_local_potential_shifts_every_plane_wave_component():
    """Catch incorrect FFT normalization for the local multiplication term."""
    constant = -0.37
    hamiltonian = make_hamiltonian()
    shifted_hamiltonian = make_hamiltonian(
        np.full(hamiltonian.grid.shape, constant)
    )
    rng = np.random.default_rng(21)
    coefficients = rng.normal(size=hamiltonian.basis.npw) + 1j * rng.normal(
        size=hamiltonian.basis.npw
    )

    actual = shifted_hamiltonian.apply(coefficients)

    assert np.allclose(
        actual,
        (shifted_hamiltonian.basis.kinetic_energies + constant) * coefficients,
        atol=1e-12,
    )


def test_hamiltonian_action_is_linear_and_batched_action_matches_apply():
    """Catch loss of complex linearity or a differently normalized batch path."""
    hamiltonian = make_hamiltonian()
    rng = np.random.default_rng(23)
    x = rng.normal(size=hamiltonian.basis.npw) + 1j * rng.normal(
        size=hamiltonian.basis.npw
    )
    y = rng.normal(size=hamiltonian.basis.npw) + 1j * rng.normal(
        size=hamiltonian.basis.npw
    )
    alpha = 0.6 - 0.2j
    beta = -0.4 + 0.7j

    combined = hamiltonian.apply(alpha * x + beta * y)
    batched = hamiltonian.apply_many(np.stack((x, y)))

    assert np.allclose(
        combined, alpha * hamiltonian.apply(x) + beta * hamiltonian.apply(y), atol=1e-12
    )
    assert np.allclose(batched, np.stack((hamiltonian.apply(x), hamiltonian.apply(y))))


def test_hamiltonian_rejects_invalid_potentials_and_coefficient_shapes():
    """Catch field/grid mismatches and accidental coefficient-axis reshaping."""
    hamiltonian = make_hamiltonian()

    with pytest.raises(ValueError, match="local_potential: expected a real-valued array"):
        make_hamiltonian(np.zeros(hamiltonian.grid.shape, dtype=np.complex128))
    with pytest.raises(ValueError, match="local_potential: expected shape"):
        make_hamiltonian(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="coefficients: expected shape"):
        hamiltonian.apply(np.zeros((2, hamiltonian.basis.npw)))
    with pytest.raises(ValueError, match="coefficients: expected shape"):
        hamiltonian.apply_many(np.zeros(hamiltonian.basis.npw))


def test_hamiltonian_rejects_grid_from_a_distinct_same_size_basis():
    """Catch reciprocal metadata mixing hidden by equal plane-wave counts."""
    lattice = Lattice.from_row_vectors(np.eye(3) * 8.0)
    basis = PlaneWaveBasis.from_cutoff(lattice, encut=1.0)
    distinct_basis = PlaneWaveBasis.from_cutoff(lattice, encut=1.0)
    grid = FFTGrid.from_basis(distinct_basis)
    assert basis is not distinct_basis
    assert basis.npw == distinct_basis.npw

    with pytest.raises(ValueError, match="grid: expected the Hamiltonian basis"):
        Hamiltonian(basis, grid, np.zeros(grid.shape))


def test_hamiltonian_is_hermitian_for_a_real_local_potential():
    """Catch conjugation errors in the FFT-mediated local-potential action."""
    reference = make_hamiltonian()
    local_potential = 0.15 * np.cos(
        2.0 * np.pi * reference.grid.fractional_mesh()[..., 0]
    )
    hamiltonian = make_hamiltonian(local_potential)
    rng = np.random.default_rng(31)
    x = rng.normal(size=hamiltonian.basis.npw) + 1j * rng.normal(
        size=hamiltonian.basis.npw
    )
    y = rng.normal(size=hamiltonian.basis.npw) + 1j * rng.normal(
        size=hamiltonian.basis.npw
    )

    lhs = np.vdot(x, hamiltonian.apply(y))
    rhs = np.vdot(hamiltonian.apply(x), y)

    assert lhs == pytest.approx(rhs, abs=1e-11)
