import numpy as np
import pytest

from mini_dft.constants import BOHR_TO_ANGSTROM, HARTREE_TO_EV, energy_to_hartree, length_to_bohr
from mini_dft.lattice import Lattice


def test_lattice_reciprocal_identity_and_coordinate_round_trip():
    """Catch row/column convention mistakes in lattice coordinate transforms."""
    rows = np.array([[4.0, 0.0, 0.0], [0.3, 5.0, 0.0], [0.1, 0.2, 6.0]])
    lattice = Lattice.from_row_vectors(rows)

    assert np.allclose(lattice.A.T @ lattice.B, 2.0 * np.pi * np.eye(3))

    frac = np.array([0.2, 0.4, 0.7])
    cartesian = lattice.fractional_to_cartesian(frac)
    assert np.allclose(lattice.cartesian_to_fractional(cartesian), frac)


def test_lattice_batched_coordinate_round_trip_matches_skew_cell_geometry():
    """Catch `(N, 3)` transforms that transpose or collapse the batch axis."""
    lattice = Lattice.from_row_vectors(
        [[4.0, 0.0, 0.0], [1.0, 5.0, 0.0], [-0.5, 0.25, 6.0]]
    )
    fractional = np.array([[0.25, 0.5, 0.75], [1.0, -1.0, 0.5]])
    expected_cartesian = np.array([[1.125, 2.6875, 4.5], [2.75, -4.875, 3.0]])

    cartesian = lattice.fractional_to_cartesian(fractional)

    assert cartesian.shape == (2, 3)
    assert cartesian == pytest.approx(expected_cartesian)
    assert lattice.cartesian_to_fractional(cartesian) == pytest.approx(fractional)


def test_unit_conversions_are_invertible():
    """Catch an inverted Angstrom or eV conversion factor."""
    assert length_to_bohr(BOHR_TO_ANGSTROM, "angstrom") == pytest.approx(1.0)
    assert energy_to_hartree(HARTREE_TO_EV, "ev") == pytest.approx(1.0)
