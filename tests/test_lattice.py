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


def test_unit_conversions_are_invertible():
    """Catch an inverted Angstrom or eV conversion factor."""
    assert length_to_bohr(BOHR_TO_ANGSTROM, "angstrom") == pytest.approx(1.0)
    assert energy_to_hartree(HARTREE_TO_EV, "ev") == pytest.approx(1.0)
