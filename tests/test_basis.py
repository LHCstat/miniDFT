import numpy as np
import pytest

from mini_dft.basis import PlaneWaveBasis
from mini_dft.lattice import Lattice


def test_cubic_basis_matches_explicit_integer_sphere():
    lattice = Lattice.from_row_vectors(np.eye(3) * 8.0)
    basis = PlaneWaveBasis.from_cutoff(lattice, encut=1.0)

    expected = []
    for h in range(-3, 4):
        for k in range(-3, 4):
            for l in range(-3, 4):
                kinetic = 0.5 * (2 * np.pi / 8.0) ** 2 * (h * h + k * k + l * l)
                if kinetic <= 1.0 + 1e-12:
                    expected.append((h, k, l))

    assert set(map(tuple, basis.g_indices)) == set(expected)
    assert np.all(basis.kinetic_energies <= 1.0 + 1e-12)


def test_skew_basis_matches_oversized_brute_force_box_and_has_pairs():
    rows = np.array([[4.0, 0.0, 0.0], [0.7, 5.0, 0.0], [0.2, 0.4, 6.0]])
    lattice = Lattice.from_row_vectors(rows)
    encut = 1.25
    basis = PlaneWaveBasis.from_cutoff(lattice, encut)

    expected = set()
    for h in range(-12, 13):
        for k in range(-12, 13):
            for l in range(-12, 13):
                index = np.array([h, k, l])
                vector = index @ lattice.B.T
                if 0.5 * np.dot(vector, vector) <= encut + 1e-12:
                    expected.add((h, k, l))

    actual = set(map(tuple, basis.g_indices))
    assert actual == expected
    assert tuple(basis.g_indices[0]) == (0, 0, 0)
    assert all(tuple(-value for value in index) in actual for index in actual if index != (0, 0, 0))
    assert np.array_equal(basis.max_abs_indices, np.max(np.abs(basis.g_indices), axis=0))


@pytest.mark.parametrize("encut", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_basis_rejects_nonpositive_or_nonfinite_cutoff(encut):
    lattice = Lattice.from_row_vectors(np.eye(3) * 8.0)

    with pytest.raises(ValueError):
        PlaneWaveBasis.from_cutoff(lattice, encut)
