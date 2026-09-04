import numpy as np
import pytest

from mini_dft.dos import gaussian_dos


def test_gaussian_dos_integrates_occupied_states_and_orders_energies():
    """Catch DOS broadening that loses occupation weight or reverses its grid."""
    result = gaussian_dos(
        eigenvalues=np.array([-0.25, 0.5]),
        occupations=np.array([2.0, 0.0]),
        points=2001,
        width=0.05,
    )

    assert result.energies.shape == (2001,)
    assert result.values.shape == (2001,)
    assert np.all(np.diff(result.energies) > 0.0)
    assert np.all(result.values >= 0.0)
    assert np.trapezoid(result.values, result.energies) == pytest.approx(2.0, abs=2.0e-5)


@pytest.mark.parametrize(
    ("eigenvalues", "occupations", "points", "width", "match"),
    [
        (np.array([-0.1]), np.array([2.0]), 0, 0.1, "points"),
        (np.array([-0.1]), np.array([2.0]), 10, 0.0, "width"),
        (np.array([-0.1]), np.array([2.0]), 10, -0.1, "width"),
        (np.array([-0.1]), np.array([2.0, 0.0]), 10, 0.1, "occupations"),
    ],
)
def test_gaussian_dos_rejects_invalid_grid_or_band_shapes(
    eigenvalues, occupations, points, width, match
):
    """Catch malformed DOS requests before they produce misleading spectra."""
    with pytest.raises(ValueError, match=match):
        gaussian_dos(eigenvalues, occupations, points, width)
