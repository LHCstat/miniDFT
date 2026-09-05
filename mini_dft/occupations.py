"""Integer occupations for non-spin-polarized spatial orbitals."""

import numpy as np


def integer_occupations(electrons: int, n_bands: int) -> np.ndarray:
    """Return two-electron occupations for the lowest requested bands."""
    if (
        not isinstance(electrons, (int, np.integer))
        or isinstance(electrons, bool)
        or electrons < 0
        or electrons % 2 != 0
    ):
        raise ValueError("electrons: expected a non-negative even integer")
    if (
        not isinstance(n_bands, (int, np.integer))
        or isinstance(n_bands, bool)
        or n_bands < 1
    ):
        raise ValueError("n_bands: expected a positive integer")
    occupied_bands = electrons // 2
    if occupied_bands > n_bands:
        raise ValueError("n_bands: insufficient bands for the requested electrons")

    occupations = np.zeros(n_bands, dtype=float)
    occupations[:occupied_bands] = 2.0
    return occupations
