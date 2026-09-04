"""Composition of the local components of the Kohn-Sham potential."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PotentialSet:
    """The ionic, Hartree, and exchange-correlation local potential fields."""

    ionic: np.ndarray
    hartree: np.ndarray
    xc: np.ndarray

    @property
    def effective(self) -> np.ndarray:
        """Return the local effective potential after exact field-shape validation."""
        ionic = np.asarray(self.ionic)
        hartree = np.asarray(self.hartree)
        xc = np.asarray(self.xc)
        if ionic.shape != hartree.shape or ionic.shape != xc.shape:
            raise ValueError("potential components: expected matching shapes")
        return ionic + hartree + xc
