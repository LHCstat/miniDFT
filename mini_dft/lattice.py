"""Lattice geometry and coordinate transformations."""

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Lattice:
    """A periodic cell whose real-space lattice vectors are matrix columns."""

    A: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.A, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError(f"lattice.A: expected shape (3, 3), received {matrix.shape}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("lattice.A: expected finite values")
        if abs(np.linalg.det(matrix)) <= 1e-12:
            raise ValueError("lattice.A: expected non-zero cell volume")
        matrix = matrix.copy()
        matrix.setflags(write=False)
        object.__setattr__(self, "A", matrix)

    @classmethod
    def from_row_vectors(cls, vectors: Any) -> "Lattice":
        """Build a lattice from the row-vector convention used by YAML input."""
        return cls(np.asarray(vectors, dtype=float).T.copy())

    @cached_property
    def B(self) -> np.ndarray:
        """Reciprocal lattice vectors as matrix columns."""
        reciprocal = 2.0 * np.pi * np.linalg.inv(self.A).T
        reciprocal.setflags(write=False)
        return reciprocal

    @cached_property
    def volume(self) -> float:
        """Positive cell volume in Bohr cubed."""
        return float(abs(np.linalg.det(self.A)))

    def fractional_to_cartesian(self, coords: Any) -> np.ndarray:
        """Convert one or more fractional coordinates to Cartesian coordinates."""
        fractional = self._coordinates(coords, "fractional coordinates")
        return fractional @ self.A.T

    def cartesian_to_fractional(self, coords: Any) -> np.ndarray:
        """Convert one or more Cartesian coordinates to fractional coordinates."""
        cartesian = self._coordinates(coords, "cartesian coordinates")
        return cartesian @ np.linalg.inv(self.A).T

    @staticmethod
    def _coordinates(coords: Any, name: str) -> np.ndarray:
        array = np.asarray(coords, dtype=float)
        if array.ndim == 0 or array.shape[-1] != 3:
            raise ValueError(f"{name}: expected an array with shape (..., 3), received {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name}: expected finite values")
        return array
