"""Plane-wave basis construction."""

from dataclasses import dataclass

import numpy as np

from .lattice import Lattice


@dataclass(frozen=True)
class PlaneWaveBasis:
    """The reciprocal-space plane waves inside a kinetic-energy cutoff."""

    lattice: Lattice
    encut: float
    g_indices: np.ndarray
    g_vectors: np.ndarray
    kinetic_energies: np.ndarray

    @classmethod
    def from_cutoff(cls, lattice: Lattice, encut: float) -> "PlaneWaveBasis":
        """Enumerate reciprocal lattice vectors with ``|G|²/2 <= encut``."""
        try:
            cutoff = float(encut)
        except (TypeError, ValueError) as exc:
            raise ValueError("encut: expected a positive finite number") from exc
        if not np.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError("encut: expected a positive finite number")

        reciprocal = np.asarray(lattice.B, dtype=float)
        inverse_reciprocal = np.linalg.inv(reciprocal)
        gmax = np.sqrt(2.0 * cutoff)

        # For G = n @ B.T, the exact component bound uses columns of B^-1.
        # Include row norms too to preserve the documented convention and make
        # the rectangular search conservative for either matrix orientation.
        row_norms = np.linalg.norm(inverse_reciprocal, axis=1)
        column_norms = np.linalg.norm(inverse_reciprocal, axis=0)
        max_abs = np.ceil(gmax * np.maximum(row_norms, column_norms)).astype(int) + 1

        axes = [
            np.arange(-int(bound), int(bound) + 1, dtype=np.int64)
            for bound in max_abs
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        indices = np.stack(mesh, axis=-1).reshape(-1, 3)
        vectors = indices @ reciprocal.T
        kinetic = 0.5 * np.einsum("ij,ij->i", vectors, vectors)

        tolerance = 32.0 * np.finfo(float).eps * max(1.0, cutoff)
        included = kinetic <= cutoff + tolerance
        indices = indices[included]
        vectors = vectors[included]
        kinetic = kinetic[included]

        # Reciprocal vectors are ordered by energy, with integer indices as
        # deterministic tie-breakers. Zero is explicitly first for callers
        # that rely on the Gamma component's position.
        rounded_kinetic = np.round(kinetic, decimals=12)
        order = np.lexsort((indices[:, 2], indices[:, 1], indices[:, 0], rounded_kinetic))
        zero_position = np.flatnonzero(np.all(indices == 0, axis=1))[0]
        order = np.concatenate(([zero_position], order[order != zero_position]))

        indices = np.ascontiguousarray(indices[order])
        vectors = np.ascontiguousarray(vectors[order])
        kinetic = np.ascontiguousarray(kinetic[order])
        for array in (indices, vectors, kinetic):
            array.setflags(write=False)

        return cls(lattice, cutoff, indices, vectors, kinetic)

    @property
    def npw(self) -> int:
        """Number of plane waves in the basis."""
        return int(self.g_indices.shape[0])

    @property
    def max_abs_indices(self) -> np.ndarray:
        """Largest absolute integer index along each reciprocal direction."""
        maximum = np.max(np.abs(self.g_indices), axis=0)
        maximum.setflags(write=False)
        return maximum
