"""Matrix-free kinetic-plus-local-potential Hamiltonian."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator

from .basis import PlaneWaveBasis
from .fft_grid import FFTGrid


@dataclass(frozen=True)
class Hamiltonian:
    """A Gamma-point plane-wave Hamiltonian applied through FFTs."""

    basis: PlaneWaveBasis
    grid: FFTGrid
    local_potential: np.ndarray

    def __post_init__(self) -> None:
        """Validate and retain a real, finite grid potential."""
        if self.grid.basis is not self.basis:
            raise ValueError("grid: expected the Hamiltonian basis by object identity")

        potential = np.asarray(self.local_potential)
        if np.iscomplexobj(potential):
            raise ValueError("local_potential: expected a real-valued array")
        if potential.shape != self.grid.shape:
            raise ValueError(
                "local_potential: expected shape "
                f"{self.grid.shape}, received {potential.shape}"
            )
        if not np.all(np.isfinite(potential)):
            raise ValueError("local_potential: expected finite values")

        object.__setattr__(self, "local_potential", potential.astype(float, copy=True))

    def apply(self, coefficients: np.ndarray) -> np.ndarray:
        """Apply the kinetic and local-potential terms to one coefficient vector."""
        coefficient_array = self._require_vector(coefficients)
        kinetic = self.basis.kinetic_energies * coefficient_array
        psi = self.grid.coefficients_to_real(coefficient_array)
        local = self.grid.real_to_coefficients(self.local_potential * psi)
        return np.asarray(kinetic + local, dtype=np.complex128)

    def apply_many(self, coefficients: np.ndarray) -> np.ndarray:
        """Apply the Hamiltonian independently to a batch of coefficient vectors."""
        coefficient_array = self._require_batch(coefficients)
        kinetic = self.basis.kinetic_energies * coefficient_array
        psi = self.grid.coefficients_to_real(coefficient_array)
        local = self.grid.real_to_coefficients(self.local_potential * psi)
        return np.asarray(kinetic + local, dtype=np.complex128)

    def as_linear_operator(self) -> LinearOperator:
        """Expose this matrix-free action to SciPy iterative solvers."""
        return LinearOperator(
            shape=(self.basis.npw, self.basis.npw),
            matvec=self.apply,
            dtype=np.complex128,
        )

    def _require_vector(self, coefficients: np.ndarray) -> np.ndarray:
        coefficient_array = np.asarray(coefficients, dtype=np.complex128)
        if coefficient_array.shape != (self.basis.npw,):
            raise ValueError(
                "coefficients: expected shape "
                f"({self.basis.npw},), received {coefficient_array.shape}"
            )
        return coefficient_array

    def _require_batch(self, coefficients: np.ndarray) -> np.ndarray:
        coefficient_array = np.asarray(coefficients, dtype=np.complex128)
        if coefficient_array.ndim != 2 or coefficient_array.shape[1:] != (self.basis.npw,):
            raise ValueError(
                "coefficients: expected shape "
                f"(n_bands, {self.basis.npw}), received {coefficient_array.shape}"
            )
        return coefficient_array
