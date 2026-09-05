"""FFT-grid mappings between plane-wave coefficients and real-space fields."""

from dataclasses import dataclass

import numpy as np
from scipy.fft import next_fast_len

from .basis import PlaneWaveBasis


@dataclass(frozen=True)
class FFTGrid:
    """A basis-compatible periodic FFT grid with explicit physical scaling."""

    basis: PlaneWaveBasis
    shape: tuple[int, int, int]

    @classmethod
    def from_basis(cls, basis: PlaneWaveBasis) -> "FFTGrid":
        """Create a grid that can represent all products of basis functions."""
        maximum = basis.max_abs_indices
        shape = tuple(
            int(next_fast_len(max(1, 4 * int(index) + 1))) for index in maximum
        )
        return cls(basis=basis, shape=shape)

    @property
    def ngrid(self) -> int:
        """Total number of grid points."""
        return int(np.prod(self.shape))

    @property
    def volume(self) -> float:
        """Real-space cell volume."""
        return self.basis.lattice.volume

    @property
    def _coefficient_indices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map signed reciprocal indices to NumPy FFT array positions."""
        indices = np.mod(self.basis.g_indices, np.asarray(self.shape, dtype=np.int64))
        return tuple(indices[:, axis] for axis in range(3))

    def coefficients_to_real(self, coefficients: np.ndarray) -> np.ndarray:
        """Transform plane-wave coefficients to normalized real-space values."""
        coefficient_array = np.asarray(coefficients, dtype=np.complex128)
        self._require_coefficient_shape(coefficient_array)

        reciprocal = np.zeros(
            (*coefficient_array.shape[:-1], *self.shape), dtype=np.complex128
        )
        reciprocal[(..., *self._coefficient_indices)] = coefficient_array
        return np.fft.ifftn(reciprocal, axes=(-3, -2, -1)) * (
            self.ngrid / np.sqrt(self.volume)
        )

    def real_to_coefficients(self, field: np.ndarray) -> np.ndarray:
        """Transform normalized real-space wavefunctions to plane-wave coefficients."""
        field_array = np.asarray(field, dtype=np.complex128)
        self._require_field_shape(field_array)

        reciprocal = np.fft.fftn(field_array, axes=(-3, -2, -1)) * (
            np.sqrt(self.volume) / self.ngrid
        )
        return reciprocal[(..., *self._coefficient_indices)]

    def field_to_fourier(self, field: np.ndarray) -> np.ndarray:
        """Transform a scalar field using its volume-independent normalization."""
        field_array = np.asarray(field)
        self._require_field_shape(field_array)
        return np.fft.fftn(field_array, axes=(-3, -2, -1)) / self.ngrid

    def fourier_to_field(self, fourier: np.ndarray) -> np.ndarray:
        """Transform scalar Fourier coefficients back to a real-space field."""
        fourier_array = np.asarray(fourier, dtype=np.complex128)
        self._require_field_shape(fourier_array)
        return np.fft.ifftn(fourier_array, axes=(-3, -2, -1)) * self.ngrid

    def integrate(self, field: np.ndarray) -> np.ndarray:
        """Apply the periodic-cell quadrature over the three grid axes."""
        field_array = np.asarray(field)
        self._require_field_shape(field_array)
        return self.volume * np.mean(field_array, axis=(-3, -2, -1))

    def fractional_mesh(self) -> np.ndarray:
        """Return fractional coordinates for each grid point in ``[0, 1)``."""
        axes = [np.arange(size, dtype=float) / size for size in self.shape]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)

    def reciprocal_squared(self) -> np.ndarray:
        """Return ``|G|²`` at every FFT reciprocal-grid point."""
        axes = [np.fft.fftfreq(size) * size for size in self.shape]
        indices = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        vectors = indices @ self.basis.lattice.B.T
        return np.einsum("...i,...i->...", vectors, vectors)

    def _require_coefficient_shape(self, coefficients: np.ndarray) -> None:
        if coefficients.ndim < 1 or coefficients.shape[-1] != self.basis.npw:
            raise ValueError(
                "coefficients: expected shape (..., "
                f"{self.basis.npw}), received {coefficients.shape}"
            )

    def _require_field_shape(self, field: np.ndarray) -> None:
        if field.ndim < 3 or tuple(field.shape[-3:]) != self.shape:
            raise ValueError(
                "field: expected shape (..., "
                f"{self.shape[0]}, {self.shape[1]}, {self.shape[2]}), "
                f"received {field.shape}"
            )
