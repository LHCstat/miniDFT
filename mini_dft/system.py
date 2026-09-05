"""Immutable, validated-at-load-time system configuration objects."""

from dataclasses import dataclass

import numpy as np

from .lattice import Lattice


@dataclass(frozen=True)
class SCFConfig:
    max_iterations: int
    density_tolerance: float
    energy_tolerance: float
    mixing_alpha: float
    eigensolver_tolerance: float
    eigensolver_max_iterations: int = 2000


@dataclass(frozen=True)
class DOSConfig:
    enabled: bool
    points: int
    width: float


@dataclass(frozen=True)
class CosinePotentialConfig:
    amplitudes: np.ndarray
    modes: np.ndarray
    phases: np.ndarray


@dataclass(frozen=True)
class GaussianAtom:
    position: np.ndarray
    depth: float
    width: float

    @property
    def fractional_position(self) -> np.ndarray:
        """The atom position in periodic fractional coordinates."""
        return self.position


@dataclass(frozen=True)
class GaussianAtomsPotentialConfig:
    atoms: tuple[GaussianAtom, ...]


@dataclass(frozen=True)
class SystemConfig:
    lattice: Lattice
    electrons: int
    encut: float
    bands: int
    potential: CosinePotentialConfig | GaussianAtomsPotentialConfig
    scf: SCFConfig
    dos: DOSConfig
    input_length_unit: str
    input_energy_unit: str
