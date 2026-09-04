"""YAML input loading and validation for Mini-DFT systems."""

from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .constants import energy_to_hartree, length_to_bohr
from .lattice import Lattice
from .system import (
    CosinePotentialConfig,
    DOSConfig,
    GaussianAtom,
    GaussianAtomsPotentialConfig,
    SCFConfig,
    SystemConfig,
)


def load_system(path: str | Path) -> SystemConfig:
    """Load a YAML system definition and convert all dimensional values to a.u."""
    with Path(path).open(encoding="utf-8") as input_file:
        raw = yaml.safe_load(input_file)
    root = _require_mapping(raw, "input")

    units = _require_mapping(_require_key(root, "units", "units"), "units")
    length_unit = _require_unit(units, "length", ("bohr", "angstrom"))
    energy_unit = _require_unit(units, "energy", ("hartree", "ev"))

    lattice = _load_lattice(root, length_unit)
    electrons = _require_integer(_require_key(root, "electrons", "electrons"), "electrons")
    if electrons < 0 or electrons % 2:
        raise ValueError("electrons: expected a non-negative even integer")
    encut = energy_to_hartree(
        _require_number(_require_key(root, "encut", "encut"), "encut"), energy_unit
    )
    if encut <= 0:
        raise ValueError("encut: expected a positive value")
    bands = _require_integer(_require_key(root, "bands", "bands"), "bands")
    if bands < 1:
        raise ValueError("bands: expected an integer of at least 1")
    if bands < electrons // 2:
        raise ValueError("bands: expected at least electrons // 2")

    potential = _load_potential(root, length_unit, energy_unit)
    scf = _load_scf(root, energy_unit)
    dos = _load_dos(root, energy_unit)
    return SystemConfig(
        lattice=lattice,
        electrons=electrons,
        encut=encut,
        bands=bands,
        potential=potential,
        scf=scf,
        dos=dos,
        input_length_unit=length_unit,
        input_energy_unit=energy_unit,
    )


def _load_lattice(root: Mapping[str, Any], length_unit: str) -> Lattice:
    lattice_mapping = _require_mapping(_require_key(root, "lattice", "lattice"), "lattice")
    rows = _require_sequence(_require_key(lattice_mapping, "vectors", "lattice.vectors"), "lattice.vectors")
    if len(rows) != 3:
        raise ValueError("lattice.vectors: expected exactly three row vectors")
    converted_rows: list[list[float]] = []
    for index, row in enumerate(rows):
        values = _require_sequence(row, f"lattice.vectors[{index}]")
        if len(values) != 3:
            raise ValueError(f"lattice.vectors[{index}]: expected exactly three values")
        converted_rows.append(
            [
                length_to_bohr(
                    _require_number(value, f"lattice.vectors[{index}][{column}]"), length_unit
                )
                for column, value in enumerate(values)
            ]
        )
    try:
        return Lattice.from_row_vectors(converted_rows)
    except ValueError as error:
        raise ValueError(f"lattice.vectors: {error}") from error


def _load_potential(
    root: Mapping[str, Any], length_unit: str, energy_unit: str
) -> CosinePotentialConfig | GaussianAtomsPotentialConfig:
    potential = _require_mapping(_require_key(root, "potential", "potential"), "potential")
    potential_type = _require_key(potential, "type", "potential.type")
    if not isinstance(potential_type, str):
        raise ValueError("potential.type: expected a string")
    if potential_type == "cosine":
        return _load_cosine_potential(potential, energy_unit)
    if potential_type == "gaussian_atoms":
        return _load_gaussian_atoms_potential(potential, length_unit, energy_unit)
    raise ValueError(f"potential.type: expected 'cosine' or 'gaussian_atoms', received {potential_type!r}")


def _load_cosine_potential(
    potential: Mapping[str, Any], energy_unit: str
) -> CosinePotentialConfig:
    modes_data = _require_sequence(_require_key(potential, "modes", "potential.modes"), "potential.modes")
    if not modes_data:
        raise ValueError("potential.modes: expected at least one mode")
    modes = np.asarray(
        [
            _integer_vector(mode, f"potential.modes[{index}]")
            for index, mode in enumerate(modes_data)
        ],
        dtype=int,
    )
    if "amplitude" in potential and "amplitudes" in potential:
        raise ValueError("potential: specify either amplitude or amplitudes, not both")
    if "amplitude" in potential:
        amplitude = energy_to_hartree(
            _require_number(potential["amplitude"], "potential.amplitude"), energy_unit
        )
        amplitudes = np.full(len(modes), amplitude, dtype=float)
    else:
        amplitudes_data = _require_sequence(
            _require_key(potential, "amplitudes", "potential.amplitudes"), "potential.amplitudes"
        )
        if len(amplitudes_data) != len(modes):
            raise ValueError("potential.amplitudes: expected one value for each potential mode")
        amplitudes = np.asarray(
            [
                energy_to_hartree(
                    _require_number(value, f"potential.amplitudes[{index}]"), energy_unit
                )
                for index, value in enumerate(amplitudes_data)
            ],
            dtype=float,
        )
    phases_data = potential.get("phases")
    if phases_data is None:
        phases = np.zeros(len(modes), dtype=float)
    else:
        phases_values = _require_sequence(phases_data, "potential.phases")
        if len(phases_values) != len(modes):
            raise ValueError("potential.phases: expected one value for each potential mode")
        phases = np.asarray(
            [
                _require_number(value, f"potential.phases[{index}]")
                for index, value in enumerate(phases_values)
            ],
            dtype=float,
        )
    return CosinePotentialConfig(
        amplitudes=_readonly_array(amplitudes),
        modes=_readonly_array(modes),
        phases=_readonly_array(phases),
    )


def _load_gaussian_atoms_potential(
    potential: Mapping[str, Any], length_unit: str, energy_unit: str
) -> GaussianAtomsPotentialConfig:
    atoms_data = _require_sequence(_require_key(potential, "atoms", "potential.atoms"), "potential.atoms")
    if not atoms_data:
        raise ValueError("potential.atoms: expected at least one atom")
    atoms: list[GaussianAtom] = []
    for index, atom_data in enumerate(atoms_data):
        path = f"potential.atoms[{index}]"
        atom = _require_mapping(atom_data, path)
        position = np.mod(
            np.asarray(
                _number_vector(_require_key(atom, "position", f"{path}.position"), f"{path}.position"),
                dtype=float,
            ),
            1.0,
        )
        depth = energy_to_hartree(
            _require_number(_require_key(atom, "depth", f"{path}.depth"), f"{path}.depth"), energy_unit
        )
        if depth >= 0:
            raise ValueError(f"{path}.depth: expected a negative energy")
        width = length_to_bohr(
            _require_number(_require_key(atom, "width", f"{path}.width"), f"{path}.width"), length_unit
        )
        if width <= 0:
            raise ValueError(f"{path}.width: expected a positive length")
        atoms.append(GaussianAtom(position=_readonly_array(position), depth=depth, width=width))
    return GaussianAtomsPotentialConfig(atoms=tuple(atoms))


def _load_scf(root: Mapping[str, Any], energy_unit: str) -> SCFConfig:
    scf = _require_mapping(_require_key(root, "scf", "scf"), "scf")
    max_iterations = _require_integer(
        _require_key(scf, "max_iterations", "scf.max_iterations"), "scf.max_iterations"
    )
    if max_iterations < 1:
        raise ValueError("scf.max_iterations: expected an integer of at least 1")
    density_tolerance = _require_positive_number(
        _require_key(scf, "density_tolerance", "scf.density_tolerance"), "scf.density_tolerance"
    )
    energy_tolerance = energy_to_hartree(
        _require_positive_number(
            _require_key(scf, "energy_tolerance", "scf.energy_tolerance"), "scf.energy_tolerance"
        ),
        energy_unit,
    )
    mixing_alpha = _require_number(
        _require_key(scf, "mixing_alpha", "scf.mixing_alpha"), "scf.mixing_alpha"
    )
    if not 0 < mixing_alpha <= 1:
        raise ValueError("scf.mixing_alpha: expected a value in (0, 1]")
    eigensolver_tolerance = _require_positive_number(
        _require_key(scf, "eigensolver_tolerance", "scf.eigensolver_tolerance"),
        "scf.eigensolver_tolerance",
    )
    max_eigensolver_iterations = _require_integer(
        scf.get("eigensolver_max_iterations", 2000), "scf.eigensolver_max_iterations"
    )
    if max_eigensolver_iterations < 1:
        raise ValueError("scf.eigensolver_max_iterations: expected an integer of at least 1")
    return SCFConfig(
        max_iterations=max_iterations,
        density_tolerance=density_tolerance,
        energy_tolerance=energy_tolerance,
        mixing_alpha=mixing_alpha,
        eigensolver_tolerance=eigensolver_tolerance,
        eigensolver_max_iterations=max_eigensolver_iterations,
    )


def _load_dos(root: Mapping[str, Any], energy_unit: str) -> DOSConfig:
    dos = _require_mapping(_require_key(root, "dos", "dos"), "dos")
    enabled = _require_key(dos, "enabled", "dos.enabled")
    if not isinstance(enabled, bool):
        raise ValueError("dos.enabled: expected a boolean")
    points = _require_integer(_require_key(dos, "points", "dos.points"), "dos.points")
    if points < 1:
        raise ValueError("dos.points: expected an integer of at least 1")
    width = energy_to_hartree(
        _require_positive_number(_require_key(dos, "width", "dos.width"), "dos.width"), energy_unit
    )
    return DOSConfig(enabled=enabled, points=points, width=width)


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a mapping, received {value!r}")
    return value


def _require_sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{path}: expected a sequence, received {value!r}")
    return value


def _require_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{path}: expected a finite number, received {value!r}")
    return float(value)


def _require_positive_number(value: Any, path: str) -> float:
    number = _require_number(value, path)
    if number <= 0:
        raise ValueError(f"{path}: expected a positive number")
    return number


def _require_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}: expected an integer, received {value!r}")
    return value


def _require_key(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{path}: required key is missing")
    return mapping[key]


def _require_unit(mapping: Mapping[str, Any], key: str, allowed: tuple[str, ...]) -> str:
    path = f"units.{key}"
    unit = _require_key(mapping, key, path)
    if not isinstance(unit, str) or unit.lower() not in allowed:
        expected = " or ".join(repr(value) for value in allowed)
        raise ValueError(f"{path}: expected {expected}, received {unit!r}")
    return unit.lower()


def _integer_vector(value: Any, path: str) -> list[int]:
    sequence = _require_sequence(value, path)
    if len(sequence) != 3:
        raise ValueError(f"{path}: expected exactly three integers")
    return [_require_integer(component, f"{path}[{index}]") for index, component in enumerate(sequence)]


def _number_vector(value: Any, path: str) -> list[float]:
    sequence = _require_sequence(value, path)
    if len(sequence) != 3:
        raise ValueError(f"{path}: expected exactly three numbers")
    return [_require_number(component, f"{path}[{index}]") for index, component in enumerate(sequence)]


def _readonly_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values).copy()
    array.setflags(write=False)
    return array
