from copy import deepcopy

import pytest
import yaml

from mini_dft.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, EV_TO_HARTREE
from mini_dft.i_o import load_system
from mini_dft.system import CosinePotentialConfig


def _valid_input() -> dict:
    return {
        "units": {"length": "angstrom", "energy": "ev"},
        "lattice": {
            "vectors": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
        },
        "electrons": 2,
        "encut": 27.211386245988,
        "bands": 2,
        "potential": {
            "type": "cosine",
            "amplitude": -13.605693122994,
            "modes": [[1, 0, 0], [0, 1, 0]],
        },
        "scf": {
            "max_iterations": 100,
            "density_tolerance": 1.0e-7,
            "energy_tolerance": 2.7211386245988e-7,
            "mixing_alpha": 0.3,
            "eigensolver_tolerance": 1.0e-10,
        },
        "dos": {"enabled": True, "points": 1000, "width": 27.211386245988},
    }


def _write_input(tmp_path, contents: dict):
    path = tmp_path / "system.yaml"
    path.write_text(yaml.safe_dump(contents), encoding="utf-8")
    return path


def test_load_system_converts_values_to_atomic_units(tmp_path):
    """Catch loaders that retain the user-facing length or energy units."""
    system = load_system(_write_input(tmp_path, _valid_input()))

    assert system.encut == pytest.approx(1.0)
    assert system.lattice.A[0, 0] == pytest.approx(5.0 * ANGSTROM_TO_BOHR)
    assert isinstance(system.potential, CosinePotentialConfig)
    assert system.potential.amplitudes.tolist() == pytest.approx([-0.5, -0.5])
    assert system.dos.width == pytest.approx(1.0)
    assert system.scf.density_tolerance == pytest.approx(
        1.0e-7 * BOHR_TO_ANGSTROM**3
    )
    assert system.scf.energy_tolerance == pytest.approx(1.0e-8)
    assert system.input_energy_unit == "ev"


@pytest.mark.parametrize(
    ("mutate", "expected_path"),
    [
        (lambda data: data.__setitem__("electrons", 3), "electrons"),
        (lambda data: data.__setitem__("electrons", 6), "bands"),
        (
            lambda data: data["lattice"].__setitem__(
                "vectors", [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
            ),
            "lattice.vectors",
        ),
        (lambda data: data["units"].__setitem__("energy", "rydberg"), "units.energy"),
        (lambda data: data["scf"].__setitem__("mixing_alpha", 0.0), "scf.mixing_alpha"),
        (lambda data: data.pop("dos"), "dos"),
    ],
)
def test_load_system_reports_the_path_of_invalid_input(tmp_path, mutate, expected_path):
    """Catch validation failures that omit the offending YAML field path."""
    contents = deepcopy(_valid_input())
    mutate(contents)

    with pytest.raises(ValueError, match=expected_path):
        load_system(_write_input(tmp_path, contents))


def test_load_system_requires_all_cosine_amplitudes(tmp_path):
    """Catch cosine mode lists whose amplitudes cannot be mapped one-to-one."""
    contents = _valid_input()
    contents["potential"].pop("amplitude")
    contents["potential"]["amplitudes"] = [-1.0]

    with pytest.raises(ValueError, match="potential.amplitudes"):
        load_system(_write_input(tmp_path, contents))


def test_load_system_requires_at_least_two_dos_points(tmp_path):
    """Catch a YAML DOS request whose single sample cannot span an interval."""
    contents = _valid_input()
    contents["dos"]["points"] = 1

    with pytest.raises(ValueError, match="dos.points: expected an integer of at least 2"):
        load_system(_write_input(tmp_path, contents))


def test_load_system_validates_gaussian_atom_parameters(tmp_path):
    """Catch Gaussian wells that are repulsive or use a nonphysical width."""
    contents = _valid_input()
    contents["potential"] = {
        "type": "gaussian_atoms",
        "atoms": [{"position": [0.0, 0.0, 0.0], "depth": 1.0, "width": 0.0}],
    }

    with pytest.raises(ValueError, match="potential.atoms\\[0\\].depth"):
        load_system(_write_input(tmp_path, contents))
