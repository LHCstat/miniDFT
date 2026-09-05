import csv
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from mini_dft.__main__ import main
from mini_dft.constants import BOHR_TO_ANGSTROM, HARTREE_TO_EV
from mini_dft.density import density_integral
from mini_dft.i_o import load_system, write_results
from mini_dft.scf import SCFRunner


def _input_data(*, max_iterations: int = 80) -> dict:
    return {
        "units": {"length": "bohr", "energy": "hartree"},
        "lattice": {"vectors": [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]]},
        "electrons": 2,
        "encut": 1.0,
        "bands": 4,
        "potential": {"type": "cosine", "amplitude": -0.05, "modes": [[1, 0, 0]]},
        "scf": {
            "max_iterations": max_iterations,
            "density_tolerance": 2.0e-7,
            "energy_tolerance": 2.0e-7,
            "mixing_alpha": 0.35,
            "eigensolver_tolerance": 1.0e-10,
        },
        "dos": {"enabled": True, "points": 101, "width": 0.05},
    }


def _write_input(tmp_path, *, max_iterations: int = 80):
    input_path = tmp_path / "system.yaml"
    input_path.write_text(yaml.safe_dump(_input_data(max_iterations=max_iterations)), encoding="utf-8")
    return input_path


@pytest.mark.parametrize(
    ("example_path", "expected_density_tolerance"),
    [
        (Path("mini_dft/input.yaml"), 2.0e-7 * BOHR_TO_ANGSTROM**3),
        (Path("examples/gaussian_atoms.yaml"), 2.0e-7),
    ],
)
def test_repository_examples_convert_density_tolerance_to_bohr_inverse_cubed(
    example_path, expected_density_tolerance
):
    """Catch example tolerances that retain their declared inverse-volume units."""
    system = load_system(example_path)

    assert system.scf.density_tolerance == pytest.approx(expected_density_tolerance)


def test_write_results_preserves_complete_converged_state(tmp_path):
    """Catch output that drops fields, units, complex coefficients, or convergence diagnostics."""
    system = load_system(_write_input(tmp_path))
    result = SCFRunner(system).run()
    assert result.converged

    output = tmp_path / "results"
    write_results(result, system, output)

    assert {path.name for path in output.iterdir()} == {
        "summary.yaml",
        "scf_history.csv",
        "eigenvalues.csv",
        "dos.csv",
        "fields.npz",
        "wavefunctions.npz",
    }
    summary = yaml.safe_load((output / "summary.yaml").read_text(encoding="utf-8"))
    assert summary["converged"] is True
    assert summary["input_units"] == {"length": "bohr", "energy": "hartree"}
    assert summary["basis"]["npw"] == result.basis.npw
    assert summary["basis"]["fft_shape"] == list(result.grid.shape)
    assert summary["hartree_g0"] == "zero_neutralizing_background"
    assert summary["charge_integral"] == pytest.approx(density_integral(result.density, result.grid))
    assert summary["energies"]["total_hartree"] == pytest.approx(result.energy.total)
    assert summary["energies"]["total_ev"] == pytest.approx(
        result.energy.total * HARTREE_TO_EV
    )

    with (output / "scf_history.csv").open(newline="", encoding="utf-8") as history_file:
        history = list(csv.DictReader(history_file))
    with (output / "eigenvalues.csv").open(newline="", encoding="utf-8") as eigenvalue_file:
        eigenvalues = list(csv.DictReader(eigenvalue_file))
    with (output / "dos.csv").open(newline="", encoding="utf-8") as dos_file:
        dos = list(csv.DictReader(dos_file))
    assert len(history) == result.iterations
    assert len(eigenvalues) == system.bands
    assert len(dos) == system.dos.points

    with np.load(output / "fields.npz") as fields, np.load(output / "wavefunctions.npz") as wavefunctions:
        assert fields["density"].shape == result.grid.shape
        assert fields["ionic_potential"].shape == result.grid.shape
        assert wavefunctions["g_indices"].shape == (result.basis.npw, 3)
        assert wavefunctions["coefficients"].shape == result.coefficients.shape
        assert np.iscomplexobj(wavefunctions["coefficients"])
        assert np.allclose(wavefunctions["coefficients"], result.coefficients)


def test_write_results_removes_only_stale_dos_from_reused_output_directory(tmp_path):
    """Catch DOS-disabled rewrites that leave a misleading prior DOS artifact."""
    system = load_system(_write_input(tmp_path))
    result = SCFRunner(system).run()
    assert result.converged
    output = tmp_path / "results"
    write_results(result, system, output)
    assert (output / "dos.csv").is_file()
    sibling_dos = tmp_path / "dos.csv"
    sibling_dos.write_text("outside output directory\n", encoding="utf-8")

    write_results(result, replace(system, dos=replace(system.dos, enabled=False)), output)

    assert not (output / "dos.csv").exists()
    assert sibling_dos.read_text(encoding="utf-8") == "outside output directory\n"


def test_cli_returns_success_and_writes_output_for_converged_system(tmp_path, capsys):
    """Catch a usable run that does not return success or serialize its result."""
    input_path = _write_input(tmp_path)
    output = tmp_path / "results"

    assert main([str(input_path), "--output", str(output), "--quiet"]) == 0
    assert (output / "summary.yaml").is_file()
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    ("max_iterations", "expected_exit", "expected_status"),
    [
        (80, 0, "SCF converged"),
        (1, 2, "maximum SCF iterations reached without convergence"),
    ],
)
def test_cli_labels_progress_energies_and_prints_final_energy_in_hartree_and_ev(
    tmp_path, capsys, max_iterations, expected_exit, expected_status
):
    """Catch ambiguous progress units or a missing final energy on either exit path."""
    input_path = _write_input(tmp_path, max_iterations=max_iterations)
    output = tmp_path / f"results-{max_iterations}"

    assert main([str(input_path), "--output", str(output)]) == expected_exit

    stdout = capsys.readouterr().out
    header = stdout.splitlines()[0]
    assert "delta_energy[Ha]" in header
    assert "model_energy[Ha]" in header
    assert expected_status in stdout
    energy_line = re.search(
        r"final model energy: (?P<hartree>[-+0-9.e]+) Ha = "
        r"(?P<ev>[-+0-9.e]+) eV",
        stdout,
    )
    assert energy_line is not None
    summary = yaml.safe_load((output / "summary.yaml").read_text(encoding="utf-8"))
    displayed_hartree = float(energy_line.group("hartree"))
    displayed_ev = float(energy_line.group("ev"))
    assert displayed_hartree == pytest.approx(
        summary["energies"]["total_hartree"], rel=1.0e-12
    )
    assert displayed_ev == pytest.approx(displayed_hartree * HARTREE_TO_EV, rel=1.0e-12)


def test_cli_reports_yaml_input_errors_without_traceback(tmp_path, capsys):
    """Catch parse errors that escape the command boundary as a traceback."""
    input_path = tmp_path / "broken.yaml"
    input_path.write_text("units: [unterminated", encoding="utf-8")

    assert main([str(input_path)]) == 1
    error = capsys.readouterr().err
    assert error.startswith("mini-dft: ")
    assert "Traceback" not in error


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        ([], "the following arguments are required: input"),
        (["--output"], "argument --output: expected one argument"),
        (["system.yaml", "--not-an-option"], "unrecognized arguments: --not-an-option"),
    ],
)
def test_cli_argument_errors_return_one_without_raising(argv, message, capsys):
    """Catch argparse SystemExit(2) leaking through main's integer-return contract."""
    assert main(argv) == 1

    error = capsys.readouterr().err
    assert "usage: mini-dft" in error
    assert message in error
    assert "Traceback" not in error


def test_cli_returns_nonconvergence_code_after_writing_diagnostics(tmp_path):
    """Catch exhausted SCF runs that lose diagnostics or report successful completion."""
    input_path = _write_input(tmp_path, max_iterations=1)
    output = tmp_path / "unconverged"

    assert main([str(input_path), "--output", str(output), "--quiet"]) == 2
    summary = yaml.safe_load((output / "summary.yaml").read_text(encoding="utf-8"))
    assert summary["converged"] is False
    assert (output / "scf_history.csv").is_file()


@pytest.mark.parametrize(
    "example_path",
    [
        Path("mini_dft/input.yaml"),
        Path("examples/gaussian_atoms.yaml"),
    ],
)
def test_repository_examples_converge_and_write_reloadable_results(tmp_path, example_path):
    """Keep the checked-in cosine and Gaussian examples runnable end to end."""
    system = load_system(example_path)
    result = SCFRunner(system).run()
    assert result.converged, result.message

    output = tmp_path / example_path.stem
    write_results(result, system, output)

    summary = yaml.safe_load((output / "summary.yaml").read_text(encoding="utf-8"))
    assert summary["converged"] is True
    assert summary["charge_integral"] == pytest.approx(float(system.electrons))
    with np.load(output / "fields.npz") as fields:
        assert fields["density"].shape == result.grid.shape
