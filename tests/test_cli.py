import csv

import numpy as np
import pytest
import yaml

from mini_dft.__main__ import main
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
    assert summary["energies"]["total_ev"] == pytest.approx(result.energy.total * 27.211386245988)

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


def test_cli_returns_success_and_writes_output_for_converged_system(tmp_path, capsys):
    """Catch a usable run that does not return success or serialize its result."""
    input_path = _write_input(tmp_path)
    output = tmp_path / "results"

    assert main([str(input_path), "--output", str(output), "--quiet"]) == 0
    assert (output / "summary.yaml").is_file()
    assert capsys.readouterr().err == ""


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
