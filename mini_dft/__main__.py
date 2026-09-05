"""Command-line entry point for Mini-DFT."""

import argparse
import sys
from collections.abc import Sequence

import yaml

from .constants import HARTREE_TO_EV
from .i_o import load_system, write_results
from .scf import SCFIteration, SCFRunner


def main(argv: Sequence[str] | None = None) -> int:
    """Run one YAML-defined SCF calculation and return a documented exit code."""
    parser = argparse.ArgumentParser(prog="mini-dft")
    parser.add_argument("input", help="YAML system definition")
    parser.add_argument("--output", default="output", help="directory for result files")
    parser.add_argument("--quiet", action="store_true", help="suppress SCF progress output")
    try:
        arguments = parser.parse_args(argv)
    except SystemExit as error:
        return 0 if error.code == 0 else 1

    callback = None if arguments.quiet else _print_iteration
    if not arguments.quiet:
        print("iter density_rms delta_energy[Ha] model_energy[Ha]")
    try:
        system = load_system(arguments.input)
        result = SCFRunner(system).run(callback=callback)
        write_results(result, system, arguments.output)
    except (OSError, yaml.YAMLError, ValueError) as error:
        print(f"mini-dft: {error}", file=sys.stderr)
        return 1

    if not arguments.quiet:
        print(result.message)
        print(
            f"final model energy: {result.energy.total:.12e} Ha = "
            f"{result.energy.total * HARTREE_TO_EV:.12e} eV"
        )
    return 0 if result.converged else 2


def _print_iteration(iteration: SCFIteration) -> None:
    print(
        f"{iteration.iteration:4d} "
        f"{iteration.density_rms:.6e} "
        f"{iteration.delta_energy:.6e} "
        f"{iteration.model_energy:.6e}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
