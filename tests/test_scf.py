from dataclasses import replace

import numpy as np
import pytest

from mini_dft.density import density_from_coefficients, density_integral
from mini_dft.energy import calculate_energy, eigenvalue_energy
from mini_dft.hamiltonian import Hamiltonian
from mini_dft.hartree import hartree_from_density
from mini_dft.lattice import Lattice
from mini_dft.mixing import density_rms_residual
from mini_dft.scf import SCFRunner
from mini_dft.system import (
    CosinePotentialConfig,
    DOSConfig,
    GaussianAtom,
    GaussianAtomsPotentialConfig,
    SCFConfig,
    SystemConfig,
)
from mini_dft.wavefunction import orthonormality_error
from mini_dft.xc import lda_pz81


def make_system(potential) -> SystemConfig:
    return SystemConfig(
        lattice=Lattice.from_row_vectors(np.eye(3) * 8.0),
        electrons=2,
        encut=1.0,
        bands=4,
        potential=potential,
        scf=SCFConfig(
            max_iterations=80,
            density_tolerance=2.0e-7,
            energy_tolerance=2.0e-7,
            mixing_alpha=0.35,
            eigensolver_tolerance=1.0e-10,
        ),
        dos=DOSConfig(enabled=False, points=100, width=0.01),
        input_length_unit="bohr",
        input_energy_unit="hartree",
    )


def assert_consistent_converged_result(result, system: SystemConfig) -> None:
    """Assert the public result represents one final self-consistent state."""
    assert result.converged
    assert result.iterations == len(result.history)
    assert result.iterations >= 3
    assert density_integral(result.density, result.grid) == pytest.approx(
        system.electrons, abs=1.0e-10
    )
    assert np.all(np.diff(result.eigenvalues) >= -1.0e-12)
    assert orthonormality_error(result.coefficients) < 1.0e-9
    assert np.isfinite(result.energy.total)
    assert result.history[-1].density_rms < system.scf.density_tolerance
    assert result.history[-1].energy_delta < system.scf.energy_tolerance

    output_density = density_from_coefficients(
        result.coefficients, result.occupations, result.grid
    )
    final_residual = density_rms_residual(result.density, output_density)
    assert final_residual == pytest.approx(result.history[-1].density_rms, abs=1.0e-13)
    assert final_residual < system.scf.density_tolerance

    hartree = hartree_from_density(result.density, result.grid)
    xc = lda_pz81(result.density, result.grid)
    assert np.allclose(result.potentials.hartree, hartree.potential, atol=1.0e-12)
    assert np.allclose(result.potentials.xc, xc.potential, atol=1.0e-12)

    hamiltonian = Hamiltonian(result.basis, result.grid, result.potentials.effective)
    eigenpair_residuals = np.linalg.norm(
        hamiltonian.apply_many(result.coefficients)
        - result.eigenvalues[:, np.newaxis] * result.coefficients,
        axis=1,
    )
    assert np.max(eigenpair_residuals) < 1.0e-8

    recomputed_energy = calculate_energy(
        result.coefficients,
        result.occupations,
        result.density,
        result.potentials.ionic,
        hartree,
        xc,
        result.grid,
    )
    assert result.energy.total == pytest.approx(recomputed_energy.total, abs=1.0e-12)

    diagnostic = eigenvalue_energy(
        result.eigenvalues,
        result.occupations,
        result.density,
        result.potentials.hartree,
        result.potentials.xc,
        xc.energy,
        result.grid,
    )
    diagnostic_bound = (
        result.grid.volume
        * final_residual
        * float(np.sqrt(np.mean(result.potentials.effective**2)))
        + 1.0e-8
    )
    assert abs(result.energy.total - diagnostic) < diagnostic_bound


def test_cosine_scf_converges_to_one_consistent_structured_result():
    """Catch convergence assembled from density, fields, and states of different steps."""
    system = make_system(
        CosinePotentialConfig(
            amplitudes=np.array([-0.05]),
            modes=np.array([[1, 0, 0]]),
            phases=np.array([0.0]),
        )
    )
    observed = []

    result = SCFRunner(system).run(callback=observed.append)

    assert_consistent_converged_result(result, system)
    assert observed == list(result.history)
    assert "converged" in result.message.lower()


def test_centered_gaussian_scf_converges_to_a_nonuniform_density():
    """Catch an SCF loop that ignores the configured Gaussian ionic potential."""
    system = make_system(
        GaussianAtomsPotentialConfig(
            atoms=(
                GaussianAtom(
                    position=np.array([0.5, 0.5, 0.5]),
                    depth=-0.05,
                    width=0.8,
                ),
            )
        )
    )

    result = SCFRunner(system).run()

    assert_consistent_converged_result(result, system)
    assert np.ptp(result.density) > 1.0e-5


def test_scf_iteration_limit_returns_a_complete_nonconverged_result():
    """Catch an exhausted SCF loop that silently succeeds or drops its last state."""
    system = make_system(
        CosinePotentialConfig(
            amplitudes=np.array([-0.05]),
            modes=np.array([[1, 0, 0]]),
            phases=np.array([0.0]),
        )
    )
    system = replace(
        system,
        scf=replace(
            system.scf,
            max_iterations=1,
            density_tolerance=1.0e-14,
            energy_tolerance=1.0e-14,
        ),
    )

    result = SCFRunner(system).run()

    assert result.converged is False
    assert result.iterations == 1
    assert len(result.history) == 1
    assert "maximum SCF iterations" in result.message
    assert result.density.shape == result.grid.shape
    assert result.potentials.effective.shape == result.grid.shape
    assert result.eigenvalues.shape == (system.bands,)
    assert result.coefficients.shape == (system.bands, result.basis.npw)
    assert result.occupations.shape == (system.bands,)
    assert np.isfinite(result.energy.total)


def test_failed_consistency_residual_returns_to_normal_scf_iterations():
    """Catch convergence claimed after an unverified final consistency solve."""
    system = make_system(
        CosinePotentialConfig(
            amplitudes=np.array([-0.05]),
            modes=np.array([[1, 0, 0]]),
            phases=np.array([0.0]),
        )
    )
    system = replace(
        system,
        scf=replace(
            system.scf,
            max_iterations=5,
            density_tolerance=1.0e-6,
            energy_tolerance=1.0,
        ),
    )
    runner = SCFRunner(system)
    physical_iterate = runner._iterate
    amplitudes = iter([5.0e-5, 1.0e-8, 5.0e-5, 1.0e-8, 1.0e-8])
    mode = np.cos(2.0 * np.pi * runner.grid.fractional_mesh()[..., 0])

    def controlled_iterate(input_density):
        step = physical_iterate(input_density)
        return replace(
            step,
            output_density=step.input_density + next(amplitudes) * mode,
        )

    runner._iterate = controlled_iterate

    result = runner.run()

    assert result.converged
    assert result.iterations == 5
    assert result.history[1].density_rms < system.scf.density_tolerance
    assert result.history[2].density_rms > system.scf.density_tolerance
    assert result.history[-1].density_rms < system.scf.density_tolerance


def test_final_consistency_solve_runs_after_normal_iteration_budget_is_met():
    """Catch omission of the mandatory final solve at the normal-loop boundary."""
    system = make_system(
        CosinePotentialConfig(
            amplitudes=np.array([0.0]),
            modes=np.array([[1, 0, 0]]),
            phases=np.array([0.0]),
        )
    )
    system = replace(system, scf=replace(system.scf, max_iterations=2))

    result = SCFRunner(system).run()

    assert result.converged
    assert result.iterations == 3
    assert len(result.history) == 3
    assert result.history[-1].density_rms < system.scf.density_tolerance
    assert result.history[-1].energy_delta < system.scf.energy_tolerance
