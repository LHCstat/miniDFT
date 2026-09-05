"""Self-consistent-field orchestration for the Mini-DFT model."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .basis import PlaneWaveBasis
from .density import density_from_coefficients
from .eigensolver import EigenResult, solve_lowest
from .energy import EnergyComponents, calculate_energy, eigenvalue_energy
from .fft_grid import FFTGrid
from .hamiltonian import Hamiltonian
from .hartree import HartreeResult, hartree_from_density
from .ionic_potential import build_ionic_potential
from .mixing import density_rms_residual, mix_density
from .occupations import integer_occupations
from .potentials import PotentialSet
from .system import SystemConfig
from .xc import XCResult, lda_pz81


@dataclass(frozen=True)
class SCFIteration:
    """One recorded input/output density comparison in the SCF loop."""

    iteration: int
    density_rms: float
    energy_delta: float
    energy: EnergyComponents
    eigenvalue_energy: float

    @property
    def delta_energy(self) -> float:
        """Alias matching the CLI history-column spelling."""
        return self.energy_delta

    @property
    def model_energy(self) -> float:
        """Model total energy for compact progress reporting."""
        return self.energy.total


@dataclass(frozen=True)
class SCFResult:
    """Complete converged or exhausted SCF state."""

    converged: bool
    iterations: int
    basis: PlaneWaveBasis
    grid: FFTGrid
    density: np.ndarray
    potentials: PotentialSet
    eigenvalues: np.ndarray
    coefficients: np.ndarray
    occupations: np.ndarray
    energy: EnergyComponents
    history: tuple[SCFIteration, ...]
    message: str


@dataclass(frozen=True)
class _SCFStep:
    """All values produced by one Hamiltonian solve from an explicit density."""

    input_density: np.ndarray
    output_density: np.ndarray
    hartree: HartreeResult
    xc: XCResult
    potentials: PotentialSet
    eigenstates: EigenResult
    energy: EnergyComponents
    eigenvalue_energy: float


class SCFRunner:
    """Build and run a Gamma-point plane-wave self-consistent-field problem."""

    def __init__(self, system: SystemConfig) -> None:
        self.system = system
        self.basis = PlaneWaveBasis.from_cutoff(system.lattice, system.encut)
        if system.bands > self.basis.npw:
            raise ValueError(
                "bands: requested "
                f"{system.bands}, but the plane-wave basis contains {self.basis.npw}"
            )
        self.grid = FFTGrid.from_basis(self.basis)
        self.occupations = integer_occupations(system.electrons, system.bands)
        self.ionic_potential = build_ionic_potential(system.potential, self.grid)
        self.density = np.full(
            self.grid.shape,
            system.electrons / system.lattice.volume,
            dtype=float,
        )

    def _iterate(self, input_density: np.ndarray) -> _SCFStep:
        """Solve once using fields constructed from ``input_density``."""
        density_input = np.asarray(input_density, dtype=float).copy()
        hartree = hartree_from_density(density_input, self.grid)
        xc = lda_pz81(density_input, self.grid)
        potentials = PotentialSet(
            ionic=self.ionic_potential,
            hartree=hartree.potential,
            xc=xc.potential,
        )
        hamiltonian = Hamiltonian(self.basis, self.grid, potentials.effective)
        eigenstates = solve_lowest(
            hamiltonian,
            n_bands=self.system.bands,
            tolerance=self.system.scf.eigensolver_tolerance,
            max_iterations=self.system.scf.eigensolver_max_iterations,
        )
        output_density = density_from_coefficients(
            eigenstates.coefficients, self.occupations, self.grid
        )
        energy = calculate_energy(
            eigenstates.coefficients,
            self.occupations,
            density_input,
            self.ionic_potential,
            hartree,
            xc,
            self.grid,
        )
        diagnostic = eigenvalue_energy(
            eigenstates.eigenvalues,
            self.occupations,
            density_input,
            hartree.potential,
            xc.potential,
            xc.energy,
            self.grid,
        )
        return _SCFStep(
            input_density=density_input,
            output_density=output_density,
            hartree=hartree,
            xc=xc,
            potentials=potentials,
            eigenstates=eigenstates,
            energy=energy,
            eigenvalue_energy=diagnostic,
        )

    def run(
        self, callback: Callable[[SCFIteration], None] | None = None
    ) -> SCFResult:
        """Run SCF until a consistency solve passes both configured tolerances."""
        scf = self.system.scf
        input_density = self.density.copy()
        history: list[SCFIteration] = []
        previous_energy: float | None = None
        latest_step: _SCFStep | None = None

        def record_step(step: _SCFStep) -> SCFIteration:
            nonlocal previous_energy
            density_rms = density_rms_residual(
                step.input_density, step.output_density
            )
            energy_delta = (
                float("inf")
                if previous_energy is None
                else abs(step.energy.total - previous_energy)
            )
            record = SCFIteration(
                iteration=len(history) + 1,
                density_rms=density_rms,
                energy_delta=energy_delta,
                energy=step.energy,
                eigenvalue_energy=step.eigenvalue_energy,
            )
            history.append(record)
            previous_energy = step.energy.total
            if callback is not None:
                callback(record)
            return record

        for normal_iteration in range(1, scf.max_iterations + 1):
            step = self._iterate(input_density)
            latest_step = step
            record = record_step(step)
            tolerances_pass = (
                normal_iteration >= 2
                and record.density_rms <= scf.density_tolerance
                and record.energy_delta <= scf.energy_tolerance
            )
            if tolerances_pass:
                # The current eigenpairs generated this accepted density but
                # were solved from the preceding input. Re-solve without
                # mixing so every stored final field belongs to one state.
                consistency_step = self._iterate(step.output_density)
                latest_step = consistency_step
                consistency_record = record_step(consistency_step)
                consistency_passes = (
                    consistency_record.density_rms <= scf.density_tolerance
                    and consistency_record.energy_delta <= scf.energy_tolerance
                )
                if consistency_passes:
                    self.density = consistency_step.input_density.copy()
                    return self._result(
                        converged=True,
                        step=consistency_step,
                        history=history,
                        message=f"SCF converged in {len(history)} iterations",
                    )
                input_density = mix_density(
                    consistency_step.input_density,
                    consistency_step.output_density,
                    alpha=scf.mixing_alpha,
                    electrons=self.system.electrons,
                    grid=self.grid,
                )
            else:
                input_density = mix_density(
                    step.input_density,
                    step.output_density,
                    alpha=scf.mixing_alpha,
                    electrons=self.system.electrons,
                    grid=self.grid,
                )
            self.density = input_density.copy()

        assert latest_step is not None
        self.density = latest_step.input_density.copy()
        return self._result(
            converged=False,
            step=latest_step,
            history=history,
            message="maximum SCF iterations reached without convergence",
        )

    def _result(
        self,
        *,
        converged: bool,
        step: _SCFStep,
        history: list[SCFIteration],
        message: str,
    ) -> SCFResult:
        """Assemble a public result from one complete, internally consistent step."""
        return SCFResult(
            converged=converged,
            iterations=len(history),
            basis=self.basis,
            grid=self.grid,
            density=step.input_density.copy(),
            potentials=step.potentials,
            eigenvalues=step.eigenstates.eigenvalues.copy(),
            coefficients=step.eigenstates.coefficients.copy(),
            occupations=self.occupations.copy(),
            energy=step.energy,
            history=tuple(history),
            message=message,
        )
