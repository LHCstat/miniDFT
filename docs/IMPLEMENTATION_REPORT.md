# Mini-DFT v0.1 Implementation Report

## Outcome and boundary

Mini-DFT v0.1 is a runnable, tested educational Gamma-point plane-wave Kohn–Sham SCF code.  It accepts validated YAML, uses Hartree atomic units internally, solves a matrix-free complex Hamiltonian, writes portable results, and includes two CLI-runnable examples.

The implementation reports a **model electronic energy** `T_s + E_ext + E_H + E_xc` for cosine or periodic Gaussian-well external fields.  It omits ion-ion energy and does not describe real materials quantitatively.  The supported physical boundary is Gamma-only, non-spin-polarized, integer occupation, local model potentials, Hartree, and unpolarized Dirac-exchange/PZ81-LDA correlation.

## Architecture and public structures

| Layer | Modules and responsibility | Primary public structures |
| --- | --- | --- |
| Input/model | `constants.py`, `i_o.py`, `system.py` convert units, validate YAML, and serialize results. | `SystemConfig`, `SCFConfig`, `DOSConfig`, `CosinePotentialConfig`, `GaussianAtom`, `GaussianAtomsPotentialConfig` |
| Geometry/representation | `lattice.py`, `basis.py`, `fft_grid.py`, `wavefunction.py` define the cell, plane waves, FFT mapping, and overlaps. | `Lattice`, `PlaneWaveBasis`, `FFTGrid` |
| Local physics | `ionic_potential.py`, `hartree.py`, `xc.py`, `potentials.py`, `occupations.py`, `density.py`, `energy.py` construct local fields, density, occupancy, and energies. | `HartreeResult`, `XCResult`, `PotentialSet`, `EnergyComponents` |
| Numerical solve | `hamiltonian.py`, `eigensolver.py`, `mixing.py` implement FFT Hamiltonian action, lowest eigenpairs, and charge-preserving linear mixing. | `Hamiltonian`, `EigenResult` |
| Orchestration/UI | `scf.py`, `dos.py`, `__main__.py`, `__init__.py` run SCF, broaden eigenvalues, expose CLI/API. | `SCFRunner`, `SCFIteration`, `SCFResult`, `DOSResult` |

`mini_dft.__init__` intentionally exposes `Lattice`, `PlaneWaveBasis`, `FFTGrid`, `SystemConfig`, `SCFRunner`, `SCFResult`, and `load_system` as the stable entry surface.  Result arrays use the contract `(npw, 3)` for reciprocal indices/vectors, `(bands, npw)` for complex coefficients, and `grid.shape` for real-space fields.

## Before and after data structures

| Concern | Original scaffold | v0.1 |
| --- | --- | --- |
| System input | Mutable `system` populated by I/O placeholder integers | Frozen `SystemConfig` with typed, validated lattice, units, SCF/DOS and potential configuration |
| Lattice and reciprocal basis | Undeclared orientation and no `G` data | Column-oriented `Lattice.A`, `Lattice.B`, `PlaneWaveBasis.g_indices/g_vectors/kinetic_energies` |
| Wavefunction | Unspecified `c` mixed with `k` | Complex normalized coefficient vector `(npw,)` or band batch `(bands, npw)` |
| FFT/data mapping | Unspecified reciprocal/real functions | `FFTGrid` with signed-index placement, separate wavefunction/scalar scaling, quadrature |
| Potentials | Placeholder scalar functions | Real grid fields in `PotentialSet(ionic, hartree, xc)` and `effective` |
| Eigenproblem | Real `LinearOperator(N, N)` with arbitrary `N` | Complex matrix-free `Hamiltonian(npw, npw)` and `EigenResult` with residual norms |
| Density/occupation | One hard-coded state, no charge contract | 0/2 integer occupations, grid density, integral and mixing renormalization |
| SCF/output | Import-time loop, eigenvalue sum, print only | Explicit `SCFResult`, iteration history, direct model energy, CLI exit status, YAML/CSV/NPZ outputs |

## Calculation trace

```text
YAML -> load_system -> SystemConfig (atomic units)
     -> Lattice -> PlaneWaveBasis -> FFTGrid
     -> ionic model potential + uniform initial density
     -> Hartree + PZ81 LDA -> PotentialSet.effective
     -> Hamiltonian.apply (G -> r FFT product -> G)
     -> solve_lowest -> eigenvalues / complex coefficients
     -> occupations + density_from_coefficients
     -> calculate_energy + density/energy convergence checks
     -> mix_density or final SCFResult
     -> write_results -> summary.yaml, CSV, NPZ (+ dos.csv when enabled)
```

`SCFRunner` performs an un-mixed consistency solve before accepting a converged state, so its stored density, fields, eigenpairs, and energies originate from one internally consistent solve.  At exhaustion it returns `converged=False`; the CLI writes the diagnostic files and exits 2.

## Equations implemented

The plane-wave basis retains \(\frac12|G|^2\le E_{cut}\), and its Hamiltonian action is

\[
HC=\tfrac12|G|^2C+\mathcal F\{V_{eff}\,\mathcal F^{-1}C\}.
\]

Density is \(n(r)=\sum_n f_n|\psi_n(r)|^2\), with `f_n` equal to 0 or 2.  The Hartree reciprocal kernel is \(4\pi/G^2\) for nonzero \(G\) and exactly zero at \(G=0\).  Exchange is Dirac exchange and correlation is unpolarized Perdew–Zunger 1981 LDA.  The direct energy is

\[
E_{model}=T_s+\int nV_{ion}+\frac12\int nV_H+\int n\epsilon_{xc}.
\]

The full normalization, lattice, FFT, shape, and energy discussion is in [MATHEMATICAL_CONVENTIONS.md](MATHEMATICAL_CONVENTIONS.md).

## Changed and created deliverables

The implementation series created or replaced all runtime modules under `mini_dft/`, package metadata in `pyproject.toml`, and tests under `tests/`.  Task 9 specifically finalizes `mini_dft/input.yaml`, creates `examples/gaussian_atoms.yaml`, extends `tests/test_cli.py` with end-to-end example coverage, and creates this report plus `README.md`, `MATHEMATICAL_CONVENTIONS.md`, and `CURRENT_CODE_PROBLEMS.md`.

The CLI accepts `python -m mini_dft INPUT.yaml --output DIRECTORY [--quiet]`.  It prints an SCF progress table unless quiet and serializes:

- `summary.yaml` with convergence, units, basis/grid, charge, zero-Gauge and energy components;
- `scf_history.csv` and `eigenvalues.csv`;
- `dos.csv` when enabled;
- `fields.npz` for the real-space fields; and
- `wavefunctions.npz` for index/vector metadata and complex coefficients.

## Test coverage and example evidence

Unit and integration coverage spans conversion/validation, lattice algebra, cutoff enumeration, FFT round trips and Parseval behavior, ionic/Hartree/XC potentials, Hamiltonian Hermiticity and free/constant-potential behavior, eigensolver residuals, occupations/density/energy invariants, mixing, SCF convergence/exhaustion, DOS, output serialization, CLI errors, and the two repository examples.

Task 9 executed the repository examples directly through the CLI:

| Example | Converged recorded iterations | `npw`, FFT grid | Charge integral | Model energy (Ha) |
| --- | ---: | --- | ---: | ---: |
| `mini_dft/input.yaml` cosine | 30 | 27, `(5, 5, 5)` | 2.0000000000000013 | -0.25851023734873285 |
| `examples/gaussian_atoms.yaml` | 23 | 27, `(5, 5, 5)` | 2.000000000000001 | -0.31789638792188146 |

The focused example/CLI regression passed after the test-first RED/GREEN cycle.  A fresh Task 9 complete-suite run measured **92 passed in 3.85 s**.  Task 10 may add its own final acceptance measurements; this report does not prestate any values that have not been run.

## Known limitations and extension seams

- Only Gamma-point calculations exist.  K-point weights, Bloch shifts, and Brillouin-zone DOS are future work.
- Only non-spin-polarized integer occupations exist.  Finite-temperature/fractional occupations and spin channels belong behind `occupations.py`, `density.py`, and XC interfaces.
- Cosine and Gaussian wells are local teaching models.  Real pseudopotential, PAW, and ion-ion/Ewald work requires a separate ionic model and energy contribution.
- LDA is intentionally local and simple; GGA/meta-GGA would extend `xc.py` and `PotentialSet` inputs.
- Linear mixing is robust for the small examples but not a production accelerator.  Pulay/Kerker strategies can replace `mix_density` without changing the SCF state/result boundary.
- The sparse eigensolver is matrix-free for normal cases; its dense fallback is intentionally limited to small dimensions where ARPACK cannot request the desired count.
- A reused output directory can retain `dos.csv` from a previous DOS-enabled run if DOS is later disabled.  Use a fresh directory until Task 10 triages stale-output cleanup.
- Task 10 will also triage the deferred edge observations: single-band `orthonormality_error`, exact Hartree/XC batched-shape error wording, explicit basis/grid identity validation, and the accepted one-point DOS case.

## Documentation inventory

- [README.md](../README.md): installation, CLI/API use, schema, outputs, tests, and interpretation warning.
- [MATHEMATICAL_CONVENTIONS.md](MATHEMATICAL_CONVENTIONS.md): exact lattice, shape, FFT, density, potential, and energy contracts.
- [CURRENT_CODE_PROBLEMS.md](CURRENT_CODE_PROBLEMS.md): evidence-based original-scaffold audit and replacements.
- This report: architecture, data flow, deliverables, evidence, limitations, and extension seams.
