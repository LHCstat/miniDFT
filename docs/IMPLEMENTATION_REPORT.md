# Mini-DFT v0.1 Implementation Report

## Outcome and boundary

Mini-DFT v0.1 is a runnable, tested educational Gamma-point plane-wave Kohn–Sham SCF code.  It accepts validated YAML, uses Hartree atomic units internally, solves a matrix-free complex Hamiltonian, writes portable results, and includes two CLI-runnable examples.

The implementation reports a **model electronic energy** `T_s + E_ext + E_H + E_xc` for cosine or periodic Gaussian-well external fields.  It omits ion-ion energy and does not describe real materials quantitatively.  The supported physical boundary is Gamma-only, non-spin-polarized, integer occupation, local model potentials, Hartree, and unpolarized Dirac-exchange/PZ81-LDA correlation.

## Architecture and public structures

| Layer | Modules and responsibility | Primary public structures |
| --- | --- | --- |
| Input/model | `constants.py`, `i_o.py`, `system.py` convert units (including density RMS to Bohr^-3), validate YAML, and serialize results. | `SystemConfig`, `SCFConfig`, `DOSConfig`, `CosinePotentialConfig`, `GaussianAtom`, `GaussianAtomsPotentialConfig` |
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

Density is \(n(r)=\sum_n f_n|\psi_n(r)|^2\), with `f_n` equal to 0 or 2.  Hartree and XC accept only exact `grid.shape` density fields.  For ordinary non-aliased nonzero FFT modes, the Hartree reciprocal kernel is \(4\pi/G^2\), and it is exactly zero at \(G=0\).  To keep the discrete kernel Hermitian on even skew FFT grids, `hartree_from_density` actually uses `0.5 * (raw_squared[m] + raw_squared[(-m) mod grid.shape])` as its denominator.  This equals \(G^2\) away from Nyquist aliases; at an aliased even-axis Nyquist pair it symmetrizes the possibly unequal Cartesian norms represented by FFT-conjugate slots, ensuring a real Hartree field.  Exchange is Dirac exchange and correlation is unpolarized Perdew–Zunger 1981 LDA.  The direct energy is

\[
E_{model}=T_s+\int nV_{ion}+\frac12\int nV_H+\int n\epsilon_{xc}.
\]

For a Gaussian well, the ionic builder wraps each fractional displacement component with `delta -= rint(delta)`, tests all 27 `delta + translation` vectors for `translation in {-1, 0, 1}³`, converts them using `A`, and uses the least squared Cartesian norm in the Gaussian.  This exact finite search is intentional v0.1 behavior, not a general closest-lattice-vector solver.

`Hamiltonian` requires its `FFTGrid` to hold the identical `PlaneWaveBasis` object, rather than merely an equal `npw`.  Normal eigenproblems stay matrix-free; a near-complete request (`n_bands >= npw - 1`) uses the deterministic dense fallback only for `npw <= 256`, and larger incompatible requests fail before dense allocation.

The full normalization, lattice, FFT, shape, and energy discussion is in [MATHEMATICAL_CONVENTIONS.md](MATHEMATICAL_CONVENTIONS.md).

## Changed and created deliverables

The implementation series created or replaced all runtime modules under `mini_dft/`, package metadata in `pyproject.toml`, and tests under `tests/`.  Task 9 specifically finalized `mini_dft/input.yaml`, created `examples/gaussian_atoms.yaml`, extended `tests/test_cli.py` with end-to-end example coverage, and created this report plus `README.md`, `MATHEMATICAL_CONVENTIONS.md`, and `CURRENT_CODE_PROBLEMS.md`.  The final fix wave added inverse-volume tolerance conversion, bounded eigensolver fallback, exact basis/field contracts, complete terminal energy reporting, deterministic DOS-output reuse, and the associated edge/invariant coverage.

The CLI accepts `python -m mini_dft INPUT.yaml --output DIRECTORY [--quiet]`.  Unless quiet, it labels progress energies in Hartree and prints the final model energy in Hartree and eV for both converged and nonconverged results.  It serializes:

- `summary.yaml` with convergence, units, basis/grid, charge, zero-Gauge and energy components;
- `scf_history.csv` and `eigenvalues.csv`;
- `dos.csv` when enabled, removing a stale file of that name from the same output directory when disabled;
- `fields.npz` for the real-space fields; and
- `wavefunctions.npz` for index/vector metadata and complex coefficients.

## Test coverage and final verification evidence

Unit and integration coverage spans conversion/validation (including inverse-volume tolerances and `dos.points >= 2`), scalar and batched lattice algebra, cutoff enumeration, FFT round trips and Parseval behavior, single/batched overlap helpers, ionic/Hartree/XC potentials and exact density shapes, Hamiltonian Hermiticity/basis identity/free/constant-potential behavior, bounded eigensolver fallback and residuals, occupations/density/energy invariants, mixing, converged and fully reconstructed exhausted SCF states, DOS, output-directory reuse, CLI units/errors, and the two repository examples.

The final fix wave ran the following acceptance commands from a clean `verification-output/` location:

```powershell
python -B -m compileall -q mini_dft tests
python -B -c "import mini_dft; print(mini_dft.__all__)"
python -B -m pytest -q
python -B -m mini_dft mini_dft/input.yaml --output verification-output/cosine
python -B -m mini_dft examples/gaussian_atoms.yaml --output verification-output/gaussian
python -B .superpowers/sdd/2026-09-04-mini-dft-v0.1/final-output-audit.py
```

Compilation was silent with exit code 0.  Import printed exactly `['FFTGrid', 'Lattice', 'PlaneWaveBasis', 'SCFResult', 'SCFRunner', 'SystemConfig', 'load_system']` and no calculation output.  The complete suite passed **106 tests in 4.22 s** without warnings.  Both CLI commands returned 0, printed the labelled progress and dual-unit final energy, reported convergence, and wrote `summary.yaml`, `scf_history.csv`, `eigenvalues.csv`, `dos.csv`, `fields.npz`, and `wavefunctions.npz`.

The independent audit loaded the YAML, CSV, and NPZ artifacts rather than trusting terminal output.  It recomputed charge, sortedness, overlaps, and kinetic energy; independently rebuilt the discrete Hartree potential and the Dirac/PZ81 XC potential and energy from saved density; reassembled the external term from saved density and the saved ionic field; rebuilt direct and corrected-eigenvalue energies using the recomputed Hartree/XC quantities; and checked convergence, effective-potential composition, finiteness, and DOS properties:

| Measured invariant | Cosine | Gaussian atoms | Acceptance |
| --- | ---: | ---: | --- |
| Recorded iterations | 30 | 23 | converged |
| `npw`, FFT grid | 27, `(5, 5, 5)` | 27, `(5, 5, 5)` | expected serialized dimensions |
| Density minimum | 0.0017707102167230596 | 0.003774769641598701 | nonnegative |
| Recomputed charge | 2.0000000000000013 | 2.000000000000001 | target 2 |
| Absolute charge error | 1.3322676295501878e-15 | 8.881784197001252e-16 | `< 1e-10` |
| Minimum adjacent eigenvalue gap (Ha) | 4.163336342344337e-17 | 7.500969084750508e-08 | `>= 0` |
| Coefficient orthogonality error | 4.441956838999741e-16 | 4.441943027585409e-16 | `< 1e-8` |
| Final density RMS / tolerance (Bohr^-3) | 9.029731492295125e-12 / 2.9636942294432554e-08 | 1.3301419498906228e-10 / 2e-07 | below converted configured tolerance |
| Final energy change / tolerance (Ha) | 5.796081947728737e-10 / 7.349864435130998e-09 | 1.0827710561489567e-08 / 2e-07 | below converted configured tolerance |
| Recomputed Hartree potential max error | 0.0 | 0.0 | exact at float precision |
| Recomputed XC potential max error | 2.7755575615628914e-17 | 2.7755575615628914e-17 | round-off only |
| Recomputed XC energy (Ha) / summary error | -0.25696280676166383 / 0.0 | -0.29733061370250574 / 0.0 | XC derived from saved density, not serialized `E_xc` |
| Recomputed direct model energy (Ha) | -0.25851023734873285 | -0.31789638792188146 | exactly matched summary/history |
| Direct vs corrected-eigenvalue error (Ha) | 1.0727330135296143e-10 | 6.624436554858448e-10 | below bounds 1.1269652817996169e-08 / 2.3863120914149249e-08 |
| Effective-potential composition error | 5.551115123125783e-17 | 5.551115123125783e-17 | round-off only |

All five stored real-space fields and all wavefunction arrays were finite in both cases.  Each 101-row DOS was finite, nonnegative, and strictly energy-sorted.  Every audit check passed; the independently recomputed XC energy and the reconstructed total model energy matched their serialized summary/history values exactly at float precision.

## Known limitations and extension seams

- Only Gamma-point calculations exist.  K-point weights, Bloch shifts, and Brillouin-zone DOS are future work.
- Only non-spin-polarized integer occupations exist.  Finite-temperature/fractional occupations and spin channels belong behind `occupations.py`, `density.py`, and XC interfaces.
- Cosine and Gaussian wells are local teaching models.  Real pseudopotential, PAW, and ion-ion/Ewald work requires a separate ionic model and energy contribution.
- Gaussian wells use the approved component-wise wrapped displacement plus a fixed 27-neighbor `{-1, 0, 1}³` image search.  This is not globally closest-image reliable for highly skew or non-reduced cells, which can require a wider lattice-vector search.
- LDA is intentionally local and simple; GGA/meta-GGA would extend `xc.py` and `PotentialSet` inputs.
- Linear mixing is robust for the small examples but not a production accelerator.  Pulay/Kerker strategies can replace `mix_density` without changing the SCF state/result boundary.
- The sparse eigensolver is matrix-free for normal cases.  Its near-complete dense fallback is capped at `npw <= 256`; for larger bases the current ARPACK interface requires `n_bands <= npw - 2`.

## Documentation inventory

- [README.md](../README.md): installation, CLI/API use, schema, outputs, tests, and interpretation warning.
- [MATHEMATICAL_CONVENTIONS.md](MATHEMATICAL_CONVENTIONS.md): exact lattice, shape, FFT, density, potential, and energy contracts.
- [CURRENT_CODE_PROBLEMS.md](CURRENT_CODE_PROBLEMS.md): evidence-based original-scaffold audit and replacements.
- This report: architecture, data flow, deliverables, evidence, limitations, and extension seams.
