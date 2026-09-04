# Mini-DFT 0.1

Mini-DFT is a small, educational three-dimensional periodic plane-wave Kohn–Sham DFT program.  It is designed to make the path from a YAML input through an FFT Hamiltonian and self-consistent field (SCF) calculation visible and testable.

> **Interpret results as model-system results only.** `total_hartree` is the electronic model energy for an educational local potential; it is not a real-material total energy and it excludes ion-ion (Ewald) energy.  Version 0.1 is Gamma-only, non-spin-polarized, integer-occupation LDA with cosine or periodic Gaussian *model* potentials.  It does not implement real elemental pseudopotentials, k-point integration, fractional occupations, forces, stress, or geometry optimisation.

## Installation

Python 3.11 or newer is required.  From a checkout, install the package and its test extra:

```powershell
python -m pip install -e ".[test]"
```

The runtime dependencies are NumPy, SciPy, and PyYAML.  The `mini-dft` console command is installed by the command above; `python -m mini_dft` also works from the checkout.

## Quick start

Run the checked-in cosine model and put the results in a new directory:

```powershell
python -m mini_dft mini_dft/input.yaml --output output/cosine
```

For a concise, script-friendly run, add `--quiet`.  Exit code `0` means SCF converged, `2` means a complete diagnostic output was written but the iteration limit was reached, and `1` means argument, input, or I/O validation failed.

Two small, regression-tested examples are included:

```powershell
python -m mini_dft mini_dft/input.yaml --output output/cosine
python -m mini_dft examples/gaussian_atoms.yaml --output output/gaussian
```

The first is expressed in eV and ångström and uses three weak cosine modes.  The second is in Hartree and bohr and uses two shallow periodic Gaussian wells.  Both comments explicitly identify their external potentials as educational models, not pseudopotentials.

## YAML schema

All dimensional values use the units declared at the top and are converted to atomic units when loaded.  `lattice.vectors` contains three real-space row vectors in the input file; the internal `Lattice.A` stores those vectors as columns.

```yaml
units:
  length: angstrom                 # bohr | angstrom
  energy: ev                       # hartree | ev
lattice:
  vectors: [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
electrons: 2                       # non-negative, even integer
encut: 20.0                        # positive kinetic cutoff
bands: 4                           # at least electrons // 2
potential:
  type: cosine                     # cosine | gaussian_atoms
  amplitudes: [-0.40, -0.40, -0.40] # alternatively one amplitude
  modes: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
  phases: [0.0, 0.0, 0.0]          # optional; defaults to zero
scf:
  max_iterations: 80
  density_tolerance: 2.0e-7
  energy_tolerance: 2.0e-7         # expressed in units.energy
  mixing_alpha: 0.35               # (0, 1]
  eigensolver_tolerance: 1.0e-10
  eigensolver_max_iterations: 2000 # optional
dos:
  enabled: true
  points: 101                      # integer >= 1
  width: 0.10                      # positive, in units.energy
```

For `gaussian_atoms`, replace the cosine fields with an `atoms` list.  Each atom has a fractional `position` (wrapped into `[0, 1)`), a negative `depth`, and a positive `width` in `units.length`:

```yaml
potential:
  type: gaussian_atoms
  atoms:
    - position: [0.25, 0.25, 0.25]
      depth: -0.05
      width: 1.50
```

Input loading validates field paths, dimensions, units, lattice volume, occupiable band count, scalar ranges, and potential-specific fields.  The complete mathematical and array contract is in [docs/MATHEMATICAL_CONVENTIONS.md](docs/MATHEMATICAL_CONVENTIONS.md).

## Output files

`--output DIR` creates the following files.  Energies are internally Hartree; files include eV fields where useful.

| File | Contents |
| --- | --- |
| `summary.yaml` | convergence status, iterations, input units, basis/grid dimensions, charge integral, `hartree_g0`, and final energy components in Hartree and eV |
| `scf_history.csv` | one row per recorded SCF solve: density RMS, energy change, model/eigenvalue energy, and component energies |
| `eigenvalues.csv` | one-based band index, eigenvalue in Hartree/eV, and occupation |
| `dos.csv` | Gamma-point Gaussian-broadened energy grid and DOS; written only when `dos.enabled: true` |
| `fields.npz` | `density`, `ionic_potential`, `hartree_potential`, `xc_potential`, and `effective_potential` arrays |
| `wavefunctions.npz` | `g_indices`, Cartesian `g_vectors`, and complex `coefficients` with shape `(bands, npw)` |

`hartree_g0: zero_neutralizing_background` records the fixed-zero Hartree average convention.  A reused output directory with DOS disabled may retain an old `dos.csv`; use a clean output directory when output-file membership matters.

## Python API

The supported starting point is the package-level API:

```python
from mini_dft import SCFRunner, load_system
from mini_dft.i_o import write_results

system = load_system("mini_dft/input.yaml")
result = SCFRunner(system).run()
if result.converged:
    write_results(result, system, "output/cosine")
```

`result` is an `SCFResult` with `basis`, `grid`, final `density`, `potentials`, eigenvalues, complex plane-wave coefficients, occupations, `EnergyComponents`, immutable iteration history, and a human-readable message.  The library returns `converged=False` on exhausted SCF iterations; the CLI maps that state to exit code 2.

## Tests

Run the complete test suite from the checkout:

```powershell
python -B -m pytest -q
```

The suite exercises unit conversion, lattice and basis construction, FFT normalization and anti-aliasing, local potentials, Hartree and LDA, Hamiltonian/eigensolver invariants, density and energy conservation, SCF behavior, CLI serialization, and both checked-in examples.

## Physical limitations and interpretation

- Gamma point only: the reported DOS is a Gaussian-broadened discrete Gamma spectrum, not a Brillouin-zone DOS.
- Non-spin-polarized: each occupied spatial orbital has occupation 2; only non-negative even electron counts are accepted.
- Local model potentials only: cosine and Gaussian wells are pedagogical fields, not transferable ionic pseudopotentials.  Do not use them to compare elements or materials.
- Exchange-correlation is unpolarized Dirac exchange plus Perdew–Zunger 1981 LDA correlation.  There is no GGA, PAW, or nonlocal pseudopotential treatment.
- The reported `T_s + E_ext + E_H + E_xc` omits ion-ion energy.  It is useful for demonstrating a self-consistent electronic model, not cohesive energies, band structures, forces, or quantitative predictions.
- Linear mixing is deliberately simple.  Difficult systems may require smaller mixing or implementation of Pulay/Kerker methods rather than interpreting nonconvergence physically.
