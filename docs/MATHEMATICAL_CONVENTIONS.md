# Mathematical and Array Conventions

This document is the implementation contract for Mini-DFT v0.1.  It names the code objects that carry each quantity so that formulas, indexing, and serialized arrays can be checked against the program.

## Units and lattice orientation

`load_system` converts `bohr`/`angstrom` lengths with `length_to_bohr` and `hartree`/`ev` energies with `energy_to_hartree`.  The absolute density RMS `scf.density_tolerance` uses the declared inverse-volume unit: an Angstrom^-3 input is multiplied by `BOHR_TO_ANGSTROM ** 3`, while a Bohr^-3 input is unchanged.  All `SystemConfig` dimensional values and every numerical calculation thereafter use Hartree atomic units: \(\hbar=m_e=e=4\pi\epsilon_0=1\).

YAML gives `lattice.vectors` as three **row** vectors.  `Lattice.from_row_vectors` transposes them, so internal `Lattice.A` has the real lattice vectors as columns:

\[
A=[\mathbf a_1\;\mathbf a_2\;\mathbf a_3],\qquad
\Omega=|\det A|=\texttt{Lattice.volume}.
\]

`Lattice.B` is `2.0 * np.pi * inv(A).T`, hence

\[
B=2\pi A^{-T},\qquad A^T B=2\pi I,
\qquad \mathbf r=A\mathbf s.
\]

The vectorized `fractional_to_cartesian` implementation is `fractional @ A.T`, equivalent to the column-vector relation above.  Cartesian reciprocal vectors are constructed as `g_indices @ lattice.B.T`.

## Plane waves, shapes, and cutoff

At Gamma, band \(n\) is represented by

\[
\psi_n(\mathbf r)=\Omega^{-1/2}\sum_{\mathbf G}
C_n(\mathbf G)e^{i\mathbf G\cdot\mathbf r},\qquad
\mathbf G=B\mathbf m,
\]

where \(\mathbf m\) is an integer triplet.  `PlaneWaveBasis.from_cutoff` retains exactly the candidates satisfying

\[
T_{\mathbf G}=\tfrac12|\mathbf G|^2\leq\texttt{encut}.
\]

The basis is stable-sorted by energy and integer components and explicitly places \(\mathbf G=0\) first.  The code's shape contract is:

| Quantity | Code symbol | Shape and dtype |
| --- | --- | --- |
| integer reciprocal index | `basis.g_indices` | `(npw, 3)`, integer |
| Cartesian reciprocal vector | `basis.g_vectors` | `(npw, 3)`, float |
| kinetic energy | `basis.kinetic_energies` | `(npw,)`, float |
| one band coefficients | `coefficients` | `(npw,)`, complex |
| many-band coefficients | `result.coefficients` | `(bands, npw)`, complex |
| scalar grid field | `density`, `PotentialSet.*` | `grid.shape`, real |

`overlap_matrix` treats a single `(npw,)` coefficient vector as a one-band batch and therefore returns a `(1, 1)` overlap; `orthonormality_error` follows the same convention.

`FFTGrid.from_basis` chooses each grid size from `next_fast_len(max(1, 4 * max_abs_index + 1))`.  This represents differences \(\mathbf G-\mathbf G'\) of retained plane waves and therefore prevents circular aliasing in density and local-potential products.

## FFT normalization and integration

`FFTGrid.coefficients_to_real` embeds plane-wave coefficients in an FFT cube and evaluates

```text
ifftn(C_grid) * ngrid / sqrt(volume).
```

`FFTGrid.real_to_coefficients` applies the inverse physical scaling:

```text
fftn(psi) * sqrt(volume) / ngrid.
```

Consequently, with `N = grid.ngrid`, the discrete quadrature `FFTGrid.integrate(field) = volume * mean(field)` has the Parseval property

\[
\frac{\Omega}{N}\sum_{\mathbf r}|\psi_n(\mathbf r)|^2
=\sum_{\mathbf G}|C_n(\mathbf G)|^2.
\]

Scalar fields use a deliberately separate convention: `field_to_fourier(field) = fftn(field) / ngrid` and `fourier_to_field(fourier) = ifftn(fourier) * ngrid`.  Separating those methods prevents the wavefunction \(\Omega^{-1/2}\) scaling from leaking into Hartree or potential fields.

## Occupation and density

`integer_occupations(electrons, n_bands)` returns an array of shape `(bands,)` with the lowest `electrons // 2` entries equal to 2.0 and the rest zero.  Only even, non-negative electron counts are accepted.  `density_from_coefficients` calculates

\[
n(\mathbf r)=\sum_n f_n|\psi_n(\mathbf r)|^2,
\qquad \int_\Omega n(\mathbf r)\,d\mathbf r
=\texttt{density_integral(density, grid)}=N_e.
\]

`mix_density` applies \(n_{\rm mix}=(1-\alpha)n_{\rm in}+\alpha n_{\rm out}\), clips only round-off-size negative values, and rescales the quadrature integral back to `electrons`.  `density_rms_residual` is the unweighted grid-point RMS of \(n_{\rm out}-n_{\rm in}\).

## Potentials and Hamiltonian

For a cosine configuration, `build_ionic_potential` evaluates

\[
V_{\rm ion}(\mathbf s)=\sum_j A_j
\cos(2\pi\mathbf m_j\!\cdot\!\mathbf s+\phi_j).
\]

For `gaussian_atoms`, the implementation first forms the component-wise wrapped fractional displacement

\[
\delta=\mathbf s-\operatorname{mod}(\mathbf s_{\rm atom},1),
\qquad \delta\mathrel{-}=\operatorname{rint}(\delta),
\]

then evaluates the 27 candidates \(A(\delta+\mathbf t)\) for
\(\mathbf t\in\{-1,0,1\}^3\), selects their smallest Cartesian squared norm, and sums `depth * exp(-distance_squared / (2 * width**2))`.  This is the approved v0.1 27-neighbor search, not a globally guaranteed closest-image algorithm.  It is reliable for the small/reasonably reduced teaching cells supplied here; highly skew or non-reduced cells can require a wider lattice-vector search to find the global closest image.  Both potential types are local educational model fields.

`hartree_from_density` and `lda_pz81` require the exact scalar-field shape `density.shape == grid.shape`; leading batch axes are rejected before any FFT, integration, or reciprocal indexing.  `hartree_from_density` computes scalar density Fourier coefficients.  For ordinary non-aliased nonzero FFT modes it uses

\[
V_H(\mathbf G)=\frac{4\pi n(\mathbf G)}{|\mathbf G|^2}.
\]

The exact code denominator is `squared = 0.5 * (raw_squared + raw_squared[conjugate])`, where `raw_squared[m] = |B m|²` at the FFT slot and `conjugate` is the slot `(-m) mod grid.shape`.  Away from even-axis Nyquist aliases, the paired vectors are negatives and the two values agree, so this reduces to the ordinary formula above.  When an even FFT axis has a Nyquist component, sign reversal can alias that component to itself while other components reverse; in a skew cell the two represented Cartesian norms can differ.  The symmetric average gives conjugate FFT slots the same real kernel, preserving Hermiticity and therefore a real Hartree field.  The implementation then applies \(4\pi n(\mathbf G)/\texttt{squared}\) to every slot with positive `squared`.

The code leaves the zero component of `potential_fourier` at zero.  Thus `V_H(G=0)=0`: a neutralizing uniform background/fixed-average-potential gauge, recorded as `hartree_g0: zero_neutralizing_background` in `summary.yaml`.  It is not an explicit ionic charge calculation.

`lda_pz81` uses unpolarized Dirac exchange

\[
\epsilon_x(n)=-\tfrac34(3/\pi)^{1/3}n^{1/3},\qquad v_x=\tfrac43\epsilon_x,
\]

and the unpolarized Perdew–Zunger 1981 correlation branches.  It uses a positive-density mask so that \(n=0\) has finite analytic-limit zero arrays.  `XCResult` stores `potential`, `energy_per_particle`, and the quadrature `energy`.

`PotentialSet.effective` is `ionic + hartree + xc`.  A `Hamiltonian` requires `grid.basis is basis` by object identity, preventing equal-sized but differently labelled reciprocal bases from being mixed.  `Hamiltonian.apply` represents

\[
(H C)_{\mathbf G}=\tfrac12|\mathbf G|^2C_{\mathbf G}
 +\mathcal F\{V_{\rm eff}(\mathbf r)\psi(\mathbf r)\}_{\mathbf G},
\]

using `coefficients_to_real`, multiplication by the real field, and `real_to_coefficients`.  `as_linear_operator()` advertises `dtype=np.complex128`; complex plane-wave coefficients are never treated as a real eigenproblem.  `solve_lowest` normally uses matrix-free `eigsh`; requests with `n_bands >= npw - 1` use a deterministic dense fallback only when `npw <= 256`.  Larger incompatible requests are rejected before allocating an `npw`-square matrix.

## Energies and SCF test

`calculate_energy` reports the model electronic energy

\[
E_{\rm model}=T_s+E_{\rm ext}+E_H+E_{xc},
\]

\[
T_s=\sum_{n\mathbf G}f_n|C_n(\mathbf G)|^2T_{\mathbf G},\quad
E_{\rm ext}=\int nV_{\rm ion},\quad
E_H=\tfrac12\int nV_H,\quad
E_{xc}=\int n\epsilon_{xc}(n).
\]

This **does not include ion-ion energy** and must not be interpreted as a real-material total energy.  `eigenvalue_energy` separately evaluates the double-counting-corrected eigenvalue expression as an internal diagnostic.

`SCFRunner` initializes `density` to `electrons / lattice.volume`, constructs a `Hamiltonian`, calls `solve_lowest`, forms `output_density`, records the direct energy, then linearly mixes.  It accepts convergence only after both `density_rms <= density_tolerance` (both in Bohr^-3 internally) and `energy_delta <= energy_tolerance` and an additional un-mixed consistency solve pass.  Exhaustion returns an `SCFResult` with `converged=False` rather than silently reporting success.
