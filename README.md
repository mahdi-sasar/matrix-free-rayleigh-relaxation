# Matrix-Free Rayleigh Relaxation

Reference implementation for the revised manuscript:

**Matrix-Free Rayleigh-Shifted Relaxation for Real-Space Schrödinger Ground States**

The code implements a normalized, matrix-free, finite-difference Rayleigh relaxation method for the real single-particle Schrödinger equation in Rydberg atomic units,

```text
-∇² ψ(r) + V(r) ψ(r) = ε ψ(r),     ψ|boundary = 0.
```

The solver never assembles the Hamiltonian matrix. It applies the finite-difference stencil directly to a grid-resident wavefunction.

## Core iteration

For the discrete Hamiltonian `H_h`, Rayleigh quotient `ε_n`, and residual `R_n = (H_h - ε_n I) ψ_n`, the solver uses

```text
q_{n+1} = ψ_n - τ R_n
ψ_{n+1} = q_{n+1} / ||q_{n+1}||_2.
```

The default step is chosen from the spectral-diameter bound

```text
D_bound = 4 d / h² + V_max - V_min,
τ = 2 σ / D_bound,       0 < σ < 1.
```

This matches the descent theorem used in the revised manuscript.

## Repository layout

```text
mrsr/                       Core TensorFlow implementation
examples/                   Reproducible paper examples
scripts/                    Plot/export utilities
tests/                      Small correctness tests
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU acceleration, install a TensorFlow build compatible with your CUDA environment or use Google Colab.

## Quick start

```bash
python examples/run_harmonic_oscillator_3d.py --n 48 --tol 1e-8 --out results/harmonic_48
python examples/run_hydrogen_box.py --n 64 --box 10.0 --tol 1e-8 --out results/hydrogen_64
```


## Main examples

```bash
# Analytical validation: 3D harmonic oscillator
python examples/run_harmonic_oscillator_3d.py --n 48 --tol 1e-8 --out results/harmonic_48

# Hydrogen atom in a finite box with an off-grid Coulomb center
python examples/run_hydrogen_box.py --n 64 --box 10.0 --tol 1e-8 --out results/hydrogen_64

# Hydrogen molecular ion potential-energy curve
python examples/run_h2plus_curve.py \
  --n 48 \
  --box 16.0 \
  --r-values 0.8 1.0 1.2 1.4 1.6 2.0 2.5 3.0 3.5 4.0 \
  --sigma 0.5 \
  --tol 1e-7 \
  --out results/h2plus_curve_48
```

For H2+, the electronic energy is computed from `-∇² - 2/r_A - 2/r_B`; the reported total Born--Oppenheimer energy adds the nuclear repulsion term `2/R` in Rydbergs.  The script writes `h2plus_curve.csv` and `h2plus_curve.png`.

## Google Colab

For Colab GPU runs, open `notebooks/colab_quickstart.ipynb`. Install the repository with `pip install -e . --no-deps` so Colab's preinstalled TensorFlow/CUDA stack is preserved. See `COLAB.md` for the recommended workflow and run order.

## Notes

- All energies are in Rydbergs.
- The implementation assumes equal spacing in x, y, and z.
- Potentials are sampled on the grid. Coulomb centers can be shifted off-grid to avoid arbitrary smoothing of the singularity.
- The main stopping criterion is the normalized eigen-residual, not consecutive energy difference.

## License

Add a license before making the repository public. MIT or BSD-3-Clause would be good choices for broad reuse.


## Important Colab debugging note

After updating the repository or uploading a new ZIP in the same Colab runtime, restart the runtime or run:

```python
import mrsr, inspect
print(mrsr.__file__)
```

to make sure Python is importing the package from the newly uploaded folder. Stale editable installs can otherwise cause the tests to use old code. For Coulomb examples, the code now prints the minimum distance from the Coulomb center to the nearest grid node; this must be comfortably nonzero.


## H2+ grid-phase note

For the H2+ curve, the Coulomb centers are placed halfway between grid nodes in
the two directions transverse to the bond axis. This avoids sampling the nuclear
singularity too close to a Cartesian node, which can otherwise produce large,
unphysical negative spikes in the curve at isolated internuclear separations.
The potential remains the unsmoothed Coulomb potential at all mesh points.
Always inspect `rmin_nucleus_A_Bohr` and `rmin_nucleus_B_Bohr` in
`h2plus_curve.csv`. If a total energy is below roughly -10 Ry for ordinary H2+
test runs, treat the run as a grid-alignment artifact and rerun with this version
or with a finer grid/larger box.

To replot an existing curve:

```bash
python scripts/plot_h2plus_curve.py results/colab_h2plus_curve_48/h2plus_curve.csv
```


## Colab plotting note

The Colab notebook now includes paper-quality plotting helpers for the H2+ energy curve and the saved transverse mid-slices, using a red-yellow (`YlOrRd`) heatmap style with contour overlays.

### Publication-ready H2+ plotting

The H2+ workflow now saves both transverse and bond-plane wavefunction slices for every sampled internuclear distance. Use:

```bash
python scripts/plot_h2plus_curve.py results/colab_h2plus_curve_48/h2plus_curve.csv \
    --out results/colab_h2plus_curve_48/h2plus_curve_pretty.png

python scripts/plot_h2plus_slices.py results/colab_h2plus_curve_48 --R best --plane bond --pdf
python scripts/plot_h2plus_slices.py results/colab_h2plus_curve_48 --R best --plane transverse --pdf
```

The bond-plane plot contains the two nuclei and is the preferred paper figure. The transverse plane is useful as a symmetry and grid-quality diagnostic. Density plots use a red-yellow `YlOrRd` heatmap with contour overlays by default.


## Full Colab workflow

The Colab notebook now preserves both main paper workflows:

1. hydrogen atom tests, including an N-sweep, energy-versus-grid-points plots, error-versus-grid-points plots, wall-time plots, and a largest-N mid-slice density heatmap;
2. H2+ molecular-ion tests, including the potential-energy curve, bond-plane density heatmap, and transverse mid-slice density heatmap.

Publication plots are saved as high-resolution PNG and PDF files where applicable.
