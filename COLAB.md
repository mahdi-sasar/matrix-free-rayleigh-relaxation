# Google Colab workflow

This repository is intended to run cleanly on Google Colab with a GPU runtime.

## Runtime setup

1. Open Colab.
2. Choose **Runtime → Change runtime type → GPU**.
3. For final paper runs, prefer `float64` on GPUs with strong double-precision performance. For debugging and exploratory plots, use `--float32`.
4. Run the notebook `notebooks/colab_quickstart.ipynb`.

## Installation in Colab

Once the repository is on GitHub:

```bash
git clone https://github.com/mahdi-sasar/matrix-free-rayleigh-relaxation.git
cd matrix-free-rayleigh-relaxation
pip install -q -r requirements-colab.txt
pip install -q -e . --no-deps
```

`--no-deps` is intentional: Colab already provides TensorFlow, and reinstalling TensorFlow inside a Colab runtime can create avoidable CUDA/library conflicts.

## Suggested run order

Start with correctness and small grids:

```bash
pytest -q tests/test_small_dense_comparison.py
python examples/run_harmonic_oscillator_3d.py --n 48 --tol 1e-8 --out results/harmonic_48
python examples/run_hydrogen_box.py --n 64 --box 10.0 --tol 1e-8 --out results/hydrogen_64
```

Then scale upward:

```bash
python scripts/run_scaling_hydrogen.py --n-values 64 80 96 112 128 --box 10.0 --tol 1e-8 --out results/hydrogen_scaling.csv
```

For very large grids, use a looser tolerance first, verify the energy/residual trend, then repeat final runs at the paper tolerance.

## Precision notes

- `float64` is the default in the examples and is recommended for final data.
- `--float32` is useful for fast development, plotting, and debugging.
- Always report the precision used in paper tables.

## Output persistence

Use Google Drive for long runs:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then write outputs into a Drive path such as:

```bash
--out /content/drive/MyDrive/mrsr_results/hydrogen_128
```


## Important Colab debugging note

After updating the repository or uploading a new ZIP in the same Colab runtime, restart the runtime or run:

```python
import mrsr, inspect
print(mrsr.__file__)
```

to make sure Python is importing the package from the newly uploaded folder. Stale editable installs can otherwise cause the tests to use old code. For Coulomb examples, the code now prints the minimum distance from the Coulomb center to the nearest grid node; this must be comfortably nonzero.

## H2+ potential-energy curve

The Colab notebook now includes a quick finite-box H2+ scan.  The electronic Hamiltonian is

```text
H = -∇² - 2/r_A - 2/r_B
```

in Rydberg units.  The total Born--Oppenheimer curve adds the proton--proton repulsion term `2/R` Ry:

```text
E_total(R) = E_electronic(R) + 2/R.
```

A modest first run is:

```bash
python examples/run_h2plus_curve.py \
  --n 48 \
  --box 16.0 \
  --r-values 0.8 1.0 1.2 1.4 1.6 2.0 2.5 3.0 3.5 4.0 \
  --sigma 0.5 \
  --tol 1e-7 \
  --max-iter 80000 \
  --check-every 20 \
  --out results/colab_h2plus_curve_48
```

For a production-quality curve, increase `--n`, enlarge the box, refine the `--r-values`, and use `--tol 1e-8` or tighter.  The script writes `h2plus_curve.csv`, per-distance convergence histories, midplane wavefunction slices, and `h2plus_curve.png`.


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

### Publication-ready H2+ plotting

The H2+ workflow now saves both transverse and bond-plane wavefunction slices for every sampled internuclear distance. Use:

```bash
python scripts/plot_h2plus_curve.py results/colab_h2plus_curve_48/h2plus_curve.csv \
    --out results/colab_h2plus_curve_48/h2plus_curve_pretty.png

python scripts/plot_h2plus_slices.py results/colab_h2plus_curve_48 --R best --plane bond --pdf
python scripts/plot_h2plus_slices.py results/colab_h2plus_curve_48 --R best --plane transverse --pdf
```

The bond-plane plot contains the two nuclei and is the preferred paper figure. The transverse plane is useful as a symmetry and grid-quality diagnostic. Density plots use a red-yellow `YlOrRd` heatmap with contour overlays by default.
