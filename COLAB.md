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
