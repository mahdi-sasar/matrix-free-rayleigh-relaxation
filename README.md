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

## Google Colab

For Colab GPU runs, open `notebooks/colab_quickstart.ipynb`. Install the repository with `pip install -e . --no-deps` so Colab's preinstalled TensorFlow/CUDA stack is preserved. See `COLAB.md` for the recommended workflow and run order.

## Notes

- All energies are in Rydbergs.
- The implementation assumes equal spacing in x, y, and z.
- Potentials are sampled on the grid. Coulomb centers can be shifted off-grid to avoid arbitrary smoothing of the singularity.
- The main stopping criterion is the normalized eigen-residual, not consecutive energy difference.

## License

Add a license before making the repository public. MIT or BSD-3-Clause would be good choices for broad reuse.
