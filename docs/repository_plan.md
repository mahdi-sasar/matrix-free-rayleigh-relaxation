# Repository plan for the revised paper

Recommended repository name:

```text
matrix-free-rayleigh-relaxation
```

Core claims supported by the code:

1. Matrix-free stencil application of `H = -Δ_h + V`.
2. Normalized Rayleigh-shifted relaxation.
3. Energy descent and residual convergence diagnostics.
4. Reproducible examples used by the manuscript.

Minimum paper examples:

- `run_harmonic_oscillator_3d.py`: analytical validation.
- `run_hydrogen_box.py`: off-grid Coulomb singularity without smoothing.
- `run_h2plus_curve.py`: one-electron molecular potential-energy curve.
- `run_double_well.py`: nonseparable localization / tunnelling example.
- `run_yukawa_scan.py`: diffuse near-threshold states.
- `run_hydrogen_centered_field.py`: corrected Stark-in-a-box example.
- `run_corrected_dipole.py`: corrected finite dipole with correct Debye conversion.

Before public release:

- Add a license.
- Add a DOI archive after the first stable tag.
- Add generated benchmark CSV files and plotting scripts.
- Add a Colab notebook for the hydrogen and harmonic oscillator examples.
