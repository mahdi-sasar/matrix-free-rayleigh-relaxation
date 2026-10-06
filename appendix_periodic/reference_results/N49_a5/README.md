# Reference periodic-hydrogen result (N=49, a=5 Bohr)

This directory records the first fully converged production band calculation from the independent Bloch-periodic appendix workflow.

## Parameters

- cubic lattice constant: `a = 5.0 Bohr`
- periodic grid: `N = 49` unique points per direction
- spacing: `h = a/N = 0.10204081632653061 Bohr`
- path: `Gamma-X-M-R-Gamma`
- points per segment: 13
- total path points: 49
- residual tolerance: `1e-8`
- relaxation safety factor: `sigma = 0.80`
- GPU used for this run: NVIDIA T4
- implementation: original eager appendix solver, before the later compiled-loop optimization
- original solver commit: `3d01bb4f7e62ef4e78b429a4c8e022641869ed18`
- original notebook commit: `acfe766fbbe4e818508c49c3aa7fc8553e22fe31`

## Converged vertex energies

| point | energy (Ry) |
|---|---:|
| Gamma | -0.1850582093356365 |
| X | 0.0044064891501604 |
| M | 0.1520149914837953 |
| R | 0.2680137462790322 |
| Gamma (return) | -0.1850582093356367 |

The return to Gamma differs from the initial Gamma energy by only about `1.94e-16 Ry`.

The sampled band width is

`0.45307195561466884 Ry`.

All 49 k-points converged. Final normalized residuals lie between approximately `9.63e-9` and `1.00e-8`.

## Timing

The original eager implementation required about `3.71 h` total wall time for the 49-point T4 run. This long runtime motivated the later compiled TensorFlow relaxation loop. The optimized implementation should be checked against this CSV before it replaces this result as the production reference.

## Figures

The run also produced:

- `periodic_hydrogen_lowest_band.pdf/png`
- `periodic_hydrogen_gamma_density_tiled.pdf/png`
- `periodic_hydrogen_potential_cut.pdf/png`

The binary figure files are not stored here by the automated GitHub connector. The numerical CSV is the authoritative record from which the band plot can be regenerated.

The potential cut is treated as a diagnostic plot; the band dispersion and tiled Gamma-point density are the primary appendix/paper candidates.
