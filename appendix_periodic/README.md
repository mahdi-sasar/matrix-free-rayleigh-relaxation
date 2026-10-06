# Bloch-periodic appendix calculation

This directory is deliberately independent of the main `mrsr` Dirichlet implementation and the main Google Colab notebook used for the paper's hydrogen and H2+ results.

It supports the reviewer-response appendix demonstrating periodic and Bloch-periodic boundary conditions for the same normalized matrix-free Rayleigh-residual relaxation.

## Scope

The model is a **single-particle periodic hydrogenic lattice**. It is not a quantitative many-electron calculation of metallic hydrogen.

For a cubic cell of side `a` containing `N^3` unique periodic grid points,

```text
h = a/N
```

and boundary-crossing nearest-neighbor stencil links carry Bloch phases. Across the positive x boundary the wrapped value is multiplied by `exp(+i k_x a)`; across the negative boundary it is multiplied by `exp(-i k_x a)`, with analogous factors in y and z.

The relaxation remains

```text
R_n       = (H_h(k) - rho_n I) psi_n
q_{n+1}   = psi_n - tau R_n
psi_{n+1} = q_{n+1} / ||q_{n+1}||
```

with the complex inner product used for nonzero Bloch vectors.

## Periodic hydrogenic potential

The external potential is the grid-resolved reciprocal-space periodic Coulomb Green function in Rydberg units,

```text
V_G = -8*pi/(Omega*G^2) exp(-i G.R_p),   G != 0
V_0 = 0
```

where `Omega=a^3`. The zero reciprocal component fixes the arbitrary mean potential / neutralized periodic Green-function convention. The inverse FFT evaluates the truncated Fourier series on the real-space grid.

The appendix defaults to odd `N`, placing the proton at `(a/2,a/2,a/2)` halfway between grid nodes in every direction. This avoids direct sampling of the Coulomb singularity and eliminates an even-grid Nyquist-phase ambiguity.

## Validation sequence

The companion notebook performs, in order:

1. periodic Coulomb-potential reality and zero-mean checks;
2. an exact discrete free-electron Bloch-dispersion test;
3. a Hermiticity test of the matrix-free Bloch Hamiltonian;
4. a Gamma-point relaxation with sampled Rayleigh-descent check;
5. the lowest band along Gamma-X-M-R-Gamma;
6. a tiled Gamma-point probability-density plot;
7. optional time-reversal and lattice-spacing checks.

The free-electron test uses the exact second-order finite-difference result

```text
E_h(k) = (4/h^2) sum_alpha sin^2(k_alpha h / 2).
```

## Google Colab

Open:

```text
notebooks/APPENDIX-PERIODIC-HYDROGEN.ipynb
```

The notebook clones the repository into `/content`, imports only this appendix module, and writes all outputs to a separate `periodic_hydrogen_appendix_results` directory. It does not edit or overwrite the main paper results.

## Plot style

The appendix intentionally mirrors the existing paper plotting conventions:

- 7.2 x 4.8 inch line plots;
- black primary curves/markers with red highlights;
- the same font-size hierarchy and light grid lines;
- 6.4 x 5.4 inch density figures;
- `YlOrRd` density maps with white contours and black proton markers;
- 300 dpi PNG plus PDF export.
