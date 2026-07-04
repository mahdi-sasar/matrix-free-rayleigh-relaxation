#!/usr/bin/env python3
"""Hydrogen atom in a finite Dirichlet box with an off-grid Coulomb center."""

from __future__ import annotations

import argparse

import tensorflow as tf

from mrsr.grid import Grid3D
from mrsr.initializers import gaussian, random_positive
from mrsr.io import ensure_dir, save_metadata, save_midplane_slice
from mrsr.potentials import hydrogen_coulomb
from mrsr.solver import save_histories, solve_ground_state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--box", type=float, default=10.0)
    ap.add_argument("--sigma", type=float, default=0.9)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--max-iter", type=int, default=100_000)
    ap.add_argument("--init", choices=["random", "gaussian"], default="gaussian")
    ap.add_argument("--out", type=str, default="results/hydrogen_box")
    ap.add_argument("--float32", action="store_true")
    args = ap.parse_args()

    dtype = tf.float32 if args.float32 else tf.float64
    L = args.box
    grid = Grid3D((args.n, args.n, args.n), ((0.0, L), (0.0, L), (0.0, L)), dtype=dtype)
    center = grid.off_grid_center()
    v = hydrogen_coulomb(grid, center=center)
    psi0 = random_positive(grid) if args.init == "random" else gaussian(grid, center=center, width=1.0)

    result = solve_ground_state(v, grid.h, psi0, sigma=args.sigma, tolerance=args.tol, max_iterations=args.max_iter)

    out = ensure_dir(args.out)
    save_histories(result, out / "history.csv")
    save_midplane_slice(result.psi, out / "psi_mid_z.csv", axis="z")
    save_metadata(
        out / "metadata.json",
        case="hydrogen_box_offgrid",
        n=args.n,
        box=args.box,
        h=grid.h,
        coulomb_center_bohr=center,
        free_hydrogen_energy_Ry=-1.0,
        computed_energy_Ry=result.energy,
        iterations=result.iterations,
        converged=result.converged,
        tau=result.tau,
        sigma=result.sigma,
        elapsed_seconds=result.elapsed_seconds,
        dtype=str(dtype.name),
    )
    print(f"Computed E = {result.energy:.12f} Ry")
    print("Free hydrogen target is -1 Ry; finite box and discretization shift this value.")


if __name__ == "__main__":
    main()
