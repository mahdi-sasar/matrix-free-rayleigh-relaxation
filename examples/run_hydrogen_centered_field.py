#!/usr/bin/env python3
"""Hydrogen atom in a centered uniform electric field inside a finite box."""

from __future__ import annotations

import argparse

import tensorflow as tf

from mrsr.grid import Grid3D
from mrsr.initializers import gaussian
from mrsr.io import ensure_dir, save_metadata, save_midplane_slice
from mrsr.potentials import hydrogen_in_centered_field
from mrsr.solver import save_histories, solve_ground_state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--box", type=float, default=10.0)
    ap.add_argument("--field", type=float, default=0.08, help="Field strength F in the term -F(x-x0), Ry/Bohr.")
    ap.add_argument("--sigma", type=float, default=0.9)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--max-iter", type=int, default=100_000)
    ap.add_argument("--out", type=str, default="results/hydrogen_field")
    ap.add_argument("--float32", action="store_true")
    args = ap.parse_args()

    dtype = tf.float32 if args.float32 else tf.float64
    L = args.box
    grid = Grid3D((args.n, args.n, args.n), ((0.0, L), (0.0, L), (0.0, L)), dtype=dtype)
    center = grid.off_grid_center()
    v = hydrogen_in_centered_field(grid, field_strength=args.field, center=center)
    psi0 = gaussian(grid, center=center, width=1.0)

    result = solve_ground_state(v, grid.h, psi0, sigma=args.sigma, tolerance=args.tol, max_iterations=args.max_iter)

    out = ensure_dir(args.out)
    save_histories(result, out / "history.csv")
    save_midplane_slice(result.psi, out / "psi_mid_z.csv", axis="z")
    save_metadata(
        out / "metadata.json",
        case="hydrogen_centered_uniform_field",
        n=args.n,
        box=args.box,
        h=grid.h,
        field_strength_Ry_per_Bohr=args.field,
        field_term="-F*(x-x0)",
        coulomb_center_bohr=center,
        computed_energy_Ry=result.energy,
        iterations=result.iterations,
        converged=result.converged,
        tau=result.tau,
        sigma=result.sigma,
        elapsed_seconds=result.elapsed_seconds,
        dtype=str(dtype.name),
    )
    print(f"Computed E = {result.energy:.12f} Ry")


if __name__ == "__main__":
    main()
