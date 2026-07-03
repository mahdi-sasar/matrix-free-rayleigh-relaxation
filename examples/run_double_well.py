#!/usr/bin/env python3
"""Tilted 3D double-well example."""

from __future__ import annotations

import argparse

import tensorflow as tf

from mrsr.grid import Grid3D
from mrsr.initializers import random_normal
from mrsr.io import ensure_dir, save_metadata, save_midplane_slice
from mrsr.potentials import double_well
from mrsr.solver import save_histories, solve_ground_state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--box", type=float, default=12.0)
    ap.add_argument("--a", type=float, default=0.02)
    ap.add_argument("--b", type=float, default=2.0)
    ap.add_argument("--omega-perp", type=float, default=0.3)
    ap.add_argument("--tilt", type=float, default=0.0)
    ap.add_argument("--sigma", type=float, default=0.9)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--max-iter", type=int, default=150_000)
    ap.add_argument("--out", type=str, default="results/double_well")
    ap.add_argument("--float32", action="store_true")
    args = ap.parse_args()

    dtype = tf.float32 if args.float32 else tf.float64
    L = args.box
    grid = Grid3D((args.n, args.n, args.n), ((-L/2, L/2), (-L/2, L/2), (-L/2, L/2)), dtype=dtype)
    v = double_well(grid, a=args.a, b=args.b, omega_perp=args.omega_perp, tilt=args.tilt)
    psi0 = random_normal(grid)

    result = solve_ground_state(v, grid.h, psi0, sigma=args.sigma, tolerance=args.tol, max_iterations=args.max_iter)

    out = ensure_dir(args.out)
    save_histories(result, out / "history.csv")
    save_midplane_slice(result.psi, out / "psi_mid_z.csv", axis="z")
    save_metadata(
        out / "metadata.json",
        case="tilted_double_well",
        n=args.n,
        box=args.box,
        h=grid.h,
        a=args.a,
        b=args.b,
        omega_perp=args.omega_perp,
        tilt=args.tilt,
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
