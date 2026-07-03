#!/usr/bin/env python3
"""Run a simple hydrogen grid-scaling series and write a summary CSV."""

from __future__ import annotations

import argparse

import pandas as pd
import tensorflow as tf

from mrsr.grid import Grid3D
from mrsr.initializers import gaussian
from mrsr.potentials import hydrogen_coulomb
from mrsr.solver import solve_ground_state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-values", type=int, nargs="+", default=[32, 40, 48, 56, 64])
    ap.add_argument("--box", type=float, default=10.0)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--sigma", type=float, default=0.9)
    ap.add_argument("--max-iter", type=int, default=100_000)
    ap.add_argument("--out", type=str, default="hydrogen_scaling.csv")
    ap.add_argument("--float32", action="store_true")
    args = ap.parse_args()

    dtype = tf.float32 if args.float32 else tf.float64
    rows = []
    for n in args.n_values:
        print(f"\n=== N = {n} per dimension ===")
        L = args.box
        grid = Grid3D((n, n, n), ((0.0, L), (0.0, L), (0.0, L)), dtype=dtype)
        center = grid.off_grid_center()
        v = hydrogen_coulomb(grid, center=center)
        psi0 = gaussian(grid, center=center, width=1.0)
        result = solve_ground_state(v, grid.h, psi0, sigma=args.sigma, tolerance=args.tol, max_iterations=args.max_iter)
        rows.append({
            "n_per_dim": n,
            "voxels_total": n**3,
            "h": grid.h,
            "energy_Ry": result.energy,
            "iterations": result.iterations,
            "converged": result.converged,
            "tau": result.tau,
            "elapsed_seconds": result.elapsed_seconds,
            "seconds_per_iteration": result.elapsed_seconds / max(1, result.iterations),
        })
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
