#!/usr/bin/env python3
"""H2+ one-electron potential-energy curve in a finite box.

The electronic energy is computed from the matrix-free solver. The optional total
Born-Oppenheimer energy adds the proton-proton repulsion term 2/R in Rydbergs.
"""

from __future__ import annotations

import argparse

import pandas as pd
import tensorflow as tf

from mrsr.grid import Grid3D
from mrsr.initializers import gaussian
from mrsr.io import ensure_dir, save_metadata, save_midplane_slice
from mrsr.potentials import h2plus
from mrsr.solver import save_histories, solve_ground_state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--box", type=float, default=16.0)
    ap.add_argument("--r-values", type=float, nargs="+", default=[1.0, 1.4, 2.0, 2.6, 3.2, 4.0])
    ap.add_argument("--sigma", type=float, default=0.9)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--max-iter", type=int, default=150_000)
    ap.add_argument("--out", type=str, default="results/h2plus_curve")
    ap.add_argument("--float32", action="store_true")
    args = ap.parse_args()

    dtype = tf.float32 if args.float32 else tf.float64
    L = args.box
    grid = Grid3D((args.n, args.n, args.n), ((-L/2, L/2), (-L/2, L/2), (-L/2, L/2)), dtype=dtype)
    out = ensure_dir(args.out)

    rows = []
    psi0 = gaussian(grid, center=grid.off_grid_center(), width=2.0)
    for R in args.r_values:
        print(f"\n=== H2+ separation R = {R:.6f} Bohr ===")
        v = h2plus(grid, separation=R, axis="z", center=grid.off_grid_center())
        result = solve_ground_state(v, grid.h, psi0, sigma=args.sigma, tolerance=args.tol, max_iterations=args.max_iter)
        # Continuation: use this solution as initial guess for the next separation.
        psi0 = result.psi
        case_dir = ensure_dir(out / f"R_{R:.4f}")
        save_histories(result, case_dir / "history.csv")
        save_midplane_slice(result.psi, case_dir / "psi_mid_z.csv", axis="z")
        total_energy = result.energy + 2.0 / R
        rows.append({
            "R_Bohr": R,
            "electronic_energy_Ry": result.energy,
            "nuclear_repulsion_Ry": 2.0 / R,
            "total_energy_Ry": total_energy,
            "iterations": result.iterations,
            "converged": result.converged,
            "elapsed_seconds": result.elapsed_seconds,
            "tau": result.tau,
        })

    pd.DataFrame(rows).to_csv(out / "h2plus_curve.csv", index=False)
    save_metadata(
        out / "metadata.json",
        case="h2plus_curve",
        n=args.n,
        box=args.box,
        h=grid.h,
        r_values_Bohr=args.r_values,
        sigma=args.sigma,
        tolerance=args.tol,
        dtype=str(dtype.name),
    )
    print(f"Wrote {out / 'h2plus_curve.csv'}")


if __name__ == "__main__":
    main()
