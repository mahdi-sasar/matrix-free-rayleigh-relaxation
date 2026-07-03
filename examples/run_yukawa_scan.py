#!/usr/bin/env python3
"""Screened Coulomb / Yukawa scan for diffuse near-threshold states."""

from __future__ import annotations

import argparse

import pandas as pd
import tensorflow as tf

from mrsr.grid import Grid3D
from mrsr.initializers import gaussian
from mrsr.io import ensure_dir, save_metadata, save_midplane_slice
from mrsr.potentials import yukawa
from mrsr.solver import save_histories, solve_ground_state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--box", type=float, default=24.0)
    ap.add_argument("--kappa-values", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.35, 0.5])
    ap.add_argument("--sigma", type=float, default=0.9)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--max-iter", type=int, default=200_000)
    ap.add_argument("--out", type=str, default="results/yukawa_scan")
    ap.add_argument("--float32", action="store_true")
    args = ap.parse_args()

    dtype = tf.float32 if args.float32 else tf.float64
    L = args.box
    grid = Grid3D((args.n, args.n, args.n), ((-L/2, L/2), (-L/2, L/2), (-L/2, L/2)), dtype=dtype)
    out = ensure_dir(args.out)

    rows = []
    psi0 = gaussian(grid, center=grid.off_grid_center(), width=2.0)
    for kappa in args.kappa_values:
        print(f"\n=== Yukawa kappa = {kappa:.6f} ===")
        v = yukawa(grid, kappa=kappa, center=grid.off_grid_center())
        result = solve_ground_state(v, grid.h, psi0, sigma=args.sigma, tolerance=args.tol, max_iterations=args.max_iter)
        psi0 = result.psi
        case_dir = ensure_dir(out / f"kappa_{kappa:.4f}")
        save_histories(result, case_dir / "history.csv")
        save_midplane_slice(result.psi, case_dir / "psi_mid_z.csv", axis="z")
        rows.append({
            "kappa": kappa,
            "energy_Ry": result.energy,
            "iterations": result.iterations,
            "converged": result.converged,
            "elapsed_seconds": result.elapsed_seconds,
            "tau": result.tau,
        })

    pd.DataFrame(rows).to_csv(out / "yukawa_scan.csv", index=False)
    save_metadata(out / "metadata.json", case="yukawa_scan", n=args.n, box=args.box, h=grid.h, kappa_values=args.kappa_values, dtype=str(dtype.name))
    print(f"Wrote {out / 'yukawa_scan.csv'}")


if __name__ == "__main__":
    main()
