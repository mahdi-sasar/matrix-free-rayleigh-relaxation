#!/usr/bin/env python3
"""Run a hydrogen grid-scaling series and write summary data.

This script is intended for the paper's finite-grid convergence and scaling plots.
It can also save wavefunction mid-slices for the largest grid, which is useful for
publication figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import tensorflow as tf

from mrsr.grid import Grid3D
from mrsr.initializers import gaussian
from mrsr.io import ensure_dir, save_metadata, save_midplane_slice
from mrsr.potentials import hydrogen_coulomb
from mrsr.solver import save_histories, solve_ground_state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-values", type=int, nargs="+", default=[32, 40, 48, 56, 64])
    ap.add_argument("--box", type=float, default=10.0)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--max-iter", type=int, default=100_000)
    ap.add_argument("--check-every", type=int, default=20)
    ap.add_argument("--out", type=str, default=None, help="Summary CSV. If omitted, writes OUTDIR/hydrogen_scaling.csv.")
    ap.add_argument("--outdir", type=str, default="results/hydrogen_n_sweep", help="Directory for summary and optional per-N outputs.")
    ap.add_argument("--save-largest-slice", action="store_true", default=True, help="Save history and mid-slice for the largest N. Enabled by default.")
    ap.add_argument("--save-all-slices", action="store_true", help="Save history and mid-slice for every N.")
    ap.add_argument("--float32", action="store_true")
    args = ap.parse_args()

    dtype = tf.float32 if args.float32 else tf.float64
    outdir = ensure_dir(args.outdir)
    out_csv = Path(args.out) if args.out else outdir / "hydrogen_scaling.csv"

    rows = []
    nmax = max(args.n_values)

    for n in args.n_values:
        print(f"\n=== Hydrogen box: N = {n} per dimension ===")
        L = args.box
        grid = Grid3D((n, n, n), ((0.0, L), (0.0, L), (0.0, L)), dtype=dtype)
        center = grid.off_grid_center()
        rmin = grid.min_distance_to_point(center)
        print(f"Coulomb center = {center}; nearest grid-node distance = {rmin:.6e} Bohr")
        v = hydrogen_coulomb(grid, center=center)
        psi0 = gaussian(grid, center=center, width=1.0)
        result = solve_ground_state(
            v,
            grid.h,
            psi0,
            sigma=args.sigma,
            tolerance=args.tol,
            max_iterations=args.max_iter,
            check_every=args.check_every,
        )

        save_this = args.save_all_slices or (args.save_largest_slice and n == nmax)
        case_dir = None
        if save_this:
            case_dir = ensure_dir(outdir / f"N_{n:04d}")
            save_histories(result, case_dir / "history.csv")
            save_midplane_slice(result.psi, case_dir / "psi_mid_z.csv", axis="z")
            save_metadata(
                case_dir / "metadata.json",
                case="hydrogen_box_sweep_member",
                n=n,
                box=args.box,
                h=grid.h,
                total_grid_points=n**3,
                interior_grid_points=max(0, n - 2) ** 3,
                coulomb_center_bohr=center,
                nearest_grid_node_distance_bohr=rmin,
                free_hydrogen_energy_Ry=-1.0,
                computed_energy_Ry=result.energy,
                iterations=result.iterations,
                converged=result.converged,
                tau=result.tau,
                sigma=result.sigma,
                elapsed_seconds=result.elapsed_seconds,
                dtype=str(dtype.name),
            )

        rows.append({
            "n_per_dim": n,
            "voxels_total": n**3,
            "interior_points": max(0, n - 2) ** 3,
            "h_Bohr": grid.h,
            "energy_Ry": result.energy,
            "energy_error_vs_free_Ry": result.energy - (-1.0),
            "abs_energy_error_vs_free_Ry": abs(result.energy + 1.0),
            "iterations": result.iterations,
            "converged": result.converged,
            "tau": result.tau,
            "sigma": result.sigma,
            "elapsed_seconds": result.elapsed_seconds,
            "seconds_per_iteration": result.elapsed_seconds / max(1, result.iterations),
            "nearest_grid_node_distance_bohr": rmin,
            "slice_dir": str(case_dir) if case_dir is not None else "",
        })

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    save_metadata(
        outdir / "metadata.json",
        case="hydrogen_n_sweep",
        n_values=args.n_values,
        box=args.box,
        tolerance=args.tol,
        sigma=args.sigma,
        max_iterations=args.max_iter,
        check_every=args.check_every,
        dtype=str(dtype.name),
        summary_csv=str(out_csv),
        note="Energy convergence should be interpreted as a combined finite-box and finite-grid study. The free hydrogen target -1 Ry is shown only as a reference.",
    )
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
