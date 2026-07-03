#!/usr/bin/env python3
"""Finite charge-pair dipole test with corrected Debye conversion."""

from __future__ import annotations

import argparse

import tensorflow as tf

from mrsr.grid import Grid3D
from mrsr.initializers import random_positive
from mrsr.io import ensure_dir, save_metadata, save_midplane_slice
from mrsr.potentials import dipole_moment_debye, point_dipole_pair
from mrsr.solver import save_histories, solve_ground_state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--box", type=float, default=16.0)
    ap.add_argument("--q", type=float, default=1.0)
    ap.add_argument("--separation", type=float, default=0.8)
    ap.add_argument("--sigma", type=float, default=0.9)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--max-iter", type=int, default=200_000)
    ap.add_argument("--out", type=str, default="results/corrected_dipole")
    ap.add_argument("--float32", action="store_true")
    args = ap.parse_args()

    dtype = tf.float32 if args.float32 else tf.float64
    L = args.box
    grid = Grid3D((args.n, args.n, args.n), ((0.0, L), (0.0, L), (0.0, L)), dtype=dtype)
    center = grid.off_grid_center()
    v = point_dipole_pair(grid, q=args.q, separation=args.separation, axis="y", center=center)
    psi0 = random_positive(grid)

    result = solve_ground_state(v, grid.h, psi0, sigma=args.sigma, tolerance=args.tol, max_iterations=args.max_iter)

    mu = dipole_moment_debye(args.q, args.separation)
    out = ensure_dir(args.out)
    save_histories(result, out / "history.csv")
    save_midplane_slice(result.psi, out / "psi_mid_z.csv", axis="z")
    save_metadata(
        out / "metadata.json",
        case="finite_charge_pair_dipole",
        n=args.n,
        box=args.box,
        h=grid.h,
        q_e=args.q,
        separation_Bohr=args.separation,
        dipole_moment_Debye=mu,
        computed_energy_Ry=result.energy,
        iterations=result.iterations,
        converged=result.converged,
        tau=result.tau,
        sigma=result.sigma,
        elapsed_seconds=result.elapsed_seconds,
        dtype=str(dtype.name),
    )
    print(f"Dipole moment = {mu:.6f} D")
    print(f"Computed E = {result.energy:.12f} Ry")


if __name__ == "__main__":
    main()
