#!/usr/bin/env python3
"""Validation case: 3D harmonic oscillator with known infinite-domain energy."""

from __future__ import annotations

import argparse
import math

import tensorflow as tf

from mrsr.grid import Grid3D
from mrsr.initializers import gaussian
from mrsr.io import ensure_dir, save_metadata, save_midplane_slice
from mrsr.potentials import harmonic_oscillator
from mrsr.solver import save_histories, solve_ground_state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--box", type=float, default=12.0)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--sigma", type=float, default=0.9)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--max-iter", type=int, default=50_000)
    ap.add_argument("--out", type=str, default="results/harmonic")
    ap.add_argument("--float32", action="store_true")
    args = ap.parse_args()

    dtype = tf.float32 if args.float32 else tf.float64
    L = args.box
    grid = Grid3D((args.n, args.n, args.n), ((-L/2, L/2), (-L/2, L/2), (-L/2, L/2)), dtype=dtype)
    v = harmonic_oscillator(grid, alpha=(args.alpha, args.alpha, args.alpha))
    psi0 = gaussian(grid, width=1.0 / math.sqrt(math.sqrt(args.alpha)))

    result = solve_ground_state(v, grid.h, psi0, sigma=args.sigma, tolerance=args.tol, max_iterations=args.max_iter)

    out = ensure_dir(args.out)
    save_histories(result, out / "history.csv")
    save_midplane_slice(result.psi, out / "psi_mid_z.csv", axis="z")

    exact_energy = 3.0 * math.sqrt(args.alpha)
    save_metadata(
        out / "metadata.json",
        case="harmonic_oscillator_3d",
        n=args.n,
        box=args.box,
        h=grid.h,
        alpha=args.alpha,
        exact_infinite_domain_energy_Ry=exact_energy,
        computed_energy_Ry=result.energy,
        absolute_energy_error_Ry=abs(result.energy - exact_energy),
        iterations=result.iterations,
        converged=result.converged,
        tau=result.tau,
        sigma=result.sigma,
        elapsed_seconds=result.elapsed_seconds,
        dtype=str(dtype.name),
    )
    print(f"Computed E = {result.energy:.12f} Ry")
    print(f"Infinite-domain exact E = {exact_energy:.12f} Ry")
    print(f"Absolute difference = {abs(result.energy - exact_energy):.3e} Ry")


if __name__ == "__main__":
    main()
