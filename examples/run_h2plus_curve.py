#!/usr/bin/env python3
"""H2+ one-electron potential-energy curve in a finite Dirichlet box.

The matrix-free solver computes the electronic energy of

    H = -∇² - 2/r_A - 2/r_B

in Rydberg atomic units.  The optional total Born-Oppenheimer energy adds
proton-proton repulsion, 2/R Ry, for an internuclear separation R in Bohr.

The Coulomb centers are placed slightly off the Cartesian grid in directions
transverse to the molecular axis.  This avoids evaluating the singularities
without introducing a pseudopotential or smoothing function.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
import tensorflow as tf

from mrsr.grid import Grid3D
from mrsr.initializers import gaussian
from mrsr.io import ensure_dir, save_metadata, save_midplane_slice
from mrsr.potentials import h2plus
from mrsr.solver import save_histories, solve_ground_state


def _cell_midpoint_near(grid: Grid3D, coordinate_axis: int, value: float) -> float:
    """Return the midpoint between two adjacent grid nodes nearest to ``value``.

    Placing Coulomb centers at cell midpoints in the transverse directions avoids
    the severe grid-phase artifact that occurs when a singularity is sampled too
    close to a Cartesian node.  This is not a smoothing or pseudopotential; the
    potential remains exactly Coulombic at the sampled mesh points.
    """
    lo, hi = grid.bounds[coordinate_axis]
    n = grid.shape[coordinate_axis]
    h = (hi - lo) / (n - 1)
    # Choose a valid cell index 0 <= j <= n-2 whose midpoint is near ``value``.
    j = int((value - lo) // h)
    j = min(max(j, 0), n - 2)
    candidate = lo + (j + 0.5) * h
    # The nearest midpoint may be in the neighboring cell when value is close to
    # an edge.  Check j-1, j, and j+1 to avoid a one-cell bias.
    best = candidate
    best_dist = abs(candidate - value)
    for jj in (j - 1, j, j + 1):
        if 0 <= jj <= n - 2:
            mid = lo + (jj + 0.5) * h
            dist = abs(mid - value)
            if dist < best_dist:
                best = mid
                best_dist = dist
    return float(best)


def transverse_offgrid_center(
    grid: Grid3D,
    axis: str = "z",
) -> Tuple[float, float, float]:
    """Return a molecular midpoint that is cell-centered transversely.

    For a diatomic molecule aligned along ``axis``, the bond midpoint is kept at
    the box center along the molecular axis.  In the two transverse directions,
    it is placed halfway between neighboring grid nodes.  Therefore, even if one
    nucleus happens to align with a grid plane along the bond coordinate, its
    nearest possible grid-node distance is at least sqrt(2) h / 2.

    This removes the large, unphysical spikes caused by subgrid Coulomb alignment
    while preserving an unsmoothed Coulomb potential.
    """
    c = list(grid.center)
    axes = {"x": 0, "y": 1, "z": 2}
    if axis not in axes:
        raise ValueError("axis must be 'x', 'y', or 'z'.")
    bond_axis = axes[axis]
    for idx in range(3):
        if idx != bond_axis:
            c[idx] = _cell_midpoint_near(grid, idx, c[idx])
    return (float(c[0]), float(c[1]), float(c[2]))


def nuclei_positions(center, separation: float, axis: str = "z"):
    cx, cy, cz = center
    half = 0.5 * separation
    if axis == "x":
        return (cx - half, cy, cz), (cx + half, cy, cz)
    if axis == "y":
        return (cx, cy - half, cz), (cx, cy + half, cz)
    if axis == "z":
        return (cx, cy, cz - half), (cx, cy, cz + half)
    raise ValueError("axis must be 'x', 'y', or 'z'.")


def make_plot(csv_path: Path, out_png: Path) -> None:
    """Plot electronic and total H2+ energies versus separation."""
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    fig = plt.figure(figsize=(6.0, 4.2))
    plt.plot(df["R_Bohr"], df["electronic_energy_Ry"], marker="o", label="electronic")
    plt.plot(df["R_Bohr"], df["total_energy_Ry"], marker="o", label="total = electronic + 2/R")
    if len(df):
        idx = int(df["total_energy_Ry"].idxmin())
        plt.scatter([df.loc[idx, "R_Bohr"]], [df.loc[idx, "total_energy_Ry"]], s=80, label="lowest sampled total")
    plt.xlabel("Internuclear separation R (Bohr)")
    plt.ylabel("Energy (Ry)")
    plt.title("H2+ finite-box potential-energy curve")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--box", type=float, default=16.0)
    ap.add_argument("--r-values", type=float, nargs="+", default=[0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0])
    ap.add_argument("--axis", choices=["x", "y", "z"], default="z")
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--max-iter", type=int, default=150_000)
    ap.add_argument("--check-every", type=int, default=10)
    ap.add_argument("--out", type=str, default="results/h2plus_curve")
    ap.add_argument("--float32", action="store_true")
    ap.add_argument("--no-plot", action="store_true", help="Do not create h2plus_curve.png.")
    args = ap.parse_args()

    dtype = tf.float32 if args.float32 else tf.float64
    L = args.box
    grid = Grid3D((args.n, args.n, args.n), ((-L/2, L/2), (-L/2, L/2), (-L/2, L/2)), dtype=dtype)
    out = ensure_dir(args.out)

    center = transverse_offgrid_center(grid, axis=args.axis)
    print(f"Molecular midpoint = {center}")
    print(f"Grid spacing h = {grid.h:.12e} Bohr")
    print(
        "H2+ Coulomb centers are cell-centered in the two transverse directions; "
        f"minimum possible transverse distance is {2**0.5 * grid.h / 2:.6e} Bohr."
    )

    rows = []
    # A broad Gaussian is a useful first guess for the covalent one-electron state.
    # Subsequent separations use continuation from the previous solution.
    psi0 = gaussian(grid, center=center, width=max(1.5, 0.20 * args.box))

    for R in args.r_values:
        c1, c2 = nuclei_positions(center, R, args.axis)
        rmin1 = grid.min_distance_to_point(c1)
        rmin2 = grid.min_distance_to_point(c2)
        print(f"\n=== H2+ separation R = {R:.6f} Bohr ===")
        print(f"nucleus A = {c1}; nearest grid-node distance = {rmin1:.6e} Bohr")
        print(f"nucleus B = {c2}; nearest grid-node distance = {rmin2:.6e} Bohr")

        v = h2plus(grid, separation=R, axis=args.axis, center=center)
        result = solve_ground_state(
            v,
            grid.h,
            psi0,
            sigma=args.sigma,
            tolerance=args.tol,
            max_iterations=args.max_iter,
            check_every=args.check_every,
        )
        # Continuation: use this solution as initial guess for the next separation.
        psi0 = result.psi
        case_dir = ensure_dir(out / f"R_{R:.4f}")
        save_histories(result, case_dir / "history.csv")
        save_midplane_slice(result.psi, case_dir / "psi_mid_z.csv", axis=args.axis)
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
            "rmin_nucleus_A_Bohr": rmin1,
            "rmin_nucleus_B_Bohr": rmin2,
        })
        print(f"E_elec = {result.energy:+.12f} Ry; E_total = {total_energy:+.12f} Ry")
        if total_energy < -10.0:
            print(
                "WARNING: unusually negative H2+ total energy. "
                "Inspect rmin_nucleus_A/B in h2plus_curve.csv and rerun with a finer grid or larger box."
            )

    csv_path = out / "h2plus_curve.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    if not args.no_plot:
        make_plot(csv_path, out / "h2plus_curve.png")

    save_metadata(
        out / "metadata.json",
        case="h2plus_curve",
        n=args.n,
        box=args.box,
        h=grid.h,
        molecular_midpoint_bohr=center,
        axis=args.axis,
        r_values_Bohr=args.r_values,
        sigma=args.sigma,
        tolerance=args.tol,
        dtype=str(dtype.name),
        note="Total energy adds proton-proton repulsion 2/R in Rydberg units. Coulomb centers are cell-centered transversely to reduce grid-phase artifacts.",
    )
    print(f"Wrote {csv_path}")
    if not args.no_plot:
        print(f"Wrote {out / 'h2plus_curve.png'}")


if __name__ == "__main__":
    main()
