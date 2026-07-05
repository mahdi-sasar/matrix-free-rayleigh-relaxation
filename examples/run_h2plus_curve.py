#!/usr/bin/env python3
"""H2+ one-electron potential-energy curve in a finite Dirichlet box.

The matrix-free solver computes the electronic energy of

    H = -∇² - 2/r_A - 2/r_B

in Rydberg atomic units.  The optional total Born-Oppenheimer energy adds
proton-proton repulsion, 2/R Ry, for an internuclear separation R in Bohr.

The Coulomb centers are placed off the Cartesian grid in directions transverse
to the molecular axis.  This avoids direct evaluation of the singularities
without introducing a pseudopotential or smoothing function.

For publication figures, each separation also saves two wavefunction slices:

  * psi_transverse_*.csv : plane perpendicular to the bond through the midpoint.
  * psi_bond_*.csv       : plane containing the two nuclei and the bond axis.

The historical psi_mid_z.csv filename is still written for compatibility.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from mrsr.grid import Grid3D
from mrsr.initializers import gaussian
from mrsr.io import ensure_dir, save_metadata
from mrsr.potentials import h2plus
from mrsr.solver import save_histories, solve_ground_state


def _cell_midpoint_near(grid: Grid3D, coordinate_axis: int, value: float) -> float:
    """Return the midpoint between two adjacent grid nodes nearest to ``value``."""
    lo, hi = grid.bounds[coordinate_axis]
    n = grid.shape[coordinate_axis]
    h = (hi - lo) / (n - 1)
    j = int((value - lo) // h)
    j = min(max(j, 0), n - 2)
    best = lo + (j + 0.5) * h
    best_dist = abs(best - value)
    for jj in (j - 1, j, j + 1):
        if 0 <= jj <= n - 2:
            mid = lo + (jj + 0.5) * h
            dist = abs(mid - value)
            if dist < best_dist:
                best = mid
                best_dist = dist
    return float(best)


def _nearest_grid_index(grid: Grid3D, coordinate_axis: int, value: float) -> int:
    lo, hi = grid.bounds[coordinate_axis]
    n = grid.shape[coordinate_axis]
    h = (hi - lo) / (n - 1)
    idx = int(round((value - lo) / h))
    return min(max(idx, 0), n - 1)


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


def _save_csv(path: Path, array: np.ndarray) -> None:
    np.savetxt(path, array, delimiter=",")


def save_h2plus_slices(
    psi: tf.Tensor,
    grid: Grid3D,
    outdir: Path,
    center: Tuple[float, float, float],
    separation: float,
    axis: str = "z",
) -> dict:
    """Save transverse and bond-plane slices for H2+.

    Returns metadata describing the saved files and their axis labels.
    """
    arr = psi.numpy()
    cx, cy, cz = center
    ix = _nearest_grid_index(grid, 0, cx)
    iy = _nearest_grid_index(grid, 1, cy)
    iz = _nearest_grid_index(grid, 2, cz)

    meta: dict[str, dict] = {}

    if axis == "z":
        # Transverse plane perpendicular to the bond: x-y at z ~ center.
        trans = arr[:, :, iz]
        trans_name = "psi_transverse_xy.csv"
        trans_labels = ("X", "Y")
        # Bond plane containing the nuclei: x-z at y ~ center.
        bond = arr[:, iy, :]
        bond_name = "psi_bond_xz.csv"
        bond_labels = ("X", "Z")
        nuclei_2d = [(cx, cz - 0.5 * separation), (cx, cz + 0.5 * separation)]
    elif axis == "x":
        # Transverse plane perpendicular to the bond: y-z at x ~ center.
        trans = arr[ix, :, :]
        trans_name = "psi_transverse_yz.csv"
        trans_labels = ("Y", "Z")
        # Bond plane containing the nuclei: x-y at z ~ center.
        bond = arr[:, :, iz]  # dimensions x,y
        bond_name = "psi_bond_xy.csv"
        bond_labels = ("X", "Y")
        nuclei_2d = [(cx - 0.5 * separation, cy), (cx + 0.5 * separation, cy)]
    elif axis == "y":
        # Transverse plane perpendicular to the bond: x-z at y ~ center.
        trans = arr[:, iy, :]
        trans_name = "psi_transverse_xz.csv"
        trans_labels = ("X", "Z")
        # Bond plane containing the nuclei: x-y at z ~ center.
        bond = arr[:, :, iz]  # dimensions x,y
        bond_name = "psi_bond_xy.csv"
        bond_labels = ("X", "Y")
        nuclei_2d = [(cx, cy - 0.5 * separation), (cx, cy + 0.5 * separation)]
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'.")

    _save_csv(outdir / trans_name, trans)
    _save_csv(outdir / bond_name, bond)

    # Historical filename used by earlier notebooks/scripts.
    _save_csv(outdir / "psi_mid_z.csv", trans)

    meta["transverse"] = {
        "file": trans_name,
        "axis_labels": trans_labels,
        "description": "Plane perpendicular to the molecular bond through the midpoint.",
    }
    meta["bond"] = {
        "file": bond_name,
        "axis_labels": bond_labels,
        "description": "Plane containing the two nuclei and the molecular bond.",
        "nuclei_positions_2d": nuclei_2d,
    }
    meta["legacy_mid_slice"] = {"file": "psi_mid_z.csv", "same_as": trans_name}
    return meta


def make_plot(csv_path: Path, out_png: Path) -> None:
    """Plot electronic and total H2+ energies versus separation and save PNG/PDF."""
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path).sort_values("R_Bohr")
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(df["R_Bohr"], df["electronic_energy_Ry"], marker="o", markersize=5, linewidth=2.0, color="black", markerfacecolor="black", markeredgecolor="black", label="Electronic energy")
    ax.plot(df["R_Bohr"], df["total_energy_Ry"], marker="s", markersize=5, linewidth=2.0, color="red", markerfacecolor="red", markeredgecolor="red", label="Total energy")
    if len(df):
        idx = int(df["total_energy_Ry"].idxmin())
        rbest = float(df.loc[idx, "R_Bohr"])
        ebest = float(df.loc[idx, "total_energy_Ry"])
        ax.scatter([rbest], [ebest], s=90, zorder=5, color="red", edgecolors="red", label=fr"Lowest sampled total ($R={rbest:.3f}$ Bohr)")
        ax.axvline(rbest, linestyle="--", linewidth=1.0, color="red", alpha=0.5)
    ax.set_xlabel("Internuclear separation $R$ (Bohr)")
    ax.set_ylabel("Energy (Ry)")
    ax.set_title(r"$\mathrm{H}_2^+$ finite-box potential-energy curve")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.legend(frameon=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
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
    psi0 = gaussian(grid, center=center, width=max(1.5, 0.20 * args.box))
    slice_catalog = {}

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
        psi0 = result.psi
        case_dir = ensure_dir(out / f"R_{R:.4f}")
        save_histories(result, case_dir / "history.csv")
        slice_catalog[f"R_{R:.4f}"] = save_h2plus_slices(result.psi, grid, case_dir, center, R, args.axis)

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
        slice_catalog=slice_catalog,
        note="Total energy adds proton-proton repulsion 2/R in Rydberg units. Coulomb centers are cell-centered transversely to reduce grid-phase artifacts.",
    )
    print(f"Wrote {csv_path}")
    if not args.no_plot:
        print(f"Wrote {out / 'h2plus_curve.png'}")


if __name__ == "__main__":
    main()
