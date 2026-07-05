#!/usr/bin/env python3
"""Create publication-ready hydrogen grid-sweep plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _fit_energy_vs_h2(df: pd.DataFrame):
    """Fit E(h) ≈ E_inf + c h^2 using rows that converged."""
    work = df.copy()
    if "converged" in work.columns:
        conv = work["converged"].astype(str).str.lower().isin(["true", "1", "yes"])
        if conv.any():
            work = work[conv]
    if len(work) < 2:
        return None
    x = np.asarray(work["h_Bohr"], dtype=float) ** 2
    y = np.asarray(work["energy_Ry"], dtype=float)
    coeff = np.polyfit(x, y, deg=1)
    c, e_inf = coeff[0], coeff[1]
    return float(e_inf), float(c)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Hydrogen sweep CSV from scripts/run_scaling_hydrogen.py")
    ap.add_argument("--outdir", default=None, help="Output directory. Defaults to CSV parent.")
    ap.add_argument("--pdf", action="store_true", help="Also save PDF copies.")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    csv_path = Path(args.csv)
    outdir = Path(args.outdir) if args.outdir else csv_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path).sort_values("voxels_total")

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    # Energy vs grid points.
    fit = _fit_energy_vs_h2(df)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(df["voxels_total"], df["energy_Ry"], marker="o", markersize=5, linewidth=2.0, label="computed energy")
    ax.axhline(-1.0, linestyle="--", linewidth=1.2, alpha=0.65, label="free H limit (-1 Ry)")
    if fit is not None:
        e_inf, c = fit
        h_dense = np.linspace(df["h_Bohr"].min(), df["h_Bohr"].max(), 300)
        # Convert h to approximate voxels using the measured box from h = L/(n-1).
        # Instead of drawing this on a transformed x-grid, annotate the estimate.
        ax.text(
            0.03, 0.05,
            rf"$E_\infty$ fit from $E(h)=E_\infty+ch^2$: {e_inf:.6f} Ry",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.8", alpha=0.9),
        )
    ax.set_xscale("log")
    ax.set_xlabel("Total grid points")
    ax.set_ylabel("Energy (Ry)")
    ax.set_title("Hydrogen in a box: finite-grid energy convergence")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.legend(frameon=True)
    out = outdir / "hydrogen_energy_vs_gridpoints.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    if args.pdf:
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        print(f"Saved {out.with_suffix('.pdf')}")
    if args.show:
        plt.show()
    plt.close(fig)

    # Absolute energy error vs grid points.
    yerr = np.abs(np.asarray(df["energy_Ry"], dtype=float) + 1.0)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(df["voxels_total"], yerr, marker="o", markersize=5, linewidth=2.0, label=r"$|E+1|$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total grid points")
    ax.set_ylabel("Absolute energy difference from -1 Ry")
    ax.set_title("Hydrogen in a box: energy approach toward the free-atom value")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=True)
    out = outdir / "hydrogen_energy_error_vs_gridpoints.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    if args.pdf:
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        print(f"Saved {out.with_suffix('.pdf')}")
    if args.show:
        plt.show()
    plt.close(fig)

    # Runtime and iterations vs grid points.
    if "elapsed_seconds" in df.columns:
        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        ax.plot(df["voxels_total"], df["elapsed_seconds"], marker="o", markersize=5, linewidth=2.0, label="wall time")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Total grid points")
        ax.set_ylabel("Wall time (s)")
        ax.set_title("Hydrogen in a box: empirical wall-time scaling")
        ax.grid(True, which="both", alpha=0.25, linewidth=0.8)
        ax.legend(frameon=True)
        out = outdir / "hydrogen_walltime_vs_gridpoints.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")
        if args.pdf:
            fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
            print(f"Saved {out.with_suffix('.pdf')}")
        if args.show:
            plt.show()
        plt.close(fig)

    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
