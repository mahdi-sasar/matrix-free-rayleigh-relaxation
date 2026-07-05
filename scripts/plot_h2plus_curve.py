#!/usr/bin/env python3
"""Replot an H2+ potential-energy curve from h2plus_curve.csv with paper-friendly styling."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=str, help="Path to h2plus_curve.csv")
    ap.add_argument("--out", type=str, default=None, help="Output PNG path")
    ap.add_argument("--show", action="store_true", help="Show the plot interactively")
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path).sort_values("R_Bohr")
    out = Path(args.out) if args.out else csv_path.with_name("h2plus_curve_replotted.png")

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(df["R_Bohr"], df["electronic_energy_Ry"], marker="o", markersize=5, linewidth=2.0, label="Electronic energy")
    ax.plot(df["R_Bohr"], df["total_energy_Ry"], marker="s", markersize=5, linewidth=2.0, label="Total energy")
    idx = int(df["total_energy_Ry"].idxmin())
    rbest = float(df.loc[idx, "R_Bohr"])
    ebest = float(df.loc[idx, "total_energy_Ry"])
    ax.scatter([rbest], [ebest], s=80, zorder=5, label=fr"Lowest sampled total ($R={rbest:.3f}$ Bohr)")
    ax.axvline(rbest, linestyle="--", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("Internuclear separation $R$ (Bohr)")
    ax.set_ylabel("Energy (Ry)")
    ax.set_title(r"$\mathrm{H}_2^+$ finite-box potential-energy curve")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.legend(frameon=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")

    cols = ["R_Bohr", "electronic_energy_Ry", "total_energy_Ry", "iterations", "converged", "rmin_nucleus_A_Bohr", "rmin_nucleus_B_Bohr"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))
    print()
    print(f"Lowest sampled total energy: R={rbest:.6f} Bohr, E={ebest:.12f} Ry")
    print(f"Saved {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
