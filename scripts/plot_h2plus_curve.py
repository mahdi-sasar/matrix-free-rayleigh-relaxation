#!/usr/bin/env python3
"""Replot an H2+ potential-energy curve from h2plus_curve.csv."""

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
    df = pd.read_csv(csv_path)
    out = Path(args.out) if args.out else csv_path.with_name("h2plus_curve_replotted.png")

    fig = plt.figure(figsize=(7.0, 4.8))
    plt.plot(df["R_Bohr"], df["electronic_energy_Ry"], marker="o", label="Electronic energy")
    plt.plot(df["R_Bohr"], df["total_energy_Ry"], marker="o", label="Total energy = electronic + 2/R")
    idx = int(df["total_energy_Ry"].idxmin())
    plt.scatter([df.loc[idx, "R_Bohr"]], [df.loc[idx, "total_energy_Ry"]], s=90, label=f"Lowest sampled: R={df.loc[idx, 'R_Bohr']:.3f} Bohr")
    plt.xlabel("Internuclear separation R (Bohr)")
    plt.ylabel("Energy (Ry)")
    plt.title("H2+ finite-box potential-energy curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out, dpi=200)

    cols = ["R_Bohr", "electronic_energy_Ry", "total_energy_Ry", "iterations", "converged", "rmin_nucleus_A_Bohr", "rmin_nucleus_B_Bohr"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))
    print(f"\nLowest sampled total energy: R={df.loc[idx, 'R_Bohr']:.6f} Bohr, E={df.loc[idx, 'total_energy_Ry']:.12f} Ry")
    print(f"Saved {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
