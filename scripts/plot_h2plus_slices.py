#!/usr/bin/env python3
"""Create publication-ready H2+ wavefunction-slice heatmaps.

The script reads an H2+ output directory produced by examples/run_h2plus_curve.py.
It can plot either the transverse midpoint slice or the bond-plane slice containing
the two nuclei.  Density plots use a red-yellow heatmap by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read_metadata(outdir: Path) -> dict:
    path = outdir / "metadata.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _choose_R(outdir: Path, requested: str | None) -> float:
    df = pd.read_csv(outdir / "h2plus_curve.csv")
    if requested is None or requested == "best":
        idx = int(df["total_energy_Ry"].idxmin())
        return float(df.loc[idx, "R_Bohr"])
    return float(requested)


def _default_file(axis: str, plane: str) -> tuple[str, tuple[str, str]]:
    if plane == "transverse":
        if axis == "z":
            return "psi_transverse_xy.csv", ("X", "Y")
        if axis == "x":
            return "psi_transverse_yz.csv", ("Y", "Z")
        if axis == "y":
            return "psi_transverse_xz.csv", ("X", "Z")
    if plane == "bond":
        if axis == "z":
            return "psi_bond_xz.csv", ("X", "Z")
        if axis == "x":
            return "psi_bond_xy.csv", ("X", "Y")
        if axis == "y":
            return "psi_bond_xy.csv", ("X", "Y")
    raise ValueError("Invalid axis/plane combination.")


def _nuclei_positions_2d(axis: str, center: list[float], R: float, labels: tuple[str, str]):
    cx, cy, cz = map(float, center)
    half = 0.5 * R
    if axis == "z":
        return [(cx, cz - half), (cx, cz + half)]
    if axis == "x":
        return [(cx - half, cy), (cx + half, cy)]
    if axis == "y":
        return [(cx, cy - half), (cx, cy + half)]
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=str, help="H2+ output directory containing h2plus_curve.csv")
    ap.add_argument("--R", default="best", help="Distance to plot, or 'best' for lowest sampled total energy.")
    ap.add_argument("--plane", choices=["bond", "transverse"], default="bond")
    ap.add_argument("--density", action="store_true", default=True, help="Plot |psi|^2. Enabled by default.")
    ap.add_argument("--amplitude", action="store_true", help="Plot psi instead of |psi|^2.")
    ap.add_argument("--cmap", default="YlOrRd", help="Matplotlib colormap.")
    ap.add_argument("--contours", type=int, default=12, help="Number of contour levels; 0 disables contours.")
    ap.add_argument("--out", default=None, help="Output image path. Default is inside the R folder.")
    ap.add_argument("--pdf", action="store_true", help="Also save a PDF copy.")
    ap.add_argument("--show", action="store_true", help="Show interactively.")
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    outdir = Path(args.outdir)
    meta = _read_metadata(outdir)
    axis = meta.get("axis", "z")
    box = float(meta.get("box", 16.0))
    center = meta.get("molecular_midpoint_bohr", [0.0, 0.0, 0.0])
    R = _choose_R(outdir, args.R)
    case_dir = outdir / f"R_{R:.4f}"

    filename, labels = _default_file(axis, args.plane)
    csv_path = case_dir / filename
    if not csv_path.exists() and args.plane == "transverse":
        csv_path = case_dir / "psi_mid_z.csv"
    if not csv_path.exists():
        available = sorted(p.name for p in case_dir.glob("*.csv"))
        raise FileNotFoundError(f"Could not find {filename} in {case_dir}. Available CSV files: {available}")

    arr = np.loadtxt(csv_path, delimiter=",")
    data = arr if args.amplitude else arr * arr
    quantity = r"$\psi$" if args.amplitude else r"$|\psi|^2$"
    suffix = "amplitude" if args.amplitude else "density"

    if args.out:
        out_png = Path(args.out)
    else:
        out_png = case_dir / f"h2plus_{args.plane}_{suffix}_pretty.png"

    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    extent = [-box / 2, box / 2, -box / 2, box / 2]
    im = ax.imshow(data.T, origin="lower", extent=extent, aspect="equal", cmap=args.cmap)

    if args.contours > 0 and np.nanmax(data) > np.nanmin(data):
        x = np.linspace(extent[0], extent[1], data.shape[0])
        y = np.linspace(extent[2], extent[3], data.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")
        levels = np.linspace(np.nanmin(data), np.nanmax(data), args.contours + 2)[1:-1]
        ax.contour(X, Y, data, levels=levels, colors="white", linewidths=0.45, alpha=0.75)

    if args.plane == "bond" and center:
        nuclei = _nuclei_positions_2d(axis, center, R, labels)
        if nuclei:
            nx, ny = zip(*nuclei)
            ax.scatter(nx, ny, marker="o", s=70, facecolor="none", edgecolor="black", linewidth=1.4, label="protons")
            ax.scatter(nx, ny, marker="+", s=90, color="black", linewidth=1.1)

    ax.set_xlabel(f"{labels[0]} (Bohr)")
    ax.set_ylabel(f"{labels[1]} (Bohr)")
    if args.plane == "bond":
        title = rf"$\mathrm{{H}}_2^+$ bond-plane {suffix}, $R={R:.3f}$ Bohr"
    else:
        title = rf"$\mathrm{{H}}_2^+$ transverse mid-slice {suffix}, $R={R:.3f}$ Bohr"
    ax.set_title(title)
    if args.plane == "bond":
        ax.legend(loc="upper right", frameon=True)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(quantity)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved {out_png}")

    if args.pdf:
        out_pdf = out_png.with_suffix(".pdf")
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"Saved {out_pdf}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
