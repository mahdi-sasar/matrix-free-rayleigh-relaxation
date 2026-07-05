#!/usr/bin/env python3
"""Plot a CSV slice produced by the examples with paper-friendly styling."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="slice.png")
    ap.add_argument("--title", default="Wavefunction slice")
    ap.add_argument("--extent", type=float, nargs=4, default=None, metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    ap.add_argument("--density", action="store_true", help="Plot |psi|^2 instead of psi.")
    ap.add_argument("--cmap", default=None, help="Matplotlib colormap. Default: viridis for amplitude, YlOrRd for density.")
    ap.add_argument("--contours", type=int, default=12, help="Number of contour levels to overlay; 0 disables contours.")
    ap.add_argument("--xlabel", default="X (Bohr)")
    ap.add_argument("--ylabel", default="Y (Bohr)")
    ap.add_argument("--pdf", action="store_true", help="Also save a PDF copy.")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    data = np.loadtxt(args.csv, delimiter=",")
    if args.density:
        data = data * data

    cmap = args.cmap or ("YlOrRd" if args.density else "viridis")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    fig, ax = plt.subplots(figsize=(6.3, 5.3), constrained_layout=True)
    im = ax.imshow(data.T, origin="lower", extent=args.extent, aspect="equal", cmap=cmap)

    if args.contours > 0 and np.nanmax(data) > np.nanmin(data):
        if args.extent:
            xs = np.linspace(args.extent[0], args.extent[1], data.shape[0])
            ys = np.linspace(args.extent[2], args.extent[3], data.shape[1])
        else:
            xs = np.arange(data.shape[0])
            ys = np.arange(data.shape[1])
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        levels = np.linspace(np.nanmin(data), np.nanmax(data), args.contours + 2)[1:-1]
        ax.contour(X, Y, data, levels=levels, colors="white", linewidths=0.45, alpha=0.72)

    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.set_title(args.title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$|\psi|^2$" if args.density else r"$\psi$")

    out = Path(args.out)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    if args.pdf:
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        print(f"Saved {out.with_suffix('.pdf')}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
