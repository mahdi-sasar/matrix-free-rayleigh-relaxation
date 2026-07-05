#!/usr/bin/env python3
"""Plot a CSV slice produced by the examples with paper-friendly styling."""

from __future__ import annotations

import argparse

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
    ap.add_argument("--contours", type=int, default=10, help="Number of contour levels to overlay; 0 disables contours.")
    args = ap.parse_args()

    data = np.loadtxt(args.csv, delimiter=",")
    if args.density:
        data = data * data

    cmap = args.cmap or ("YlOrRd" if args.density else "viridis")
    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    im = ax.imshow(data.T, origin="lower", extent=args.extent, aspect="equal", cmap=cmap)
    if args.contours and np.nanmax(data) > np.nanmin(data):
        levels = np.linspace(np.nanmin(data), np.nanmax(data), args.contours + 2)[1:-1]
        if len(levels):
            xs = np.linspace(args.extent[0], args.extent[1], data.shape[0]) if args.extent else np.arange(data.shape[0])
            ys = np.linspace(args.extent[2], args.extent[3], data.shape[1]) if args.extent else np.arange(data.shape[1])
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            ax.contour(X, Y, data, levels=levels, colors="white", linewidths=0.5, alpha=0.7)
    ax.set_xlabel("X (Bohr)")
    ax.set_ylabel("Y (Bohr)")
    ax.set_title(args.title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("|ψ|²" if args.density else "ψ")
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
