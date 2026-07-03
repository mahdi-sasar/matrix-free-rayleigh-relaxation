#!/usr/bin/env python3
"""Plot a CSV midplane slice produced by the examples."""

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
    args = ap.parse_args()

    data = np.loadtxt(args.csv, delimiter=",")
    if args.density:
        data = data * data
    plt.figure(figsize=(6, 5))
    plt.imshow(data.T, origin="lower", extent=args.extent, aspect="equal")
    plt.xlabel("X (Bohr)")
    plt.ylabel("Y (Bohr)")
    plt.title(args.title)
    plt.colorbar(label="|ψ|²" if args.density else "ψ")
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
