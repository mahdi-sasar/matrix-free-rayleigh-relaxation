#!/usr/bin/env python3
"""Create publication-ready hydrogen grid-sweep plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _fit_energy_exponential_logN(df: pd.DataFrame):
    """Fit E(N) = E_inf + A exp[-k log(N)] using total grid points N.

    This is the exponential-in-log(N) form used for the hydrogen convergence
    plot, equivalent to a power-law approach in N.  It is numerically better
    conditioned than fitting exp(-k N) directly because N can be very large.
    """
    work = df.copy()
    if "converged" in work.columns:
        conv = work["converged"].astype(str).str.lower().isin(["true", "1", "yes"])
        if conv.any():
            work = work[conv]
    work = work.dropna(subset=["voxels_total", "energy_Ry"])
    if len(work) < 3:
        return None

    x = np.log(np.asarray(work["voxels_total"], dtype=float))
    y = np.asarray(work["energy_Ry"], dtype=float)

    def model(xv, e_inf, amp, k):
        return e_inf + amp * np.exp(-k * xv)

    # Prefer scipy's nonlinear least squares when available.  Colab normally
    # ships with scipy, but this fallback keeps the script usable elsewhere.
    try:
        from scipy.optimize import curve_fit

        e0 = float(y[-1])
        a0 = float(y[0] - e0) if abs(float(y[0] - e0)) > 1e-12 else 0.1
        p0 = [e0, a0, 1.0]
        popt, _ = curve_fit(
            model,
            x,
            y,
            p0=p0,
            bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf]),
            maxfev=100000,
        )
        e_inf, amp, k = map(float, popt)
        yhat = model(x, e_inf, amp, k)
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        return {
            "e_inf": e_inf,
            "amp": amp,
            "k": k,
            "rmse": rmse,
            "model": model,
            "x_fit_min": float(x.min()),
            "x_fit_max": float(x.max()),
        }
    except Exception:
        # Fallback: fix E_inf slightly beyond the last point and fit log|E-E_inf|.
        e_inf = float(y[-1])
        delta = np.sign(y[0] - e_inf) or 1.0
        e_inf = e_inf - delta * 0.05 * max(1e-12, abs(y[0] - y[-1]))
        z = y - e_inf
        mask = np.isfinite(z) & (np.abs(z) > 1e-14) & (np.sign(z) == np.sign(z[0]))
        if mask.sum() < 2:
            return None
        slope, intercept = np.polyfit(x[mask], np.log(np.abs(z[mask])), deg=1)
        k = float(max(0.0, -slope))
        amp = float(np.sign(z[0]) * np.exp(intercept))
        yhat = model(x, e_inf, amp, k)
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        return {
            "e_inf": e_inf,
            "amp": amp,
            "k": k,
            "rmse": rmse,
            "model": model,
            "x_fit_min": float(x.min()),
            "x_fit_max": float(x.max()),
        }


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

    # Energy vs grid points with exponential-decay fit.
    fit = _fit_energy_exponential_logN(df)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(
        df["voxels_total"],
        df["energy_Ry"],
        marker="o",
        markersize=5,
        linewidth=2.0,
        color="black",
        markerfacecolor="black",
        markeredgecolor="black",
        label="computed energy",
    )

    if fit is not None:
        x_smooth = np.linspace(fit["x_fit_min"], fit["x_fit_max"], 400)
        n_smooth = np.exp(x_smooth)
        e_smooth = fit["model"](x_smooth, fit["e_inf"], fit["amp"], fit["k"])
        ax.plot(n_smooth, e_smooth, linewidth=2.0, color="red", label="exponential fit")
        # Red fit symbols at the measured N values, matching the requested plot style.
        measured_x = np.log(np.asarray(df["voxels_total"], dtype=float))
        measured_fit = fit["model"](measured_x, fit["e_inf"], fit["amp"], fit["k"])
        ax.plot(
            df["voxels_total"],
            measured_fit,
            linestyle="none",
            marker="s",
            markersize=4,
            color="red",
            markerfacecolor="red",
            markeredgecolor="red",
        )
        equation = (
            r"$E(N)=E_{\infty}+A\exp[-k\log N]$" "\n"
            rf"$E_{{\infty}}={fit['e_inf']:.6f}\,\mathrm{{Ry}}$" "\n"
            rf"$A={fit['amp']:.3g},\ k={fit['k']:.3g}$"
        )
        ax.text(
            0.03,
            0.97,
            equation,
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.75", alpha=0.92),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Total grid points $N$")
    ax.set_ylabel("Energy (Ry)")
    ax.set_title("Hydrogen in a box: finite-grid energy convergence")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.legend(frameon=True, loc="best")
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
    ax.plot(
        df["voxels_total"],
        yerr,
        marker="o",
        markersize=5,
        linewidth=2.0,
        color="black",
        markerfacecolor="black",
        markeredgecolor="black",
        label=r"$|E+1|$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total grid points $N$")
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
        ax.plot(
            df["voxels_total"],
            df["elapsed_seconds"],
            marker="o",
            markersize=5,
            linewidth=2.0,
            color="black",
            markerfacecolor="black",
            markeredgecolor="black",
            label="wall time",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Total grid points $N$")
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
