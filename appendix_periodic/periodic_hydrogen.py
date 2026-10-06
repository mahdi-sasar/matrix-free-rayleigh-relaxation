"""Independent Bloch-periodic appendix calculation for matrix-free Rayleigh relaxation.

This module is intentionally separate from the main mrsr package used by the
paper's Dirichlet-boundary examples. It implements the same normalized Rayleigh
residual step on a periodic N x N x N cell, with Bloch phases applied only to
finite-difference links that cross a cell boundary.

The periodic hydrogenic external potential is the grid-resolved reciprocal-space
representation of the periodic Coulomb Green function with the G=0 component
removed. In Rydberg units for Z=1,

    V_G = -8*pi/(Omega*|G|^2) exp(-i G.R_p),   G != 0,
    V_0 = 0.

The appendix is a single-particle boundary-condition demonstration, not a
many-electron model of metallic hydrogen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

import json
import numpy as np
import pandas as pd
import tensorflow as tf


ArrayLike3 = Sequence[float]


@dataclass
class BlochResult:
    psi: tf.Tensor
    energy: float
    residual: float
    iterations: int
    converged: bool
    elapsed_seconds: float
    tau: float
    sigma: float
    kvec: tuple[float, float, float]
    iteration_history: np.ndarray
    energy_history: np.ndarray
    residual_history: np.ndarray


def periodic_grid_coordinates(n: int, a: float):
    """Return the N unique points of a cubic periodic cell [0,a)^3."""
    if n < 5:
        raise ValueError("n must be at least 5.")
    h = a / n
    x = np.arange(n, dtype=float) * h
    return np.meshgrid(x, x, x, indexing="ij")


def periodic_coulomb_potential(
    n: int,
    a: float,
    *,
    center: ArrayLike3 | None = None,
    require_odd_n: bool = True,
) -> tuple[np.ndarray, dict]:
    """Construct the periodic hydrogenic potential on an N^3 grid.

    V(r) = sum_{G != 0} [-8*pi/(Omega G^2)] exp[i G.(r-R_p)]

    NumPy's ifftn(..., norm="forward") evaluates this Fourier sum directly.
    Odd N is used by default so R_p=(a/2,a/2,a/2) lies between grid nodes and
    the reciprocal grid contains no self-conjugate Nyquist plane.
    """
    if a <= 0.0:
        raise ValueError("a must be positive.")
    if require_odd_n and n % 2 == 0:
        raise ValueError(
            "Use an odd periodic grid size for this appendix calculation. "
            "This keeps the centered proton off all grid nodes."
        )

    h = a / n
    if center is None:
        center_arr = np.array([0.5 * a, 0.5 * a, 0.5 * a], dtype=float)
    else:
        center_arr = np.asarray(center, dtype=float)
        if center_arr.shape != (3,):
            raise ValueError("center must contain exactly three coordinates.")
        center_arr = np.mod(center_arr, a)

    g = 2.0 * np.pi * np.fft.fftfreq(n, d=h)
    gx, gy, gz = np.meshgrid(g, g, g, indexing="ij")
    g2 = gx * gx + gy * gy + gz * gz
    phase = np.exp(-1j * (gx * center_arr[0] + gy * center_arr[1] + gz * center_arr[2]))

    coeff = np.zeros((n, n, n), dtype=np.complex128)
    mask = g2 > 0.0
    coeff[mask] = (-8.0 * np.pi / (a ** 3)) * phase[mask] / g2[mask]

    potential_complex = np.fft.ifftn(coeff, norm="forward")
    imag_max = float(np.max(np.abs(potential_complex.imag)))
    real_scale = max(1.0, float(np.max(np.abs(potential_complex.real))))
    if imag_max > 5e-12 * real_scale:
        raise FloatingPointError(
            f"Periodic Coulomb potential is not real to roundoff: max |Im V|={imag_max:.3e}."
        )
    potential = potential_complex.real.astype(np.float64, copy=False)

    x1 = np.arange(n, dtype=float) * h
    mins = []
    for c in center_arr:
        d = np.abs(x1 - c)
        d = np.minimum(d, a - d)
        mins.append(float(np.min(d)))
    rmin = float(np.sqrt(sum(v * v for v in mins)))

    meta = {
        "n": int(n),
        "a_Bohr": float(a),
        "h_Bohr": float(h),
        "center_Bohr": [float(v) for v in center_arr],
        "cell_volume_Bohr3": float(a ** 3),
        "G0_component": 0.0,
        "mean_potential_Ry": float(np.mean(potential)),
        "potential_min_Ry": float(np.min(potential)),
        "potential_max_Ry": float(np.max(potential)),
        "max_imaginary_roundoff_Ry": imag_max,
        "nearest_grid_distance_to_proton_Bohr": rmin,
    }
    return potential, meta


def _complex_dtype_for(real_dtype: tf.dtypes.DType) -> tf.dtypes.DType:
    if real_dtype == tf.float32:
        return tf.complex64
    if real_dtype == tf.float64:
        return tf.complex128
    raise TypeError("Supported real dtypes are float32 and float64.")


def l2_inner(a: tf.Tensor, b: tf.Tensor, h: float) -> tf.Tensor:
    """Complex discrete L2 inner product <a|b> including h^3."""
    return tf.reduce_sum(tf.math.conj(a) * b) * tf.cast(h ** 3, a.dtype)


def l2_norm(a: tf.Tensor, h: float) -> tf.Tensor:
    value = tf.math.real(l2_inner(a, a, h))
    return tf.sqrt(tf.maximum(value, tf.cast(0.0, value.dtype)))


def normalize(a: tf.Tensor, h: float, eps: float = 1e-300) -> tf.Tensor:
    nrm = l2_norm(a, h)
    safe = tf.maximum(nrm, tf.cast(eps, a.dtype.real_dtype))
    return a / tf.cast(safe, a.dtype)


def _phase(theta: float | tf.Tensor, complex_dtype: tf.dtypes.DType) -> tf.Tensor:
    real_dtype = complex_dtype.real_dtype
    return tf.exp(tf.complex(tf.cast(0.0, real_dtype), tf.cast(theta, real_dtype)))


def bloch_neighbor_sum(psi: tf.Tensor, kvec: ArrayLike3, a: float) -> tf.Tensor:
    """Sum the six nearest neighbors with Bloch phases on wrapped links."""
    if not psi.dtype.is_complex:
        raise TypeError("psi must be complex for the Bloch-periodic operator.")

    kx, ky, kz = [float(v) for v in kvec]
    px = _phase(kx * a, psi.dtype)
    py = _phase(ky * a, psi.dtype)
    pz = _phase(kz * a, psi.dtype)

    xp = tf.concat([psi[1:, :, :], px * psi[:1, :, :]], axis=0)
    xm = tf.concat([tf.math.conj(px) * psi[-1:, :, :], psi[:-1, :, :]], axis=0)
    yp = tf.concat([psi[:, 1:, :], py * psi[:, :1, :]], axis=1)
    ym = tf.concat([tf.math.conj(py) * psi[:, -1:, :], psi[:, :-1, :]], axis=1)
    zp = tf.concat([psi[:, :, 1:], pz * psi[:, :, :1]], axis=2)
    zm = tf.concat([tf.math.conj(pz) * psi[:, :, -1:], psi[:, :, :-1]], axis=2)
    return xp + xm + yp + ym + zp + zm


def apply_hamiltonian(
    psi: tf.Tensor,
    potential: tf.Tensor,
    h: float,
    a: float,
    kvec: ArrayLike3,
) -> tf.Tensor:
    """Apply H_k=-Delta_h+V with Bloch-periodic boundary links."""
    neigh = bloch_neighbor_sum(psi, kvec, a)
    kinetic = (tf.cast(6.0, psi.dtype) * psi - neigh) / tf.cast(h ** 2, psi.dtype)
    return kinetic + tf.cast(potential, psi.dtype) * psi


def rayleigh_quotient(
    psi: tf.Tensor,
    potential: tf.Tensor,
    h: float,
    a: float,
    kvec: ArrayLike3,
) -> tf.Tensor:
    hpsi = apply_hamiltonian(psi, potential, h, a, kvec)
    rq = l2_inner(psi, hpsi, h) / l2_inner(psi, psi, h)
    return tf.math.real(rq)


def normalized_residual(
    psi: tf.Tensor,
    potential: tf.Tensor,
    h: float,
    a: float,
    kvec: ArrayLike3,
    energy: tf.Tensor | None = None,
) -> tf.Tensor:
    hpsi = apply_hamiltonian(psi, potential, h, a, kvec)
    if energy is None:
        energy = tf.math.real(l2_inner(psi, hpsi, h) / l2_inner(psi, psi, h))
    r = hpsi - tf.cast(energy, psi.dtype) * psi
    denom = (
        l2_norm(hpsi, h)
        + tf.abs(energy) * l2_norm(psi, h)
        + tf.cast(1e-300, energy.dtype)
    )
    return l2_norm(r, h) / denom


def spectral_diameter_bound(potential: tf.Tensor, h: float) -> tf.Tensor:
    """Conservative 3D second-order periodic-Hamiltonian spectral bound."""
    vmax = tf.reduce_max(potential)
    vmin = tf.reduce_min(potential)
    return tf.cast(12.0 / (h ** 2), potential.dtype) + vmax - vmin


def _random_complex(shape, real_dtype: tf.dtypes.DType, seed: int) -> tf.Tensor:
    r = tf.random.stateless_normal(shape, seed=[seed, 0], dtype=real_dtype)
    im = tf.random.stateless_normal(shape, seed=[seed, 1], dtype=real_dtype)
    return tf.complex(r, im)


def solve_ground_state(
    potential: np.ndarray | tf.Tensor,
    a: float,
    kvec: ArrayLike3,
    psi0: np.ndarray | tf.Tensor | None = None,
    *,
    sigma: float = 0.8,
    tau: float | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 100_000,
    check_every: int = 20,
    seed: int = 1234,
    verbose: bool = True,
    real_dtype: tf.dtypes.DType = tf.float64,
) -> BlochResult:
    """Solve the lowest state at one Bloch vector by normalized Rayleigh relaxation."""
    if not (0.0 < sigma < 1.0) and tau is None:
        raise ValueError("For the theorem-backed bound, sigma must satisfy 0 < sigma < 1.")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    if check_every <= 0:
        raise ValueError("check_every must be positive.")

    v = tf.convert_to_tensor(potential, dtype=real_dtype)
    if v.shape.rank != 3 or len(set(v.shape.as_list())) != 1:
        raise ValueError("potential must be a cubic N x N x N array.")
    n = int(v.shape[0])
    h = a / n
    complex_dtype = _complex_dtype_for(real_dtype)

    if psi0 is None:
        psi = _random_complex(v.shape, real_dtype, seed)
    else:
        p0 = tf.convert_to_tensor(psi0)
        if p0.dtype.is_complex:
            psi = tf.cast(p0, complex_dtype)
        else:
            p0r = tf.cast(p0, real_dtype)
            psi = tf.complex(p0r, tf.zeros_like(p0r))
    psi = normalize(psi, h)

    if tau is None:
        dbound = spectral_diameter_bound(v, h)
        tau_tf = tf.cast(2.0 * sigma, real_dtype) / dbound
    else:
        if tau <= 0.0:
            raise ValueError("tau must be positive.")
        tau_tf = tf.cast(tau, real_dtype)

    iterations = []
    energies = []
    residuals = []
    converged = False
    start = perf_counter()

    for it in range(max_iterations + 1):
        if it % check_every == 0 or it == max_iterations:
            e = rayleigh_quotient(psi, v, h, a, kvec)
            rr = normalized_residual(psi, v, h, a, kvec, e)
            e_float = float(e.numpy())
            rr_float = float(rr.numpy())
            if not np.isfinite(e_float) or not np.isfinite(rr_float):
                raise FloatingPointError("Non-finite Bloch energy or residual encountered.")
            iterations.append(it)
            energies.append(e_float)
            residuals.append(rr_float)
            if verbose:
                print(
                    f"iter={it:7d}  E={e_float:+.12e} Ry  "
                    f"residual={rr_float:.3e}  k={tuple(float(x) for x in kvec)}"
                )
            if rr_float < tolerance:
                converged = True
                break

        if it == max_iterations:
            break

        e = rayleigh_quotient(psi, v, h, a, kvec)
        hpsi = apply_hamiltonian(psi, v, h, a, kvec)
        r = hpsi - tf.cast(e, complex_dtype) * psi
        psi = normalize(psi - tf.cast(tau_tf, complex_dtype) * r, h)

    elapsed = perf_counter() - start
    final_energy = float(rayleigh_quotient(psi, v, h, a, kvec).numpy())
    final_residual = float(normalized_residual(psi, v, h, a, kvec).numpy())

    return BlochResult(
        psi=psi,
        energy=final_energy,
        residual=final_residual,
        iterations=it,
        converged=converged,
        elapsed_seconds=elapsed,
        tau=float(tau_tf.numpy()),
        sigma=float(sigma),
        kvec=tuple(float(x) for x in kvec),
        iteration_history=np.asarray(iterations, dtype=int),
        energy_history=np.asarray(energies, dtype=float),
        residual_history=np.asarray(residuals, dtype=float),
    )


def free_electron_fd_energy(kvec: ArrayLike3, h: float) -> float:
    """Exact lowest-plane-wave energy of the second-order stencil at k."""
    return float((4.0 / h ** 2) * sum(np.sin(0.5 * float(k) * h) ** 2 for k in kvec))


def plane_wave(n: int, a: float, kvec: ArrayLike3) -> np.ndarray:
    x, y, z = periodic_grid_coordinates(n, a)
    kx, ky, kz = [float(v) for v in kvec]
    return np.exp(1j * (kx * x + ky * y + kz * z))


def validate_free_electron(
    n: int,
    a: float,
    kvecs: Iterable[ArrayLike3],
    *,
    real_dtype: tf.dtypes.DType = tf.float64,
) -> pd.DataFrame:
    """Validate Bloch wrapping against the exact discrete free-electron dispersion."""
    h = a / n
    v = tf.zeros((n, n, n), dtype=real_dtype)
    complex_dtype = _complex_dtype_for(real_dtype)
    rows = []
    for kvec in kvecs:
        psi_np = plane_wave(n, a, kvec)
        psi = normalize(tf.cast(tf.convert_to_tensor(psi_np), complex_dtype), h)
        e = float(rayleigh_quotient(psi, v, h, a, kvec).numpy())
        exact = free_electron_fd_energy(kvec, h)
        rr = float(normalized_residual(psi, v, h, a, kvec).numpy())
        rows.append({
            "kx": float(kvec[0]),
            "ky": float(kvec[1]),
            "kz": float(kvec[2]),
            "computed_Ry": e,
            "exact_discrete_Ry": exact,
            "abs_error_Ry": abs(e - exact),
            "normalized_residual": rr,
        })
    return pd.DataFrame(rows)


def hermiticity_error(
    potential: np.ndarray | tf.Tensor,
    a: float,
    kvec: ArrayLike3,
    *,
    seed: int = 991,
    real_dtype: tf.dtypes.DType = tf.float64,
) -> float:
    """Relative sesquilinear-form Hermiticity error."""
    v = tf.convert_to_tensor(potential, dtype=real_dtype)
    n = int(v.shape[0])
    h = a / n
    aa = normalize(_random_complex(v.shape, real_dtype, seed), h)
    bb = normalize(_random_complex(v.shape, real_dtype, seed + 1), h)
    hab = apply_hamiltonian(bb, v, h, a, kvec)
    haa = apply_hamiltonian(aa, v, h, a, kvec)
    lhs = l2_inner(aa, hab, h)
    rhs = l2_inner(haa, bb, h)
    denom = max(1.0, float(tf.abs(lhs).numpy()), float(tf.abs(rhs).numpy()))
    return float(tf.abs(lhs - rhs).numpy()) / denom


def high_symmetry_path(a: float, points_per_segment: int = 16) -> dict:
    """Simple-cubic Gamma-X-M-R-Gamma path."""
    if points_per_segment < 2:
        raise ValueError("points_per_segment must be at least 2.")
    vertices = [
        ("Γ", np.array([0.0, 0.0, 0.0])),
        ("X", np.array([0.5, 0.0, 0.0])),
        ("M", np.array([0.5, 0.5, 0.0])),
        ("R", np.array([0.5, 0.5, 0.5])),
        ("Γ", np.array([0.0, 0.0, 0.0])),
    ]
    scale = 2.0 * np.pi / a
    kvecs = []
    point_vertex_label = []
    for seg in range(len(vertices) - 1):
        label0, q0 = vertices[seg]
        label1, q1 = vertices[seg + 1]
        ts = np.linspace(0.0, 1.0, points_per_segment, endpoint=True)
        if seg > 0:
            ts = ts[1:]
        for j, t in enumerate(ts):
            q = (1.0 - t) * q0 + t * q1
            kvecs.append(scale * q)
            if seg == 0 and j == 0:
                point_vertex_label.append(label0)
            elif np.isclose(t, 1.0):
                point_vertex_label.append(label1)
            else:
                point_vertex_label.append(None)

    karr = np.asarray(kvecs, dtype=float)
    distance = np.zeros(len(karr), dtype=float)
    if len(karr) > 1:
        distance[1:] = np.cumsum(np.linalg.norm(np.diff(karr, axis=0), axis=1))

    ticks = []
    ticklabels = []
    for i, label in enumerate(point_vertex_label):
        if label is not None:
            ticks.append(float(distance[i]))
            ticklabels.append(label)

    return {
        "kvecs": karr,
        "distance": distance,
        "ticks": np.asarray(ticks, dtype=float),
        "ticklabels": ticklabels,
        "vertex_labels": point_vertex_label,
    }


def run_band_path(
    potential: np.ndarray,
    a: float,
    path: dict,
    *,
    psi0: np.ndarray | tf.Tensor | None = None,
    sigma: float = 0.8,
    tolerance: float = 1e-8,
    max_iterations: int = 100_000,
    check_every: int = 20,
    seed: int = 1234,
    verbose: bool = False,
    real_dtype: tf.dtypes.DType = tf.float64,
):
    """Solve the lowest Bloch state along a path, warm-starting each k."""
    kvecs = np.asarray(path["kvecs"], dtype=float)
    distance = np.asarray(path["distance"], dtype=float)
    labels = path["vertex_labels"]
    rows = []
    results = []
    current = psi0

    for idx, kvec in enumerate(kvecs):
        if verbose:
            print(f"\n=== k-point {idx + 1}/{len(kvecs)}: {kvec} ===")
        result = solve_ground_state(
            potential, a, kvec, current,
            sigma=sigma,
            tolerance=tolerance,
            max_iterations=max_iterations,
            check_every=check_every,
            seed=seed + idx,
            verbose=verbose,
            real_dtype=real_dtype,
        )
        current = result.psi
        results.append(result)
        rows.append({
            "index": idx,
            "path_distance": float(distance[idx]),
            "label": labels[idx] or "",
            "kx": float(kvec[0]),
            "ky": float(kvec[1]),
            "kz": float(kvec[2]),
            "energy_Ry": result.energy,
            "normalized_residual": result.residual,
            "iterations": result.iterations,
            "converged": result.converged,
            "elapsed_seconds": result.elapsed_seconds,
            "tau": result.tau,
        })

    return pd.DataFrame(rows), results


def _set_publication_rcparams() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def plot_band_structure(df: pd.DataFrame, path: dict, out: str | Path, *, show: bool = False) -> None:
    """Paper-style lowest-band plot matching the existing black/red theme."""
    import matplotlib.pyplot as plt
    _set_publication_rcparams()
    out = Path(out)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(
        df["path_distance"], df["energy_Ry"],
        marker="o", markersize=4, linewidth=2.0,
        color="black", markerfacecolor="black", markeredgecolor="black",
        label="Lowest Bloch state",
    )
    for xpos in path["ticks"]:
        ax.axvline(float(xpos), linewidth=0.8, color="0.80", zorder=0)
    for xpos in path["ticks"]:
        rows = df[np.isclose(df["path_distance"], float(xpos))]
        if len(rows):
            ax.scatter(
                [float(xpos)], [float(rows.iloc[0]["energy_Ry"])],
                marker="s", s=34, color="red", edgecolors="red", zorder=5,
            )

    ax.set_xticks(path["ticks"])
    ax.set_xticklabels(path["ticklabels"])
    ax.set_xlabel("Bloch wave-vector path")
    ax.set_ylabel("Energy (Ry)")
    ax.set_title("Periodic hydrogenic lattice: lowest Bloch band")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=True, loc="best")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_tiled_density(
    psi: np.ndarray | tf.Tensor,
    a: float,
    out: str | Path,
    *,
    tiles: int = 3,
    plane_axis: str = "z",
    contours: int = 12,
    show: bool = False,
) -> None:
    """Plot a tiled |psi|^2 mid-plane using the paper's YlOrRd density theme."""
    import matplotlib.pyplot as plt
    _set_publication_rcparams()

    if tiles < 1 or tiles % 2 == 0:
        raise ValueError("tiles must be a positive odd integer.")

    arr = psi.numpy() if tf.is_tensor(psi) else np.asarray(psi)
    density = np.abs(arr) ** 2
    n = density.shape[0]

    if plane_axis == "z":
        sl = density[:, :, n // 2]
        labels = ("X", "Y")
    elif plane_axis == "y":
        sl = density[:, n // 2, :]
        labels = ("X", "Z")
    elif plane_axis == "x":
        sl = density[n // 2, :, :]
        labels = ("Y", "Z")
    else:
        raise ValueError("plane_axis must be x, y, or z.")

    tiled = np.tile(sl, (tiles, tiles))
    half = tiles // 2
    extent = [-half * a, (half + 1) * a, -half * a, (half + 1) * a]

    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    im = ax.imshow(tiled.T, origin="lower", extent=extent, aspect="equal", cmap="YlOrRd")

    if contours > 0 and np.nanmax(tiled) > np.nanmin(tiled):
        x = np.linspace(extent[0], extent[1], tiled.shape[0], endpoint=False)
        y = np.linspace(extent[2], extent[3], tiled.shape[1], endpoint=False)
        X, Y = np.meshgrid(x, y, indexing="ij")
        levels = np.linspace(np.nanmin(tiled), np.nanmax(tiled), contours + 2)[1:-1]
        ax.contour(X, Y, tiled, levels=levels, colors="white", linewidths=0.45, alpha=0.75)

    centers = []
    for ix in range(-half, half + 1):
        for iy in range(-half, half + 1):
            centers.append((ix * a + 0.5 * a, iy * a + 0.5 * a))
    cx, cy = zip(*centers)
    ax.scatter(cx, cy, marker="o", s=55, facecolor="none", edgecolor="black", linewidth=1.2, label="proton sites")
    ax.scatter(cx, cy, marker="+", s=70, color="black", linewidth=1.0)

    ax.set_xlabel(f"{labels[0]} (Bohr)")
    ax.set_ylabel(f"{labels[1]} (Bohr)")
    ax.set_title(r"Periodic hydrogenic lattice: tiled $|\psi_{\Gamma}|^2$")
    ax.legend(loc="upper right", frameon=True)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$|\psi|^2$")

    out = Path(out)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_periodic_potential_cut(
    potential: np.ndarray,
    a: float,
    out: str | Path,
    *,
    show: bool = False,
) -> None:
    """Diagnostic central-line cut through the periodic potential."""
    import matplotlib.pyplot as plt
    _set_publication_rcparams()

    n = potential.shape[0]
    h = a / n
    x = np.arange(n, dtype=float) * h
    j = int(np.argmin(np.abs(x - 0.5 * a)))
    cut = potential[:, j, j]

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(
        x, cut, marker="o", markersize=3, linewidth=1.8,
        color="black", label="Periodic Coulomb potential",
    )
    ax.axvline(
        0.5 * a, linestyle="--", linewidth=1.0,
        color="red", alpha=0.6, label="Proton position",
    )
    ax.set_xlabel("X (Bohr)")
    ax.set_ylabel("Potential (Ry)")
    ax.set_title("Periodic hydrogenic lattice: potential cut through one cell")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.legend(frameon=True)

    out = Path(out)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def save_metadata(path: str | Path, **items) -> None:
    with open(Path(path), "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, sort_keys=True)
