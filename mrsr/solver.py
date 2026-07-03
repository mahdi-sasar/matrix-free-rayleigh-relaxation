"""Normalized Rayleigh-shifted matrix-free relaxation solver."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional

import numpy as np
import tensorflow as tf

from .operators import (
    apply_hamiltonian,
    l2_norm,
    normalize,
    normalized_residual,
    rayleigh_quotient,
    residual,
    spectral_diameter_bound,
    zero_boundary,
)


@dataclass
class SolverResult:
    psi: tf.Tensor
    energy: float
    iterations: int
    converged: bool
    elapsed_seconds: float
    tau: float
    sigma: float
    energy_history: np.ndarray
    residual_history: np.ndarray
    norm_history: np.ndarray


def solve_ground_state(
    potential: tf.Tensor,
    h: float,
    psi0: Optional[tf.Tensor] = None,
    *,
    sigma: float = 0.9,
    tau: Optional[float] = None,
    tolerance: float = 1e-8,
    max_iterations: int = 50_000,
    check_every: int = 10,
    seed: int = 1234,
    verbose: bool = True,
) -> SolverResult:
    """Solve for the lowest eigenstate by normalized Rayleigh relaxation.

    Parameters
    ----------
    potential:
        Full 3D potential array, including boundary points.
    h:
        Uniform grid spacing.
    psi0:
        Optional initial wavefunction. If omitted, a random normal vector is used.
    sigma:
        Safety factor for the rigorous step bound. Used only when `tau` is None.
    tau:
        Optional explicit time step / relaxation parameter.
    tolerance:
        Stop when the normalized residual is below this value.
    check_every:
        Record diagnostics and test convergence every this many iterations.

    Returns
    -------
    SolverResult
        Final wavefunction and diagnostic histories.
    """
    tf.debugging.assert_all_finite(potential, "Potential contains NaN or Inf. A singularity may be on the grid.")
    dtype = potential.dtype
    if psi0 is None:
        tf.random.set_seed(seed)
        psi = tf.random.normal(tf.shape(potential), dtype=dtype)
    else:
        psi = tf.convert_to_tensor(psi0, dtype=dtype)

    psi = normalize(zero_boundary(psi), h)

    if tau is None:
        dbound = spectral_diameter_bound(potential, h, dimension=3)
        tau_tf = 2.0 * tf.cast(sigma, dtype) / dbound
    else:
        tau_tf = tf.cast(tau, dtype)

    energies: list[float] = []
    residuals: list[float] = []
    norms: list[float] = []
    converged = False
    start = perf_counter()

    for it in range(max_iterations + 1):
        if it % check_every == 0:
            e = rayleigh_quotient(psi, potential, h)
            rn = normalized_residual(psi, potential, h, e)
            nm = l2_norm(psi, h)
            e_float = float(e.numpy())
            rn_float = float(rn.numpy())
            nm_float = float(nm.numpy())
            energies.append(e_float)
            residuals.append(rn_float)
            norms.append(nm_float)
            if verbose:
                print(f"iter={it:7d}  E={e_float:+.12e}  residual={rn_float:.3e}  norm={nm_float:.6f}")
            if not np.isfinite(e_float) or not np.isfinite(rn_float):
                raise FloatingPointError("Energy or residual became non-finite; reduce sigma/tau and check the potential.")
            if rn_float < tolerance:
                converged = True
                break

        e = rayleigh_quotient(psi, potential, h)
        r = residual(psi, potential, h, e)
        q = psi - tau_tf * r
        psi = normalize(q, h)

    elapsed = perf_counter() - start
    final_energy = float(rayleigh_quotient(psi, potential, h).numpy())

    return SolverResult(
        psi=psi,
        energy=final_energy,
        iterations=it,
        converged=converged,
        elapsed_seconds=elapsed,
        tau=float(tau_tf.numpy()),
        sigma=sigma,
        energy_history=np.array(energies, dtype=float),
        residual_history=np.array(residuals, dtype=float),
        norm_history=np.array(norms, dtype=float),
    )


def save_histories(result: SolverResult, path: str) -> None:
    """Save diagnostic histories as CSV."""
    import pandas as pd

    n = len(result.energy_history)
    df = pd.DataFrame(
        {
            "sample": np.arange(n),
            "energy_Ry": result.energy_history,
            "normalized_residual": result.residual_history,
            "norm": result.norm_history,
        }
    )
    df.to_csv(path, index=False)
