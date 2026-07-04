"""Potential-energy functions in Rydberg atomic units."""

from __future__ import annotations

from typing import Iterable, Tuple

import tensorflow as tf

from .grid import Grid3D

Tensor = tf.Tensor


def _distance(x: Tensor, y: Tensor, z: Tensor, center: Tuple[float, float, float], dtype) -> Tensor:
    cx, cy, cz = [tf.cast(c, dtype) for c in center]
    return tf.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)


def _assert_off_grid(r: Tensor, name: str, tol: float = 1e-12) -> None:
    """Raise a clear error if a singular center is sampled on the grid.

    This is not a smoothing or pseudopotential.  It is only a guard against the
    accidental parity bug where a Coulomb center is placed exactly on a mesh node.
    """
    rmin = float(tf.reduce_min(r).numpy())
    if not tf.math.reduce_all(tf.math.is_finite(r)).numpy() or rmin < tol:
        raise ValueError(
            f"{name} singularity is on or too close to the grid: min distance = {rmin:.3e}. "
            "Move the center off the mesh, e.g. use grid.off_grid_center()."
        )


def harmonic_oscillator(grid: Grid3D, alpha=(0.25, 0.25, 0.25), center=None) -> Tensor:
    """V = αx (x-cx)^2 + αy (y-cy)^2 + αz (z-cz)^2.

    For the infinite-domain operator -d²/dx² + α x², the 1D ground energy is sqrt(α).
    The 3D infinite-domain ground energy is sqrt(αx)+sqrt(αy)+sqrt(αz).
    """
    if center is None:
        center = grid.center
    x, y, z = grid.coordinates()
    cx, cy, cz = [tf.cast(c, grid.dtype) for c in center]
    ax, ay, az = [tf.cast(a, grid.dtype) for a in alpha]
    return ax * (x - cx) ** 2 + ay * (y - cy) ** 2 + az * (z - cz) ** 2


def hydrogen_coulomb(grid: Grid3D, center=None, charge: float = 1.0) -> Tensor:
    """Hydrogenic Coulomb potential V = -2 Z / r in Rydberg units.

    Use an off-grid center to avoid evaluating the singularity at r=0 without smoothing.
    """
    if center is None:
        center = grid.off_grid_center()
    x, y, z = grid.coordinates()
    r = _distance(x, y, z, center, grid.dtype)
    _assert_off_grid(r, "hydrogen_coulomb")
    return -2.0 * tf.cast(charge, grid.dtype) / r


def hydrogen_in_centered_field(grid: Grid3D, field_strength: float = 0.08, center=None, charge: float = 1.0) -> Tensor:
    """Hydrogen in a box plus a centered uniform electric-field term.

    The field term is -F (x - x0). Centering removes the arbitrary constant energy shift
    that appears when one writes -F x on a domain [0, L].
    """
    if center is None:
        center = grid.off_grid_center()
    x, _, _ = grid.coordinates()
    x0 = tf.cast(grid.center[0], grid.dtype)
    return hydrogen_coulomb(grid, center=center, charge=charge) - tf.cast(field_strength, grid.dtype) * (x - x0)


def h2plus(grid: Grid3D, separation: float, axis: str = "z", center=None, charge: float = 1.0) -> Tensor:
    """One-electron H2+ potential in Rydberg units.

    V(r) = -2/r_A - 2/r_B for two protons separated by `separation` Bohr.
    """
    if center is None:
        center = grid.off_grid_center()
    cx, cy, cz = center
    half = 0.5 * separation
    if axis == "x":
        c1, c2 = (cx - half, cy, cz), (cx + half, cy, cz)
    elif axis == "y":
        c1, c2 = (cx, cy - half, cz), (cx, cy + half, cz)
    elif axis == "z":
        c1, c2 = (cx, cy, cz - half), (cx, cy, cz + half)
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'.")
    x, y, z = grid.coordinates()
    r1 = _distance(x, y, z, c1, grid.dtype)
    r2 = _distance(x, y, z, c2, grid.dtype)
    _assert_off_grid(r1, "h2plus center 1")
    _assert_off_grid(r2, "h2plus center 2")
    zcharge = tf.cast(charge, grid.dtype)
    return -2.0 * zcharge / r1 - 2.0 * zcharge / r2


def double_well(grid: Grid3D, a: float = 0.02, b: float = 2.0, omega_perp: float = 0.3, tilt: float = 0.0, center=None) -> Tensor:
    """A 3D tilted double-well potential.

    V = a[(x-x0)^2-b^2]^2 + 0.5 omega_perp^2 [(y-y0)^2+(z-z0)^2] + tilt (x-x0).
    """
    if center is None:
        center = grid.center
    x, y, z = grid.coordinates()
    cx, cy, cz = [tf.cast(c, grid.dtype) for c in center]
    aa = tf.cast(a, grid.dtype)
    bb = tf.cast(b, grid.dtype)
    op = tf.cast(omega_perp, grid.dtype)
    tt = tf.cast(tilt, grid.dtype)
    xx = x - cx
    yy = y - cy
    zz = z - cz
    return aa * (xx ** 2 - bb ** 2) ** 2 + 0.5 * op ** 2 * (yy ** 2 + zz ** 2) + tt * xx


def yukawa(grid: Grid3D, kappa: float = 0.2, center=None, charge: float = 1.0) -> Tensor:
    """Screened Coulomb potential V = -2 Z exp(-κ r) / r."""
    if center is None:
        center = grid.off_grid_center()
    x, y, z = grid.coordinates()
    r = _distance(x, y, z, center, grid.dtype)
    _assert_off_grid(r, "yukawa")
    return -2.0 * tf.cast(charge, grid.dtype) * tf.exp(-tf.cast(kappa, grid.dtype) * r) / r


def point_dipole_pair(grid: Grid3D, q: float = 1.0, separation: float = 0.8, axis: str = "y", center=None) -> Tensor:
    """Finite point-charge dipole potential for an electron in Rydberg units.

    The positive charge attracts the electron and contributes -2q/r_plus;
    the negative charge repels it and contributes +2q/r_minus.

    The dipole moment magnitude is q * separation in e a0, or
    q * separation * 2.541746 Debye.
    """
    if center is None:
        center = grid.off_grid_center()
    cx, cy, cz = center
    half = 0.5 * separation
    if axis == "x":
        plus, minus = (cx - half, cy, cz), (cx + half, cy, cz)
    elif axis == "y":
        plus, minus = (cx, cy - half, cz), (cx, cy + half, cz)
    elif axis == "z":
        plus, minus = (cx, cy, cz - half), (cx, cy, cz + half)
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'.")
    x, y, z = grid.coordinates()
    rp = _distance(x, y, z, plus, grid.dtype)
    rm = _distance(x, y, z, minus, grid.dtype)
    _assert_off_grid(rp, "dipole positive charge")
    _assert_off_grid(rm, "dipole negative charge")
    qq = tf.cast(q, grid.dtype)
    return -2.0 * qq / rp + 2.0 * qq / rm


def dipole_moment_debye(q: float, separation: float) -> float:
    return q * separation * 2.541746
