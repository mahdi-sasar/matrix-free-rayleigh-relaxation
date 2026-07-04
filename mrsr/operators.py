"""Matrix-free finite-difference Hamiltonian operations."""

from __future__ import annotations

import tensorflow as tf


Tensor = tf.Tensor


def zero_boundary(p: Tensor) -> Tensor:
    """Set all boundary faces of a 3D tensor to zero."""
    interior = p[1:-1, 1:-1, 1:-1]
    nx = tf.shape(p)[0]
    ny = tf.shape(p)[1]
    nz = tf.shape(p)[2]
    q = tf.pad(interior, paddings=[[1, 1], [1, 1], [1, 1]])
    q.set_shape(p.shape)
    return q


def l2_inner(a: Tensor, b: Tensor, h: float) -> Tensor:
    """Discrete L2 inner product including the volume element."""
    return tf.reduce_sum(a * b) * tf.cast(h ** 3, a.dtype)


def l2_norm(a: Tensor, h: float) -> Tensor:
    """Discrete L2 norm including the volume element."""
    return tf.sqrt(tf.maximum(l2_inner(a, a, h), tf.cast(0.0, a.dtype)))


def normalize(p: Tensor, h: float, eps: float = 1e-300) -> Tensor:
    """Normalize a grid function in the discrete L2 norm."""
    p = zero_boundary(p)
    nrm = l2_norm(p, h)
    return p / tf.maximum(nrm, tf.cast(eps, p.dtype))


def apply_hamiltonian(p: Tensor, v: Tensor, h: float) -> Tensor:
    """Apply H = -Δ_h + V to a 3D wavefunction with Dirichlet boundaries.

    Parameters
    ----------
    p:
        Full 3D wavefunction including boundary points. Boundary values are
        treated as zero Dirichlet data. The function explicitly zeros them so
        dense-matrix tests and solver calls agree even when a caller supplies
        a tensor with nonzero boundary entries.
    v:
        Full 3D potential sampled on the same grid.
    h:
        Uniform grid spacing.
    """
    p = zero_boundary(p)
    c = p[1:-1, 1:-1, 1:-1]
    neigh = (
        p[2:, 1:-1, 1:-1]
        + p[:-2, 1:-1, 1:-1]
        + p[1:-1, 2:, 1:-1]
        + p[1:-1, :-2, 1:-1]
        + p[1:-1, 1:-1, 2:]
        + p[1:-1, 1:-1, :-2]
    )
    hp_int = (6.0 * c - neigh) / tf.cast(h ** 2, p.dtype) + v[1:-1, 1:-1, 1:-1] * c
    hp = tf.pad(hp_int, paddings=[[1, 1], [1, 1], [1, 1]])
    hp.set_shape(p.shape)
    return hp


def rayleigh_quotient(p: Tensor, v: Tensor, h: float) -> Tensor:
    hp = apply_hamiltonian(p, v, h)
    return l2_inner(p, hp, h) / l2_inner(p, p, h)


def residual(p: Tensor, v: Tensor, h: float, energy: Tensor | None = None) -> Tensor:
    if energy is None:
        energy = rayleigh_quotient(p, v, h)
    return apply_hamiltonian(p, v, h) - energy * p


def residual_norm(p: Tensor, v: Tensor, h: float, energy: Tensor | None = None) -> Tensor:
    return l2_norm(residual(p, v, h, energy), h)


def normalized_residual(p: Tensor, v: Tensor, h: float, energy: Tensor | None = None) -> Tensor:
    if energy is None:
        energy = rayleigh_quotient(p, v, h)
    hp = apply_hamiltonian(p, v, h)
    r = hp - energy * p
    denom = l2_norm(hp, h) + tf.abs(energy) * l2_norm(p, h) + tf.cast(1e-300, p.dtype)
    return l2_norm(r, h) / denom


def spectral_diameter_bound(v: Tensor, h: float, dimension: int = 3) -> Tensor:
    """Upper bound on λ_max - λ_min for H = -Δ_h + V."""
    vint = v[1:-1, 1:-1, 1:-1]
    vmax = tf.reduce_max(vint)
    vmin = tf.reduce_min(vint)
    return tf.cast(4.0 * dimension / (h ** 2), v.dtype) + vmax - vmin
