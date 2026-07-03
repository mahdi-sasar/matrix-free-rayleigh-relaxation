"""Initial wavefunction generators."""

from __future__ import annotations

from typing import Optional, Tuple

import tensorflow as tf

from .grid import Grid3D
from .operators import normalize, zero_boundary


def random_positive(grid: Grid3D, seed: Optional[int] = 1234) -> tf.Tensor:
    if seed is not None:
        tf.random.set_seed(seed)
    p = tf.random.uniform(grid.shape, dtype=grid.dtype)
    return normalize(p, grid.h)


def random_normal(grid: Grid3D, seed: Optional[int] = 1234) -> tf.Tensor:
    if seed is not None:
        tf.random.set_seed(seed)
    p = tf.random.normal(grid.shape, dtype=grid.dtype)
    return normalize(p, grid.h)


def gaussian(grid: Grid3D, center: Tuple[float, float, float] | None = None, width: float = 1.0) -> tf.Tensor:
    if center is None:
        center = grid.center
    x, y, z = grid.coordinates()
    cx, cy, cz = [tf.cast(c, grid.dtype) for c in center]
    w = tf.cast(width, grid.dtype)
    p = tf.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2.0 * w ** 2))
    return normalize(zero_boundary(p), grid.h)
