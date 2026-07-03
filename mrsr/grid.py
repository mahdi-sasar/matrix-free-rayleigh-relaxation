"""Uniform Cartesian grids for matrix-free Schrödinger relaxation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf


@dataclass(frozen=True)
class Grid3D:
    """Uniform 3D grid with equal spacing in all directions.

    Parameters
    ----------
    shape:
        Number of grid points `(nx, ny, nz)`, including boundary points.
    bounds:
        Domain bounds `((xmin, xmax), (ymin, ymax), (zmin, zmax))`.

    Notes
    -----
    The finite-difference operator assumes Dirichlet zero boundary conditions.
    """

    shape: Tuple[int, int, int]
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    dtype: tf.dtypes.DType = tf.float64

    def __post_init__(self) -> None:
        nx, ny, nz = self.shape
        if min(self.shape) < 4:
            raise ValueError("Each grid dimension must contain at least four points.")
        hx = (self.bounds[0][1] - self.bounds[0][0]) / (nx - 1)
        hy = (self.bounds[1][1] - self.bounds[1][0]) / (ny - 1)
        hz = (self.bounds[2][1] - self.bounds[2][0]) / (nz - 1)
        if not (abs(hx - hy) <= 1e-12 * max(1.0, abs(hx)) and abs(hx - hz) <= 1e-12 * max(1.0, abs(hx))):
            raise ValueError(
                "This reference implementation assumes equal spacing: hx = hy = hz."
            )

    @property
    def h(self) -> float:
        nx = self.shape[0]
        return (self.bounds[0][1] - self.bounds[0][0]) / (nx - 1)

    @property
    def volume_element(self) -> float:
        return self.h ** 3

    @property
    def dimension(self) -> int:
        return 3

    @property
    def center(self) -> Tuple[float, float, float]:
        return tuple(0.5 * (lo + hi) for lo, hi in self.bounds)  # type: ignore[return-value]

    def coordinates(self):
        """Return full coordinate arrays X, Y, Z as TensorFlow tensors."""
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.bounds
        nx, ny, nz = self.shape
        x = tf.linspace(tf.cast(xmin, self.dtype), tf.cast(xmax, self.dtype), nx)
        y = tf.linspace(tf.cast(ymin, self.dtype), tf.cast(ymax, self.dtype), ny)
        z = tf.linspace(tf.cast(zmin, self.dtype), tf.cast(zmax, self.dtype), nz)
        return tf.meshgrid(x, y, z, indexing="ij")

    def off_grid_center(self, offset_fraction: float = 0.5) -> Tuple[float, float, float]:
        """Return a domain-center position shifted by a fraction of one grid spacing.

        This is useful for Coulomb centers. With an odd number of grid points, the
        exact center would fall on a grid node; shifting by `h/2` avoids sampling the
        singularity while preserving the intended centered geometry up to O(h).
        """
        cx, cy, cz = self.center
        shift = offset_fraction * self.h
        return (cx + shift, cy + shift, cz + shift)
