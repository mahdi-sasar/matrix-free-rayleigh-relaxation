"""Small-grid dense checks for the matrix-free operator.

These tests are intentionally small. They are meant to verify signs, normalization,
and agreement between the matrix-free stencil and an explicitly assembled matrix.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from mrsr.grid import Grid3D
from mrsr.initializers import random_normal
from mrsr.operators import apply_hamiltonian, rayleigh_quotient
from mrsr.potentials import harmonic_oscillator
from mrsr.solver import solve_ground_state


def assemble_dense_hamiltonian_3d(n: int, h: float, v_full: np.ndarray) -> np.ndarray:
    """Assemble the interior Hamiltonian for a cubic n x n x n grid."""
    ni = n - 2
    m = ni**3
    H = np.zeros((m, m), dtype=float)

    def idx(i, j, k):
        return (i * ni + j) * ni + k

    for i in range(ni):
        for j in range(ni):
            for k in range(ni):
                row = idx(i, j, k)
                H[row, row] = 6.0 / h**2 + v_full[i+1, j+1, k+1]
                for di, dj, dk in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
                    ii, jj, kk = i + di, j + dj, k + dk
                    if 0 <= ii < ni and 0 <= jj < ni and 0 <= kk < ni:
                        H[row, idx(ii, jj, kk)] = -1.0 / h**2
    return H


def test_operator_matches_dense_matrix():
    n = 6
    grid = Grid3D((n, n, n), ((-2, 2), (-2, 2), (-2, 2)), dtype=tf.float64)
    v = harmonic_oscillator(grid, alpha=(0.1, 0.2, 0.3))
    p = random_normal(grid, seed=7)
    hp = apply_hamiltonian(p, v, grid.h).numpy()[1:-1, 1:-1, 1:-1].reshape(-1, order="C")
    H = assemble_dense_hamiltonian_3d(n, grid.h, v.numpy())
    p_int = p.numpy()[1:-1, 1:-1, 1:-1].reshape(-1, order="C")
    dense_hp = H @ p_int
    diff = hp - dense_hp
    max_abs = np.max(np.abs(diff))
    assert np.allclose(hp, dense_hp, rtol=1e-9, atol=1e-9), (
        f"max_abs_diff={max_abs:.3e}; "
        f"argmax={np.argmax(np.abs(diff))}; "
        f"hp={hp[np.argmax(np.abs(diff))]:.16e}; "
        f"dense={dense_hp[np.argmax(np.abs(diff))]:.16e}"
    )


def test_solver_approaches_lowest_dense_eigenvalue():
    n = 8
    grid = Grid3D((n, n, n), ((-4, 4), (-4, 4), (-4, 4)), dtype=tf.float64)
    v = harmonic_oscillator(grid, alpha=(0.25, 0.25, 0.25))
    H = assemble_dense_hamiltonian_3d(n, grid.h, v.numpy())
    lam0 = np.linalg.eigvalsh(H)[0]
    result = solve_ground_state(v, grid.h, random_normal(grid, seed=5), tolerance=1e-7, max_iterations=20_000, check_every=20, verbose=False)
    assert result.converged
    assert abs(result.energy - lam0) < 1e-5
