"""Input/output helpers for examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_midplane_slice(psi: tf.Tensor, out_csv: str | Path, axis: str = "z") -> None:
    arr = psi.numpy()
    if axis == "x":
        sl = arr[arr.shape[0] // 2, :, :]
    elif axis == "y":
        sl = arr[:, arr.shape[1] // 2, :]
    elif axis == "z":
        sl = arr[:, :, arr.shape[2] // 2]
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'.")
    np.savetxt(out_csv, sl, delimiter=",")


def save_metadata(path: str | Path, **items: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, sort_keys=True)
