from __future__ import annotations

import numpy as np
import pandas as pd


def target_values(frame: pd.DataFrame, target_scale: str) -> pd.Series:
    values = pd.to_numeric(frame["DOC_mgC_L"], errors="coerce")
    if target_scale == "raw":
        return values
    if target_scale == "log":
        return np.log(values)
    raise ValueError(f"Unknown target scale: {target_scale}")


def inverse_target(values, target_scale: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if target_scale == "raw":
        return values
    if target_scale == "log":
        return np.exp(values)
    raise ValueError(f"Unknown target scale: {target_scale}")
