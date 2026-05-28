from __future__ import annotations

import numpy as np
import pandas as pd


SEASON_WINDOW_ORDER = ["early_season", "spring_freshet_provisional", "summer", "late_season", "other"]


def assign_season_window(month: pd.Series) -> pd.Series:
    values = pd.to_numeric(month, errors="coerce")
    conditions = [
        values.isin([5, 6]),
        values.isin([5, 6, 7]),
        values.isin([7, 8]),
        values.isin([9, 10]),
    ]
    out = np.select(conditions, SEASON_WINDOW_ORDER[:-1], default="other")
    return pd.Series(out, index=month.index, dtype="object")


def add_nonlinear_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["log_Q_squared"] = pd.to_numeric(out["log_Q"], errors="coerce") ** 2
    out["season_window"] = assign_season_window(out["month"])

    q = pd.to_numeric(out["Q_m3s"], errors="coerce")
    river_q75 = q.groupby(out["river"]).transform(lambda values: values.quantile(0.75))
    out["high_flow_flag"] = ((q >= river_q75) & q.notna()).astype(float)
    out["log_Q_high_flow"] = pd.to_numeric(out["log_Q"], errors="coerce") * out["high_flow_flag"]
    return out
