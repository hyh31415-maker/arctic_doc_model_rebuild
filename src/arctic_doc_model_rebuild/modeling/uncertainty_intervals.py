from __future__ import annotations

import numpy as np
import pandas as pd

from .residual_diagnostics import SEASON_WINDOWS


QUANTILES = {
    "q02_5": 0.025,
    "q05": 0.05,
    "q10": 0.10,
    "q25": 0.25,
    "q50": 0.50,
    "q75": 0.75,
    "q90": 0.90,
    "q95": 0.95,
    "q97_5": 0.975,
}


def _interval_row(scope: str, group_value: str, subset: pd.DataFrame) -> dict[str, object]:
    residual = pd.to_numeric(subset["residual_mgC_L"], errors="coerce").dropna()
    row: dict[str, object] = {
        "scope": scope,
        "group_value": group_value,
        "n": int(len(residual)),
        "interval_source": "leave_one_year_out_cv_residuals",
        "is_production_prediction_interval": False,
    }
    for name, quantile in QUANTILES.items():
        row[name] = float(residual.quantile(quantile)) if not residual.empty else np.nan
    row["empirical_80pct_interval_lower"] = row["q10"]
    row["empirical_80pct_interval_upper"] = row["q90"]
    row["empirical_90pct_interval_lower"] = row["q05"]
    row["empirical_90pct_interval_upper"] = row["q95"]
    row["empirical_95pct_interval_lower"] = row["q02_5"]
    row["empirical_95pct_interval_upper"] = row["q97_5"]
    return row


def empirical_residual_intervals(primary_loyo: pd.DataFrame) -> pd.DataFrame:
    rows = [_interval_row("overall", "overall", primary_loyo)]
    for river, subset in primary_loyo.groupby("river", dropna=False):
        if len(subset) >= 30:
            rows.append(_interval_row("river", str(river), subset))
    for window, months in SEASON_WINDOWS.items():
        subset = primary_loyo[primary_loyo["month"].isin(months)]
        if len(subset) >= 30:
            rows.append(_interval_row("season_window", window, subset))
    return pd.DataFrame(rows)


def empirical_interval_coverage(primary_loyo: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("overall", "overall", primary_loyo),
        *[
            ("river", str(river), subset)
            for river, subset in primary_loyo.groupby("river", dropna=False)
            if len(subset) >= 30
        ],
    ]
    for scope, group_value, subset in specs:
        interval = intervals[intervals["scope"].eq(scope) & intervals["group_value"].astype(str).eq(str(group_value))]
        if interval.empty:
            continue
        record = interval.iloc[0]
        residual = pd.to_numeric(subset["residual_mgC_L"], errors="coerce").dropna()
        row = {"scope": scope, "group_value": group_value, "n": int(len(residual))}
        for level in ["80", "90", "95"]:
            lower = float(record[f"empirical_{level}pct_interval_lower"])
            upper = float(record[f"empirical_{level}pct_interval_upper"])
            row[f"empirical_{level}pct_interval_coverage"] = float(((residual >= lower) & (residual <= upper)).mean()) if len(residual) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
