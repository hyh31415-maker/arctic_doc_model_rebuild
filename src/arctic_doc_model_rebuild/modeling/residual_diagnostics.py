from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import metric_row


SEASON_WINDOWS = {
    "early_season": [5, 6],
    "spring_freshet_provisional": [5, 6, 7],
    "summer": [7, 8],
    "late_season": [9, 10],
}


def add_residual_bins(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for source, destination in [("DOC_observed_mgC_L", "doc_quantile"), ("Q_m3s", "q_quantile")]:
        values = pd.to_numeric(out[source], errors="coerce")
        try:
            out[destination] = pd.qcut(values, q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        except ValueError:
            out[destination] = pd.NA
    return out


def add_season_windows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window, months in SEASON_WINDOWS.items():
        subset = frame[frame["month"].isin(months)].copy()
        subset["season_window"] = window
        rows.append(subset)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def residual_metric_row(subset: pd.DataFrame) -> dict[str, float]:
    metrics = metric_row(subset["DOC_observed_mgC_L"], subset["DOC_cv_predicted_mgC_L"])
    residual = pd.to_numeric(subset["residual_mgC_L"], errors="coerce").dropna()
    row = {
        "n": int(len(subset)),
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "bias_mean": metrics["bias_mean"],
        "bias_median": metrics["bias_median"],
        "residual_std": float(residual.std(ddof=1)) if len(residual) > 1 else np.nan,
        "p05_residual": float(residual.quantile(0.05)) if not residual.empty else np.nan,
        "p25_residual": float(residual.quantile(0.25)) if not residual.empty else np.nan,
        "p50_residual": float(residual.quantile(0.50)) if not residual.empty else np.nan,
        "p75_residual": float(residual.quantile(0.75)) if not residual.empty else np.nan,
        "p95_residual": float(residual.quantile(0.95)) if not residual.empty else np.nan,
    }
    return row


def grouped_residual_summary(frame: pd.DataFrame, *, validation_scheme: str = "leave_one_year_out") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    base = frame[frame["validation_scheme"].eq(validation_scheme)].copy()
    base = add_residual_bins(base)
    season = add_season_windows(base)
    group_specs = [
        ("overall", []),
        ("river", ["river"]),
        ("year", ["year"]),
        ("month", ["month"]),
        ("DOC quantile", ["doc_quantile"]),
        ("Q quantile", ["q_quantile"]),
    ]
    rows: list[dict[str, object]] = []
    base_columns = ["model_role", "feature_set", "model_id", "target_scale", "validation_scheme"]
    for group_type, group_columns in group_specs:
        for keys, subset in base.groupby([*base_columns, *group_columns], dropna=False, observed=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip([*base_columns, *group_columns], keys))
            row["group_type"] = group_type
            row["group_value"] = "overall" if not group_columns else ";".join(str(row[column]) for column in group_columns)
            row.update(residual_metric_row(subset))
            rows.append(row)
    if not season.empty:
        for keys, subset in season.groupby([*base_columns, "season_window"], dropna=False, observed=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip([*base_columns, "season_window"], keys))
            row["group_type"] = "season_window"
            row["group_value"] = row["season_window"]
            row.update(residual_metric_row(subset))
            rows.append(row)
    return pd.DataFrame(rows)


def high_doc_groups(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    values = pd.to_numeric(out["DOC_observed_mgC_L"], errors="coerce")
    q25 = values.quantile(0.25)
    q75 = values.quantile(0.75)
    q90 = values.quantile(0.90)
    conditions = [
        values <= q25,
        (values > q25) & (values < q75),
        (values >= q75) & (values < q90),
        values >= q90,
    ]
    out["doc_behavior_group"] = np.select(conditions, ["low", "mid", "high", "extreme_high"], default="unclassified")
    return out


def high_doc_residual_review(frame: pd.DataFrame, *, validation_scheme: str = "leave_one_year_out") -> pd.DataFrame:
    base = high_doc_groups(frame[frame["validation_scheme"].eq(validation_scheme)].copy())
    rows: list[dict[str, object]] = []
    for keys, subset in base.groupby(["model_role", "feature_set", "model_id", "target_scale", "doc_behavior_group"], dropna=False):
        model_role, feature_set, model_id, target_scale, group = keys
        row = {
            "model_role": model_role,
            "feature_set": feature_set,
            "model_id": model_id,
            "target_scale": target_scale,
            "doc_behavior_group": group,
        }
        row.update(residual_metric_row(subset))
        row["log_target_promotion_status"] = "sensitivity_only" if target_scale == "log" else "not_log_target"
        rows.append(row)
    return pd.DataFrame(rows)
