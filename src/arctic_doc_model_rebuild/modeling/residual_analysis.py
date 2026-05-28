from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import metric_row


def add_season_windows(frame: pd.DataFrame) -> pd.DataFrame:
    windows = {
        "spring_freshet_provisional": [5, 6, 7],
        "early_season": [5, 6],
        "summer": [7, 8],
        "late_season": [9, 10],
    }
    rows = []
    for window, months in windows.items():
        subset = frame[frame["month"].isin(months)].copy()
        subset["season_window"] = window
        rows.append(subset)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def add_quantiles(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for source, destination in [("DOC_observed_mgC_L", "doc_quantile"), ("Q_m3s", "q_quantile")]:
        values = pd.to_numeric(out[source], errors="coerce")
        try:
            out[destination] = pd.qcut(values, q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        except ValueError:
            out[destination] = pd.NA
    return out


def residual_summary(frame: pd.DataFrame, group_columns: list[str], overall_rmse: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    if frame.empty:
        return pd.DataFrame()
    overall_lookup = {}
    if overall_rmse is not None and not overall_rmse.empty:
        overall_lookup = overall_rmse.set_index(["model_id", "feature_set", "target_scale"])["rmse"].to_dict()
    for keys, subset in frame.groupby(["model_id", "feature_set", "target_scale", *group_columns], dropna=False, observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(["model_id", "feature_set", "target_scale", *group_columns], keys))
        metrics = metric_row(subset["DOC_observed_mgC_L"], subset["DOC_cv_predicted_mgC_L"])
        residual = pd.to_numeric(subset["residual_mgC_L"], errors="coerce").dropna()
        row.update(
            {
                "n": len(subset),
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "bias_mean": metrics["bias_mean"],
                "bias_median": metrics["bias_median"],
                "p05_residual": float(residual.quantile(0.05)) if not residual.empty else np.nan,
                "p95_residual": float(residual.quantile(0.95)) if not residual.empty else np.nan,
            }
        )
        base_rmse = overall_lookup.get((row["model_id"], row["feature_set"], row["target_scale"]), np.nan)
        flags = []
        if row["n"] < 10:
            flags.append("small_n")
        if np.isfinite(row["bias_mean"]) and row["bias_mean"] > 0.5:
            flags.append("systematic_positive_bias")
        if np.isfinite(row["bias_mean"]) and row["bias_mean"] < -0.5:
            flags.append("systematic_negative_bias")
        if np.isfinite(base_rmse) and np.isfinite(row["rmse"]) and row["rmse"] > base_rmse * 1.5:
            flags.append("high_rmse_relative_to_overall")
        row["flags"] = ";".join(flags)
        rows.append(row)
    return pd.DataFrame(rows)


def fold_stability(frame: pd.DataFrame, overall: pd.DataFrame) -> pd.DataFrame:
    rows = []
    overall_lookup = overall.set_index(["model_id", "feature_set", "target_scale"])["rmse"].to_dict() if not overall.empty else {}
    loyo = frame[frame["validation_scheme"].eq("leave_one_year_out")].copy()
    for (model_id, feature_set, target_scale, fold_id), subset in loyo.groupby(["model_id", "feature_set", "target_scale", "fold_id"], dropna=False):
        metrics = metric_row(subset["DOC_observed_mgC_L"], subset["DOC_cv_predicted_mgC_L"])
        overall_rmse = overall_lookup.get((model_id, feature_set, target_scale), np.nan)
        fold_year = subset["year"].iloc[0] if "year" in subset.columns and not subset.empty else ""
        flags = []
        if len(subset) < 5:
            flags.append("n_test_lt_5")
        if np.isfinite(overall_rmse) and np.isfinite(metrics["rmse"]) and metrics["rmse"] > overall_rmse * 1.5:
            flags.append("rmse_gt_1_5x_overall")
        rows.append(
            {
                "model_id": model_id,
                "feature_set": feature_set,
                "target_scale": target_scale,
                "fold_id": fold_id,
                "fold_year": int(fold_year) if pd.notna(fold_year) else "",
                "n_test": len(subset),
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "bias": metrics["bias_mean"],
                "r2": metrics["r2"],
                "rivers_in_fold": ";".join(sorted(subset["river"].astype(str).unique())),
                "overall_rmse": overall_rmse,
                "flags": ";".join(flags),
            }
        )
    return pd.DataFrame(rows)
