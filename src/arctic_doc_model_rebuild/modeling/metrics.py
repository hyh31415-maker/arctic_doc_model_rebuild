from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, r2_score


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    subset = pd.DataFrame({"x": x, "y": y}).apply(pd.to_numeric, errors="coerce").dropna()
    if len(subset) < 2 or subset["x"].nunique() < 2 or subset["y"].nunique() < 2:
        return float("nan")
    if method == "spearman":
        return float(stats.spearmanr(subset["x"], subset["y"]).statistic)
    return float(stats.pearsonr(subset["x"], subset["y"]).statistic)


def metric_row(y_true, y_pred) -> dict[str, float]:
    y_true = pd.to_numeric(pd.Series(y_true), errors="coerce").reset_index(drop=True)
    y_pred = pd.to_numeric(pd.Series(y_pred), errors="coerce").reset_index(drop=True)
    subset = pd.DataFrame({"observed": y_true, "predicted": y_pred}).dropna()
    if subset.empty:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "median_absolute_error": float("nan"),
            "r2": float("nan"),
            "bias_mean": float("nan"),
            "bias_median": float("nan"),
            "spearman_r": float("nan"),
            "pearson_r": float("nan"),
        }
    observed = subset["observed"]
    predicted = subset["predicted"]
    residual = observed - predicted
    return {
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "median_absolute_error": float(median_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)) if len(subset) > 1 and observed.nunique() > 1 else float("nan"),
        "bias_mean": float(residual.mean()),
        "bias_median": float(residual.median()),
        "spearman_r": _safe_corr(observed, predicted, "spearman"),
        "pearson_r": _safe_corr(observed, predicted, "pearson"),
    }


def grouped_metrics(predictions: pd.DataFrame, fold_summary: pd.DataFrame, group_columns: list[str] | None = None) -> pd.DataFrame:
    group_columns = group_columns or []
    base_columns = ["model_id", "feature_set", "validation_scheme"]
    rows = []
    if predictions.empty:
        return pd.DataFrame()
    for keys, subset in predictions.groupby([*base_columns, *group_columns], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip([*base_columns, *group_columns], keys))
        row["n_test_total"] = len(subset)
        row["n_folds"] = int(subset["fold_id"].nunique()) if "fold_id" in subset.columns else ""
        fold_key = {column: row[column] for column in base_columns}
        folds = fold_summary
        for column, value in fold_key.items():
            folds = folds[folds[column].eq(value)]
        row["n_train_total"] = int(folds["n_train"].sum()) if not folds.empty and not group_columns else ""
        row.update(metric_row(subset["DOC_observed_mgC_L"], subset["DOC_cv_predicted_mgC_L"]))
        rows.append(row)
    return pd.DataFrame(rows)
