from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..gold_contract import require_gold_data_dir, sha256_file, table_path, verification_problem_counts, verify_all_gold_tables
from ..paths import path
from ..schema_checks import issue_count, run_all_schema_checks
from .diagnostics import assert_gold_hash_unchanged, assert_no_forbidden_outputs, make_baseline_figures
from .feature_sets import FEATURE_SETS, REQUIRED_HYDROCORE_COLUMNS, TARGET_COLUMN, FeatureSet, feature_set_registry
from .metrics import grouped_metrics, metric_row
from .reports import BASELINE_REPORT_DIR, BASELINE_TABLE_DIR, write_baseline_report_from_tables
from .validation import validation_scheme_registry, validation_splits


BASELINE_FIGURE_DIR = path("outputs", "figures", "baseline")


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_family: str
    model_class: str
    alpha: float | None = None
    caveat: str = ""


MODEL_SPECS = [
    ModelSpec("mean_baseline", "mean", "MeanBaseline", caveat="Predicts training-fold mean DOC."),
    ModelSpec("linear_regression", "linear", "LinearRegression"),
    ModelSpec("ridge_alpha_0.1", "ridge", "Ridge", alpha=0.1),
    ModelSpec("ridge_alpha_1", "ridge", "Ridge", alpha=1.0),
    ModelSpec("ridge_alpha_10", "ridge", "Ridge", alpha=10.0),
]


class MeanBaseline:
    def fit(self, x: pd.DataFrame, y: pd.Series) -> "MeanBaseline":
        self.mean_ = float(pd.to_numeric(y, errors="coerce").mean())
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.full(len(x), self.mean_, dtype=float)


def model_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_id": spec.model_id,
                "model_family": spec.model_family,
                "model_class": spec.model_class,
                "alpha": spec.alpha if spec.alpha is not None else "",
                "simple_model_only": True,
                "validation_only": True,
                "caveat": spec.caveat,
            }
            for spec in MODEL_SPECS
        ]
    )


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _make_estimator(spec: ModelSpec, feature_set: FeatureSet):
    if spec.model_id == "mean_baseline":
        return MeanBaseline()
    numeric_features = list(feature_set.numeric_features)
    categorical_features = list(feature_set.categorical_features)
    transformers = []
    if numeric_features:
        transformers.append(("numeric", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(("categorical", _one_hot_encoder(), categorical_features))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    if spec.model_family == "linear":
        estimator = LinearRegression()
    elif spec.model_family == "ridge":
        estimator = Ridge(alpha=float(spec.alpha))
    else:
        raise KeyError(f"Unsupported model spec: {spec.model_id}")
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def _read_hydrocore() -> pd.DataFrame:
    gold_dir = require_gold_data_dir()
    destination = table_path("training_matrix_hydrocore.csv", gold_dir=gold_dir)
    frame = pd.read_csv(destination, low_memory=False)
    missing = sorted(set(REQUIRED_HYDROCORE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"training_matrix_hydrocore.csv is missing required columns: {missing}")
    return frame


def _prepare_hydrocore(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["month"] = out["date"].dt.month
    out["is_may_july"] = out["month"].isin([5, 6, 7])
    out["Q_m3s"] = pd.to_numeric(out["Q_m3s"], errors="coerce")
    out["log_Q"] = np.where(out["Q_m3s"] > 0, np.log(out["Q_m3s"]), np.nan)
    out["log_DOC"] = np.where(pd.to_numeric(out[TARGET_COLUMN], errors="coerce") > 0, np.log(pd.to_numeric(out[TARGET_COLUMN], errors="coerce")), np.nan)
    for column in [
        TARGET_COLUMN,
        "sin_doy",
        "cos_doy",
        "temperature_2m_C",
        "positive_degree_day_Cday",
        "snow_cover_fraction",
        "snow_depletion_rate_7d",
        "surface_runoff_m",
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _usable_frame(frame: pd.DataFrame, feature_set: FeatureSet) -> pd.DataFrame:
    required = [TARGET_COLUMN, "label_id", "river", "date", "year", "month", *feature_set.required_features]
    required = list(dict.fromkeys(required))
    usable = frame.dropna(subset=[column for column in required if column in frame.columns]).copy()
    if "log_Q" in feature_set.required_features:
        usable = usable[usable["Q_m3s"] > 0].copy()
    return usable


def _model_specs_for_feature_set(feature_set: FeatureSet) -> list[ModelSpec]:
    if feature_set.feature_set == "F0_intercept_only":
        return [MODEL_SPECS[0]]
    return [spec for spec in MODEL_SPECS if spec.model_id != "mean_baseline"]


def _season_window_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    windows = {
        "spring_freshet_provisional": [5, 6, 7],
        "early_season": [5, 6],
        "summer": [7, 8],
        "late_season": [9, 10],
    }
    rows = []
    for name, months in windows.items():
        subset = predictions[predictions["month"].isin(months)].copy()
        subset["season_window"] = name
        rows.append(subset)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _missingness_used(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(frame)
    for feature_set in FEATURE_SETS:
        usable = _usable_frame(frame, feature_set)
        rows.append(
            {
                "feature_set": feature_set.feature_set,
                "required_columns": ";".join(feature_set.required_features),
                "rows_total": total,
                "rows_available": len(usable),
                "rows_dropped": total - len(usable),
                "drop_rate": (total - len(usable)) / total if total else 0.0,
                "sensitivity_only": feature_set.sensitivity_only,
                "caveat": feature_set.caveat,
            }
        )
    return pd.DataFrame(rows)


def _rank_models(overall: pd.DataFrame, missingness: pd.DataFrame) -> pd.DataFrame:
    loyo = overall[overall["validation_scheme"].eq("leave_one_year_out")].copy()
    if loyo.empty:
        return pd.DataFrame()
    available = missingness.set_index("feature_set")["rows_available"].to_dict()
    sensitivity = missingness.set_index("feature_set")["sensitivity_only"].to_dict()
    max_rows = max(available.values()) if available else 0
    loyo["rows_available_for_feature_set"] = loyo["feature_set"].map(available)
    loyo["sample_penalty"] = np.where(loyo["rows_available_for_feature_set"] < 400, 1.0, 0.0)
    loyo["snow_complete_case_penalty"] = np.where(
        loyo["feature_set"].map(sensitivity).fillna(False) & (loyo["rows_available_for_feature_set"] < 0.8 * max_rows),
        1.0,
        0.0,
    )
    loyo["ranking_score"] = loyo["rmse"] + loyo["sample_penalty"] + loyo["snow_complete_case_penalty"]
    loyo = loyo.sort_values(["ranking_score", "rmse", "mae"], kind="mergesort").reset_index(drop=True)
    loyo["primary_rank"] = np.arange(1, len(loyo) + 1)
    loyo["ranking_basis"] = "leave_one_year_out_rmse_mae_with_sample_penalty"
    loyo["selection_caveat"] = np.where(
        loyo["feature_set"].str.contains("snow"),
        "Snow complete-case sensitivity; sample loss penalized.",
        "Primary LOYO ranking candidate.",
    )
    return loyo


def _write_csv(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8")
    return destination


def run_baseline_models() -> dict[str, Any]:
    BASELINE_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    assert_no_forbidden_outputs()

    verification = verify_all_gold_tables()
    schema = run_all_schema_checks()
    counts = verification_problem_counts(verification)
    if any(counts.values()) or issue_count(schema) > 0:
        raise RuntimeError("Gold data contract or schema checks failed; refusing to train validation-only baselines.")

    gold_dir = require_gold_data_dir()
    hydrocore_path = table_path("training_matrix_hydrocore.csv", gold_dir=gold_dir)
    before_hash = sha256_file(hydrocore_path)
    hydrocore = _prepare_hydrocore(_read_hydrocore())

    cv_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for feature_set in FEATURE_SETS:
        usable = _usable_frame(hydrocore, feature_set).reset_index(drop=True)
        if usable.empty:
            continue
        for spec in _model_specs_for_feature_set(feature_set):
            for scheme, fold_id, train_idx, test_idx, fold_info in validation_splits(usable):
                train = usable.iloc[train_idx].copy()
                test = usable.iloc[test_idx].copy()
                estimator = _make_estimator(spec, feature_set)
                x_train = train[list(feature_set.required_features)]
                y_train = train[TARGET_COLUMN]
                x_test = test[list(feature_set.required_features)]
                estimator.fit(x_train, y_train)
                pred = estimator.predict(x_test)
                fold_metric = metric_row(test[TARGET_COLUMN], pred)
                fold_rows.append(
                    {
                        "model_id": spec.model_id,
                        "feature_set": feature_set.feature_set,
                        "validation_scheme": scheme,
                        "fold_id": fold_id,
                        "fold_group": fold_info.get("fold_group", ""),
                        "fold_year": fold_info.get("fold_year", ""),
                        "fold_river": fold_info.get("fold_river", ""),
                        "n_train": len(train),
                        "n_test": len(test),
                        "rivers_in_test": ";".join(sorted(test["river"].astype(str).unique())),
                        "rmse": fold_metric["rmse"],
                        "mae": fold_metric["mae"],
                        "bias_mean": fold_metric["bias_mean"],
                    }
                )
                for row, predicted in zip(test.to_dict("records"), pred):
                    observed = float(row[TARGET_COLUMN])
                    cv_rows.append(
                        {
                            "label_id": row["label_id"],
                            "river": row["river"],
                            "date": pd.Timestamp(row["date"]).date().isoformat(),
                            "year": int(row["year"]) if pd.notna(row["year"]) else "",
                            "month": int(row["month"]) if pd.notna(row["month"]) else "",
                            "DOC_observed_mgC_L": observed,
                            "DOC_cv_predicted_mgC_L": float(predicted),
                            "residual_mgC_L": observed - float(predicted),
                            "model_id": spec.model_id,
                            "feature_set": feature_set.feature_set,
                            "validation_scheme": scheme,
                            "fold_id": fold_id,
                            "is_cv_prediction": True,
                            "is_production_prediction": False,
                        }
                    )

    predictions = pd.DataFrame(cv_rows)
    fold_summary = pd.DataFrame(fold_rows)
    residuals = predictions.copy()
    residuals["abs_residual_mgC_L"] = residuals["residual_mgC_L"].abs()
    residuals["squared_residual_mgC_L"] = residuals["residual_mgC_L"] ** 2
    missingness = _missingness_used(hydrocore)
    overall = grouped_metrics(predictions, fold_summary)
    by_river = grouped_metrics(predictions, fold_summary, ["river"])
    by_year = grouped_metrics(predictions, fold_summary, ["year"])
    by_month = grouped_metrics(predictions, fold_summary, ["month"])
    by_season = grouped_metrics(_season_window_rows(predictions), fold_summary, ["season_window"])
    ranking = _rank_models(overall, missingness)

    table_paths = [
        _write_csv(model_registry(), BASELINE_TABLE_DIR / "model_registry.csv"),
        _write_csv(feature_set_registry(), BASELINE_TABLE_DIR / "feature_set_registry.csv"),
        _write_csv(validation_scheme_registry(), BASELINE_TABLE_DIR / "validation_scheme_registry.csv"),
        _write_csv(overall, BASELINE_TABLE_DIR / "baseline_metrics_overall.csv"),
        _write_csv(by_river, BASELINE_TABLE_DIR / "baseline_metrics_by_river.csv"),
        _write_csv(by_year, BASELINE_TABLE_DIR / "baseline_metrics_by_year.csv"),
        _write_csv(by_month, BASELINE_TABLE_DIR / "baseline_metrics_by_month.csv"),
        _write_csv(by_season, BASELINE_TABLE_DIR / "baseline_metrics_by_season_window.csv"),
        _write_csv(fold_summary, BASELINE_TABLE_DIR / "baseline_fold_summary.csv"),
        _write_csv(predictions, BASELINE_TABLE_DIR / "baseline_cv_predictions.csv"),
        _write_csv(residuals, BASELINE_TABLE_DIR / "baseline_residuals.csv"),
        _write_csv(missingness, BASELINE_TABLE_DIR / "baseline_missingness_used_by_model.csv"),
        _write_csv(ranking, BASELINE_TABLE_DIR / "baseline_model_ranking.csv"),
    ]
    figure_paths = make_baseline_figures(predictions, overall, by_river, by_month, fold_summary, ranking, BASELINE_FIGURE_DIR)
    report_path = write_baseline_report_from_tables()

    assert_gold_hash_unchanged(hydrocore_path, before_hash)
    assert_no_forbidden_outputs()
    return {"tables": table_paths, "figures": figure_paths, "report": report_path, "ranking": ranking}
