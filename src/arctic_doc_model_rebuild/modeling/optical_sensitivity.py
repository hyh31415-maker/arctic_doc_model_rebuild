from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..gold_contract import load_contract, require_gold_data_dir, sha256_file, table_path
from ..paths import CONFIG_DIR, REPORT_DIR, TABLE_DIR, path
from .diagnostics import assert_no_forbidden_outputs
from .feature_sets import TARGET_COLUMN
from .metrics import metric_row
from .optical_features import (
    BASELINE_COMPARATOR,
    BASELINE_MODEL_ID,
    OPTICAL_DATASETS,
    OPTICAL_FEATURE_SETS,
    OPTICAL_MODEL_SPECS,
    OpticalDataset,
    OpticalFeatureSet,
    OpticalModelSpec,
    optical_dataset_registry,
    optical_feature_set_registry,
    optical_feature_sets_for_dataset,
    optical_model_registry,
    validate_optical_feature_sets,
)
from .optical_reports import write_optical_sensitivity_report
from .validation import validation_scheme_registry, validation_splits


OPTICAL_TABLE_DIR = TABLE_DIR / "optical_sensitivity"
OPTICAL_REPORT_DIR = REPORT_DIR / "optical_sensitivity"
OPTICAL_FIGURE_DIR = path("outputs", "figures", "optical_sensitivity")
OPTICAL_REPORT_PATH = OPTICAL_REPORT_DIR / "optical_sensitivity_report.md"
BASELINE_DECISION_PATH = TABLE_DIR / "baseline_final" / "baseline_model_decision.csv"
PRIMARY_BASELINE_SPEC_PATH = CONFIG_DIR / "model_specs" / "primary_baseline_f3_q_season_river_fixed_ridge_alpha_1.yaml"

ALLOWED_OPTICAL_TABLES = {dataset.table_name for dataset in OPTICAL_DATASETS}
BASE_REQUIRED_COLUMNS = [
    "label_id",
    "river",
    "date",
    TARGET_COLUMN,
    "Q_m3s",
    "sin_doy",
    "cos_doy",
    "sensor",
    "days_offset",
    "pct_valid_water_pixels",
]
OPTICAL_NUMERIC_COLUMNS = [
    TARGET_COLUMN,
    "Q_m3s",
    "sin_doy",
    "cos_doy",
    "days_offset",
    "abs_days_offset",
    "pct_valid_water_pixels",
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
    "ndwi",
    "mndwi",
    "red_green_ratio",
    "green_blue_ratio",
]
VALIDATION_MIN_TEST_ROWS = 2
VALIDATION_MIN_TRAIN_ROWS = 5


def _ensure_dirs() -> None:
    for directory in [OPTICAL_TABLE_DIR, OPTICAL_REPORT_DIR, OPTICAL_FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def _write_csv(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8")
    return destination


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _make_estimator(model_spec: OpticalModelSpec, feature_set: OpticalFeatureSet):
    numeric_features = list(feature_set.numeric_features)
    categorical_features = list(feature_set.categorical_features)
    transformers = []
    if numeric_features:
        transformers.append(("numeric", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(("categorical", _one_hot_encoder(), categorical_features))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    if model_spec.model_family == "linear":
        estimator = LinearRegression()
    elif model_spec.model_family == "ridge":
        estimator = Ridge(alpha=float(model_spec.alpha))
    else:
        raise KeyError(f"Unsupported optical model: {model_spec.model_id}")
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def _baseline_model_spec() -> OpticalModelSpec:
    for model in OPTICAL_MODEL_SPECS:
        if model.model_id == BASELINE_MODEL_ID:
            return model
    raise KeyError(f"Missing baseline optical model spec: {BASELINE_MODEL_ID}")


def _load_baseline_metadata() -> tuple[pd.DataFrame, dict[str, Any]]:
    if not BASELINE_DECISION_PATH.exists():
        raise FileNotFoundError("Baseline final decision table is missing. Run finalize-baseline before optical sensitivity.")
    if not PRIMARY_BASELINE_SPEC_PATH.exists():
        raise FileNotFoundError("Primary baseline model spec is missing. Run finalize-baseline before optical sensitivity.")
    decision = pd.read_csv(BASELINE_DECISION_PATH)
    primary = decision[decision["decision_type"].eq("primary_baseline")]
    if primary.empty or primary.iloc[0]["feature_set"] != "F3_q_season_river_fixed" or primary.iloc[0]["model_id"] != "ridge_alpha_1":
        raise RuntimeError("Baseline comparator is not the finalized F3 + ridge_alpha_1 decision.")
    spec = yaml.safe_load(PRIMARY_BASELINE_SPEC_PATH.read_text(encoding="utf-8"))
    if spec.get("production_prediction_allowed") is not False or spec.get("flux_allowed") is not False:
        raise RuntimeError("Primary baseline spec must disallow production prediction and flux in this phase.")
    return decision, spec


def _verify_contract_snapshot() -> pd.DataFrame:
    verification_path = TABLE_DIR / "gold_table_verification.csv"
    if not verification_path.exists():
        raise FileNotFoundError("Run `python -m arctic_doc_model_rebuild.cli verify-gold-data` before optical sensitivity.")
    verification = pd.read_csv(verification_path)
    contract = load_contract()
    expected = set(contract.get("expected_tables", {}))
    if not expected.issubset(set(verification["table_name"].astype(str))):
        raise RuntimeError("Gold table verification snapshot is incomplete.")
    rows = verification[verification["table_name"].isin(expected)]
    if len(rows) != len(expected) or not rows["status"].eq("ok").all():
        raise RuntimeError("Gold data contract snapshot is not fully passing.")
    return rows


def _verify_allowed_optical_hashes() -> dict[str, str]:
    contract = load_contract()
    gold_dir = require_gold_data_dir()
    hashes: dict[str, str] = {}
    for table_name in sorted(ALLOWED_OPTICAL_TABLES):
        if table_name not in contract.get("expected_tables", {}):
            raise KeyError(f"Optical sensitivity table is not in the gold contract: {table_name}")
        destination = table_path(table_name, gold_dir=gold_dir)
        actual = sha256_file(destination)
        expected = str(contract["expected_tables"][table_name]["sha256"]).lower()
        if actual != expected:
            raise RuntimeError(f"Gold optical table hash mismatch: {table_name}")
        hashes[table_name] = actual
    return hashes


def _read_optical_table(dataset: OpticalDataset) -> pd.DataFrame:
    if dataset.table_name not in ALLOWED_OPTICAL_TABLES:
        raise RuntimeError(f"Refusing to read non-optical sensitivity table: {dataset.table_name}")
    gold_dir = require_gold_data_dir()
    destination = table_path(dataset.table_name, gold_dir=gold_dir)
    frame = pd.read_csv(destination, low_memory=False)
    missing = sorted(set(BASE_REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{dataset.table_name} is missing required optical sensitivity columns: {missing}")
    return frame


def _prepare_optical_frame(frame: pd.DataFrame, dataset: OpticalDataset) -> pd.DataFrame:
    out = frame.copy()
    out["dataset_id"] = dataset.dataset_id
    out["window"] = dataset.window
    out["sensor_scope"] = dataset.sensor_scope
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["Q_m3s"] = pd.to_numeric(out["Q_m3s"], errors="coerce")
    out["log_Q"] = np.where(out["Q_m3s"] > 0, np.log(out["Q_m3s"]), np.nan)
    out["days_offset"] = pd.to_numeric(out["days_offset"], errors="coerce")
    out["abs_days_offset"] = out["days_offset"].abs()
    out["sensor"] = out["sensor"].astype(str)
    for column in OPTICAL_NUMERIC_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
        out[column] = out[column].replace([np.inf, -np.inf], np.nan)
    return out


def _usable_same_subset(frame: pd.DataFrame, candidate: OpticalFeatureSet) -> pd.DataFrame:
    required = [
        TARGET_COLUMN,
        "label_id",
        "river",
        "date",
        "year",
        "month",
        "sensor",
        "Q_m3s",
        *BASELINE_COMPARATOR.required_features,
        *candidate.required_features,
    ]
    required = list(dict.fromkeys(required))
    usable = frame.dropna(subset=[column for column in required if column in frame.columns]).copy()
    usable = usable[usable["Q_m3s"] > 0].copy()
    usable = usable.reset_index(drop=True)
    return usable


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


def _run_cv(
    *,
    frame: pd.DataFrame,
    dataset: OpticalDataset,
    feature_set: OpticalFeatureSet,
    model_spec: OpticalModelSpec,
    comparison_feature_set: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cv_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    comparison_id = f"{dataset.dataset_id}__{comparison_feature_set}"
    for scheme, fold_id, train_idx, test_idx, fold_info in validation_splits(frame):
        train = frame.iloc[train_idx].copy()
        test = frame.iloc[test_idx].copy()
        skip_reason = ""
        if len(test) < VALIDATION_MIN_TEST_ROWS:
            skip_reason = f"n_test_lt_{VALIDATION_MIN_TEST_ROWS}"
        elif len(train) < VALIDATION_MIN_TRAIN_ROWS:
            skip_reason = f"n_train_lt_{VALIDATION_MIN_TRAIN_ROWS}"

        fold_base = {
            "dataset_id": dataset.dataset_id,
            "window": dataset.window,
            "sensor_scope": dataset.sensor_scope,
            "comparison_id": comparison_id,
            "comparison_feature_set": comparison_feature_set,
            "feature_set": feature_set.feature_set,
            "model_id": model_spec.model_id,
            "validation_scheme": scheme,
            "fold_id": fold_id,
            "fold_group": fold_info.get("fold_group", ""),
            "fold_year": fold_info.get("fold_year", ""),
            "fold_river": fold_info.get("fold_river", ""),
            "n_train": len(train),
            "n_test": len(test),
            "rivers_in_test": ";".join(sorted(test["river"].astype(str).unique())) if len(test) else "",
            "sensors_in_test": ";".join(sorted(test["sensor"].astype(str).unique())) if len(test) else "",
            "fold_skipped": bool(skip_reason),
            "skip_reason": skip_reason,
        }
        if skip_reason:
            fold_base.update({"rmse": np.nan, "mae": np.nan, "bias_mean": np.nan, "r2": np.nan})
            fold_rows.append(fold_base)
            continue

        estimator = _make_estimator(model_spec, feature_set)
        x_train = train[list(feature_set.required_features)]
        x_test = test[list(feature_set.required_features)]
        y_train = train[TARGET_COLUMN]
        estimator.fit(x_train, y_train)
        pred = estimator.predict(x_test)
        fold_metric = metric_row(test[TARGET_COLUMN], pred)
        fold_base.update(
            {
                "rmse": fold_metric["rmse"],
                "mae": fold_metric["mae"],
                "bias_mean": fold_metric["bias_mean"],
                "r2": fold_metric["r2"],
            }
        )
        fold_rows.append(fold_base)
        for row, predicted in zip(test.to_dict("records"), pred):
            observed = float(row[TARGET_COLUMN])
            cv_rows.append(
                {
                    "label_id": row["label_id"],
                    "river": row["river"],
                    "date": pd.Timestamp(row["date"]).date().isoformat(),
                    "year": int(row["year"]) if pd.notna(row["year"]) else "",
                    "month": int(row["month"]) if pd.notna(row["month"]) else "",
                    "sensor": row.get("sensor", ""),
                    "window": dataset.window,
                    "DOC_observed_mgC_L": observed,
                    "DOC_cv_predicted_mgC_L": float(predicted),
                    "residual_mgC_L": observed - float(predicted),
                    "Q_m3s": row.get("Q_m3s", np.nan),
                    "days_offset": row.get("days_offset", np.nan),
                    "abs_days_offset": row.get("abs_days_offset", np.nan),
                    "pct_valid_water_pixels": row.get("pct_valid_water_pixels", np.nan),
                    "dataset_id": dataset.dataset_id,
                    "sensor_scope": dataset.sensor_scope,
                    "comparison_id": comparison_id,
                    "comparison_feature_set": comparison_feature_set,
                    "feature_set": feature_set.feature_set,
                    "model_id": model_spec.model_id,
                    "validation_scheme": scheme,
                    "fold_id": fold_id,
                    "is_cv_prediction": True,
                    "is_production_prediction": False,
                }
            )
    return pd.DataFrame(cv_rows), pd.DataFrame(fold_rows)


def _run_dataset(dataset: OpticalDataset, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    subset_rows: list[dict[str, Any]] = []
    baseline_model = _baseline_model_spec()
    for candidate in optical_feature_sets_for_dataset(dataset):
        subset = _usable_same_subset(frame, candidate)
        subset_rows.append(
            {
                "dataset_id": dataset.dataset_id,
                "window": dataset.window,
                "sensor_scope": dataset.sensor_scope,
                "comparison_feature_set": candidate.feature_set,
                "rows_total": len(frame),
                "same_sample_n": len(subset),
                "n_rivers": int(subset["river"].nunique()) if not subset.empty else 0,
                "n_years": int(subset["year"].nunique()) if not subset.empty else 0,
                "n_sensors": int(subset["sensor"].nunique()) if not subset.empty else 0,
                "underpowered": len(subset) < dataset.underpowered_threshold,
            }
        )
        if len(subset) < VALIDATION_MIN_TEST_ROWS + VALIDATION_MIN_TRAIN_ROWS:
            continue
        baseline_cv, baseline_folds = _run_cv(
            frame=subset,
            dataset=dataset,
            feature_set=BASELINE_COMPARATOR,
            model_spec=baseline_model,
            comparison_feature_set=candidate.feature_set,
        )
        pred_frames.append(baseline_cv)
        fold_frames.append(baseline_folds)
        for model_spec in OPTICAL_MODEL_SPECS:
            candidate_cv, candidate_folds = _run_cv(
                frame=subset,
                dataset=dataset,
                feature_set=candidate,
                model_spec=model_spec,
                comparison_feature_set=candidate.feature_set,
            )
            pred_frames.append(candidate_cv)
            fold_frames.append(candidate_folds)
    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    folds = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    subsets = pd.DataFrame(subset_rows)
    return predictions, folds, subsets


def _metric_groups(predictions: pd.DataFrame, fold_summary: pd.DataFrame, group_columns: list[str] | None = None) -> pd.DataFrame:
    group_columns = group_columns or []
    base_columns = ["dataset_id", "window", "sensor_scope", "comparison_feature_set", "feature_set", "model_id", "validation_scheme"]
    rows: list[dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame()
    for keys, subset in predictions.groupby([*base_columns, *group_columns], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip([*base_columns, *group_columns], keys))
        row["n_test_total"] = len(subset)
        row["n_folds"] = int(subset["fold_id"].nunique()) if "fold_id" in subset.columns else 0
        folds = fold_summary.copy()
        for column in base_columns:
            folds = folds[folds[column].eq(row[column])]
        folds = folds[~folds.get("fold_skipped", pd.Series(False, index=folds.index)).astype(bool)] if not folds.empty else folds
        row["n_train_total"] = int(folds["n_train"].sum()) if not folds.empty and not group_columns else ""
        row.update(metric_row(subset["DOC_observed_mgC_L"], subset["DOC_cv_predicted_mgC_L"]))
        rows.append(row)
    return pd.DataFrame(rows)


def _classification(row: pd.Series) -> str:
    if bool(row.get("underpowered", False)):
        return "underpowered"
    meaningful = float(row.get("rmse_reduction", 0.0)) >= 0.10 or float(row.get("mae_reduction", 0.0)) >= 0.05
    positive = float(row.get("rmse_reduction", 0.0)) > 0.0 or float(row.get("mae_reduction", 0.0)) > 0.0
    worse = float(row.get("rmse_reduction", 0.0)) < -0.05 and float(row.get("mae_reduction", 0.0)) < -0.02
    major_bias = float(row.get("bias_change_abs", 0.0)) > 0.25
    severe_fold_loss = bool(row.get("severe_fold_loss", False))
    secondary_ok = float(row.get("secondary_rmse_reduction", 0.0)) > 0.0 or float(row.get("secondary_mae_reduction", 0.0)) > 0.0
    if meaningful and not major_bias and not severe_fold_loss and secondary_ok:
        return "optical_improves_baseline"
    if meaningful or positive:
        return "optical_marginal"
    if worse:
        return "optical_worse_than_baseline"
    return "optical_no_improvement"


def _same_sample_deltas(overall: pd.DataFrame, subset_audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline = overall[overall["feature_set"].eq(BASELINE_COMPARATOR.feature_set) & overall["model_id"].eq(BASELINE_MODEL_ID)].copy()
    candidates = overall[~overall["feature_set"].eq(BASELINE_COMPARATOR.feature_set)].copy()
    subset_lookup = subset_audit.set_index(["dataset_id", "comparison_feature_set"]) if not subset_audit.empty else pd.DataFrame()
    for candidate in candidates.to_dict("records"):
        base = baseline[
            baseline["dataset_id"].eq(candidate["dataset_id"])
            & baseline["comparison_feature_set"].eq(candidate["comparison_feature_set"])
            & baseline["validation_scheme"].eq(candidate["validation_scheme"])
        ]
        if base.empty:
            continue
        base_row = base.iloc[0]
        subset_key = (candidate["dataset_id"], candidate["comparison_feature_set"])
        subset_row = subset_lookup.loc[subset_key] if not subset_lookup.empty and subset_key in subset_lookup.index else pd.Series(dtype=object)
        severe_fold_loss = int(candidate["n_folds"]) < max(3, int(float(base_row["n_folds"]) * 0.7))
        rows.append(
            {
                "dataset_id": candidate["dataset_id"],
                "window": candidate["window"],
                "sensor_scope": candidate["sensor_scope"],
                "comparison_feature_set": candidate["comparison_feature_set"],
                "baseline_feature_set": BASELINE_COMPARATOR.feature_set,
                "candidate_feature_set": candidate["feature_set"],
                "is_optical_proxy_feature_set": candidate["feature_set"] != "O1_quality_only",
                "candidate_model_id": candidate["model_id"],
                "validation_scheme": candidate["validation_scheme"],
                "same_sample_n": int(subset_row.get("same_sample_n", candidate["n_test_total"])),
                "baseline_n_folds": base_row["n_folds"],
                "candidate_n_folds": candidate["n_folds"],
                "baseline_rmse": base_row["rmse"],
                "candidate_rmse": candidate["rmse"],
                "rmse_reduction": base_row["rmse"] - candidate["rmse"],
                "baseline_mae": base_row["mae"],
                "candidate_mae": candidate["mae"],
                "mae_reduction": base_row["mae"] - candidate["mae"],
                "baseline_r2": base_row["r2"],
                "candidate_r2": candidate["r2"],
                "r2_gain": candidate["r2"] - base_row["r2"],
                "baseline_bias_mean": base_row["bias_mean"],
                "candidate_bias_mean": candidate["bias_mean"],
                "bias_change_abs": abs(candidate["bias_mean"]) - abs(base_row["bias_mean"]),
                "underpowered": bool(subset_row.get("underpowered", False)),
                "severe_fold_loss": severe_fold_loss,
            }
        )
    deltas = pd.DataFrame(rows)
    if deltas.empty:
        return deltas
    secondary = deltas[deltas["validation_scheme"].eq("river_year_groupkfold")][
        ["dataset_id", "comparison_feature_set", "candidate_feature_set", "candidate_model_id", "rmse_reduction", "mae_reduction"]
    ].rename(columns={"rmse_reduction": "secondary_rmse_reduction", "mae_reduction": "secondary_mae_reduction"})
    deltas = deltas.merge(secondary, on=["dataset_id", "comparison_feature_set", "candidate_feature_set", "candidate_model_id"], how="left")
    deltas[["secondary_rmse_reduction", "secondary_mae_reduction"]] = deltas[["secondary_rmse_reduction", "secondary_mae_reduction"]].fillna(0.0)
    deltas["classification"] = deltas.apply(_classification, axis=1)
    rank_map = {
        "optical_improves_baseline": 1,
        "optical_marginal": 2,
        "optical_no_improvement": 3,
        "optical_worse_than_baseline": 4,
        "underpowered": 5,
    }
    deltas["classification_rank"] = deltas["classification"].map(rank_map).fillna(9).astype(int)
    deltas["comparison"] = "candidate_minus_B0_F3_same_subset"
    return deltas


def _model_ranking(deltas: pd.DataFrame) -> pd.DataFrame:
    if deltas.empty:
        return pd.DataFrame()
    loyo = deltas[deltas["validation_scheme"].eq("leave_one_year_out")].copy()
    loyo["primary_ranking_scope"] = loyo["dataset_id"].eq("any_sensor_3d")
    loyo["selection_basis"] = np.where(
        loyo["primary_ranking_scope"],
        "any_sensor_3d_leave_one_year_out_against_B0_F3_same_subset",
        "secondary_window_or_sensor_sensitivity",
    )
    loyo["ranking_score"] = loyo["classification_rank"] - loyo["rmse_reduction"].fillna(0.0) * 0.01
    loyo = loyo.sort_values(
        ["primary_ranking_scope", "classification_rank", "rmse_reduction", "mae_reduction"],
        ascending=[False, True, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    loyo["rank"] = np.arange(1, len(loyo) + 1)
    return loyo


def _bias_flag(row: dict[str, Any], overall_rmse: float | None = None) -> str:
    flags = []
    n = int(row.get("n", 0))
    bias = float(row.get("bias_mean", np.nan))
    rmse = float(row.get("rmse", np.nan))
    if n < 10:
        flags.append("small_n")
    if np.isfinite(bias) and bias > 0.5:
        flags.append("systematic_positive_bias")
    if np.isfinite(bias) and bias < -0.5:
        flags.append("systematic_negative_bias")
    if overall_rmse is not None and np.isfinite(overall_rmse) and np.isfinite(rmse) and rmse > overall_rmse * 1.5:
        flags.append("high_rmse_relative_to_overall")
    return ";".join(flags)


def _optical_bias_audit(predictions: pd.DataFrame, overall: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    loyo = predictions[predictions["validation_scheme"].eq("leave_one_year_out")].copy()
    loyo["abs_days_offset_bin"] = pd.cut(
        pd.to_numeric(loyo["abs_days_offset"], errors="coerce"),
        bins=[-0.1, 0, 1, 3, 7, np.inf],
        labels=["0d", "1d", "2-3d", "4-7d", "gt7d"],
    )
    try:
        loyo["valid_water_pixel_bin"] = pd.qcut(
            pd.to_numeric(loyo["pct_valid_water_pixels"], errors="coerce"),
            q=4,
            duplicates="drop",
        ).astype(str)
    except ValueError:
        loyo["valid_water_pixel_bin"] = pd.NA
    overall_lookup = overall.set_index(["dataset_id", "comparison_feature_set", "feature_set", "model_id", "validation_scheme"])["rmse"].to_dict()
    group_specs = [("sensor", "sensor"), ("abs_days_offset_bin", "days_offset_bin"), ("valid_water_pixel_bin", "valid_water_pixel_bin")]
    base_columns = ["dataset_id", "window", "sensor_scope", "comparison_feature_set", "feature_set", "model_id", "validation_scheme"]
    for source_column, group_type in group_specs:
        for keys, subset in loyo.groupby([*base_columns, source_column], dropna=False, observed=False):
            row = dict(zip([*base_columns, source_column], keys))
            metrics = metric_row(subset["DOC_observed_mgC_L"], subset["DOC_cv_predicted_mgC_L"])
            residual = pd.to_numeric(subset["residual_mgC_L"], errors="coerce").dropna()
            output = {
                **{column: row[column] for column in base_columns},
                "group_type": group_type,
                "group_value": row[source_column],
                "n": len(subset),
                **metrics,
                "p05_residual": float(residual.quantile(0.05)) if not residual.empty else np.nan,
                "p95_residual": float(residual.quantile(0.95)) if not residual.empty else np.nan,
            }
            output["flags"] = _bias_flag(
                output,
                overall_lookup.get((output["dataset_id"], output["comparison_feature_set"], output["feature_set"], output["model_id"], output["validation_scheme"])),
            )
            rows.append(output)
    return pd.DataFrame(rows)


def _make_figures(
    *,
    deltas: pd.DataFrame,
    predictions: pd.DataFrame,
    fold_summary: pd.DataFrame,
    prepared_frames: dict[str, pd.DataFrame],
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    OPTICAL_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def save(fig, name: str) -> None:
        fig.tight_layout()
        destination = OPTICAL_FIGURE_DIR / name
        fig.savefig(destination, dpi=120)
        plt.close(fig)
        paths.append(destination)

    loyo = deltas[deltas["validation_scheme"].eq("leave_one_year_out")].copy()
    any_sensor = loyo[loyo["sensor_scope"].eq("any_sensor")]
    if not any_sensor.empty:
        best_by_window = any_sensor.sort_values(["dataset_id", "classification_rank", "rmse_reduction"], ascending=[True, True, False]).groupby("dataset_id", as_index=False).first()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(best_by_window["window"], best_by_window["rmse_reduction"])
        ax.axhline(0, linestyle="--")
        ax.set_xlabel("Optical match window")
        ax.set_ylabel("Best RMSE reduction vs F3")
        ax.set_title("Best optical RMSE delta by window")
        save(fig, "rmse_delta_by_window.png")

    sensors = loyo[loyo["dataset_id"].isin(["hls_3d", "landsat_3d", "sentinel2_3d"])]
    if not sensors.empty:
        best_sensor = sensors.sort_values(["dataset_id", "classification_rank", "rmse_reduction"], ascending=[True, True, False]).groupby("dataset_id", as_index=False).first()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(best_sensor["sensor_scope"], best_sensor["rmse_reduction"])
        ax.axhline(0, linestyle="--")
        ax.set_xlabel("Sensor-specific 3d subset")
        ax.set_ylabel("Best RMSE reduction vs F3")
        ax.set_title("Best optical RMSE delta by sensor")
        save(fig, "rmse_delta_by_sensor_3d.png")

    primary_ranked = loyo[loyo["dataset_id"].eq("any_sensor_3d")].sort_values(["classification_rank", "rmse_reduction"], ascending=[True, False])
    if not primary_ranked.empty and not predictions.empty:
        best = primary_ranked.iloc[0]
        plot = predictions[
            predictions["dataset_id"].eq("any_sensor_3d")
            & predictions["comparison_feature_set"].eq(best["comparison_feature_set"])
            & predictions["validation_scheme"].eq("leave_one_year_out")
            & (
                (predictions["feature_set"].eq(BASELINE_COMPARATOR.feature_set) & predictions["model_id"].eq(BASELINE_MODEL_ID))
                | (predictions["feature_set"].eq(best["candidate_feature_set"]) & predictions["model_id"].eq(best["candidate_model_id"]))
            )
        ].copy()
        if not plot.empty:
            fig, ax = plt.subplots(figsize=(6, 5))
            for label, subset in plot.groupby(plot["feature_set"] + ":" + plot["model_id"]):
                ax.scatter(subset["DOC_observed_mgC_L"], subset["DOC_cv_predicted_mgC_L"], s=14, alpha=0.55, label=label)
            lo = min(plot["DOC_observed_mgC_L"].min(), plot["DOC_cv_predicted_mgC_L"].min())
            hi = max(plot["DOC_observed_mgC_L"].max(), plot["DOC_cv_predicted_mgC_L"].max())
            ax.plot([lo, hi], [lo, hi], linestyle="--")
            ax.set_xlabel("Observed DOC mg C/L")
            ax.set_ylabel("Validation-only CV predicted DOC mg C/L")
            ax.set_title("Any-sensor 3d observed vs CV predicted")
            ax.legend(fontsize="x-small")
            save(fig, "observed_vs_predicted_any_sensor_3d.png")

            fig, ax = plt.subplots(figsize=(7, 4.5))
            sensor_data = [group["residual_mgC_L"].to_numpy() for _, group in plot.groupby("sensor")]
            labels = [str(name) for name, _ in plot.groupby("sensor")]
            if sensor_data:
                try:
                    ax.boxplot(sensor_data, tick_labels=labels, showfliers=False)
                except TypeError:
                    ax.boxplot(sensor_data, labels=labels, showfliers=False)
                ax.axhline(0, linestyle="--")
                ax.set_ylabel("Residual mg C/L")
                ax.set_title("Residuals by sensor")
                save(fig, "residuals_by_sensor.png")

            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.scatter(plot["abs_days_offset"], plot["residual_mgC_L"], s=12, alpha=0.55)
            ax.axhline(0, linestyle="--")
            ax.set_xlabel("Absolute days offset")
            ax.set_ylabel("Residual mg C/L")
            ax.set_title("Residuals by optical days offset")
            save(fig, "residuals_by_days_offset.png")

    source = prepared_frames.get("any_sensor_3d", pd.DataFrame())
    corr_columns = ["blue", "green", "red", "nir", "swir1", "swir2", "ndwi", "mndwi", "red_green_ratio", "green_blue_ratio"]
    corr_source = source[[column for column in corr_columns if column in source.columns]].dropna()
    if len(corr_source) >= 3:
        corr = corr_source.corr()
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(corr, vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)
        ax.set_title("Optical feature correlation")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        save(fig, "optical_feature_correlation_heatmap.png")

    if not primary_ranked.empty and not fold_summary.empty:
        best = primary_ranked.iloc[0]
        folds = fold_summary[
            fold_summary["dataset_id"].eq("any_sensor_3d")
            & fold_summary["comparison_feature_set"].eq(best["comparison_feature_set"])
            & fold_summary["validation_scheme"].eq("leave_one_year_out")
            & ~fold_summary["fold_skipped"].astype(bool)
            & (
                (fold_summary["feature_set"].eq(BASELINE_COMPARATOR.feature_set) & fold_summary["model_id"].eq(BASELINE_MODEL_ID))
                | (fold_summary["feature_set"].eq(best["candidate_feature_set"]) & fold_summary["model_id"].eq(best["candidate_model_id"]))
            )
        ].copy()
        if not folds.empty:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            for label, subset in folds.groupby(folds["feature_set"] + ":" + folds["model_id"]):
                ax.plot(subset["fold_year"].astype(str), subset["rmse"], marker="o", label=label)
            ax.set_xlabel("Held-out year")
            ax.set_ylabel("Fold RMSE")
            ax.set_title("Any-sensor 3d LOYO fold performance")
            ax.tick_params(axis="x", rotation=45)
            ax.legend(fontsize="x-small")
            save(fig, "fold_performance_any_sensor_3d.png")

    return paths


def run_optical_sensitivity() -> dict[str, Any]:
    _ensure_dirs()
    assert_no_forbidden_outputs()
    validate_optical_feature_sets()
    _load_baseline_metadata()
    verification = _verify_contract_snapshot()
    before_hashes = _verify_allowed_optical_hashes()

    prepared_frames: dict[str, pd.DataFrame] = {}
    pred_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    subset_frames: list[pd.DataFrame] = []
    dataset_rows: list[dict[str, Any]] = []
    for dataset in OPTICAL_DATASETS:
        raw = _read_optical_table(dataset)
        prepared = _prepare_optical_frame(raw, dataset)
        prepared_frames[dataset.dataset_id] = prepared
        pred, folds, subsets = _run_dataset(dataset, prepared)
        pred_frames.append(pred)
        fold_frames.append(folds)
        subset_frames.append(subsets)
        dataset_rows.append(
            {
                **dataset.__dict__,
                "row_count": len(prepared),
                "n_rivers": int(prepared["river"].nunique()),
                "n_years": int(prepared["year"].nunique()),
                "n_sensors": int(prepared["sensor"].nunique()),
                "rows_with_any_optical_band": int(prepared[["blue", "green", "red", "nir"]].notna().any(axis=1).sum()),
                "median_abs_days_offset": float(prepared["abs_days_offset"].median()) if len(prepared) else np.nan,
                "median_pct_valid_water_pixels": float(prepared["pct_valid_water_pixels"].median()) if len(prepared) else np.nan,
            }
        )

    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    fold_summary = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    subset_audit = pd.concat(subset_frames, ignore_index=True) if subset_frames else pd.DataFrame()
    residuals = predictions.copy()
    if not residuals.empty:
        residuals["abs_residual_mgC_L"] = residuals["residual_mgC_L"].abs()
        residuals["squared_residual_mgC_L"] = residuals["residual_mgC_L"] ** 2

    overall = _metric_groups(predictions, fold_summary)
    by_river = _metric_groups(predictions, fold_summary, ["river"])
    by_year = _metric_groups(predictions, fold_summary, ["year"])
    by_month = _metric_groups(predictions, fold_summary, ["month"])
    by_sensor = _metric_groups(predictions, fold_summary, ["sensor"])
    by_season = _metric_groups(_season_window_rows(predictions), fold_summary, ["season_window"])
    deltas = _same_sample_deltas(overall, subset_audit)
    ranking = _model_ranking(deltas)
    bias_audit = _optical_bias_audit(predictions, overall)

    dataset_registry = pd.DataFrame(dataset_rows)
    if not subset_audit.empty:
        dataset_registry = dataset_registry.merge(
            subset_audit.groupby("dataset_id", as_index=False).agg(
                min_same_sample_n=("same_sample_n", "min"),
                max_same_sample_n=("same_sample_n", "max"),
                underpowered_feature_sets=("underpowered", "sum"),
            ),
            on="dataset_id",
            how="left",
        )

    table_paths = [
        _write_csv(dataset_registry, OPTICAL_TABLE_DIR / "optical_dataset_registry.csv"),
        _write_csv(optical_feature_set_registry(), OPTICAL_TABLE_DIR / "optical_feature_set_registry.csv"),
        _write_csv(optical_model_registry(), OPTICAL_TABLE_DIR / "optical_model_registry.csv"),
        _write_csv(validation_scheme_registry(), OPTICAL_TABLE_DIR / "optical_validation_registry.csv"),
        _write_csv(overall, OPTICAL_TABLE_DIR / "optical_metrics_overall.csv"),
        _write_csv(by_river, OPTICAL_TABLE_DIR / "optical_metrics_by_river.csv"),
        _write_csv(by_year, OPTICAL_TABLE_DIR / "optical_metrics_by_year.csv"),
        _write_csv(by_month, OPTICAL_TABLE_DIR / "optical_metrics_by_month.csv"),
        _write_csv(by_sensor, OPTICAL_TABLE_DIR / "optical_metrics_by_sensor.csv"),
        _write_csv(by_season, OPTICAL_TABLE_DIR / "optical_metrics_by_season_window.csv"),
        _write_csv(predictions, OPTICAL_TABLE_DIR / "optical_cv_predictions.csv"),
        _write_csv(residuals, OPTICAL_TABLE_DIR / "optical_residuals.csv"),
        _write_csv(deltas, OPTICAL_TABLE_DIR / "optical_same_sample_deltas.csv"),
        _write_csv(ranking, OPTICAL_TABLE_DIR / "optical_model_ranking.csv"),
        _write_csv(bias_audit, OPTICAL_TABLE_DIR / "optical_bias_audit.csv"),
        _write_csv(fold_summary, OPTICAL_TABLE_DIR / "optical_fold_summary.csv"),
    ]
    figure_paths = _make_figures(deltas=deltas, predictions=predictions, fold_summary=fold_summary, prepared_frames=prepared_frames)
    report_path = write_optical_sensitivity_report(OPTICAL_TABLE_DIR, OPTICAL_REPORT_DIR, OPTICAL_REPORT_PATH)

    after_hashes = _verify_allowed_optical_hashes()
    if before_hashes != after_hashes:
        raise RuntimeError("One or more frozen optical gold table hashes changed during optical sensitivity.")
    assert_no_forbidden_outputs()
    return {
        "tables": table_paths,
        "figures": figure_paths,
        "report": report_path,
        "verification": verification,
        "ranking": ranking,
    }


def write_optical_report_from_tables() -> Path:
    return write_optical_sensitivity_report(OPTICAL_TABLE_DIR, OPTICAL_REPORT_DIR, OPTICAL_REPORT_PATH)
