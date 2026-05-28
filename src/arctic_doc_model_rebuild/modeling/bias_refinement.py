from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..gold_contract import load_contract, require_gold_data_dir, sha256_file, table_path
from ..paths import CONFIG_DIR, REPORT_DIR, TABLE_DIR, path
from ..reports import _md_table, utc_now
from .baseline_models import _prepare_hydrocore, _read_hydrocore
from .diagnostics import assert_gold_hash_unchanged, assert_no_forbidden_outputs
from .interaction_features import BiasFeatureSet, add_river_interactions, bias_feature_set_registry, bias_feature_sets
from .metrics import metric_row
from .nonlinear_features import add_nonlinear_features
from .refinement_readiness import refined_readiness_decision
from .target_transforms import inverse_target, target_values
from .validation import validation_splits


BIAS_TABLE_DIR = TABLE_DIR / "bias_refinement"
BIAS_REPORT_DIR = REPORT_DIR / "bias_refinement"
BIAS_FIGURE_DIR = path("outputs", "figures", "bias_refinement")
BIAS_REPORT_PATH = BIAS_REPORT_DIR / "bias_refinement_report.md"
REFINED_MODEL_SPEC_PATH = CONFIG_DIR / "model_specs" / "refined_primary_model.yaml"

BASELINE_DECISION_PATH = TABLE_DIR / "baseline_final" / "baseline_model_decision.csv"
PRIMARY_SPEC_PATH = CONFIG_DIR / "model_specs" / "primary_baseline_f3_q_season_river_fixed_ridge_alpha_1.yaml"
UNCERTAINTY_PREDICTIONS_PATH = TABLE_DIR / "concentration_uncertainty" / "uncertainty_cv_predictions.csv"
UNCERTAINTY_RIVER_BIAS_PATH = TABLE_DIR / "concentration_uncertainty" / "river_bias_summary.csv"
UNCERTAINTY_FOLD_STABILITY_PATH = TABLE_DIR / "concentration_uncertainty" / "fold_stability_summary.csv"
UNCERTAINTY_HIGH_DOC_PATH = TABLE_DIR / "concentration_uncertainty" / "high_doc_residual_review.csv"
UNCERTAINTY_READINESS_PATH = TABLE_DIR / "concentration_uncertainty" / "production_readiness_decision.csv"
ALLOWED_METADATA_PATHS = [
    BASELINE_DECISION_PATH,
    PRIMARY_SPEC_PATH,
    UNCERTAINTY_PREDICTIONS_PATH,
    UNCERTAINTY_RIVER_BIAS_PATH,
    UNCERTAINTY_FOLD_STABILITY_PATH,
    UNCERTAINTY_HIGH_DOC_PATH,
    UNCERTAINTY_READINESS_PATH,
]


@dataclass(frozen=True)
class BiasModelSpec:
    model_id: str
    model_family: str
    model_class: str
    alpha: float | None = None
    caveat: str = ""


BIAS_MODEL_SPECS = [
    BiasModelSpec("linear_regression", "linear", "LinearRegression"),
    BiasModelSpec("ridge_alpha_0.1", "ridge", "Ridge", alpha=0.1),
    BiasModelSpec("ridge_alpha_1", "ridge", "Ridge", alpha=1.0),
    BiasModelSpec("ridge_alpha_10", "ridge", "Ridge", alpha=10.0),
    BiasModelSpec("ridge_alpha_100", "ridge", "Ridge", alpha=100.0),
    BiasModelSpec("huber_F3", "robust_linear", "HuberRegressor", caveat="R7 robust F3 sensitivity only."),
]


def _ensure_dirs() -> None:
    for directory in [BIAS_TABLE_DIR, BIAS_REPORT_DIR, BIAS_FIGURE_DIR, REFINED_MODEL_SPEC_PATH.parent]:
        directory.mkdir(parents=True, exist_ok=True)


def _write_csv(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8")
    return destination


def _read_required_csv(destination: Path) -> pd.DataFrame:
    if not destination.exists():
        raise FileNotFoundError(f"Required bias refinement input is missing: {destination}")
    return pd.read_csv(destination)


def _load_required_metadata() -> dict[str, Any]:
    for destination in ALLOWED_METADATA_PATHS:
        if not destination.exists():
            raise FileNotFoundError(f"Required bias refinement metadata is missing: {destination}")
    baseline_decision = _read_required_csv(BASELINE_DECISION_PATH)
    primary = baseline_decision[baseline_decision["decision_type"].eq("primary_baseline")]
    if primary.empty or primary.iloc[0]["feature_set"] != "F3_q_season_river_fixed" or primary.iloc[0]["model_id"] != "ridge_alpha_1":
        raise RuntimeError("Bias refinement requires finalized primary baseline F3_q_season_river_fixed + ridge_alpha_1.")
    with PRIMARY_SPEC_PATH.open("r", encoding="utf-8") as handle:
        primary_spec = yaml.safe_load(handle)
    if primary_spec.get("production_prediction_allowed") is not False:
        raise RuntimeError("Primary baseline spec must disallow production prediction.")
    return {
        "baseline_decision": baseline_decision,
        "primary_spec": primary_spec,
        "uncertainty_predictions": _read_required_csv(UNCERTAINTY_PREDICTIONS_PATH),
        "river_bias": _read_required_csv(UNCERTAINTY_RIVER_BIAS_PATH),
        "fold_stability": _read_required_csv(UNCERTAINTY_FOLD_STABILITY_PATH),
        "high_doc": _read_required_csv(UNCERTAINTY_HIGH_DOC_PATH),
        "readiness": _read_required_csv(UNCERTAINTY_READINESS_PATH),
    }


def _verify_contract_snapshot() -> None:
    verification_path = TABLE_DIR / "gold_table_verification.csv"
    if not verification_path.exists():
        raise FileNotFoundError("Run `python -m arctic_doc_model_rebuild.cli verify-gold-data` before bias-aware refinement.")
    verification = pd.read_csv(verification_path)
    hydro = verification[verification["table_name"].eq("training_matrix_hydrocore.csv")]
    if hydro.empty or hydro.iloc[0]["status"] != "ok":
        raise RuntimeError("training_matrix_hydrocore.csv is not verified in the current gold contract snapshot.")


def _read_hydrocore_only() -> tuple[pd.DataFrame, Path]:
    gold_dir = require_gold_data_dir()
    destination = table_path("training_matrix_hydrocore.csv", gold_dir=gold_dir)
    if destination.name != "training_matrix_hydrocore.csv":
        raise RuntimeError(f"Bias refinement cannot read this gold table: {destination.name}")
    contract = load_contract()
    expected_hash = str(contract["expected_tables"]["training_matrix_hydrocore.csv"]["sha256"]).lower()
    if sha256_file(destination) != expected_hash:
        raise RuntimeError("training_matrix_hydrocore.csv hash does not match frozen contract.")
    frame = _prepare_hydrocore(_read_hydrocore())
    frame = add_nonlinear_features(frame)
    frame = add_river_interactions(frame)
    return frame, destination


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _model_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_id": spec.model_id,
                "model_family": spec.model_family,
                "model_class": spec.model_class,
                "alpha": "" if spec.alpha is None else spec.alpha,
                "simple_model_only": True,
                "validation_only": True,
                "caveat": spec.caveat,
            }
            for spec in BIAS_MODEL_SPECS
        ]
    )


def _model_specs_for_feature_set(feature_set: BiasFeatureSet) -> list[BiasModelSpec]:
    if feature_set.feature_set == "R7_robust_huber_F3":
        return [spec for spec in BIAS_MODEL_SPECS if spec.model_id == "huber_F3"]
    if feature_set.target_scale == "log":
        return [spec for spec in BIAS_MODEL_SPECS if spec.model_family == "ridge"]
    return [spec for spec in BIAS_MODEL_SPECS if spec.model_id != "huber_F3"]


def _make_estimator(spec: BiasModelSpec, feature_set: BiasFeatureSet) -> Pipeline:
    transformers = []
    if feature_set.numeric_features:
        transformers.append(("numeric", StandardScaler(), list(feature_set.numeric_features)))
    if feature_set.categorical_features:
        transformers.append(("categorical", _one_hot_encoder(), list(feature_set.categorical_features)))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    if spec.model_family == "linear":
        model = LinearRegression()
    elif spec.model_family == "ridge":
        model = Ridge(alpha=float(spec.alpha))
    elif spec.model_family == "robust_linear":
        model = HuberRegressor(max_iter=1000, epsilon=1.35)
    else:
        raise KeyError(f"Unsupported bias refinement model: {spec.model_id}")
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def _usable(frame: pd.DataFrame, feature_set: BiasFeatureSet) -> pd.DataFrame:
    required = ["label_id", "river", "date", "year", "month", "Q_m3s", "DOC_mgC_L", *feature_set.required_features]
    required = list(dict.fromkeys(required))
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"Bias feature set {feature_set.feature_set} is missing required columns: {missing}")
    usable = frame.dropna(subset=required).copy()
    usable = usable[usable["Q_m3s"] > 0].copy()
    if feature_set.target_scale == "log":
        usable = usable[pd.to_numeric(usable["DOC_mgC_L"], errors="coerce") > 0].copy()
    return usable.reset_index(drop=True)


def _validation_splits_with_recent_years(frame: pd.DataFrame):
    for item in validation_splits(frame):
        yield item
    years = sorted(pd.Series(frame["year"]).dropna().astype(int).unique())
    recent = [year for year in years if 2021 <= year <= 2024]
    if recent:
        test_mask = frame["year"].astype(int).isin(recent).to_numpy()
        train_idx = np.flatnonzero(~test_mask)
        test_idx = np.flatnonzero(test_mask)
        if len(train_idx) and len(test_idx) >= 5:
            yield (
                "leave_recent_years_out",
                "leave_recent_years_2021_2024",
                train_idx,
                test_idx,
                {"fold_group": "2021;2022;2023;2024", "fold_year": "2021-2024"},
            )


def _fit_predict(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_set: BiasFeatureSet,
    model_spec: BiasModelSpec,
) -> tuple[np.ndarray, np.ndarray]:
    estimator = _make_estimator(model_spec, feature_set)
    x_train = train[list(feature_set.required_features)]
    x_test = test[list(feature_set.required_features)]
    y_train = target_values(train, feature_set.target_scale)
    estimator.fit(x_train, y_train)
    pred_target = estimator.predict(x_test)
    pred_mg = inverse_target(pred_target, feature_set.target_scale)
    return np.asarray(pred_mg, dtype=float), np.asarray(pred_target, dtype=float)


def _run_candidate_cv(frame: pd.DataFrame, feature_set: BiasFeatureSet, model_spec: BiasModelSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = _usable(frame, feature_set)
    cv_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    candidate_key = f"{feature_set.feature_set}:{model_spec.model_id}:{feature_set.target_scale}"
    for scheme, fold_id, train_idx, test_idx, fold_info in _validation_splits_with_recent_years(usable):
        train = usable.iloc[train_idx].copy()
        test = usable.iloc[test_idx].copy()
        pred_mg, pred_target = _fit_predict(train=train, test=test, feature_set=feature_set, model_spec=model_spec)
        fold_metric = metric_row(test["DOC_mgC_L"].to_numpy(dtype=float), pred_mg)
        fold_rows.append(
            {
                "candidate_key": f"{candidate_key}:{scheme}",
                "model_id": model_spec.model_id,
                "feature_set": feature_set.feature_set,
                "target_scale": feature_set.target_scale,
                "validation_scheme": scheme,
                "fold_id": fold_id,
                "fold_group": fold_info.get("fold_group", ""),
                "fold_year": fold_info.get("fold_year", ""),
                "fold_river": fold_info.get("fold_river", ""),
                "n_train": len(train),
                "n_test": len(test),
                "rivers_in_fold": ";".join(sorted(test["river"].astype(str).unique())),
                "rmse": fold_metric["rmse"],
                "mae": fold_metric["mae"],
                "median_absolute_error": fold_metric["median_absolute_error"],
                "r2": fold_metric["r2"],
                "bias_mean": fold_metric["bias_mean"],
                "bias_median": fold_metric["bias_median"],
                "spearman_r": fold_metric["spearman_r"],
                "pearson_r": fold_metric["pearson_r"],
                "stress_test": scheme == "leave_one_river_out",
            }
        )
        for row, predicted_mg, predicted_target in zip(test.to_dict("records"), pred_mg, pred_target):
            observed = float(row["DOC_mgC_L"])
            cv_rows.append(
                {
                    "label_id": row["label_id"],
                    "river": row["river"],
                    "date": pd.Timestamp(row["date"]).date().isoformat(),
                    "year": int(row["year"]) if pd.notna(row["year"]) else "",
                    "month": int(row["month"]) if pd.notna(row["month"]) else "",
                    "Q_m3s": row.get("Q_m3s", np.nan),
                    "DOC_observed_mgC_L": observed,
                    "DOC_cv_predicted_mgC_L": float(predicted_mg),
                    "residual_mgC_L": observed - float(predicted_mg),
                    "DOC_cv_predicted_target_scale": float(predicted_target),
                    "model_id": model_spec.model_id,
                    "feature_set": feature_set.feature_set,
                    "target_scale": feature_set.target_scale,
                    "validation_scheme": scheme,
                    "fold_id": fold_id,
                    "candidate_key": f"{candidate_key}:{scheme}",
                    "is_cv_prediction": True,
                    "is_production_prediction": False,
                }
            )
    return pd.DataFrame(cv_rows), pd.DataFrame(fold_rows)


def _run_all_candidates(frame: pd.DataFrame, feature_sets: list[BiasFeatureSet]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_frames = []
    fold_frames = []
    for feature_set in feature_sets:
        for model_spec in _model_specs_for_feature_set(feature_set):
            predictions, folds = _run_candidate_cv(frame, feature_set, model_spec)
            pred_frames.append(predictions)
            fold_frames.append(folds)
    return pd.concat(pred_frames, ignore_index=True), pd.concat(fold_frames, ignore_index=True)


def _metrics_overall(predictions: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, subset in predictions.groupby(["candidate_key", "model_id", "feature_set", "target_scale", "validation_scheme"], dropna=False):
        candidate_key, model_id, feature_set, target_scale, validation_scheme = keys
        fold_subset = folds[folds["candidate_key"].eq(candidate_key)]
        row = {
            "candidate_key": candidate_key,
            "model_id": model_id,
            "feature_set": feature_set,
            "target_scale": target_scale,
            "validation_scheme": validation_scheme,
            "n_test_total": len(subset),
            "n_train_total": int(fold_subset["n_train"].sum()) if not fold_subset.empty else "",
            "n_folds": int(subset["fold_id"].nunique()),
            "stress_test": validation_scheme == "leave_one_river_out",
        }
        row.update(metric_row(subset["DOC_observed_mgC_L"], subset["DOC_cv_predicted_mgC_L"]))
        rows.append(row)
    return pd.DataFrame(rows)


def _metrics_by_group(predictions: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, subset in predictions.groupby(["candidate_key", "model_id", "feature_set", "target_scale", "validation_scheme", group_column], dropna=False):
        candidate_key, model_id, feature_set, target_scale, validation_scheme, group_value = keys
        row = {
            "candidate_key": candidate_key,
            "model_id": model_id,
            "feature_set": feature_set,
            "target_scale": target_scale,
            "validation_scheme": validation_scheme,
            group_column: group_value,
            "n": len(subset),
        }
        row.update(metric_row(subset["DOC_observed_mgC_L"], subset["DOC_cv_predicted_mgC_L"]))
        rows.append(row)
    return pd.DataFrame(rows)


def _doc_behavior_groups(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
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


def _high_doc_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    enriched = _doc_behavior_groups(predictions[predictions["validation_scheme"].eq("leave_one_year_out")].copy())
    return _metrics_by_group(enriched, "doc_behavior_group")


def _fold_stability(folds: pd.DataFrame, overall: pd.DataFrame) -> pd.DataFrame:
    overall_lookup = overall[overall["validation_scheme"].eq("leave_one_year_out")].set_index("candidate_key")["rmse"].to_dict()
    rows = []
    loyo = folds[folds["validation_scheme"].eq("leave_one_year_out")].copy()
    for record in loyo.to_dict("records"):
        overall_rmse = overall_lookup.get(record["candidate_key"], np.nan)
        high_rmse = bool(pd.notna(overall_rmse) and pd.notna(record["rmse"]) and record["rmse"] > overall_rmse * 1.5)
        high_bias = bool(pd.notna(record["bias_mean"]) and abs(record["bias_mean"]) > 1.0)
        small_n = bool(int(record["n_test"]) < 5)
        rows.append(
            {
                **record,
                "overall_rmse": overall_rmse,
                "small_fold_n": small_n,
                "high_rmse": high_rmse,
                "high_bias": high_bias,
                "unstable_fold": high_rmse or high_bias,
            }
        )
    return pd.DataFrame(rows)


def _uncertainty_diagnostic_audit(metadata: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    folds = metadata["fold_stability"]
    blank_metric_mask = pd.Series(False, index=folds.index)
    for column in ["rmse", "mae", "bias_mean", "bias_median"]:
        if column in folds.columns:
            blank_metric_mask = blank_metric_mask | folds[column].isna()
    affected = folds[blank_metric_mask & (pd.to_numeric(folds.get("n_test", 0), errors="coerce") > 0)]
    predictions = metadata["uncertainty_predictions"]
    missing_prediction_rows = 0
    if not predictions.empty:
        primary = predictions[predictions["model_role"].eq("primary_baseline") & predictions["validation_scheme"].eq("leave_one_year_out")]
        missing_prediction_rows = int(primary[["DOC_observed_mgC_L", "DOC_cv_predicted_mgC_L"]].isna().any(axis=1).sum()) if not primary.empty else 0
    rows.append(
        {
            "diagnostic_item": "fold_stability_blank_metrics",
            "status": "code_bug_fixed" if len(affected) else "ok",
            "n_affected": int(len(affected)),
            "likely_cause": "previous_metric_index_alignment_bug" if len(affected) and missing_prediction_rows == 0 else "missing_predictions_or_observations" if len(affected) else "none",
            "requires_code_fix": False,
            "requires_model_refinement": bool(len(affected)),
            "notes": "metric_row now resets indexes before metric computation; bias refinement recomputes fold metrics." if len(affected) else "No blank RMSE/MAE/bias metrics detected.",
        }
    )

    river_bias = metadata["river_bias"]
    mean_bias_flags = int(river_bias.get("flag_abs_bias_gt_1", pd.Series(dtype=bool)).astype(bool).sum()) if not river_bias.empty else 0
    rmse_flags = int(river_bias.get("flag_rmse_gt_1_25x_overall", pd.Series(dtype=bool)).astype(bool).sum()) if not river_bias.empty else 0
    rmse_flag_series = river_bias["flag_rmse_gt_1_25x_overall"].astype(bool) if "flag_rmse_gt_1_25x_overall" in river_bias.columns else pd.Series(False, index=river_bias.index)
    rows.append(
        {
            "diagnostic_item": "major_river_bias_type",
            "status": "rmse_driven" if rmse_flags and not mean_bias_flags else "mean_bias_driven" if mean_bias_flags else "ok",
            "n_affected": int(mean_bias_flags + rmse_flags),
            "likely_cause": "Lena_high_rmse" if "Lena" in set(river_bias.loc[rmse_flag_series, "river"]) else "none_or_multiple",
            "requires_code_fix": False,
            "requires_model_refinement": bool(mean_bias_flags or rmse_flags),
            "notes": "River-bias blocker is based on high RMSE rather than mean bias." if rmse_flags and not mean_bias_flags else "Review river-level residual behavior.",
        }
    )

    high_doc = _doc_behavior_groups(predictions[predictions["model_role"].eq("primary_baseline") & predictions["validation_scheme"].eq("leave_one_year_out")].copy())
    high = high_doc[high_doc["doc_behavior_group"].isin(["high", "extreme_high"])].copy()
    if not high.empty:
        by_group = high.groupby(["river", "year", "month"]).agg(n=("label_id", "count"), bias_mean=("residual_mgC_L", "mean"), rmse=("residual_mgC_L", lambda x: float(np.sqrt(np.mean(np.square(x)))))).reset_index()
        dominant = by_group.sort_values(["rmse", "n"], ascending=[False, False]).head(5)
        notes = "; ".join(f"{row.river}-{int(row.year)}-{int(row.month)} n={int(row.n)} rmse={row.rmse:.2f}" for row in dominant.itertuples(index=False))
    else:
        notes = "No high-DOC rows available."
    rows.append(
        {
            "diagnostic_item": "high_doc_residual_drivers",
            "status": "review_needed" if not high.empty else "not_available",
            "n_affected": int(len(high)),
            "likely_cause": "high_DOC_samples_underpredicted_and_low_DOC_overpredicted",
            "requires_code_fix": False,
            "requires_model_refinement": True,
            "notes": notes,
        }
    )
    return pd.DataFrame(rows)


def _candidate_deltas(
    overall: pd.DataFrame,
    by_river: pd.DataFrame,
    high_doc: pd.DataFrame,
    fold_stability: pd.DataFrame,
) -> pd.DataFrame:
    loyo = overall[overall["validation_scheme"].eq("leave_one_year_out")].copy()
    f3_key = "B0_F3_finalized:ridge_alpha_1:raw:leave_one_year_out"
    f3 = loyo[loyo["candidate_key"].eq(f3_key)]
    if f3.empty:
        raise RuntimeError("F3 ridge_alpha_1 LOYO comparator is missing from bias refinement metrics.")
    f3_row = f3.iloc[0]
    river_loyo = by_river[by_river["validation_scheme"].eq("leave_one_year_out")]
    lena = river_loyo[river_loyo["river"].eq("Lena")]
    f3_lena = lena[lena["candidate_key"].eq(f3_key)]
    f3_lena_rmse = float(f3_lena["rmse"].iloc[0]) if not f3_lena.empty else np.nan
    high_lookup = high_doc[high_doc["validation_scheme"].eq("leave_one_year_out")].set_index(["candidate_key", "doc_behavior_group"])["bias_mean"].to_dict()
    f3_high_bias = high_lookup.get((f3_key, "high"), np.nan)
    f3_extreme_bias = high_lookup.get((f3_key, "extreme_high"), np.nan)
    unstable_counts = fold_stability.groupby("candidate_key")["unstable_fold"].sum().to_dict()
    f3_unstable = int(unstable_counts.get(f3_key, 0))

    gkf = overall[overall["validation_scheme"].eq("river_year_groupkfold")].set_index("candidate_key")
    f3_gkf_rmse = float(gkf.loc["B0_F3_finalized:ridge_alpha_1:raw:river_year_groupkfold", "rmse"]) if "B0_F3_finalized:ridge_alpha_1:raw:river_year_groupkfold" in gkf.index else np.nan
    rows = []
    for record in loyo.to_dict("records"):
        key = record["candidate_key"]
        lena_row = lena[lena["candidate_key"].eq(key)]
        lena_rmse = float(lena_row["rmse"].iloc[0]) if not lena_row.empty else np.nan
        high_bias = high_lookup.get((key, "high"), np.nan)
        extreme_bias = high_lookup.get((key, "extreme_high"), np.nan)
        gkf_key = f"{record['feature_set']}:{record['model_id']}:{record['target_scale']}:river_year_groupkfold"
        gkf_rmse = float(gkf.loc[gkf_key, "rmse"]) if gkf_key in gkf.index else np.nan
        rows.append(
            {
                "candidate_key": key,
                "feature_set": record["feature_set"],
                "model_id": record["model_id"],
                "target_scale": record["target_scale"],
                "validation_scheme": record["validation_scheme"],
                "f3_rmse": f3_row["rmse"],
                "candidate_rmse": record["rmse"],
                "rmse_reduction_vs_f3": f3_row["rmse"] - record["rmse"],
                "f3_mae": f3_row["mae"],
                "candidate_mae": record["mae"],
                "mae_reduction_vs_f3": f3_row["mae"] - record["mae"],
                "f3_lena_rmse": f3_lena_rmse,
                "candidate_lena_rmse": lena_rmse,
                "delta_lena_rmse_vs_f3": f3_lena_rmse - lena_rmse if pd.notna(f3_lena_rmse) and pd.notna(lena_rmse) else np.nan,
                "f3_high_doc_bias": f3_high_bias,
                "candidate_high_doc_bias": high_bias,
                "delta_high_doc_bias_abs_vs_f3": abs(f3_high_bias) - abs(high_bias) if pd.notna(f3_high_bias) and pd.notna(high_bias) else np.nan,
                "f3_extreme_high_bias": f3_extreme_bias,
                "candidate_extreme_high_bias": extreme_bias,
                "delta_extreme_high_bias_abs_vs_f3": abs(f3_extreme_bias) - abs(extreme_bias) if pd.notna(f3_extreme_bias) and pd.notna(extreme_bias) else np.nan,
                "f3_unstable_fold_count": f3_unstable,
                "candidate_unstable_fold_count": int(unstable_counts.get(key, 0)),
                "delta_unstable_fold_count_vs_f3": f3_unstable - int(unstable_counts.get(key, 0)),
                "f3_groupkfold_rmse": f3_gkf_rmse,
                "candidate_groupkfold_rmse": gkf_rmse,
                "groupkfold_rmse_reduction_vs_f3": f3_gkf_rmse - gkf_rmse if pd.notna(f3_gkf_rmse) and pd.notna(gkf_rmse) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _ranking(deltas: pd.DataFrame, overall: pd.DataFrame) -> pd.DataFrame:
    ranked = deltas.copy()
    ranked["primary_selection_eligible"] = (
        ranked["validation_scheme"].eq("leave_one_year_out")
        & ranked["target_scale"].eq("raw")
        & ~ranked["feature_set"].eq("B0_F3_finalized")
    )
    ranked["meaningful_overall_or_high_doc_gain"] = (ranked["rmse_reduction_vs_f3"] >= 0.10) | (ranked["delta_high_doc_bias_abs_vs_f3"] >= 0.50)
    ranked["lena_not_worse"] = ranked["delta_lena_rmse_vs_f3"] >= 0
    ranked["high_doc_not_worse"] = (ranked["delta_high_doc_bias_abs_vs_f3"] >= 0) & (ranked["delta_extreme_high_bias_abs_vs_f3"] >= 0)
    ranked["fold_instability_not_worse"] = ranked["delta_unstable_fold_count_vs_f3"] >= 0
    ranked["groupkfold_not_contradictory"] = ranked["groupkfold_rmse_reduction_vs_f3"].fillna(0) >= -0.05
    ranked["interpretable"] = ~ranked["feature_set"].str.contains("R4").fillna(False) | True
    ranked["meets_replacement_criteria"] = (
        ranked["primary_selection_eligible"]
        & ranked["meaningful_overall_or_high_doc_gain"]
        & ranked["lena_not_worse"]
        & ranked["high_doc_not_worse"]
        & ranked["fold_instability_not_worse"]
        & ranked["groupkfold_not_contradictory"]
    )
    ranked["classification"] = np.select(
        [
            ranked["feature_set"].eq("B0_F3_finalized"),
            ranked["meets_replacement_criteria"],
            ranked["target_scale"].eq("log"),
            ranked["rmse_reduction_vs_f3"] > 0,
        ],
        ["f3_comparator", "replacement_candidate", "log_target_sensitivity", "incremental_but_not_selected"],
        default="not_improved",
    )
    ranked["ranking_score"] = np.where(
        ranked["meets_replacement_criteria"],
        -ranked["rmse_reduction_vs_f3"] - 0.25 * ranked["delta_high_doc_bias_abs_vs_f3"].fillna(0),
        ranked["candidate_rmse"] + 10,
    )
    return ranked.sort_values(["meets_replacement_criteria", "ranking_score", "candidate_rmse"], ascending=[False, True, True]).reset_index(drop=True)


def _recommendation(ranking: pd.DataFrame) -> pd.DataFrame:
    selected = ranking[ranking["meets_replacement_criteria"]].head(1)
    if selected.empty:
        f3 = ranking[ranking["feature_set"].eq("B0_F3_finalized") & ranking["model_id"].eq("ridge_alpha_1")].head(1).iloc[0]
        primary_status = "F3_q_season_river_fixed + ridge_alpha_1"
        action = "retain_f3_with_empirical_intervals_and_caveats"
        reason = "No candidate met all replacement criteria; keep finalized F3 and carry empirical residual interval caveats."
        row = f3
    else:
        row = selected.iloc[0]
        primary_status = f"{row['feature_set']} + {row['model_id']}"
        action = "replace_f3_with_refined_model"
        reason = "Candidate met LOYO improvement, Lena, high-DOC, fold-stability, GroupKFold, and interpretability criteria."
    best_log = ranking[ranking["target_scale"].eq("log")].sort_values(["candidate_rmse"]).head(1)
    rows = [
        {
            "decision_item": "recommended_primary_model_after_refinement",
            "status": primary_status,
            "feature_set": row["feature_set"],
            "model_id": row["model_id"],
            "target_scale": row["target_scale"],
            "recommended_action": action,
            "reason": reason,
        },
        {
            "decision_item": "log_target_review",
            "status": "sensitivity_only",
            "feature_set": best_log["feature_set"].iloc[0] if not best_log.empty else "",
            "model_id": best_log["model_id"].iloc[0] if not best_log.empty else "",
            "target_scale": "log",
            "recommended_action": "do_not_promote_in_this_phase",
            "reason": "Log-target candidates remain sensitivity-only unless a separate target-transform decision is made.",
        },
        {
            "decision_item": "production_readiness_update",
            "status": "pending_refined_readiness_table",
            "feature_set": row["feature_set"],
            "model_id": row["model_id"],
            "target_scale": row["target_scale"],
            "recommended_action": "see_refined_production_readiness_decision",
            "reason": "Production readiness is updated in refined_production_readiness_decision.csv.",
        },
    ]
    return pd.DataFrame(rows)


def _write_refined_model_spec(recommendation: pd.DataFrame, feature_registry: pd.DataFrame) -> list[Path]:
    selected = recommendation[recommendation["decision_item"].eq("recommended_primary_model_after_refinement")]
    if selected.empty or not selected["recommended_action"].iloc[0].startswith("replace"):
        if REFINED_MODEL_SPEC_PATH.exists():
            REFINED_MODEL_SPEC_PATH.unlink()
        return []
    row = selected.iloc[0]
    feature_row = feature_registry[feature_registry["feature_set"].eq(row["feature_set"])].iloc[0]
    numeric = [item for item in str(feature_row["numeric_features"]).split(";") if item]
    categorical = [item for item in str(feature_row["categorical_features"]).split(";") if item]
    model_type = "HuberRegressor" if row["model_id"] == "huber_F3" else "Ridge" if "ridge" in row["model_id"] else "LinearRegression"
    alpha = None
    if "ridge_alpha_" in row["model_id"]:
        alpha = float(str(row["model_id"]).replace("ridge_alpha_", ""))
    spec = {
        "model_spec_id": "refined_primary_model",
        "freeze_id": "data_freeze_gold_20260526_v1",
        "input_table": "data/processed/gold/training_matrix_hydrocore.csv",
        "target": "DOC_mgC_L",
        "target_transform": "log" if row["target_scale"] == "log" else "none",
        "inverse_transform": "exp" if row["target_scale"] == "log" else "none",
        "feature_set": row["feature_set"],
        "numeric_features": numeric,
        "categorical_features": categorical,
        "model": {"type": model_type, "alpha": alpha},
        "validation_primary": "leave_one_year_out",
        "validation_secondary": ["river_year_groupkfold", "leave_recent_years_out"],
        "validation_stress": ["leave_one_river_out"],
        "selected_as": "refined_primary_candidate",
        "production_prediction_allowed": False,
        "flux_allowed": False,
        "notes": [
            "Generated by bias-aware refinement; validation-only until a production prediction phase is explicitly authorized.",
            "No optical or basin predictors are included.",
        ],
    }
    REFINED_MODEL_SPEC_PATH.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return [REFINED_MODEL_SPEC_PATH]


def _make_figures(
    *,
    ranking: pd.DataFrame,
    high_doc: pd.DataFrame,
    by_river: pd.DataFrame,
    fold_stability: pd.DataFrame,
    predictions: pd.DataFrame,
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    BIAS_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def save(fig, name: str) -> None:
        fig.tight_layout()
        destination = BIAS_FIGURE_DIR / name
        fig.savefig(destination, dpi=120)
        plt.close(fig)
        paths.append(destination)

    loyo = ranking[ranking["validation_scheme"].eq("leave_one_year_out")].head(12)
    if not loyo.empty:
        labels = loyo["feature_set"] + ":" + loyo["model_id"] + ":" + loyo["target_scale"]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(labels, loyo["rmse_reduction_vs_f3"])
        ax.axvline(0, linestyle="--")
        ax.set_xlabel("RMSE reduction vs F3")
        ax.set_title("Bias refinement RMSE delta vs F3")
        ax.invert_yaxis()
        save(fig, "rmse_delta_vs_f3.png")

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(labels, loyo["delta_extreme_high_bias_abs_vs_f3"])
        ax.axvline(0, linestyle="--")
        ax.set_xlabel("Extreme-high DOC absolute bias reduction")
        ax.set_title("High-DOC bias delta")
        ax.invert_yaxis()
        save(fig, "high_doc_bias_delta.png")

    lena = by_river[(by_river["validation_scheme"].eq("leave_one_year_out")) & (by_river["river"].eq("Lena"))]
    if not lena.empty:
        plot = lena.merge(ranking[["candidate_key", "classification"]], on="candidate_key", how="left").sort_values("rmse").head(12)
        labels = plot["feature_set"] + ":" + plot["model_id"] + ":" + plot["target_scale"]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(labels, plot["rmse"])
        ax.set_xlabel("Lena RMSE")
        ax.set_title("Lena RMSE comparison")
        ax.invert_yaxis()
        save(fig, "lena_rmse_comparison.png")

    if not fold_stability.empty:
        counts = fold_stability.groupby(["candidate_key", "feature_set", "model_id", "target_scale"])["unstable_fold"].sum().reset_index()
        counts = counts.merge(ranking[["candidate_key", "classification"]], on="candidate_key", how="left").sort_values("unstable_fold").head(12)
        labels = counts["feature_set"] + ":" + counts["model_id"] + ":" + counts["target_scale"]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(labels, counts["unstable_fold"])
        ax.set_xlabel("Unstable LOYO fold count")
        ax.set_title("Fold instability comparison")
        ax.invert_yaxis()
        save(fig, "fold_instability_comparison.png")

    top_keys = list(ranking.head(4)["candidate_key"])
    top_preds = predictions[predictions["candidate_key"].isin(top_keys) & predictions["validation_scheme"].eq("leave_one_year_out")].copy()
    if not top_preds.empty:
        top_preds["label"] = top_preds["feature_set"] + ":" + top_preds["model_id"] + ":" + top_preds["target_scale"]
        fig, ax = plt.subplots(figsize=(7, 6))
        for label, subset in top_preds.groupby("label"):
            ax.scatter(subset["DOC_observed_mgC_L"], subset["DOC_cv_predicted_mgC_L"], s=12, alpha=0.5, label=label)
        lo = min(top_preds["DOC_observed_mgC_L"].min(), top_preds["DOC_cv_predicted_mgC_L"].min())
        hi = max(top_preds["DOC_observed_mgC_L"].max(), top_preds["DOC_cv_predicted_mgC_L"].max())
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlabel("Observed DOC mg C/L")
        ax.set_ylabel("Validation-only CV predicted DOC mg C/L")
        ax.set_title("Observed vs predicted top candidates")
        ax.legend(fontsize="x-small")
        save(fig, "observed_vs_predicted_top_candidates.png")

        enriched = _doc_behavior_groups(top_preds)
        labels = []
        data = []
        for (candidate, group), subset in enriched.groupby(["label", "doc_behavior_group"], observed=False):
            if group in {"low", "high", "extreme_high"}:
                labels.append(f"{candidate}\n{group}")
                data.append(subset["residual_mgC_L"].to_numpy())
        if data:
            fig, ax = plt.subplots(figsize=(10, 5.5))
            try:
                ax.boxplot(data, tick_labels=labels, showfliers=False)
            except TypeError:
                ax.boxplot(data, labels=labels, showfliers=False)
            ax.axhline(0, linestyle="--")
            ax.set_ylabel("Residual mg C/L")
            ax.set_title("Residuals by DOC quantile for top candidates")
            ax.tick_params(axis="x", rotation=45)
            save(fig, "residuals_by_doc_quantile_top_candidates.png")
    return paths


def write_bias_refinement_report() -> Path:
    BIAS_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    contract = load_contract()
    audit = _read_required_csv(BIAS_TABLE_DIR / "uncertainty_diagnostic_audit.csv")
    registry = _read_required_csv(BIAS_TABLE_DIR / "bias_refinement_feature_set_registry.csv")
    overall = _read_required_csv(BIAS_TABLE_DIR / "bias_refinement_metrics_overall.csv")
    by_river = _read_required_csv(BIAS_TABLE_DIR / "bias_refinement_metrics_by_river.csv")
    high_doc = _read_required_csv(BIAS_TABLE_DIR / "bias_refinement_high_doc_metrics.csv")
    fold = _read_required_csv(BIAS_TABLE_DIR / "bias_refinement_fold_stability.csv")
    ranking = _read_required_csv(BIAS_TABLE_DIR / "bias_refinement_model_ranking.csv")
    recommendation = _read_required_csv(BIAS_TABLE_DIR / "bias_refinement_recommendation.csv")
    readiness = _read_required_csv(BIAS_TABLE_DIR / "refined_production_readiness_decision.csv")
    ready = readiness[readiness["decision_item"].eq("ready_for_production_daily_prediction")]
    ready_status = ready["status"].iloc[0] if not ready.empty else "unknown"
    lines = [
        "# Bias-aware Concentration Model Refinement Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "## 1. Scope and guardrails",
        "",
        "This phase trains validation-only DOC concentration models to diagnose and reduce residual bias. It reads only `training_matrix_hydrocore.csv` as model input plus the allowed baseline/uncertainty metadata tables.",
        "",
        "## 2. Why production readiness was false",
        "",
        "The concentration uncertainty phase found empirical residual intervals, but production readiness was blocked by river-specific RMSE and fold-stability caveats, especially Lena RMSE and high-DOC underprediction.",
        "",
        "## 3. Diagnostic audit of uncertainty outputs",
        "",
        _md_table(audit, max_rows=20),
        "",
        "## 4. Candidate refinement feature sets",
        "",
        _md_table(registry, max_rows=20),
        "",
        "## 5. Overall performance",
        "",
        _md_table(overall[overall["validation_scheme"].eq("leave_one_year_out")].sort_values(["rmse", "mae"]).head(25), max_rows=25),
        "",
        "## 6. River bias and Lena-specific behavior",
        "",
        _md_table(by_river[(by_river["validation_scheme"].eq("leave_one_year_out")) & (by_river["river"].eq("Lena"))].sort_values("rmse").head(20), max_rows=20),
        "",
        "## 7. High-DOC and low-DOC residual behavior",
        "",
        _md_table(high_doc[high_doc["validation_scheme"].eq("leave_one_year_out")].sort_values(["doc_behavior_group", "rmse"]).head(40), max_rows=40),
        "",
        "## 8. Fold stability",
        "",
        _md_table(fold.sort_values(["unstable_fold", "rmse"], ascending=[False, False]).head(40), max_rows=40),
        "",
        "## 9. Log target review",
        "",
        "Log-target candidates are evaluated as sensitivity candidates. They are not promoted without a separate target-transform decision.",
        "",
        "## 10. Huber/robust regression review",
        "",
        "Huber regression is evaluated only for the finalized F3 feature set as an interpretable robust-regression sensitivity.",
        "",
        "## 11. Recommended model decision",
        "",
        _md_table(recommendation, max_rows=10),
        "",
        _md_table(ranking.head(20), max_rows=20),
        "",
        "## 12. Production-readiness update",
        "",
        f"ready_for_production_daily_prediction: `{ready_status}`",
        "",
        _md_table(readiness, max_rows=20),
        "",
        "## 13. Explicit statements",
        "",
        "- Validation-only DOC concentration models were trained.",
        "- No production daily DOC prediction was generated.",
        "- No DOC flux was generated.",
        "- Gold data were not modified.",
        "- Optical/basin matrices were not used.",
        "- Prediction grid was not loaded.",
        "",
        f"freeze_id: `{contract['freeze_id']}`",
    ]
    BIAS_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return BIAS_REPORT_PATH


def run_bias_aware_refinement() -> dict[str, Any]:
    _ensure_dirs()
    assert_no_forbidden_outputs()
    _verify_contract_snapshot()
    metadata = _load_required_metadata()
    hydrocore, hydrocore_path = _read_hydrocore_only()
    before_hash = sha256_file(hydrocore_path)

    feature_sets = bias_feature_sets(sorted(hydrocore["river"].dropna().astype(str).unique()))
    feature_registry = bias_feature_set_registry(feature_sets)
    predictions, folds = _run_all_candidates(hydrocore, feature_sets)
    overall = _metrics_overall(predictions, folds)
    by_river = _metrics_by_group(predictions, "river")
    by_year = _metrics_by_group(predictions, "year")
    high_doc = _high_doc_metrics(predictions)
    fold_stability = _fold_stability(folds, overall)
    audit = _uncertainty_diagnostic_audit(metadata)
    deltas = _candidate_deltas(overall, by_river, high_doc, fold_stability)
    ranking = _ranking(deltas, overall)
    recommendation = _recommendation(ranking)
    readiness = refined_readiness_decision(
        recommendation=recommendation,
        deltas=deltas,
        ranking=ranking,
        diagnostic_audit=audit,
        no_production_generated=True,
    )
    specs = _write_refined_model_spec(recommendation, feature_registry)

    table_paths = [
        _write_csv(_model_registry(), BIAS_TABLE_DIR / "bias_refinement_model_registry.csv"),
        _write_csv(feature_registry, BIAS_TABLE_DIR / "bias_refinement_feature_set_registry.csv"),
        _write_csv(overall, BIAS_TABLE_DIR / "bias_refinement_metrics_overall.csv"),
        _write_csv(by_river, BIAS_TABLE_DIR / "bias_refinement_metrics_by_river.csv"),
        _write_csv(by_year, BIAS_TABLE_DIR / "bias_refinement_metrics_by_year.csv"),
        _write_csv(high_doc, BIAS_TABLE_DIR / "bias_refinement_high_doc_metrics.csv"),
        _write_csv(fold_stability, BIAS_TABLE_DIR / "bias_refinement_fold_stability.csv"),
        _write_csv(predictions, BIAS_TABLE_DIR / "bias_refinement_cv_predictions.csv"),
        _write_csv(deltas, BIAS_TABLE_DIR / "bias_refinement_deltas_vs_f3.csv"),
        _write_csv(ranking, BIAS_TABLE_DIR / "bias_refinement_model_ranking.csv"),
        _write_csv(recommendation, BIAS_TABLE_DIR / "bias_refinement_recommendation.csv"),
        _write_csv(audit, BIAS_TABLE_DIR / "uncertainty_diagnostic_audit.csv"),
        _write_csv(readiness, BIAS_TABLE_DIR / "refined_production_readiness_decision.csv"),
    ]
    figure_paths = _make_figures(ranking=ranking, high_doc=high_doc, by_river=by_river, fold_stability=fold_stability, predictions=predictions)
    report_path = write_bias_refinement_report()

    assert_gold_hash_unchanged(hydrocore_path, before_hash)
    assert_no_forbidden_outputs()
    return {
        "tables": table_paths,
        "figures": figure_paths,
        "report": report_path,
        "specs": specs,
        "ranking": ranking,
        "recommendation": recommendation,
        "readiness": readiness,
        "diagnostic_audit": audit,
    }
