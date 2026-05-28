from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..gold_contract import load_contract, require_gold_data_dir, sha256_file, table_path
from ..paths import CONFIG_DIR, REPORT_DIR, TABLE_DIR, path
from ..reports import _md_table, utc_now
from .baseline_models import _prepare_hydrocore, _read_hydrocore
from .diagnostics import assert_gold_hash_unchanged
from .metrics import metric_row


FREEZE_ID = "data_freeze_gold_20260526_v1"
MODEL_SPEC_ID = "production_candidate_r4_river_specific_q_and_season_linear"
MODEL_FIT_ID = "production_candidate_r4_linear_fit_data_freeze_gold_20260526_v1"

PRODUCTION_CANDIDATE_REPORT_DIR = REPORT_DIR / "production_candidate"
PRODUCTION_CANDIDATE_TABLE_DIR = TABLE_DIR / "production_candidate"
DAILY_DOC_REPORT_DIR = REPORT_DIR / "daily_doc_prediction"
DAILY_DOC_TABLE_DIR = TABLE_DIR / "daily_doc_prediction"
DAILY_DOC_FIGURE_DIR = path("outputs", "figures", "daily_doc_prediction")
MODEL_DIR = path("outputs", "models")

PRODUCTION_SPEC_PATH = CONFIG_DIR / "model_specs" / "production_candidate_r4_river_specific_q_and_season_linear.yaml"
PRODUCTION_DECISION_PATH = PRODUCTION_CANDIDATE_TABLE_DIR / "production_candidate_model_decision.csv"
PRODUCTION_HANDOFF_PATH = PRODUCTION_CANDIDATE_REPORT_DIR / "production_candidate_handoff.md"
DAILY_DOC_REPORT_PATH = DAILY_DOC_REPORT_DIR / "daily_doc_prediction_report.md"
MODEL_ARTIFACT_PATH = MODEL_DIR / "production_candidate_r4_daily_doc_model.joblib"
MODEL_METADATA_PATH = MODEL_DIR / "production_candidate_r4_daily_doc_model_metadata.json"

BIAS_RECOMMENDATION_PATH = TABLE_DIR / "bias_refinement" / "bias_refinement_recommendation.csv"
BIAS_READINESS_PATH = TABLE_DIR / "bias_refinement" / "refined_production_readiness_decision.csv"
BIAS_CV_PREDICTIONS_PATH = TABLE_DIR / "bias_refinement" / "bias_refinement_cv_predictions.csv"
BIAS_REPORT_PATH = REPORT_DIR / "bias_refinement" / "bias_refinement_report.md"
INTERVALS_PATH = TABLE_DIR / "concentration_uncertainty" / "empirical_residual_intervals.csv"
INTERVAL_COVERAGE_PATH = TABLE_DIR / "concentration_uncertainty" / "empirical_interval_coverage.csv"
ROI_QC_REPORT_PATH = REPORT_DIR / "roi_qc" / "roi_final_qc_report.md"
ROI_QC_SUMMARY_PATH = TABLE_DIR / "roi_qc" / "roi_final_qc_summary.csv"
OPTICAL_RANKING_PATH = TABLE_DIR / "optical_sensitivity" / "optical_model_ranking.csv"

ALLOWED_METADATA_PATHS = [
    BIAS_RECOMMENDATION_PATH,
    BIAS_READINESS_PATH,
    BIAS_CV_PREDICTIONS_PATH,
    BIAS_REPORT_PATH,
    INTERVALS_PATH,
    INTERVAL_COVERAGE_PATH,
    ROI_QC_REPORT_PATH,
    ROI_QC_SUMMARY_PATH,
    OPTICAL_RANKING_PATH,
]

BASE_FEATURES = ("log_Q", "sin_doy", "cos_doy")
INTERACTION_VARIABLES = ("log_Q", "sin_doy", "cos_doy")
FORBIDDEN_PREDICTION_COLUMNS = {"daily_flux", "Mg_day", "TgC", "kg_day", "DOC_flux", "annual_flux", "snowmelt_flux"}


def _ensure_candidate_dirs() -> None:
    for directory in [PRODUCTION_CANDIDATE_REPORT_DIR, PRODUCTION_CANDIDATE_TABLE_DIR, CONFIG_DIR / "model_specs"]:
        directory.mkdir(parents=True, exist_ok=True)


def _ensure_prediction_dirs() -> None:
    for directory in [DAILY_DOC_REPORT_DIR, DAILY_DOC_TABLE_DIR, DAILY_DOC_FIGURE_DIR, MODEL_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def _write_csv(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8")
    return destination


def _read_required_csv(destination: Path) -> pd.DataFrame:
    if not destination.exists():
        raise FileNotFoundError(f"Required daily DOC prediction input is missing: {destination}")
    return pd.read_csv(destination)


def _read_required_text(destination: Path) -> str:
    if not destination.exists():
        raise FileNotFoundError(f"Required daily DOC prediction input is missing: {destination}")
    return destination.read_text(encoding="utf-8")


def _verify_contract_snapshot() -> None:
    verification_path = TABLE_DIR / "gold_table_verification.csv"
    if not verification_path.exists():
        raise FileNotFoundError("Run `python -m arctic_doc_model_rebuild.cli verify-gold-data` before daily DOC prediction.")
    verification = pd.read_csv(verification_path)
    required = {"training_matrix_hydrocore.csv", "prediction_grid_daily_hydrocore.csv"}
    status = verification[verification["table_name"].isin(required)]
    if set(status["table_name"]) != required or not status["status"].eq("ok").all():
        raise RuntimeError("Training matrix and prediction grid must be verified against the gold contract before prediction.")


def _load_metadata() -> dict[str, Any]:
    for destination in ALLOWED_METADATA_PATHS:
        if not destination.exists():
            raise FileNotFoundError(f"Required daily DOC prediction metadata is missing: {destination}")
    recommendation = _read_required_csv(BIAS_RECOMMENDATION_PATH)
    selected = recommendation[recommendation["decision_item"].eq("recommended_primary_model_after_refinement")]
    if selected.empty or selected.iloc[0]["feature_set"] != "R4_river_specific_Q_and_season" or selected.iloc[0]["model_id"] != "linear_regression":
        raise RuntimeError("Daily DOC prediction requires refined candidate R4_river_specific_Q_and_season + linear_regression.")
    readiness = _read_required_csv(BIAS_READINESS_PATH)
    ready = readiness[readiness["decision_item"].eq("ready_for_production_daily_prediction")]
    if ready.empty or ready.iloc[0]["status"] not in {"true", "true_with_caveats"}:
        raise RuntimeError("Bias refinement did not authorize a guarded daily DOC prediction phase.")
    return {
        "recommendation": recommendation,
        "readiness": readiness,
        "bias_cv_predictions": _read_required_csv(BIAS_CV_PREDICTIONS_PATH),
        "bias_report": _read_required_text(BIAS_REPORT_PATH),
        "legacy_intervals": _read_required_csv(INTERVALS_PATH),
        "legacy_interval_coverage": _read_required_csv(INTERVAL_COVERAGE_PATH),
        "roi_qc": _read_required_csv(ROI_QC_SUMMARY_PATH),
        "roi_report": _read_required_text(ROI_QC_REPORT_PATH),
        "optical_ranking": _read_required_csv(OPTICAL_RANKING_PATH),
    }


def _interaction_columns(rivers: list[str]) -> list[str]:
    return [f"{river}__x__{variable}" for variable in INTERACTION_VARIABLES for river in rivers]


def _add_interactions_for_rivers(frame: pd.DataFrame, rivers: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for river in rivers:
        mask = out["river"].astype(str).eq(river)
        for variable in INTERACTION_VARIABLES:
            out[f"{river}__x__{variable}"] = pd.to_numeric(out[variable], errors="coerce").where(mask, 0.0)
    return out


def _feature_columns(rivers: list[str]) -> tuple[list[str], list[str]]:
    numeric = [*BASE_FEATURES, *_interaction_columns(rivers)]
    categorical = ["river"]
    return numeric, categorical


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _make_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [
                        ("numeric", StandardScaler(), numeric_features),
                        ("categorical", _one_hot_encoder(), categorical_features),
                    ],
                    remainder="drop",
                ),
            ),
            ("model", LinearRegression()),
        ]
    )


def _read_training_and_grid() -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    gold_dir = require_gold_data_dir()
    train_path = table_path("training_matrix_hydrocore.csv", gold_dir=gold_dir)
    grid_path = table_path("prediction_grid_daily_hydrocore.csv", gold_dir=gold_dir)
    contract = load_contract()
    for table_name, destination in [("training_matrix_hydrocore.csv", train_path), ("prediction_grid_daily_hydrocore.csv", grid_path)]:
        expected = str(contract["expected_tables"][table_name]["sha256"]).lower()
        if sha256_file(destination) != expected:
            raise RuntimeError(f"{table_name} hash does not match frozen contract.")
    training = _prepare_hydrocore(_read_hydrocore())
    grid = pd.read_csv(grid_path, low_memory=False)
    return training, grid, train_path, grid_path


def _prepare_grid(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["year"] = pd.to_numeric(out.get("year", out["date"].dt.year), errors="coerce")
    out["month"] = out["date"].dt.month
    out["doy"] = pd.to_numeric(out.get("doy", out["date"].dt.dayofyear), errors="coerce")
    for column in ["Q_m3s", "sin_doy", "cos_doy"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["log_Q"] = np.where(out["Q_m3s"] > 0, np.log(out["Q_m3s"]), np.nan)
    return out


def _eligible_training(frame: pd.DataFrame, rivers: list[str]) -> pd.DataFrame:
    prepared = _add_interactions_for_rivers(frame, rivers)
    numeric, categorical = _feature_columns(rivers)
    required = ["label_id", "river", "date", "year", "month", "doy", "DOC_mgC_L", "Q_m3s", *numeric, *categorical]
    usable = prepared.dropna(subset=list(dict.fromkeys(required))).copy()
    usable = usable[usable["Q_m3s"] > 0].copy()
    return usable.reset_index(drop=True)


def _training_ranges(training: pd.DataFrame) -> pd.DataFrame:
    return (
        training.groupby("river", dropna=False)
        .agg(
            train_n=("label_id", "count"),
            train_log_Q_min=("log_Q", "min"),
            train_log_Q_max=("log_Q", "max"),
            train_doy_min=("doy", "min"),
            train_doy_max=("doy", "max"),
            train_year_min=("year", "min"),
            train_year_max=("year", "max"),
        )
        .reset_index()
    )


def _production_spec(rivers: list[str]) -> dict[str, Any]:
    numeric_expanded, categorical = _feature_columns(rivers)
    return {
        "model_spec_id": MODEL_SPEC_ID,
        "freeze_id": FREEZE_ID,
        "input_training_table": "data/processed/gold/training_matrix_hydrocore.csv",
        "input_prediction_grid": "data/processed/gold/prediction_grid_daily_hydrocore.csv",
        "target": "DOC_mgC_L",
        "target_transform": "none",
        "feature_set": "R4_river_specific_Q_and_season",
        "numeric_features": ["log_Q", "sin_doy", "cos_doy", "river_x_log_Q", "river_x_sin_doy", "river_x_cos_doy"],
        "expanded_numeric_features": numeric_expanded,
        "categorical_features": categorical,
        "model": {"type": "LinearRegression"},
        "validation_basis": ["leave_one_year_out", "river_year_groupkfold"],
        "caveats": [
            "within_six_arcticgro_rivers_only",
            "no_cross_river_extrapolation",
            "high_doc_bias_improved_but_caveated",
            "fold_stability_caveated",
            "empirical_intervals_from_cv_residuals",
        ],
        "excluded_features": ["optical", "basin_context", "lab_optical_cdom", "flux"],
        "production_daily_doc_prediction_allowed": True,
        "flux_allowed": False,
    }


def freeze_production_candidate() -> dict[str, Any]:
    _ensure_candidate_dirs()
    _verify_contract_snapshot()
    metadata = _load_metadata()
    training, _, train_path, _ = _read_training_and_grid()
    before_hash = sha256_file(train_path)
    rivers = sorted(training["river"].dropna().astype(str).unique())
    spec = _production_spec(rivers)
    PRODUCTION_SPEC_PATH.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    decision = pd.DataFrame(
        [
            {
                "decision_type": "production_candidate",
                "model_spec_id": MODEL_SPEC_ID,
                "feature_set": "R4_river_specific_Q_and_season",
                "model_id": "linear_regression",
                "target_scale": "raw",
                "source_decision": "bias_refinement",
                "production_daily_doc_prediction_allowed": True,
                "flux_allowed": False,
                "decision": "frozen_for_guarded_daily_doc_prediction",
                "reason": "Bias-aware refinement selected R4 river-specific Q and season interactions; readiness is true_with_caveats.",
            },
            {
                "decision_type": "flux_guardrail",
                "model_spec_id": MODEL_SPEC_ID,
                "feature_set": "not_applicable",
                "model_id": "not_applicable",
                "target_scale": "not_applicable",
                "source_decision": "phase_guardrail",
                "production_daily_doc_prediction_allowed": True,
                "flux_allowed": False,
                "decision": "flux_not_allowed_in_this_phase",
                "reason": "This phase generates DOC concentration predictions only.",
            },
        ]
    )
    decision_path = _write_csv(decision, PRODUCTION_DECISION_PATH)
    handoff_lines = [
        "# Production Candidate Handoff",
        "",
        f"Generated: {utc_now()}",
        "",
        f"- freeze_id: `{FREEZE_ID}`",
        f"- model_spec_id: `{MODEL_SPEC_ID}`",
        "- model: `R4_river_specific_Q_and_season + LinearRegression`",
        "- production daily DOC prediction: allowed in guarded mode",
        "- flux: not allowed",
        "",
        "## Inputs",
        "",
        "- `data/processed/gold/training_matrix_hydrocore.csv`",
        "- `data/processed/gold/prediction_grid_daily_hydrocore.csv`",
        "",
        "## Caveats",
        "",
        "- within six ArcticGRO rivers only",
        "- no cross-river extrapolation",
        "- high-DOC caveat carried forward",
        "- fold stability caveat carried forward",
        "- empirical intervals are derived from validation residuals",
        "- optical, basin, lab optical/CDOM, and flux features are excluded",
    ]
    PRODUCTION_HANDOFF_PATH.write_text("\n".join(handoff_lines) + "\n", encoding="utf-8")
    assert_gold_hash_unchanged(train_path, before_hash)
    return {"spec": PRODUCTION_SPEC_PATH, "decision": decision_path, "handoff": PRODUCTION_HANDOFF_PATH, "metadata": metadata}


def _coefficients(estimator: Pipeline) -> pd.DataFrame:
    preprocess = estimator.named_steps["preprocess"]
    model = estimator.named_steps["model"]
    names = preprocess.get_feature_names_out()
    rows = [{"feature": "intercept", "coefficient": float(model.intercept_)}]
    rows.extend({"feature": str(name), "coefficient": float(value)} for name, value in zip(names, model.coef_))
    return pd.DataFrame(rows)


def _refined_residual_intervals(cv_predictions: pd.DataFrame) -> pd.DataFrame:
    subset = cv_predictions[
        cv_predictions["feature_set"].eq("R4_river_specific_Q_and_season")
        & cv_predictions["model_id"].eq("linear_regression")
        & cv_predictions["target_scale"].eq("raw")
        & cv_predictions["validation_scheme"].eq("leave_one_year_out")
    ].copy()
    if subset.empty:
        raise RuntimeError("Refined R4 LOYO CV predictions are required for empirical intervals.")

    rows: list[dict[str, Any]] = []

    def add_row(scope: str, group_value: str, residuals: pd.Series) -> None:
        residuals = pd.to_numeric(residuals, errors="coerce").dropna()
        if residuals.empty:
            return
        qs = residuals.quantile([0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975])
        rows.append(
            {
                "scope": scope,
                "group_value": group_value,
                "n": int(len(residuals)),
                "interval_source": "bias_refinement_leave_one_year_out_cv_residuals",
                "is_production_prediction_interval": False,
                "q02_5": float(qs.loc[0.025]),
                "q05": float(qs.loc[0.05]),
                "q10": float(qs.loc[0.10]),
                "q25": float(qs.loc[0.25]),
                "q50": float(qs.loc[0.50]),
                "q75": float(qs.loc[0.75]),
                "q90": float(qs.loc[0.90]),
                "q95": float(qs.loc[0.95]),
                "q97_5": float(qs.loc[0.975]),
                "empirical_80pct_interval_lower": float(qs.loc[0.10]),
                "empirical_80pct_interval_upper": float(qs.loc[0.90]),
                "empirical_90pct_interval_lower": float(qs.loc[0.05]),
                "empirical_90pct_interval_upper": float(qs.loc[0.95]),
                "empirical_95pct_interval_lower": float(qs.loc[0.025]),
                "empirical_95pct_interval_upper": float(qs.loc[0.975]),
            }
        )

    add_row("overall", "overall", subset["residual_mgC_L"])
    for river, group in subset.groupby("river", dropna=False):
        if len(group) >= 30:
            add_row("river", str(river), group["residual_mgC_L"])
    return pd.DataFrame(rows)


def _prediction_intervals(predictions: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    overall = intervals[intervals["scope"].eq("overall")].iloc[0].to_dict()
    river_intervals = intervals[intervals["scope"].eq("river")].set_index("group_value").to_dict("index")
    lower_any = pd.Series(False, index=out.index)
    for idx, row in out.iterrows():
        source = river_intervals.get(str(row["river"]), overall)
        out.at[idx, "interval_source_scope"] = "river" if str(row["river"]) in river_intervals else "overall"
        out.at[idx, "interval_source_n"] = int(source["n"])
        point = row["DOC_predicted_mgC_L"]
        if pd.isna(point):
            for label in ["80", "90", "95"]:
                out.at[idx, f"DOC_prediction_interval_{label}_lower"] = np.nan
                out.at[idx, f"DOC_prediction_interval_{label}_upper"] = np.nan
            continue
        for label, lower_col, upper_col in [
            ("80", "empirical_80pct_interval_lower", "empirical_80pct_interval_upper"),
            ("90", "empirical_90pct_interval_lower", "empirical_90pct_interval_upper"),
            ("95", "empirical_95pct_interval_lower", "empirical_95pct_interval_upper"),
        ]:
            lower_raw = float(point) + float(source[lower_col])
            upper_raw = float(point) + float(source[upper_col])
            out.at[idx, f"DOC_prediction_interval_{label}_lower_raw"] = lower_raw
            out.at[idx, f"DOC_prediction_interval_{label}_upper_raw"] = upper_raw
            out.at[idx, f"DOC_prediction_interval_{label}_lower"] = max(lower_raw, 0.0)
            out.at[idx, f"DOC_prediction_interval_{label}_upper"] = max(upper_raw, 0.0)
            lower_any.at[idx] = bool(lower_any.at[idx] or lower_raw < 0)
    out["lower_interval_clipped_at_zero"] = lower_any
    return out


def _make_predictions(training: pd.DataFrame, grid: pd.DataFrame) -> tuple[pd.DataFrame, Pipeline, pd.DataFrame, pd.DataFrame]:
    rivers = sorted(training["river"].dropna().astype(str).unique())
    numeric, categorical = _feature_columns(rivers)
    training_eligible = _eligible_training(training, rivers)
    estimator = _make_pipeline(numeric, categorical)
    estimator.fit(training_eligible[numeric + categorical], training_eligible["DOC_mgC_L"])

    prepared_grid = _add_interactions_for_rivers(_prepare_grid(grid), rivers)
    known_rivers = set(rivers)
    required = ["river", "date", "year", "month", "doy", "Q_m3s", "log_Q", "sin_doy", "cos_doy", *numeric, *categorical]
    range_table = _training_ranges(training_eligible)
    range_lookup = range_table.set_index("river").to_dict("index")

    rows: list[dict[str, Any]] = []
    for source_index, row in prepared_grid.iterrows():
        reasons = []
        if str(row.get("river", "")) not in known_rivers:
            reasons.append("unknown_river")
        for column in ["date", "Q_m3s", "log_Q", "sin_doy", "cos_doy", "doy"]:
            value = row.get(column, np.nan)
            if pd.isna(value):
                reasons.append(f"missing_{column}")
        if pd.notna(row.get("Q_m3s", np.nan)) and float(row["Q_m3s"]) <= 0:
            reasons.append("Q_m3s_nonpositive")
        missing_features = [column for column in required if column in prepared_grid.columns and pd.isna(row.get(column, np.nan))]
        for column in missing_features:
            if not any(column in reason for reason in reasons):
                reasons.append(f"missing_{column}")

        prediction_status = "predicted" if not reasons else "missing_predictor"
        raw_prediction = np.nan
        clipped_prediction = np.nan
        point_clipped = False
        if prediction_status == "predicted":
            feature_row = pd.DataFrame([row[numeric + categorical].to_dict()])
            raw_prediction = float(estimator.predict(feature_row)[0])
            clipped_prediction = max(raw_prediction, 0.0)
            point_clipped = raw_prediction < 0
        river = str(row.get("river", ""))
        train_range = range_lookup.get(river, {})
        outside_logq = bool(
            prediction_status == "predicted"
            and (float(row["log_Q"]) < float(train_range.get("train_log_Q_min", np.nan)) or float(row["log_Q"]) > float(train_range.get("train_log_Q_max", np.nan)))
        )
        outside_doy = bool(
            prediction_status == "predicted"
            and (float(row["doy"]) < float(train_range.get("train_doy_min", np.nan)) or float(row["doy"]) > float(train_range.get("train_doy_max", np.nan)))
        )
        outside_year = bool(
            prediction_status == "predicted"
            and (float(row["year"]) < float(train_range.get("train_year_min", np.nan)) or float(row["year"]) > float(train_range.get("train_year_max", np.nan)))
        )
        date_text = pd.Timestamp(row["date"]).date().isoformat() if pd.notna(row.get("date", pd.NaT)) else ""
        rows.append(
            {
                "prediction_id": f"{MODEL_SPEC_ID}_{river}_{date_text}",
                "river": river,
                "date": date_text,
                "year": int(row["year"]) if pd.notna(row.get("year", np.nan)) else "",
                "month": int(row["month"]) if pd.notna(row.get("month", np.nan)) else "",
                "doy": int(row["doy"]) if pd.notna(row.get("doy", np.nan)) else "",
                "Q_m3s": row.get("Q_m3s", np.nan),
                "log_Q": row.get("log_Q", np.nan),
                "DOC_predicted_raw_mgC_L": raw_prediction,
                "DOC_predicted_mgC_L": clipped_prediction,
                "point_prediction_clipped_at_zero": point_clipped,
                "prediction_status": prediction_status,
                "missing_predictor_reason": ";".join(sorted(set(reasons))),
                "outside_training_logQ_range": outside_logq,
                "outside_training_doy_range": outside_doy,
                "outside_training_year_range": outside_year,
                "model_spec_id": MODEL_SPEC_ID,
                "model_fit_id": MODEL_FIT_ID,
                "freeze_id": FREEZE_ID,
                "is_production_daily_doc_prediction": True,
                "is_flux": False,
                "notes": "guarded_daily_doc_concentration_prediction_not_flux",
                "source_grid_row_index": int(source_index),
            }
        )
    return pd.DataFrame(rows), estimator, training_eligible, range_table


def _fit_summary(training_eligible: pd.DataFrame, estimator: Pipeline, train_path: Path, grid_path: Path) -> pd.DataFrame:
    pred = estimator.predict(training_eligible[_feature_columns(sorted(training_eligible["river"].astype(str).unique()))[0] + ["river"]])
    metrics = metric_row(training_eligible["DOC_mgC_L"], pred)
    return pd.DataFrame(
        [
            {
                "model_fit_id": MODEL_FIT_ID,
                "model_spec_id": MODEL_SPEC_ID,
                "freeze_id": FREEZE_ID,
                "training_rows_used": len(training_eligible),
                "training_rivers": ";".join(sorted(training_eligible["river"].astype(str).unique())),
                "training_date_min": pd.Timestamp(training_eligible["date"].min()).date().isoformat(),
                "training_date_max": pd.Timestamp(training_eligible["date"].max()).date().isoformat(),
                "training_table_sha256": sha256_file(train_path),
                "prediction_grid_sha256": sha256_file(grid_path),
                "in_sample_rmse_diagnostic": metrics["rmse"],
                "in_sample_mae_diagnostic": metrics["mae"],
                "production_candidate_not_flux_model": True,
                "flux_allowed": False,
            }
        ]
    )


def _qc_tables(predictions: pd.DataFrame, training_eligible: pd.DataFrame, intervals: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train_doc = pd.to_numeric(training_eligible["DOC_mgC_L"], errors="coerce")
    max_train = float(train_doc.max())
    thresholds = {
        "train_DOC_min": float(train_doc.min()),
        "train_DOC_max": max_train,
        "train_DOC_p01": float(train_doc.quantile(0.01)),
        "train_DOC_p99": float(train_doc.quantile(0.99)),
        "high_doc_threshold_1_5x_train_max": max_train * 1.5,
        "hard_high_doc_threshold": 30.0,
    }
    summary = (
        predictions.groupby("river", dropna=False)
        .agg(
            n_grid_rows=("prediction_id", "count"),
            n_predicted_rows=("prediction_status", lambda x: int((x == "predicted").sum())),
            date_min=("date", "min"),
            date_max=("date", "max"),
            n_years=("year", lambda x: int(pd.Series(x).replace("", np.nan).dropna().nunique())),
            outside_training_logQ_rows=("outside_training_logQ_range", "sum"),
            outside_training_doy_rows=("outside_training_doy_range", "sum"),
            outside_training_year_rows=("outside_training_year_range", "sum"),
        )
        .reset_index()
    )
    summary["prediction_coverage_rate"] = summary["n_predicted_rows"] / summary["n_grid_rows"]

    by_river_year = (
        predictions.groupby(["river", "year"], dropna=False)
        .agg(
            n_grid_rows=("prediction_id", "count"),
            n_predicted_rows=("prediction_status", lambda x: int((x == "predicted").sum())),
            doc_predicted_mean=("DOC_predicted_mgC_L", "mean"),
            doc_predicted_min=("DOC_predicted_mgC_L", "min"),
            doc_predicted_max=("DOC_predicted_mgC_L", "max"),
            interval_95_width_mean=("DOC_prediction_interval_95_upper", lambda x: np.nan),
        )
        .reset_index()
    )
    width = predictions["DOC_prediction_interval_95_upper"] - predictions["DOC_prediction_interval_95_lower"]
    predictions = predictions.assign(interval_95_width=width)
    by_river_year = (
        predictions.groupby(["river", "year"], dropna=False)
        .agg(
            n_grid_rows=("prediction_id", "count"),
            n_predicted_rows=("prediction_status", lambda x: int((x == "predicted").sum())),
            doc_predicted_mean=("DOC_predicted_mgC_L", "mean"),
            doc_predicted_min=("DOC_predicted_mgC_L", "min"),
            doc_predicted_max=("DOC_predicted_mgC_L", "max"),
            interval_95_width_mean=("interval_95_width", "mean"),
        )
        .reset_index()
    )

    range_rows = []
    pred = predictions[predictions["prediction_status"].eq("predicted")].copy()
    flag_specs = [
        ("point_prediction_raw_lt_0", pred["DOC_predicted_raw_mgC_L"] < 0),
        ("point_prediction_gt_train_max_1_5x", pred["DOC_predicted_mgC_L"] > thresholds["high_doc_threshold_1_5x_train_max"]),
        ("point_prediction_gt_30", pred["DOC_predicted_mgC_L"] > thresholds["hard_high_doc_threshold"]),
        ("interval_lower_lt_0_before_clipping", pred["DOC_prediction_interval_95_lower_raw"] < 0),
        ("interval_width_95_gt_15", (pred["DOC_prediction_interval_95_upper"] - pred["DOC_prediction_interval_95_lower"]) > 15.0),
        ("outside_training_logQ_range", pred["outside_training_logQ_range"].astype(bool)),
        ("outside_training_doy_range", pred["outside_training_doy_range"].astype(bool)),
        ("outside_training_year_range", pred["outside_training_year_range"].astype(bool)),
    ]
    for flag_name, mask in flag_specs:
        subset = pred[mask.fillna(False)].copy()
        if subset.empty:
            range_rows.append({"flag_type": flag_name, "n_rows": 0, "example_prediction_ids": "", **thresholds})
        else:
            range_rows.append(
                {
                    "flag_type": flag_name,
                    "n_rows": len(subset),
                    "example_prediction_ids": ";".join(subset["prediction_id"].head(5).astype(str)),
                    **thresholds,
                }
            )
    range_flags = pd.DataFrame(range_rows)
    missing = predictions[predictions["prediction_status"].ne("predicted")].copy()
    interval_summary = intervals.copy()
    interval_summary["used_for_daily_doc_prediction"] = True
    return {
        "daily_doc_prediction_qc_summary": summary,
        "daily_doc_prediction_by_river_year": by_river_year,
        "daily_doc_prediction_range_flags": range_flags,
        "daily_doc_prediction_missing_predictor_rows": missing,
        "daily_doc_prediction_interval_summary": interval_summary,
    }


def _flux_readiness(qc: dict[str, pd.DataFrame], predictions: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    summary = qc["daily_doc_prediction_qc_summary"]
    range_flags = qc["daily_doc_prediction_range_flags"]
    coverage_min = float(summary["prediction_coverage_rate"].min()) if not summary.empty else 0.0
    severe_range_count = int(range_flags[range_flags["flag_type"].isin(["point_prediction_gt_30", "point_prediction_gt_train_max_1_5x"])]["n_rows"].sum()) if not range_flags.empty else 0
    extrap_count = int(range_flags[range_flags["flag_type"].str.contains("outside_training")]["n_rows"].sum()) if not range_flags.empty else 0
    rows = [
        {
            "decision_item": "daily_doc_prediction_generated",
            "status": "true" if predictions["prediction_status"].eq("predicted").any() else "false",
            "evidence": f"predicted_rows={int(predictions['prediction_status'].eq('predicted').sum())}",
            "recommendation": "Use only as DOC concentration input to a separate flux phase.",
            "blocking_for_flux": not predictions["prediction_status"].eq("predicted").any(),
        },
        {
            "decision_item": "prediction_coverage_acceptable",
            "status": "true" if coverage_min >= 0.99 else "true_with_caveats" if coverage_min >= 0.95 else "false",
            "evidence": f"minimum_river_coverage={coverage_min:.4f}",
            "recommendation": "Review missing predictor rows before flux." if coverage_min < 0.99 else "Coverage is complete or near-complete.",
            "blocking_for_flux": coverage_min < 0.95,
        },
        {
            "decision_item": "range_flags_acceptable",
            "status": "true" if severe_range_count == 0 else "true_with_caveats",
            "evidence": f"severe_range_flag_rows={severe_range_count}",
            "recommendation": "Review high DOC range flags before flux." if severe_range_count else "No severe range flags.",
            "blocking_for_flux": False,
        },
        {
            "decision_item": "extrapolation_flags_acceptable",
            "status": "true" if extrap_count == 0 else "true_with_caveats",
            "evidence": f"extrapolation_flag_rows={extrap_count}",
            "recommendation": "Carry extrapolation flags into flux uncertainty." if extrap_count else "No extrapolation flags.",
            "blocking_for_flux": False,
        },
        {
            "decision_item": "intervals_available",
            "status": "true" if not intervals.empty else "false",
            "evidence": f"interval_rows={len(intervals)}",
            "recommendation": "Use empirical residual intervals as concentration uncertainty input.",
            "blocking_for_flux": intervals.empty,
        },
    ]
    blockers = [row for row in rows if row["blocking_for_flux"]]
    if blockers:
        ready = "false"
        rec = "Resolve blocking prediction QC issues before flux calculation."
    elif severe_range_count or extrap_count or coverage_min < 0.99:
        ready = "true_with_caveats"
        rec = "Proceed to flux only with concentration caveats and uncertainty intervals."
    else:
        ready = "true"
        rec = "Proceed to a separate flux calculation phase."
    rows.append(
        {
            "decision_item": "ready_for_flux_calculation",
            "status": ready,
            "evidence": "No flux was calculated in this phase.",
            "recommendation": rec,
            "blocking_for_flux": ready == "false",
        }
    )
    return pd.DataFrame(rows)


def _save_model_artifact(estimator: Pipeline, metadata: dict[str, Any]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, MODEL_ARTIFACT_PATH)
    metadata = {
        **metadata,
        "model_artifact": "outputs/models/production_candidate_r4_daily_doc_model.joblib",
        "model_artifact_sha256": sha256_file(MODEL_ARTIFACT_PATH),
        "production_candidate_not_flux_model": True,
        "flux_allowed": False,
        "created_at_utc": utc_now(),
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _make_figures(predictions: pd.DataFrame, qc: dict[str, pd.DataFrame]) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    DAILY_DOC_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    pred = predictions[predictions["prediction_status"].eq("predicted")].copy()
    if pred.empty:
        return []
    pred["date"] = pd.to_datetime(pred["date"], errors="coerce")

    def save(fig, name: str) -> None:
        fig.tight_layout()
        destination = DAILY_DOC_FIGURE_DIR / name
        fig.savefig(destination, dpi=120)
        plt.close(fig)
        paths.append(destination)

    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=False)
    for ax, (river, subset) in zip(axes.ravel(), pred.groupby("river")):
        ax.plot(subset["date"], subset["DOC_predicted_mgC_L"], linewidth=0.8)
        ax.set_title(str(river))
        ax.set_ylabel("DOC mg C/L")
    save(fig, "daily_doc_prediction_timeseries_by_river.png")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    pred["interval_95_width"] = pred["DOC_prediction_interval_95_upper"] - pred["DOC_prediction_interval_95_lower"]
    data = [subset["interval_95_width"].dropna().to_numpy() for _, subset in pred.groupby("river")]
    labels = [str(river) for river, _ in pred.groupby("river")]
    if data:
        try:
            ax.boxplot(data, tick_labels=labels, showfliers=False)
        except TypeError:
            ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_ylabel("95% interval width")
        ax.set_title("Daily DOC interval width by river")
        save(fig, "daily_doc_prediction_interval_width_by_river.png")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    month_data = [subset["DOC_predicted_mgC_L"].dropna().to_numpy() for _, subset in pred.groupby("month")]
    month_labels = [str(month) for month, _ in pred.groupby("month")]
    if month_data:
        try:
            ax.boxplot(month_data, tick_labels=month_labels, showfliers=False)
        except TypeError:
            ax.boxplot(month_data, labels=month_labels, showfliers=False)
        ax.set_xlabel("Month")
        ax.set_ylabel("Predicted DOC mg C/L")
        ax.set_title("Daily DOC prediction by month")
        save(fig, "daily_doc_prediction_by_month_boxplot.png")

    coverage = qc["daily_doc_prediction_by_river_year"]
    if not coverage.empty:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        yearly = coverage.groupby("year")["n_predicted_rows"].sum().reset_index()
        ax.bar(yearly["year"].astype(str), yearly["n_predicted_rows"])
        ax.set_xlabel("Year")
        ax.set_ylabel("Predicted daily rows")
        ax.set_title("Prediction grid coverage by year")
        ax.tick_params(axis="x", rotation=45)
        save(fig, "prediction_grid_coverage_by_year.png")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for river, subset in pred.groupby("river"):
        sample = subset.sample(min(len(subset), 3000), random_state=31415)
        ax.scatter(sample["Q_m3s"], sample["DOC_predicted_mgC_L"], s=4, alpha=0.35, label=str(river))
    ax.set_xscale("log")
    ax.set_xlabel("Q m3/s")
    ax.set_ylabel("Predicted DOC mg C/L")
    ax.set_title("Predicted DOC vs Q by river")
    ax.legend(fontsize="x-small")
    save(fig, "predicted_doc_vs_q_by_river.png")
    return paths


def write_daily_doc_prediction_report() -> Path:
    DAILY_DOC_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    spec = yaml.safe_load(PRODUCTION_SPEC_PATH.read_text(encoding="utf-8"))
    fit_summary = _read_required_csv(PRODUCTION_CANDIDATE_TABLE_DIR / "production_model_fit_summary.csv")
    qc_summary = _read_required_csv(DAILY_DOC_TABLE_DIR / "daily_doc_prediction_qc_summary.csv")
    range_flags = _read_required_csv(DAILY_DOC_TABLE_DIR / "daily_doc_prediction_range_flags.csv")
    interval_summary = _read_required_csv(DAILY_DOC_TABLE_DIR / "daily_doc_prediction_interval_summary.csv")
    flux_ready = _read_required_csv(DAILY_DOC_TABLE_DIR / "flux_readiness_decision.csv")
    prediction_path = DAILY_DOC_TABLE_DIR / "daily_doc_prediction.csv"
    prediction_hash = sha256_file(prediction_path) if prediction_path.exists() else ""
    ready = flux_ready[flux_ready["decision_item"].eq("ready_for_flux_calculation")]
    ready_status = ready["status"].iloc[0] if not ready.empty else "unknown"
    lines = [
        "# Daily DOC Prediction Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "## 1. Scope and guardrails",
        "",
        "This phase fits the frozen refined R4 DOC concentration candidate and generates guarded daily DOC concentration predictions. It does not calculate DOC flux.",
        "",
        "## 2. Model spec and freeze ID",
        "",
        f"- freeze_id: `{spec['freeze_id']}`",
        f"- model_spec_id: `{spec['model_spec_id']}`",
        f"- prediction_output_sha256: `{prediction_hash}`",
        "",
        "## 3. Training rows used",
        "",
        _md_table(fit_summary, max_rows=10),
        "",
        "## 4. Prediction grid coverage",
        "",
        _md_table(qc_summary, max_rows=20),
        "",
        "## 5. Daily DOC prediction output",
        "",
        f"- path: `{prediction_path}`",
        "- output is DOC concentration only.",
        "",
        "## 6. Prediction intervals",
        "",
        "Intervals are empirical validation residual intervals attached to the daily predictions; they are not flux uncertainty estimates.",
        "",
        _md_table(interval_summary, max_rows=20),
        "",
        "## 7. Extrapolation flags",
        "",
        _md_table(range_flags[range_flags["flag_type"].astype(str).str.contains("outside_training")], max_rows=20),
        "",
        "## 8. Range flags",
        "",
        _md_table(range_flags[~range_flags["flag_type"].astype(str).str.contains("outside_training")], max_rows=20),
        "",
        "## 9. River/year coverage",
        "",
        "See `outputs/tables/daily_doc_prediction/daily_doc_prediction_by_river_year.csv`.",
        "",
        "## 10. Caveats carried forward",
        "",
        "- within_six_arcticgro_rivers_only",
        "- no_cross_river_extrapolation",
        "- fold_stability_caveated",
        "- high_doc_bias_caveated",
        "- ROI caveat is not directly relevant to the hydrocore prediction path",
        "- optical excluded",
        "",
        "## 11. Readiness for flux phase",
        "",
        f"ready_for_flux_calculation: `{ready_status}`",
        "",
        _md_table(flux_ready, max_rows=20),
        "",
        "## 12. Explicit statements",
        "",
        "- Production daily DOC prediction was generated.",
        "- No DOC flux was generated.",
        "- Gold data were not modified.",
        "- Optical/basin/lab features were not used.",
    ]
    DAILY_DOC_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return DAILY_DOC_REPORT_PATH


def run_daily_doc_prediction() -> dict[str, Any]:
    _ensure_candidate_dirs()
    _ensure_prediction_dirs()
    _verify_contract_snapshot()
    candidate = freeze_production_candidate()
    metadata = _load_metadata()
    training, grid, train_path, grid_path = _read_training_and_grid()
    train_hash = sha256_file(train_path)
    grid_hash = sha256_file(grid_path)

    predictions, estimator, training_eligible, training_ranges = _make_predictions(training, grid)
    intervals = _refined_residual_intervals(metadata["bias_cv_predictions"])
    predictions = _prediction_intervals(predictions, intervals)
    if FORBIDDEN_PREDICTION_COLUMNS.intersection(predictions.columns):
        raise RuntimeError("Daily DOC prediction output contains forbidden flux columns.")

    qc = _qc_tables(predictions, training_eligible, intervals)
    flux_readiness = _flux_readiness(qc, predictions, intervals)
    coefficients = _coefficients(estimator)
    fit_summary = _fit_summary(training_eligible, estimator, train_path, grid_path)

    training_rows = training_eligible[
        ["label_id", "river", "date", "year", "month", "doy", "DOC_mgC_L", "Q_m3s", "log_Q"]
    ].copy()
    training_rows["used_for_production_candidate_fit"] = True
    training_rows["model_fit_id"] = MODEL_FIT_ID

    table_paths = [
        _write_csv(training_rows, PRODUCTION_CANDIDATE_TABLE_DIR / "production_training_rows_used.csv"),
        _write_csv(coefficients, PRODUCTION_CANDIDATE_TABLE_DIR / "production_model_coefficients.csv"),
        _write_csv(fit_summary, PRODUCTION_CANDIDATE_TABLE_DIR / "production_model_fit_summary.csv"),
        _write_csv(predictions, DAILY_DOC_TABLE_DIR / "daily_doc_prediction.csv"),
        _write_csv(qc["daily_doc_prediction_qc_summary"], DAILY_DOC_TABLE_DIR / "daily_doc_prediction_qc_summary.csv"),
        _write_csv(qc["daily_doc_prediction_by_river_year"], DAILY_DOC_TABLE_DIR / "daily_doc_prediction_by_river_year.csv"),
        _write_csv(qc["daily_doc_prediction_range_flags"], DAILY_DOC_TABLE_DIR / "daily_doc_prediction_range_flags.csv"),
        _write_csv(qc["daily_doc_prediction_missing_predictor_rows"], DAILY_DOC_TABLE_DIR / "daily_doc_prediction_missing_predictor_rows.csv"),
        _write_csv(qc["daily_doc_prediction_interval_summary"], DAILY_DOC_TABLE_DIR / "daily_doc_prediction_interval_summary.csv"),
        _write_csv(flux_readiness, DAILY_DOC_TABLE_DIR / "flux_readiness_decision.csv"),
        _write_csv(training_ranges, PRODUCTION_CANDIDATE_TABLE_DIR / "production_training_ranges_by_river.csv"),
    ]
    metadata_json = {
        "model_spec_id": MODEL_SPEC_ID,
        "model_fit_id": MODEL_FIT_ID,
        "freeze_id": FREEZE_ID,
        "training_rows_used": int(len(training_eligible)),
        "prediction_rows_total": int(len(predictions)),
        "prediction_rows_generated": int(predictions["prediction_status"].eq("predicted").sum()),
        "training_table": "data/processed/gold/training_matrix_hydrocore.csv",
        "prediction_grid": "data/processed/gold/prediction_grid_daily_hydrocore.csv",
        "production_candidate_not_flux_model": True,
        "is_flux_model": False,
    }
    _save_model_artifact(estimator, metadata_json)
    figure_paths = _make_figures(predictions, qc)
    report_path = write_daily_doc_prediction_report()
    assert_gold_hash_unchanged(train_path, train_hash)
    assert_gold_hash_unchanged(grid_path, grid_hash)
    return {
        "candidate": candidate,
        "tables": table_paths,
        "figures": figure_paths,
        "report": report_path,
        "predictions": predictions,
        "fit_summary": fit_summary,
        "flux_readiness": flux_readiness,
        "model_artifact": MODEL_ARTIFACT_PATH,
        "model_metadata": MODEL_METADATA_PATH,
    }
