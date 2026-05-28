from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..gold_contract import load_contract, require_gold_data_dir, sha256_file, table_path
from ..paths import CONFIG_DIR, REPORT_DIR, TABLE_DIR, path
from ..reports import _md_table, utc_now
from .diagnostics import assert_gold_hash_unchanged, assert_no_forbidden_outputs


BASELINE_FINAL_TABLE_DIR = TABLE_DIR / "baseline_final"
BASELINE_FINAL_REPORT_DIR = REPORT_DIR / "baseline_final"
BASELINE_FINAL_FIGURE_DIR = path("outputs", "figures", "baseline_final")
MODEL_SPEC_DIR = CONFIG_DIR / "model_specs"

BASELINE_MODEL_DECISION_PATH = BASELINE_FINAL_TABLE_DIR / "baseline_model_decision.csv"
LOG_TARGET_BIAS_REVIEW_PATH = BASELINE_FINAL_TABLE_DIR / "log_target_bias_review.csv"
BASELINE_FINAL_REPORT_PATH = BASELINE_FINAL_REPORT_DIR / "baseline_final_report.md"
NEXT_PHASE_HANDOFF_PATH = BASELINE_FINAL_REPORT_DIR / "next_phase_handoff.md"

REFINEMENT_TABLE_DIR = TABLE_DIR / "baseline_refinement"
BASELINE_TABLE_DIR = TABLE_DIR / "baseline"

PRIMARY_SPEC_PATH = MODEL_SPEC_DIR / "primary_baseline_f3_q_season_river_fixed_ridge_alpha_1.yaml"
HYDROCLIMATE_SPEC_PATH = MODEL_SPEC_DIR / "hydroclimate_extension_f6_ridge_alpha_1.yaml"
LOG_TARGET_SPEC_PATH = MODEL_SPEC_DIR / "log_target_sensitivity_f6_ridge_alpha_1.yaml"


DECISION_ROWS: list[dict[str, Any]] = [
    {
        "decision_name": "primary_baseline",
        "decision_type": "primary_baseline",
        "feature_set": "F3_q_season_river_fixed",
        "model_id": "ridge_alpha_1",
        "target_scale": "raw",
        "validation_basis": "same_sample_leave_one_year_out",
        "rmse": 2.2420822729715915,
        "mae": 1.668080954460471,
        "decision": "selected",
        "reason": (
            "Simpler Q+season+river model; F6 hydroclimate extension improved RMSE by only "
            "0.0013 mg C/L and MAE by only 0.0032 mg C/L on same sample."
        ),
    },
    {
        "decision_name": "hydroclimate_extension",
        "decision_type": "hydroclimate_extension",
        "feature_set": "F6_reduced_hydroclimate_river_fixed",
        "model_id": "ridge_alpha_1",
        "target_scale": "raw",
        "validation_basis": "same_sample_leave_one_year_out",
        "rmse": 2.2407770032345224,
        "mae": 1.6649203993655768,
        "decision": "extension_not_primary",
        "reason": "Hydroclimate adds negligible same-sample improvement over F3; retain as process sensitivity / extension.",
    },
    {
        "decision_name": "log_target_sensitivity",
        "decision_type": "log_target_sensitivity",
        "feature_set": "F6_reduced_hydroclimate_river_fixed",
        "model_id": "ridge_alpha_1",
        "target_scale": "log",
        "validation_basis": "log_target_leave_one_year_out",
        "rmse": 2.2159252904772724,
        "mae": 1.56283545754769,
        "decision": "sensitivity_candidate_only",
        "reason": (
            "Improves RMSE/MAE but requires residual-bias review before any promotion; "
            "not selected as primary baseline now."
        ),
    },
    {
        "decision_name": "leave_one_river_out",
        "decision_type": "leave_one_river_out_stress",
        "feature_set": "any",
        "model_id": "any",
        "target_scale": "raw",
        "validation_basis": "leave_one_river_out",
        "rmse": "",
        "mae": "",
        "decision": "stress_test_only",
        "reason": (
            "Only six rivers and unseen river fixed-effect categories; do not use LORO winner "
            "for primary model selection."
        ),
    },
]

KEY_BIAS_MODELS = [
    {
        "model_role": "primary_baseline",
        "feature_set": "F3_q_season_river_fixed",
        "model_id": "ridge_alpha_1",
        "target_scale": "raw",
    },
    {
        "model_role": "hydroclimate_extension",
        "feature_set": "F6_reduced_hydroclimate_river_fixed",
        "model_id": "ridge_alpha_1",
        "target_scale": "raw",
    },
    {
        "model_role": "log_target_sensitivity",
        "feature_set": "F6_reduced_hydroclimate_river_fixed",
        "model_id": "ridge_alpha_1",
        "target_scale": "log",
    },
]


def _ensure_dirs() -> None:
    for directory in [BASELINE_FINAL_TABLE_DIR, BASELINE_FINAL_REPORT_DIR, BASELINE_FINAL_FIGURE_DIR, MODEL_SPEC_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def _write_csv(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8")
    return destination


def _read_required_csv(path_: Path) -> pd.DataFrame:
    if not path_.exists():
        raise FileNotFoundError(f"Required baseline finalization input is missing: {path_}")
    return pd.read_csv(path_)


def _hash_all_gold_tables() -> dict[str, str]:
    contract = load_contract()
    gold_dir = require_gold_data_dir()
    hashes: dict[str, str] = {}
    for table_name, spec in contract.get("expected_tables", {}).items():
        destination = table_path(table_name, gold_dir=gold_dir)
        if not destination.exists():
            raise FileNotFoundError(f"Missing expected frozen gold table: {destination}")
        actual_hash = sha256_file(destination)
        expected_hash = str(spec.get("sha256", "")).lower()
        if actual_hash != expected_hash:
            raise RuntimeError(f"Frozen gold table hash mismatch for {table_name}.")
        hashes[table_name] = actual_hash
    return hashes


def _verify_gold_contract_snapshot() -> pd.DataFrame:
    contract = load_contract()
    verification_path = TABLE_DIR / "gold_table_verification.csv"
    if not verification_path.exists():
        raise FileNotFoundError("Run `python -m arctic_doc_model_rebuild.cli verify-gold-data` before finalizing baseline.")
    verification = _read_required_csv(verification_path)
    expected_tables = set(contract.get("expected_tables", {}))
    verified_tables = set(verification.get("table_name", pd.Series(dtype=str)).astype(str))
    missing_from_snapshot = sorted(expected_tables - verified_tables)
    if missing_from_snapshot:
        raise RuntimeError(f"Gold verification snapshot is incomplete: {missing_from_snapshot}")
    expected_rows = verification[verification["table_name"].isin(expected_tables)].copy()
    if len(expected_rows) != len(expected_tables) or not expected_rows["status"].eq("ok").all():
        raise RuntimeError("Gold data contract snapshot is not fully passing; rerun verify-gold-data.")

    gold_dir = require_gold_data_dir()
    hydrocore_path = table_path("training_matrix_hydrocore.csv", gold_dir=gold_dir)
    hydro_spec = contract["expected_tables"]["training_matrix_hydrocore.csv"]
    if sha256_file(hydrocore_path) != str(hydro_spec["sha256"]).lower():
        raise RuntimeError("training_matrix_hydrocore.csv hash no longer matches the frozen contract.")
    actual_rows = len(pd.read_csv(hydrocore_path, usecols=[0], low_memory=False))
    if actual_rows != int(hydro_spec["row_count"]):
        raise RuntimeError("training_matrix_hydrocore.csv row count no longer matches the frozen contract.")
    return expected_rows


def _decision_table() -> pd.DataFrame:
    recommendation = _read_required_csv(REFINEMENT_TABLE_DIR / "refined_model_recommendation.csv")
    required_types = {
        "recommended_primary_baseline",
        "recommended_hydroclimate_extension",
        "recommended_sensitivity_only",
        "not_recommended",
    }
    if not required_types.issubset(set(recommendation["recommendation_type"])):
        raise RuntimeError("Refined model recommendation table is incomplete.")
    return pd.DataFrame(DECISION_ROWS)


def _model_specs() -> dict[Path, dict[str, Any]]:
    common = {
        "freeze_id": "data_freeze_gold_20260526_v1",
        "input_table": "data/processed/gold/training_matrix_hydrocore.csv",
        "validation_primary": "leave_one_year_out",
        "validation_secondary": ["river_year_groupkfold"],
        "validation_stress": ["leave_one_river_out"],
        "production_prediction_allowed": False,
        "flux_allowed": False,
    }
    primary = {
        "model_spec_id": "primary_baseline_f3_q_season_river_fixed_ridge_alpha_1",
        **common,
        "target": "DOC_mgC_L",
        "target_transform": "none",
        "feature_set": "F3_q_season_river_fixed",
        "numeric_features": ["log_Q", "sin_doy", "cos_doy"],
        "categorical_features": ["river"],
        "model": {"type": "Ridge", "alpha": 1.0},
        "selected_as": "primary_baseline",
        "notes": [
            "Selected because F6 hydroclimate extension gives negligible same-sample improvement over F3.",
            "River fixed effects mean this is a within-six-river model, not arbitrary river extrapolation.",
        ],
    }
    hydroclimate = {
        "model_spec_id": "hydroclimate_extension_f6_ridge_alpha_1",
        **common,
        "target": "DOC_mgC_L",
        "target_transform": "none",
        "feature_set": "F6_reduced_hydroclimate_river_fixed",
        "numeric_features": [
            "log_Q",
            "sin_doy",
            "cos_doy",
            "temperature_2m_C",
            "positive_degree_day_Cday",
            "surface_runoff_m",
        ],
        "categorical_features": ["river"],
        "model": {"type": "Ridge", "alpha": 1.0},
        "selected_as": "hydroclimate_extension",
        "notes": ["Process extension only; not primary because incremental improvement over F3 is negligible."],
    }
    log_target = {
        "model_spec_id": "log_target_sensitivity_f6_ridge_alpha_1",
        **common,
        "target": "DOC_mgC_L",
        "target_transform": "log",
        "inverse_transform": "exp",
        "feature_set": "F6_reduced_hydroclimate_river_fixed",
        "numeric_features": hydroclimate["numeric_features"],
        "categorical_features": ["river"],
        "model": {"type": "Ridge", "alpha": 1.0},
        "selected_as": "sensitivity_candidate",
        "promotion_requires": ["residual_bias_review", "river_specific_bias_check", "high_DOC_residual_check"],
    }
    return {
        PRIMARY_SPEC_PATH: primary,
        HYDROCLIMATE_SPEC_PATH: hydroclimate,
        LOG_TARGET_SPEC_PATH: log_target,
    }


def _write_model_specs() -> list[Path]:
    paths: list[Path] = []
    for destination, payload in _model_specs().items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        paths.append(destination)
    return paths


def _metric_filter(frame: pd.DataFrame, model: dict[str, str]) -> pd.Series:
    return (
        frame["feature_set"].eq(model["feature_set"])
        & frame["model_id"].eq(model["model_id"])
        & frame["target_scale"].eq(model["target_scale"])
    )


def _overall_bias_rows(same_sample: pd.DataFrame, log_metrics: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in KEY_BIAS_MODELS:
        source = log_metrics if model["target_scale"] == "log" else same_sample
        subset = source[
            _metric_filter(source, model)
            & source["validation_scheme"].eq("leave_one_year_out")
        ]
        if subset.empty:
            continue
        record = subset.iloc[0]
        rows.append(
            {
                "model_role": model["model_role"],
                "scope": "overall",
                "group_type": "overall",
                "group_value": "all",
                "feature_set": model["feature_set"],
                "model_id": model["model_id"],
                "target_scale": model["target_scale"],
                "n": int(record.get("n_test_total", 0)),
                "rmse": record.get("rmse", np.nan),
                "mae": record.get("mae", np.nan),
                "bias_mean": record.get("bias_mean", np.nan),
                "bias_median": record.get("bias_median", np.nan),
                "p05_residual": np.nan,
                "p95_residual": np.nan,
                "flags": "",
                "log_target_status": "sensitivity_candidate_only" if model["target_scale"] == "log" else "not_log_target",
            }
        )
    return rows


def _group_bias_rows(path_: Path, group_type: str, group_column: str) -> list[dict[str, Any]]:
    frame = _read_required_csv(path_)
    rows: list[dict[str, Any]] = []
    for model in KEY_BIAS_MODELS:
        subset = frame[_metric_filter(frame, model)].copy()
        for record in subset.to_dict("records"):
            rows.append(
                {
                    "model_role": model["model_role"],
                    "scope": group_type,
                    "group_type": group_type,
                    "group_value": record.get(group_column, ""),
                    "feature_set": model["feature_set"],
                    "model_id": model["model_id"],
                    "target_scale": model["target_scale"],
                    "n": record.get("n", ""),
                    "rmse": record.get("rmse", np.nan),
                    "mae": record.get("mae", np.nan),
                    "bias_mean": record.get("bias_mean", np.nan),
                    "bias_median": record.get("bias_median", np.nan),
                    "p05_residual": record.get("p05_residual", np.nan),
                    "p95_residual": record.get("p95_residual", np.nan),
                    "flags": record.get("flags", ""),
                    "log_target_status": "sensitivity_candidate_only" if model["target_scale"] == "log" else "not_log_target",
                }
            )
    return rows


def _log_target_bias_review() -> pd.DataFrame:
    same_sample = _read_required_csv(REFINEMENT_TABLE_DIR / "same_sample_ablation_metrics.csv")
    log_metrics = _read_required_csv(REFINEMENT_TABLE_DIR / "log_target_sensitivity_metrics.csv")
    rows = _overall_bias_rows(same_sample, log_metrics)
    rows.extend(_group_bias_rows(REFINEMENT_TABLE_DIR / "residual_summary_by_river.csv", "river", "river"))
    rows.extend(_group_bias_rows(REFINEMENT_TABLE_DIR / "residual_summary_by_doc_quantile.csv", "DOC quantile", "doc_quantile"))
    rows.extend(_group_bias_rows(REFINEMENT_TABLE_DIR / "residual_summary_by_month.csv", "month", "month"))
    review = pd.DataFrame(rows)
    review["decision_note"] = np.where(
        review["target_scale"].eq("log"),
        "Do not promote in baseline finalization; retain for later residual-bias review.",
        "Comparator for log-target bias review.",
    )
    return review


def write_baseline_final_report() -> Path:
    BASELINE_FINAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    contract = load_contract()
    decision = _read_required_csv(BASELINE_MODEL_DECISION_PATH)
    bias_review = _read_required_csv(LOG_TARGET_BIAS_REVIEW_PATH)
    refined_recommendation = _read_required_csv(REFINEMENT_TABLE_DIR / "refined_model_recommendation.csv")
    ablation_deltas = _read_required_csv(REFINEMENT_TABLE_DIR / "same_sample_ablation_deltas.csv")
    baseline_ranking_path = BASELINE_TABLE_DIR / "baseline_model_ranking.csv"
    baseline_ranking = pd.read_csv(baseline_ranking_path) if baseline_ranking_path.exists() else pd.DataFrame()
    f6_vs_f3 = ablation_deltas[
        ablation_deltas["comparison"].eq("F6_minus_F3_incremental_hydroclimate_after_river")
        & ablation_deltas["model_id"].eq("ridge_alpha_1")
        & ablation_deltas["validation_scheme"].eq("leave_one_year_out")
    ]
    log_rows = bias_review[bias_review["target_scale"].eq("log")]
    lines = [
        "# Baseline Final Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "## 1. Scope and guardrails",
        "",
        "This phase finalizes the baseline model decision from existing validation-only diagnostics. It does not add model families, does not run optical sensitivity, does not use basin context, does not generate daily DOC prediction, and does not compute flux.",
        "",
        "## 2. Inputs and freeze contract",
        "",
        f"- freeze_id: `{contract['freeze_id']}`",
        "- model input for baseline decision: `data/processed/gold/training_matrix_hydrocore.csv`",
        "- source diagnostics: `outputs/tables/baseline_refinement/` and `outputs/reports/baseline_refinement/`",
        "- gold data contract snapshot: passing before finalization.",
        "",
        "## 3. Baseline refinement summary",
        "",
        _md_table(refined_recommendation, max_rows=10),
        "",
        "Same-sample F6-vs-F3 delta:",
        "",
        _md_table(f6_vs_f3, max_rows=5),
        "",
        "## 4. Final primary baseline decision",
        "",
        "The selected primary baseline is `F3_q_season_river_fixed + ridge_alpha_1` with raw `DOC_mgC_L` target. It should be interpreted as a within-six-river ArcticGRO baseline, not a model for arbitrary Arctic river extrapolation.",
        "",
        _md_table(decision[decision["decision_type"].eq("primary_baseline")], max_rows=5),
        "",
        "## 5. Hydroclimate extension decision",
        "",
        "Hydroclimate extension `F6_reduced_hydroclimate_river_fixed + ridge_alpha_1` is retained for process sensitivity, but its incremental same-sample gain over F3 is negligible.",
        "",
        _md_table(decision[decision["decision_type"].eq("hydroclimate_extension")], max_rows=5),
        "",
        "## 6. Log-target sensitivity status",
        "",
        "The log-target F6 candidate remains `sensitivity_candidate_only`. It improves aggregate RMSE/MAE, but the residual-bias review still needs river-specific and high-DOC checks before any later promotion.",
        "",
        _md_table(decision[decision["decision_type"].eq("log_target_sensitivity")], max_rows=5),
        "",
        _md_table(log_rows.head(30), max_rows=30),
        "",
        "## 7. Leave-one-river-out stress-test interpretation",
        "",
        "Leave-one-river-out is retained as a stress test only. With six rivers and river fixed effects, held-out river categories are unseen and should not drive primary selection.",
        "",
        _md_table(decision[decision["decision_type"].eq("leave_one_river_out_stress")], max_rows=5),
        "",
        "## 8. What this baseline can and cannot claim",
        "",
        "It can support within-six-river validation comparisons among simple DOC concentration baselines. It cannot claim Arctic-wide extrapolation, production daily DOC prediction readiness, optical proxy benefit, basin sensitivity, or flux performance.",
        "",
        "## 9. Recommended next phase",
        "",
        "Recommended next phase: Optical Sensitivity Phase, using the selected F3 primary baseline as the comparator on identical optical-matched subsets.",
        "",
        "## 10. Explicit statements",
        "",
        "- Validation-only DOC concentration models were used for model selection.",
        "- No production daily DOC prediction was generated.",
        "- No DOC flux was generated.",
        "- Gold data were not modified.",
        "- Optical and basin matrices were not used.",
    ]
    if not baseline_ranking.empty:
        lines.extend(["", "## Baseline phase 1 ranking snapshot", "", _md_table(baseline_ranking.head(10), max_rows=10)])
    BASELINE_FINAL_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return BASELINE_FINAL_REPORT_PATH


def write_next_phase_handoff() -> Path:
    BASELINE_FINAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Next Phase Handoff",
        "",
        "Recommended next phase: Optical Sensitivity Phase.",
        "",
        "Use:",
        "",
        "- `training_matrix_optical_matched_3d.csv`",
        "- `training_matrix_optical_matched_3d_hls.csv`",
        "- `training_matrix_optical_matched_3d_landsat.csv`",
        "- `training_matrix_optical_matched_3d_sentinel2.csv`",
        "",
        "Baseline comparator:",
        "",
        "- `primary_baseline_f3_q_season_river_fixed_ridge_alpha_1`",
        "",
        "Question:",
        "",
        "Does optical proxy improve validation metrics over F3 on the same optical-matched subset?",
        "",
        "Guardrails:",
        "",
        "- Optical is proxy, not DOC observation.",
        "- Compare on the same sample.",
        "- Sensor-specific sensitivity is required.",
        "- Do not generate daily DOC prediction.",
        "- Do not compute flux.",
    ]
    NEXT_PHASE_HANDOFF_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return NEXT_PHASE_HANDOFF_PATH


def finalize_baseline() -> dict[str, Any]:
    _ensure_dirs()
    assert_no_forbidden_outputs()
    before_hashes = _hash_all_gold_tables()
    verification = _verify_gold_contract_snapshot()
    hydrocore_path = table_path("training_matrix_hydrocore.csv", gold_dir=require_gold_data_dir())
    before_hydrocore_hash = sha256_file(hydrocore_path)

    decision = _decision_table()
    bias_review = _log_target_bias_review()
    table_paths = [
        _write_csv(decision, BASELINE_MODEL_DECISION_PATH),
        _write_csv(bias_review, LOG_TARGET_BIAS_REVIEW_PATH),
    ]
    spec_paths = _write_model_specs()
    report_paths = [write_baseline_final_report(), write_next_phase_handoff()]

    assert_gold_hash_unchanged(hydrocore_path, before_hydrocore_hash)
    after_hashes = _hash_all_gold_tables()
    if before_hashes != after_hashes:
        raise RuntimeError("One or more frozen gold table hashes changed during baseline finalization.")
    assert_no_forbidden_outputs()
    return {
        "tables": table_paths,
        "specs": spec_paths,
        "reports": report_paths,
        "verification": verification,
        "decision": decision,
        "bias_review": bias_review,
    }
