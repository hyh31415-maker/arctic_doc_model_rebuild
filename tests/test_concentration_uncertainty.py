from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from arctic_doc_model_rebuild.gold_contract import load_contract, require_gold_data_dir, sha256_file, table_path
from arctic_doc_model_rebuild.modeling.concentration_uncertainty import (
    UNCERTAINTY_REPORT_PATH,
    UNCERTAINTY_TABLE_DIR,
    run_concentration_uncertainty,
)
from arctic_doc_model_rebuild.modeling.diagnostics import ALLOWED_DOC_MODEL_ARTIFACTS
from arctic_doc_model_rebuild.paths import project_root


@pytest.fixture(scope="session")
def uncertainty_result():
    return run_concentration_uncertainty()


def test_uncertainty_outputs_exist(uncertainty_result) -> None:
    assert UNCERTAINTY_REPORT_PATH.exists()
    required = [
        "uncertainty_model_registry.csv",
        "uncertainty_cv_predictions.csv",
        "residual_distribution_summary.csv",
        "empirical_residual_intervals.csv",
        "empirical_interval_coverage.csv",
        "production_readiness_decision.csv",
    ]
    for name in required:
        assert (UNCERTAINTY_TABLE_DIR / name).exists()


def test_uncertainty_cv_predictions_validation_only(uncertainty_result) -> None:
    predictions = pd.read_csv(UNCERTAINTY_TABLE_DIR / "uncertainty_cv_predictions.csv")
    assert not predictions.empty
    assert predictions["is_cv_prediction"].astype(str).str.lower().isin({"true", "1"}).all()
    assert predictions["is_production_prediction"].astype(str).str.lower().isin({"false", "0"}).all()


def test_empirical_residual_intervals_exist(uncertainty_result) -> None:
    intervals = pd.read_csv(UNCERTAINTY_TABLE_DIR / "empirical_residual_intervals.csv")
    assert not intervals.empty
    assert {"q02_5", "q05", "q10", "q90", "q95", "q97_5"}.issubset(intervals.columns)
    assert intervals["is_production_prediction_interval"].astype(str).str.lower().isin({"false", "0"}).all()


def test_empirical_interval_coverage_exists(uncertainty_result) -> None:
    coverage = pd.read_csv(UNCERTAINTY_TABLE_DIR / "empirical_interval_coverage.csv")
    assert not coverage.empty
    assert "empirical_90pct_interval_coverage" in coverage.columns


def test_production_readiness_decision_exists(uncertainty_result) -> None:
    readiness = pd.read_csv(UNCERTAINTY_TABLE_DIR / "production_readiness_decision.csv")
    assert "ready_for_production_daily_prediction" in set(readiness["decision_item"])


def test_no_production_predictions(uncertainty_result) -> None:
    root = project_root()
    assert not (root / "outputs" / "predictions").exists()


def test_no_flux_outputs(uncertainty_result) -> None:
    root = project_root()
    assert not (root / "outputs" / "flux").exists()
    forbidden = [
        item
        for item in (root / "outputs").rglob("*")
        if item.is_file() and (item.name.lower() in {"daily_flux.csv", "annual_flux.csv", "snowmelt_flux.csv"} or item.name.lower().endswith("_flux.csv"))
    ]
    assert forbidden == []


def test_no_gold_data_modified() -> None:
    gold_dir = require_gold_data_dir()
    contract = load_contract()
    before = {table_name: sha256_file(table_path(table_name, gold_dir=gold_dir)) for table_name in contract["expected_tables"]}
    run_concentration_uncertainty()
    after = {table_name: sha256_file(table_path(table_name, gold_dir=gold_dir)) for table_name in contract["expected_tables"]}
    assert before == after


def test_no_optical_or_basin_matrices_used_as_predictors(monkeypatch) -> None:
    import pandas as pd_module

    original_read_csv = pd_module.read_csv
    read_paths: list[str] = []

    def spy_read_csv(filepath, *args, **kwargs):
        read_paths.append(str(filepath))
        return original_read_csv(filepath, *args, **kwargs)

    monkeypatch.setattr(pd_module, "read_csv", spy_read_csv)
    run_concentration_uncertainty()
    forbidden = [
        "training_matrix_optical_matched",
        "training_matrix_basin_context",
        "basin_attributes_curated",
        "lab_optical_proxy_gold",
    ]
    assert not [path for path in read_paths if any(token in Path(path).name for token in forbidden)]


def test_prediction_grid_not_loaded(monkeypatch) -> None:
    import pandas as pd_module

    original_read_csv = pd_module.read_csv
    read_paths: list[str] = []

    def spy_read_csv(filepath, *args, **kwargs):
        read_paths.append(str(filepath))
        return original_read_csv(filepath, *args, **kwargs)

    monkeypatch.setattr(pd_module, "read_csv", spy_read_csv)
    run_concentration_uncertainty()
    assert not [path for path in read_paths if "prediction_grid_daily" in Path(path).name]


def test_ready_for_prediction_status_valid(uncertainty_result) -> None:
    readiness = pd.read_csv(UNCERTAINTY_TABLE_DIR / "production_readiness_decision.csv")
    ready = readiness[readiness["decision_item"].eq("ready_for_production_daily_prediction")].iloc[0]
    assert ready["status"] in {"true", "true_with_caveats", "false"}


def test_log_target_status_not_promoted_without_review(uncertainty_result) -> None:
    readiness = pd.read_csv(UNCERTAINTY_TABLE_DIR / "production_readiness_decision.csv")
    log_status = readiness[readiness["decision_item"].eq("log_target_status")].iloc[0]
    assert log_status["status"] == "sensitivity_only"
    high_doc = pd.read_csv(UNCERTAINTY_TABLE_DIR / "high_doc_residual_review.csv")
    log_rows = high_doc[high_doc["target_scale"].eq("log")]
    assert not log_rows.empty
    assert set(log_rows["log_target_promotion_status"]) == {"sensitivity_only"}


def test_roi_qc_caveat_carried_forward(uncertainty_result) -> None:
    river_bias = pd.read_csv(UNCERTAINTY_TABLE_DIR / "river_bias_summary.csv")
    assert "roi_decision" in river_bias.columns
    assert river_bias["roi_decision"].astype(str).str.contains("caveat").any()


def test_no_model_binary_artifacts(uncertainty_result) -> None:
    root = project_root()
    forbidden = [
        item
        for item in root.rglob("*")
        if item.is_file() and item.suffix.lower() in {".joblib", ".pkl", ".pickle"} and item.name not in ALLOWED_DOC_MODEL_ARTIFACTS
    ]
    assert forbidden == []
