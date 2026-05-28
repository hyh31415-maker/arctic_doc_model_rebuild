from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from arctic_doc_model_rebuild.gold_contract import load_contract, require_gold_data_dir, sha256_file, table_path
from arctic_doc_model_rebuild.modeling.bias_refinement import BIAS_REPORT_PATH, BIAS_TABLE_DIR, run_bias_aware_refinement
from arctic_doc_model_rebuild.paths import project_root


@pytest.fixture(scope="session")
def bias_refinement_result():
    return run_bias_aware_refinement()


def test_bias_refinement_outputs_exist(bias_refinement_result) -> None:
    assert BIAS_REPORT_PATH.exists()
    required = [
        "bias_refinement_model_registry.csv",
        "bias_refinement_feature_set_registry.csv",
        "bias_refinement_metrics_overall.csv",
        "bias_refinement_cv_predictions.csv",
        "bias_refinement_deltas_vs_f3.csv",
        "bias_refinement_model_ranking.csv",
        "bias_refinement_recommendation.csv",
        "refined_production_readiness_decision.csv",
        "uncertainty_diagnostic_audit.csv",
    ]
    for name in required:
        assert (BIAS_TABLE_DIR / name).exists()


def test_bias_refinement_cv_predictions_validation_only(bias_refinement_result) -> None:
    predictions = pd.read_csv(BIAS_TABLE_DIR / "bias_refinement_cv_predictions.csv")
    assert not predictions.empty
    assert predictions["is_cv_prediction"].astype(str).str.lower().isin({"true", "1"}).all()
    assert predictions["is_production_prediction"].astype(str).str.lower().isin({"false", "0"}).all()


def test_no_production_predictions(bias_refinement_result) -> None:
    root = project_root()
    assert not (root / "outputs" / "predictions").exists()
    assert [item for item in root.rglob("*daily_doc_prediction*") if item.is_file()] == []


def test_no_flux_outputs(bias_refinement_result) -> None:
    root = project_root()
    assert not (root / "outputs" / "flux").exists()
    forbidden = [
        item
        for item in (root / "outputs").rglob("*")
        if item.is_file()
        and (item.name.lower() in {"daily_flux.csv", "annual_flux.csv", "snowmelt_flux.csv"} or item.name.lower().endswith("_flux.csv"))
    ]
    assert forbidden == []


def test_no_gold_data_modified() -> None:
    gold_dir = require_gold_data_dir()
    contract = load_contract()
    before = {table_name: sha256_file(table_path(table_name, gold_dir=gold_dir)) for table_name in contract["expected_tables"]}
    run_bias_aware_refinement()
    after = {table_name: sha256_file(table_path(table_name, gold_dir=gold_dir)) for table_name in contract["expected_tables"]}
    assert before == after


def test_no_optical_or_basin_matrices_used(monkeypatch) -> None:
    import pandas as pd_module

    original_read_csv = pd_module.read_csv
    read_paths: list[str] = []

    def spy_read_csv(filepath, *args, **kwargs):
        read_paths.append(str(filepath))
        return original_read_csv(filepath, *args, **kwargs)

    monkeypatch.setattr(pd_module, "read_csv", spy_read_csv)
    run_bias_aware_refinement()
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
    run_bias_aware_refinement()
    assert not [path for path in read_paths if "prediction_grid_daily" in Path(path).name]


def test_diagnostic_audit_checks_blank_fold_metrics(bias_refinement_result) -> None:
    audit = pd.read_csv(BIAS_TABLE_DIR / "uncertainty_diagnostic_audit.csv")
    row = audit[audit["diagnostic_item"].eq("fold_stability_blank_metrics")]
    assert not row.empty
    assert "likely_cause" in row.columns


def test_model_ranking_has_f3_comparator(bias_refinement_result) -> None:
    ranking = pd.read_csv(BIAS_TABLE_DIR / "bias_refinement_model_ranking.csv")
    assert "f3_comparator" in set(ranking["classification"])
    assert "B0_F3_finalized:ridge_alpha_1:raw:leave_one_year_out" in set(ranking["candidate_key"])


def test_recommendation_exists(bias_refinement_result) -> None:
    recommendation = pd.read_csv(BIAS_TABLE_DIR / "bias_refinement_recommendation.csv")
    assert "recommended_primary_model_after_refinement" in set(recommendation["decision_item"])


def test_refined_readiness_status_valid(bias_refinement_result) -> None:
    readiness = pd.read_csv(BIAS_TABLE_DIR / "refined_production_readiness_decision.csv")
    ready = readiness[readiness["decision_item"].eq("ready_for_production_daily_prediction")].iloc[0]
    assert ready["status"] in {"true", "true_with_caveats", "false"}


def test_no_model_binary_artifacts(bias_refinement_result) -> None:
    root = project_root()
    forbidden = [item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in {".joblib", ".pkl", ".pickle"}]
    assert forbidden == []
