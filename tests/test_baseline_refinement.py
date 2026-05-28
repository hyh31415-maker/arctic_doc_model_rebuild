from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arctic_doc_model_rebuild.gold_contract import load_contract, require_gold_data_dir, sha256_file, table_path
from arctic_doc_model_rebuild.modeling.refinement import (
    REFINEMENT_REPORT_PATH,
    REFINEMENT_TABLE_DIR,
    run_baseline_refinement,
)
from arctic_doc_model_rebuild.modeling.validation import validation_scheme_registry
from arctic_doc_model_rebuild.paths import project_root


@pytest.fixture(scope="session")
def refinement_result():
    return run_baseline_refinement()


def test_refinement_outputs_exist(refinement_result) -> None:
    assert REFINEMENT_REPORT_PATH.exists()
    required = [
        "same_sample_ablation_metrics.csv",
        "same_sample_ablation_deltas.csv",
        "log_target_sensitivity_metrics.csv",
        "log_target_cv_predictions.csv",
        "fold_stability_leave_one_year_out.csv",
        "refined_model_recommendation.csv",
    ]
    for name in required:
        assert (REFINEMENT_TABLE_DIR / name).exists()


def test_same_sample_ablation_common_n_consistent(refinement_result) -> None:
    metrics = pd.read_csv(REFINEMENT_TABLE_DIR / "same_sample_ablation_metrics.csv")
    assert metrics["same_sample_n"].nunique() == 1
    assert int(metrics["same_sample_n"].iloc[0]) == 511
    loyo = metrics[metrics["validation_scheme"].eq("leave_one_year_out")]
    assert set(loyo["n_test_total"].astype(int)) == {511}


def test_ablation_deltas_have_expected_comparisons(refinement_result) -> None:
    deltas = pd.read_csv(REFINEMENT_TABLE_DIR / "same_sample_ablation_deltas.csv")
    required = {
        "F2_minus_F1_value_of_Q",
        "F3_minus_F2_value_of_river_fixed_effects",
        "F4_minus_F2_value_of_hydroclimate_without_river",
        "F6_minus_F3_incremental_hydroclimate_after_river",
        "F6_minus_F4_value_of_river_after_hydroclimate",
    }
    assert required.issubset(set(deltas["comparison"]))
    assert {"rmse_reduction", "mae_reduction", "r2_gain"}.issubset(deltas.columns)


def test_log_target_predictions_inverse_transformed(refinement_result) -> None:
    predictions = pd.read_csv(REFINEMENT_TABLE_DIR / "log_target_cv_predictions.csv")
    log_rows = predictions[predictions["target_scale"].eq("log")].dropna(subset=["DOC_cv_predicted_log"])
    assert not log_rows.empty
    assert (log_rows["DOC_cv_predicted_mgC_L"] > 0).all()
    expected = np.exp(log_rows["DOC_cv_predicted_log"].astype(float))
    assert np.allclose(expected, log_rows["DOC_cv_predicted_mgC_L"].astype(float))


def test_refinement_no_gold_data_modified() -> None:
    gold_dir = require_gold_data_dir()
    contract = load_contract()
    before = {table_name: sha256_file(table_path(table_name, gold_dir=gold_dir)) for table_name in contract["expected_tables"]}
    run_baseline_refinement()
    after = {table_name: sha256_file(table_path(table_name, gold_dir=gold_dir)) for table_name in contract["expected_tables"]}
    assert before == after


def test_refinement_no_production_predictions(refinement_result) -> None:
    root = project_root()
    assert not (root / "outputs" / "predictions").exists()
    for name in ["same_sample_ablation_cv_predictions.csv", "log_target_cv_predictions.csv"]:
        frame = pd.read_csv(REFINEMENT_TABLE_DIR / name)
        assert frame["is_cv_prediction"].astype(str).str.lower().isin({"true", "1"}).all()
        assert frame["is_production_prediction"].astype(str).str.lower().isin({"false", "0"}).all()


def test_refinement_no_flux_outputs(refinement_result) -> None:
    root = project_root()
    assert not (root / "outputs" / "flux").exists()
    forbidden_names = [
        item
        for item in (root / "outputs").rglob("*")
        if item.is_file() and item.name.lower() in {"daily_flux.csv", "annual_flux.csv", "snowmelt_flux.csv"}
    ]
    assert forbidden_names == []
    table_mentions = []
    for item in REFINEMENT_TABLE_DIR.rglob("*.csv"):
        if "flux" in item.read_text(encoding="utf-8", errors="ignore").lower():
            table_mentions.append(item)
    assert table_mentions == []


def test_refinement_does_not_load_optical_or_basin_matrices(monkeypatch) -> None:
    import pandas as pd_module

    original_read_csv = pd_module.read_csv
    read_paths: list[str] = []

    def spy_read_csv(filepath, *args, **kwargs):
        read_paths.append(str(filepath))
        return original_read_csv(filepath, *args, **kwargs)

    monkeypatch.setattr(pd_module, "read_csv", spy_read_csv)
    run_baseline_refinement()
    forbidden = [
        "training_matrix_optical",
        "optical_timeseries",
        "training_matrix_basin_context",
        "basin_attributes_curated",
        "prediction_grid_daily",
    ]
    assert not [path for path in read_paths if any(token in Path(path).name for token in forbidden)]


def test_model_recommendation_exists(refinement_result) -> None:
    recommendation = pd.read_csv(REFINEMENT_TABLE_DIR / "refined_model_recommendation.csv")
    assert {
        "recommended_primary_baseline",
        "recommended_hydroclimate_extension",
        "recommended_sensitivity_only",
        "not_recommended",
    }.issubset(set(recommendation["recommendation_type"]))
    primary = recommendation[recommendation["recommendation_type"].eq("recommended_primary_baseline")].iloc[0]
    assert primary["feature_set"] in {"F3_q_season_river_fixed", "F6_reduced_hydroclimate_river_fixed"}


def test_loro_marked_stress_not_primary() -> None:
    registry = validation_scheme_registry()
    row = registry[registry["validation_scheme"].eq("leave_one_river_out")].iloc[0]
    assert bool(row["stress_test"])
    assert not bool(row["primary_for_model_selection"])
