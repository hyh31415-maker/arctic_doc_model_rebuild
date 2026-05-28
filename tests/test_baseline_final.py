from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from arctic_doc_model_rebuild.gold_contract import load_contract, require_gold_data_dir, sha256_file, table_path
from arctic_doc_model_rebuild.modeling.baseline_final import (
    BASELINE_FINAL_REPORT_PATH,
    BASELINE_FINAL_TABLE_DIR,
    HYDROCLIMATE_SPEC_PATH,
    LOG_TARGET_SPEC_PATH,
    NEXT_PHASE_HANDOFF_PATH,
    PRIMARY_SPEC_PATH,
    finalize_baseline,
)
from arctic_doc_model_rebuild.paths import project_root


@pytest.fixture(scope="session")
def baseline_final_result():
    return finalize_baseline()


def _decision() -> pd.DataFrame:
    return pd.read_csv(BASELINE_FINAL_TABLE_DIR / "baseline_model_decision.csv")


def test_baseline_final_decision_exists(baseline_final_result) -> None:
    assert BASELINE_FINAL_REPORT_PATH.exists()
    assert (BASELINE_FINAL_TABLE_DIR / "baseline_model_decision.csv").exists()
    assert (BASELINE_FINAL_TABLE_DIR / "log_target_bias_review.csv").exists()
    decision = _decision()
    assert {
        "primary_baseline",
        "hydroclimate_extension",
        "log_target_sensitivity",
        "leave_one_river_out",
    }.issubset(set(decision["decision_name"]))


def test_primary_baseline_is_f3(baseline_final_result) -> None:
    row = _decision()[lambda frame: frame["decision_type"].eq("primary_baseline")].iloc[0]
    assert row["feature_set"] == "F3_q_season_river_fixed"
    assert row["model_id"] == "ridge_alpha_1"
    assert row["target_scale"] == "raw"
    assert row["decision"] == "selected"


def test_hydroclimate_extension_is_f6_not_primary(baseline_final_result) -> None:
    row = _decision()[lambda frame: frame["decision_type"].eq("hydroclimate_extension")].iloc[0]
    assert row["feature_set"] == "F6_reduced_hydroclimate_river_fixed"
    assert row["model_id"] == "ridge_alpha_1"
    assert row["decision"] == "extension_not_primary"


def test_log_target_is_sensitivity_candidate_only(baseline_final_result) -> None:
    row = _decision()[lambda frame: frame["decision_type"].eq("log_target_sensitivity")].iloc[0]
    assert row["feature_set"] == "F6_reduced_hydroclimate_river_fixed"
    assert row["target_scale"] == "log"
    assert row["decision"] == "sensitivity_candidate_only"
    review = pd.read_csv(BASELINE_FINAL_TABLE_DIR / "log_target_bias_review.csv")
    log_rows = review[review["target_scale"].eq("log")]
    assert not log_rows.empty
    assert set(log_rows["log_target_status"]) == {"sensitivity_candidate_only"}


def test_loro_is_stress_test_only(baseline_final_result) -> None:
    row = _decision()[lambda frame: frame["decision_type"].eq("leave_one_river_out_stress")].iloc[0]
    assert row["validation_basis"] == "leave_one_river_out"
    assert row["decision"] == "stress_test_only"


def test_model_specs_exist(baseline_final_result) -> None:
    for path in [PRIMARY_SPEC_PATH, HYDROCLIMATE_SPEC_PATH, LOG_TARGET_SPEC_PATH]:
        assert path.exists()


def test_model_specs_disallow_prediction_and_flux(baseline_final_result) -> None:
    for path in [PRIMARY_SPEC_PATH, HYDROCLIMATE_SPEC_PATH, LOG_TARGET_SPEC_PATH]:
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert spec["production_prediction_allowed"] is False
        assert spec["flux_allowed"] is False
    primary = yaml.safe_load(PRIMARY_SPEC_PATH.read_text(encoding="utf-8"))
    assert primary["feature_set"] == "F3_q_season_river_fixed"
    assert primary["selected_as"] == "primary_baseline"


def test_gold_data_not_modified(baseline_final_result) -> None:
    gold_dir = require_gold_data_dir()
    contract = load_contract()
    before = {table_name: sha256_file(table_path(table_name, gold_dir=gold_dir)) for table_name in contract["expected_tables"]}
    finalize_baseline()
    after = {table_name: sha256_file(table_path(table_name, gold_dir=gold_dir)) for table_name in contract["expected_tables"]}
    assert before == after


def test_no_predictions_or_flux_outputs(baseline_final_result) -> None:
    root = project_root()
    assert not (root / "outputs" / "predictions").exists()
    assert not (root / "outputs" / "flux").exists()
    forbidden = [
        item
        for item in root.rglob("*")
        if item.is_file()
        and (
            item.name.lower() in {"daily_doc_prediction.csv", "daily_flux.csv", "annual_flux.csv", "snowmelt_flux.csv"}
            or item.suffix.lower() in {".joblib", ".pkl", ".pickle"}
        )
    ]
    assert forbidden == []


def test_next_phase_handoff_exists(baseline_final_result) -> None:
    assert NEXT_PHASE_HANDOFF_PATH.exists()
    text = NEXT_PHASE_HANDOFF_PATH.read_text(encoding="utf-8")
    assert "Optical Sensitivity Phase" in text
    assert "primary_baseline_f3_q_season_river_fixed_ridge_alpha_1" in text


def test_baseline_final_does_not_load_optical_or_basin_matrices(monkeypatch) -> None:
    import pandas as pd_module

    original_read_csv = pd_module.read_csv
    read_paths: list[str] = []

    def spy_read_csv(filepath, *args, **kwargs):
        read_paths.append(str(filepath))
        return original_read_csv(filepath, *args, **kwargs)

    monkeypatch.setattr(pd_module, "read_csv", spy_read_csv)
    finalize_baseline()
    forbidden = [
        "training_matrix_optical",
        "optical_timeseries",
        "training_matrix_basin_context",
        "basin_attributes_curated",
        "prediction_grid_daily",
    ]
    assert not [path for path in read_paths if any(token in Path(path).name for token in forbidden)]
