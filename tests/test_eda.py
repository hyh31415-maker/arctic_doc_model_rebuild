from __future__ import annotations

import pandas as pd
import pytest

from arctic_doc_model_rebuild.eda import EDA_TABLE_DIR, run_eda
from arctic_doc_model_rebuild.gold_contract import load_contract, require_gold_data_dir, sha256_file, table_path
from arctic_doc_model_rebuild.modeling.diagnostics import ALLOWED_DOC_MODEL_ARTIFACTS
from arctic_doc_model_rebuild.paths import project_root


@pytest.fixture(scope="session")
def eda_result():
    return run_eda()


def test_run_eda_outputs_report(eda_result) -> None:
    assert eda_result.report_path.exists()
    text = eda_result.report_path.read_text(encoding="utf-8")
    assert "No model was trained" in text
    assert "No DOC prediction was generated" in text
    assert "No flux was generated" in text


def test_eda_tables_exist(eda_result) -> None:
    required = {
        "gold_matrix_inventory.csv",
        "doc_label_counts_by_river.csv",
        "hydrocore_missingness_by_column.csv",
        "doc_q_spearman_by_river.csv",
        "model_scope_feasibility.csv",
    }
    existing = {path.name for path in eda_result.table_paths}
    assert required.issubset(existing)


def test_eda_no_model_outputs(eda_result) -> None:
    root = project_root()
    model_dir = root / "outputs" / "models"
    if model_dir.exists():
        assert {item.name for item in model_dir.iterdir() if item.is_file()}.issubset(ALLOWED_DOC_MODEL_ARTIFACTS)
    assert not (root / "outputs" / "predictions").exists()
    assert not (root / "outputs" / "flux").exists()
    forbidden = [
        item
        for item in root.rglob("*")
        if item.is_file() and item.suffix.lower() in {".joblib", ".pkl", ".pickle"} and item.name not in ALLOWED_DOC_MODEL_ARTIFACTS
    ]
    assert forbidden == []


def test_eda_does_not_modify_gold_data() -> None:
    gold_dir = require_gold_data_dir()
    contract = load_contract()
    before = {
        table_name: sha256_file(table_path(table_name, gold_dir=gold_dir))
        for table_name in contract["expected_tables"]
    }
    run_eda()
    after = {
        table_name: sha256_file(table_path(table_name, gold_dir=gold_dir))
        for table_name in contract["expected_tables"]
    }
    assert before == after


def test_model_scope_feasibility_exists(eda_result) -> None:
    path = EDA_TABLE_DIR / "model_scope_feasibility.csv"
    assert path.exists()
    frame = pd.read_csv(path)
    required_scopes = {
        "season_only_baseline",
        "q_season_baseline",
        "hydroclimate_complete_case",
        "hydroclimate_missingness_aware",
        "river_effects_model",
        "leave_one_year_out_cv",
        "leave_one_river_out_cv",
        "optical_3d_any_sensor",
        "optical_3d_hls_only",
        "optical_3d_landsat_only",
        "optical_3d_sentinel2_only",
        "basin_context_sensitivity",
        "daily_prediction_grid_ready",
    }
    assert required_scopes.issubset(set(frame["scope"]))


def test_optical_bias_audit_exists(eda_result) -> None:
    path = EDA_TABLE_DIR / "optical_match_bias_audit.csv"
    assert path.exists()
    frame = pd.read_csv(path)
    assert not frame.empty
    assert {"doc_median_difference", "q_median_difference", "month_distribution_difference"}.issubset(frame.columns)


def test_prediction_grid_audit_has_no_predictions(eda_result) -> None:
    path = EDA_TABLE_DIR / "prediction_grid_coverage_by_river.csv"
    assert path.exists()
    frame = pd.read_csv(path)
    forbidden = [column for column in frame.columns if "prediction" in column.lower() or "flux" in column.lower() or column == "DOC_mgC_L"]
    assert forbidden == []
