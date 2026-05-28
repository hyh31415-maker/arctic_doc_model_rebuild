from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from arctic_doc_model_rebuild.gold_contract import load_contract, require_gold_data_dir, sha256_file, table_path
from arctic_doc_model_rebuild.modeling.optical_sensitivity import (
    OPTICAL_REPORT_PATH,
    OPTICAL_TABLE_DIR,
    run_optical_sensitivity,
)
from arctic_doc_model_rebuild.paths import project_root


@pytest.fixture(scope="session")
def optical_result():
    return run_optical_sensitivity()


def test_optical_outputs_exist(optical_result) -> None:
    assert OPTICAL_REPORT_PATH.exists()
    required = [
        "optical_dataset_registry.csv",
        "optical_feature_set_registry.csv",
        "optical_model_registry.csv",
        "optical_validation_registry.csv",
        "optical_metrics_overall.csv",
        "optical_same_sample_deltas.csv",
        "optical_model_ranking.csv",
        "optical_cv_predictions.csv",
    ]
    for name in required:
        assert (OPTICAL_TABLE_DIR / name).exists()


def test_optical_metrics_nonempty(optical_result) -> None:
    metrics = pd.read_csv(OPTICAL_TABLE_DIR / "optical_metrics_overall.csv")
    assert not metrics.empty
    assert {"leave_one_year_out", "river_year_groupkfold", "leave_one_river_out"}.issubset(set(metrics["validation_scheme"]))


def test_optical_same_sample_deltas_exist(optical_result) -> None:
    deltas = pd.read_csv(OPTICAL_TABLE_DIR / "optical_same_sample_deltas.csv")
    assert not deltas.empty
    assert {"rmse_reduction", "mae_reduction", "r2_gain", "bias_change_abs"}.issubset(deltas.columns)
    assert deltas["baseline_feature_set"].eq("B0_F3_same_subset").all()


def test_any_sensor_3d_in_ranking(optical_result) -> None:
    ranking = pd.read_csv(OPTICAL_TABLE_DIR / "optical_model_ranking.csv")
    subset = ranking[ranking["dataset_id"].eq("any_sensor_3d")]
    assert not subset.empty
    assert subset["validation_scheme"].eq("leave_one_year_out").all()


def test_sensor_specific_3d_outputs_exist(optical_result) -> None:
    metrics = pd.read_csv(OPTICAL_TABLE_DIR / "optical_metrics_overall.csv")
    assert {"hls_3d", "landsat_3d", "sentinel2_3d"}.issubset(set(metrics["dataset_id"]))


def test_cv_predictions_validation_only(optical_result) -> None:
    predictions = pd.read_csv(OPTICAL_TABLE_DIR / "optical_cv_predictions.csv")
    assert not predictions.empty
    assert predictions["is_cv_prediction"].astype(str).str.lower().isin({"true", "1"}).all()
    assert predictions["is_production_prediction"].astype(str).str.lower().isin({"false", "0"}).all()


def test_no_production_predictions(optical_result) -> None:
    root = project_root()
    assert not (root / "outputs" / "predictions").exists()
    assert [item for item in root.rglob("*daily_doc_prediction*") if item.is_file()] == []


def test_no_flux_outputs(optical_result) -> None:
    root = project_root()
    assert not (root / "outputs" / "flux").exists()
    forbidden_names = [
        item
        for item in (root / "outputs").rglob("*")
        if item.is_file() and (item.name.lower() in {"daily_flux.csv", "annual_flux.csv", "snowmelt_flux.csv"} or item.name.lower().endswith("_flux.csv"))
    ]
    assert forbidden_names == []


def test_no_gold_data_modified(optical_result) -> None:
    gold_dir = require_gold_data_dir()
    contract = load_contract()
    before = {table_name: sha256_file(table_path(table_name, gold_dir=gold_dir)) for table_name in contract["expected_tables"]}
    run_optical_sensitivity()
    after = {table_name: sha256_file(table_path(table_name, gold_dir=gold_dir)) for table_name in contract["expected_tables"]}
    assert before == after


def test_no_basin_context_loaded(monkeypatch) -> None:
    import pandas as pd_module

    original_read_csv = pd_module.read_csv
    read_paths: list[str] = []

    def spy_read_csv(filepath, *args, **kwargs):
        read_paths.append(str(filepath))
        return original_read_csv(filepath, *args, **kwargs)

    monkeypatch.setattr(pd_module, "read_csv", spy_read_csv)
    run_optical_sensitivity()
    forbidden = ["training_matrix_basin_context", "basin_attributes_curated", "prediction_grid_daily"]
    assert not [path for path in read_paths if any(token in Path(path).name for token in forbidden)]


def test_no_lab_optical_loaded(monkeypatch) -> None:
    import pandas as pd_module

    original_read_csv = pd_module.read_csv
    read_paths: list[str] = []

    def spy_read_csv(filepath, *args, **kwargs):
        read_paths.append(str(filepath))
        return original_read_csv(filepath, *args, **kwargs)

    monkeypatch.setattr(pd_module, "read_csv", spy_read_csv)
    run_optical_sensitivity()
    assert not [path for path in read_paths if "lab_optical_proxy_gold" in Path(path).name]


def test_loro_marked_stress_test(optical_result) -> None:
    registry = pd.read_csv(OPTICAL_TABLE_DIR / "optical_validation_registry.csv")
    row = registry[registry["validation_scheme"].eq("leave_one_river_out")].iloc[0]
    assert str(row["stress_test"]).lower() in {"true", "1"}
    assert row["validation_role"] == "stress_test"


def test_sentinel2_underpowered_flag_if_needed(optical_result) -> None:
    ranking = pd.read_csv(OPTICAL_TABLE_DIR / "optical_model_ranking.csv")
    sentinel = ranking[ranking["dataset_id"].eq("sentinel2_3d")]
    assert not sentinel.empty
    assert "underpowered" in set(sentinel["classification"])


def test_optical_report_answers_incremental_value(optical_result) -> None:
    text = OPTICAL_REPORT_PATH.read_text(encoding="utf-8")
    assert "Does optical improve F3 baseline?" in text
    assert "Answer:" in text
    assert any(f"Answer: `{status}`" in text for status in ["yes", "no", "marginal", "sensor-specific only"])
