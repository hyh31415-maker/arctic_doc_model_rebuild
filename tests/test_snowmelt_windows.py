from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arctic_doc_model_rebuild.flux.snowmelt_window_metrics import run_snowmelt_window_flux
from arctic_doc_model_rebuild.flux.snowmelt_window_reports import SNOWMELT_REPORT_PATH, SNOWMELT_TABLE_DIR
from arctic_doc_model_rebuild.flux.snowmelt_windows import define_snowmelt_windows, required_snowmelt_input_paths
from arctic_doc_model_rebuild.gold_contract import load_contract, require_gold_data_dir, sha256_file, table_path
from arctic_doc_model_rebuild.paths import project_root


@pytest.fixture(scope="session")
def snowmelt_result():
    define_snowmelt_windows()
    return run_snowmelt_window_flux()


def test_snowmelt_window_definitions_exist(snowmelt_result) -> None:
    path = SNOWMELT_TABLE_DIR / "snowmelt_window_definitions_by_river_year.csv"
    assert path.exists()
    definitions = pd.read_csv(path)
    assert {"fixed_may_july_reference", "discharge_centered_freshet", "q75_peak_contiguous", "snow_depletion_assisted", "common_overlap_w1_w2"}.issubset(
        set(definitions["window_id"])
    )


def test_window_start_before_end(snowmelt_result) -> None:
    definitions = pd.read_csv(SNOWMELT_TABLE_DIR / "snowmelt_window_definitions_by_river_year.csv")
    valid = definitions[definitions["start_date"].notna() & definitions["end_date"].notna() & definitions["start_date"].astype(str).ne("")]
    assert not valid.empty
    assert (pd.to_datetime(valid["start_date"]) <= pd.to_datetime(valid["end_date"])).all()


def test_peak_q_inside_window_when_available(snowmelt_result) -> None:
    definitions = pd.read_csv(SNOWMELT_TABLE_DIR / "snowmelt_window_definitions_by_river_year.csv")
    valid = definitions[
        definitions["peak_q_date"].notna()
        & definitions["peak_q_date"].astype(str).ne("")
        & definitions["start_date"].astype(str).ne("")
        & definitions["end_date"].astype(str).ne("")
    ]
    assert not valid.empty
    peak = pd.to_datetime(valid["peak_q_date"])
    assert ((pd.to_datetime(valid["start_date"]) <= peak) & (peak <= pd.to_datetime(valid["end_date"]))).all()


def test_window_flux_summary_exists(snowmelt_result) -> None:
    summary_path = SNOWMELT_TABLE_DIR / "snowmelt_window_flux_summary.csv"
    assert summary_path.exists()
    summary = pd.read_csv(summary_path)
    required = {"window_flux_TgC", "window_fraction_of_annual", "window_confidence_tier", "core_2003_2024"}
    assert required.issubset(summary.columns)
    assert not summary.empty


def test_window_flux_sums_existing_daily_flux(snowmelt_result) -> None:
    summary = pd.read_csv(SNOWMELT_TABLE_DIR / "snowmelt_window_flux_summary.csv")
    daily = pd.read_csv(project_root() / "outputs" / "tables" / "doc_flux" / "daily_doc_flux.csv", low_memory=False)
    daily["date"] = pd.to_datetime(daily["date"])
    row = summary[summary["window_id"].eq("q75_peak_contiguous") & summary["window_flux_TgC"].notna()].iloc[0]
    subset = daily[
        daily["river"].astype(str).eq(str(row["river"]))
        & daily["date"].between(pd.to_datetime(row["start_date"]), pd.to_datetime(row["end_date"]), inclusive="both")
        & daily["flux_status"].astype(str).eq("calculated")
    ]
    assert np.isclose(subset["daily_flux_TgC_day"].sum(), row["window_flux_TgC"])


def test_no_daily_flux_recalculation() -> None:
    daily_flux = project_root() / "outputs" / "tables" / "doc_flux" / "daily_doc_flux.csv"
    before = sha256_file(daily_flux)
    define_snowmelt_windows()
    run_snowmelt_window_flux()
    assert sha256_file(daily_flux) == before


def test_no_model_retraining(snowmelt_result) -> None:
    models_dir = project_root() / "outputs" / "models"
    allowed = {"production_candidate_r4_daily_doc_model.joblib", "production_candidate_r4_daily_doc_model_metadata.json"}
    model_files = {item.name for item in models_dir.glob("*") if item.is_file()} if models_dir.exists() else set()
    assert model_files.issubset(allowed)


def test_no_new_doc_prediction() -> None:
    daily_prediction = project_root() / "outputs" / "tables" / "daily_doc_prediction" / "daily_doc_prediction.csv"
    before = sha256_file(daily_prediction)
    define_snowmelt_windows()
    run_snowmelt_window_flux()
    assert sha256_file(daily_prediction) == before


def test_yukon_annual_vs_window_comparison_exists(snowmelt_result) -> None:
    comparison = pd.read_csv(SNOWMELT_TABLE_DIR / "annual_vs_snowmelt_signal_comparison.csv")
    yukon = comparison[comparison["river"].astype(str).eq("Yukon")]
    assert not yukon.empty
    assert yukon["does_window_explain_annual_signal"].isin({"yes", "no", "partial", "uncertain", "not_applicable"}).all()


def test_fixed_may_july_labeled_reference_only(snowmelt_result) -> None:
    definitions = pd.read_csv(SNOWMELT_TABLE_DIR / "snowmelt_window_definitions_by_river_year.csv")
    fixed = definitions[definitions["window_id"].eq("fixed_may_july_reference")]
    assert not fixed.empty
    assert fixed["definition_status"].eq("reference_only").all()
    assert fixed["caveat_reason"].astype(str).str.contains("reference_only").all()


def test_report_does_not_call_may_july_final_snowmelt(snowmelt_result) -> None:
    text = SNOWMELT_REPORT_PATH.read_text(encoding="utf-8").lower()
    assert "fixed may-july remains a provisional reference" in text
    assert "final snowmelt" not in text


def test_window_confidence_tier_exists(snowmelt_result) -> None:
    definitions = pd.read_csv(SNOWMELT_TABLE_DIR / "snowmelt_window_definitions_by_river_year.csv")
    assert definitions["window_confidence_tier"].isin({"high", "medium", "low"}).all()


def test_no_forbidden_inputs_loaded(monkeypatch) -> None:
    import pandas as pd_module

    original_read_csv = pd_module.read_csv
    read_paths: list[str] = []

    def spy_read_csv(filepath, *args, **kwargs):
        read_paths.append(str(filepath))
        return original_read_csv(filepath, *args, **kwargs)

    monkeypatch.setattr(pd_module, "read_csv", spy_read_csv)
    define_snowmelt_windows()
    forbidden = [
        "training_matrix_optical_matched",
        "training_matrix_basin_context",
        "lab_optical_proxy_gold",
        "data/raw",
        "data/interim",
        "data/canonical",
    ]
    assert not [read_path for read_path in read_paths if any(token in read_path.replace("\\", "/") for token in forbidden)]


def test_gold_data_unchanged_by_snowmelt_windows() -> None:
    gold_dir = require_gold_data_dir()
    contract = load_contract()
    before = {table_name: sha256_file(table_path(table_name, gold_dir=gold_dir)) for table_name in contract["expected_tables"]}
    define_snowmelt_windows()
    run_snowmelt_window_flux()
    after = {table_name: sha256_file(table_path(table_name, gold_dir=gold_dir)) for table_name in contract["expected_tables"]}
    assert after == before


def test_required_input_hashes_stable(snowmelt_result) -> None:
    before = {name: sha256_file(path) for name, path in required_snowmelt_input_paths().items()}
    define_snowmelt_windows()
    run_snowmelt_window_flux()
    after = {name: sha256_file(path) for name, path in required_snowmelt_input_paths().items()}
    assert after == before
