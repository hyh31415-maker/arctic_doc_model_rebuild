from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from arctic_doc_model_rebuild.flux.flux_calculation import DOC_FLUX_REPORT_PATH, DOC_FLUX_TABLE_DIR, run_doc_flux
from arctic_doc_model_rebuild.gold_contract import load_contract, require_gold_data_dir, sha256_file, table_path
from arctic_doc_model_rebuild.paths import project_root


@pytest.fixture(scope="session")
def doc_flux_result():
    return run_doc_flux()


def test_daily_doc_flux_exists(doc_flux_result) -> None:
    assert (DOC_FLUX_TABLE_DIR / "daily_doc_flux.csv").exists()
    daily = pd.read_csv(DOC_FLUX_TABLE_DIR / "daily_doc_flux.csv")
    assert not daily.empty
    assert daily["is_flux"].astype(str).str.lower().isin({"true", "1"}).all()
    assert daily["is_doc_prediction"].astype(str).str.lower().isin({"true", "1"}).all()


def test_flux_formula_unit_conversion(doc_flux_result) -> None:
    daily = pd.read_csv(DOC_FLUX_TABLE_DIR / "daily_doc_flux.csv")
    row = daily.iloc[0]
    expected_kg = row["DOC_predicted_mgC_L"] * row["Q_m3s"] * 86.4
    assert abs(row["daily_flux_kgC_day"] - expected_kg) < 1e-8
    assert abs(row["daily_flux_MgC_day"] - row["daily_flux_kgC_day"] / 1000.0) < 1e-12
    assert abs(row["daily_flux_TgC_day"] - row["daily_flux_kgC_day"] / 1e9) < 1e-15


def test_flux_nonnegative_or_flagged(doc_flux_result) -> None:
    daily = pd.read_csv(DOC_FLUX_TABLE_DIR / "daily_doc_flux.csv")
    flags = pd.read_csv(DOC_FLUX_TABLE_DIR / "doc_flux_range_flags.csv")
    negative_rows = daily[daily["daily_flux_kgC_day"] < 0]
    if negative_rows.empty:
        assert "daily_flux_negative" not in set(flags["flag_type"])
    else:
        assert "daily_flux_negative" in set(flags["flag_type"])


def test_annual_flux_summary_exists(doc_flux_result) -> None:
    annual = pd.read_csv(DOC_FLUX_TABLE_DIR / "annual_doc_flux_summary.csv")
    assert not annual.empty
    assert {"river", "year", "annual_flux_TgC", "coverage_rate", "annual_confidence_tier"}.issubset(annual.columns)


def test_may_july_flux_summary_exists(doc_flux_result) -> None:
    may_july = pd.read_csv(DOC_FLUX_TABLE_DIR / "provisional_may_july_flux_summary.csv")
    assert not may_july.empty
    assert "window_label" in may_july.columns
    assert may_july["window_label"].astype(str).str.contains("not_final_snowmelt").all()


def test_flux_intervals_exist(doc_flux_result) -> None:
    daily = pd.read_csv(DOC_FLUX_TABLE_DIR / "daily_doc_flux.csv")
    required = {
        "daily_flux_80_lower_kgC_day",
        "daily_flux_80_upper_kgC_day",
        "daily_flux_90_lower_TgC_day",
        "daily_flux_90_upper_TgC_day",
        "daily_flux_95_lower_TgC_day",
        "daily_flux_95_upper_TgC_day",
    }
    assert required.issubset(daily.columns)
    assert daily[list(required)].notna().all().all()


def test_flux_carries_extrapolation_flags(doc_flux_result) -> None:
    daily = pd.read_csv(DOC_FLUX_TABLE_DIR / "daily_doc_flux.csv")
    required = {
        "outside_training_logQ_range",
        "outside_training_doy_range",
        "outside_training_year_range",
        "point_prediction_clipped_at_zero",
        "interval_lower_clipped_at_zero",
    }
    assert required.issubset(daily.columns)


def test_flux_confidence_tier_exists(doc_flux_result) -> None:
    daily = pd.read_csv(DOC_FLUX_TABLE_DIR / "daily_doc_flux.csv")
    summary = pd.read_csv(DOC_FLUX_TABLE_DIR / "doc_flux_confidence_tier_summary.csv")
    assert "daily_confidence_tier" in daily.columns
    assert set(daily["daily_confidence_tier"]).issubset({"high", "medium", "low"})
    assert not summary.empty


def test_no_gold_data_modified() -> None:
    gold_dir = require_gold_data_dir()
    contract = load_contract()
    before = {table_name: sha256_file(table_path(table_name, gold_dir=gold_dir)) for table_name in contract["expected_tables"]}
    run_doc_flux()
    after = {table_name: sha256_file(table_path(table_name, gold_dir=gold_dir)) for table_name in contract["expected_tables"]}
    assert before == after


def test_no_new_doc_model_trained(doc_flux_result) -> None:
    qc = pd.read_csv(DOC_FLUX_TABLE_DIR / "doc_flux_qc_summary.csv")
    row = qc[qc["qc_item"].eq("new_doc_model_trained")].iloc[0]
    assert row["status"] == "false"


def test_may_july_not_labeled_final_snowmelt(doc_flux_result) -> None:
    text = DOC_FLUX_REPORT_PATH.read_text(encoding="utf-8")
    assert "May-July flux is provisional, not final snowmelt contribution" in text


def test_discharge_uncertainty_not_claimed(doc_flux_result) -> None:
    text = DOC_FLUX_REPORT_PATH.read_text(encoding="utf-8")
    assert "Discharge uncertainty was not propagated" in text
    qc = pd.read_csv(DOC_FLUX_TABLE_DIR / "doc_flux_qc_summary.csv")
    row = qc[qc["qc_item"].eq("discharge_uncertainty_propagated")].iloc[0]
    assert row["status"] == "false"


def test_flux_report_exists(doc_flux_result) -> None:
    assert DOC_FLUX_REPORT_PATH.exists()
    text = DOC_FLUX_REPORT_PATH.read_text(encoding="utf-8")
    assert "Daily DOC flux was calculated" in text


def test_no_legacy_flux_outputs(doc_flux_result) -> None:
    root = project_root()
    assert not (root / "outputs" / "flux").exists()
    forbidden = [
        item
        for item in (root / "outputs").rglob("*")
        if item.is_file() and item.name.lower() in {"daily_flux.csv", "annual_flux.csv", "snowmelt_flux.csv"}
    ]
    assert forbidden == []


def test_no_forbidden_inputs_loaded(monkeypatch) -> None:
    import pandas as pd_module

    original_read_csv = pd_module.read_csv
    read_paths: list[str] = []

    def spy_read_csv(filepath, *args, **kwargs):
        read_paths.append(str(filepath))
        return original_read_csv(filepath, *args, **kwargs)

    monkeypatch.setattr(pd_module, "read_csv", spy_read_csv)
    run_doc_flux()
    forbidden = [
        "training_matrix_hydrocore",
        "prediction_grid_daily",
        "training_matrix_optical_matched",
        "training_matrix_basin_context",
        "basin_attributes_curated",
        "lab_optical_proxy_gold",
    ]
    assert not [read_path for read_path in read_paths if any(token in Path(read_path).name for token in forbidden)]
