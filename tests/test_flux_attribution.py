from __future__ import annotations

import pandas as pd
import pytest

from arctic_doc_model_rebuild.flux.flux_attribution import (
    FLUX_ATTRIBUTION_REPORT_PATH,
    FLUX_ATTRIBUTION_TABLE_DIR,
    REQUIRED_ATTRIBUTION_INPUTS,
    YUKON_ATTRIBUTION_REPORT_PATH,
    run_flux_attribution,
)
from arctic_doc_model_rebuild.gold_contract import load_contract, require_gold_data_dir, sha256_file, table_path
from arctic_doc_model_rebuild.paths import project_root


@pytest.fixture(scope="session")
def flux_attribution_result():
    return run_flux_attribution()


def test_attribution_outputs_exist(flux_attribution_result) -> None:
    assert FLUX_ATTRIBUTION_REPORT_PATH.exists()
    assert YUKON_ATTRIBUTION_REPORT_PATH.exists()
    required = [
        "annual_flux_attribution_by_river_year.csv",
        "q_doc_component_trends_by_river.csv",
        "flux_driver_classification.csv",
        "monthly_flux_by_river_year.csv",
        "monthly_flux_trends_by_river.csv",
        "seasonal_flux_decomposition_by_river_year.csv",
        "seasonal_flux_trends_by_river.csv",
        "export_phenology_by_river_year.csv",
        "export_phenology_trends_by_river.csv",
        "yukon_flux_attribution_summary.csv",
    ]
    for name in required:
        assert (FLUX_ATTRIBUTION_TABLE_DIR / name).exists()


def test_annual_q_volume_computed(flux_attribution_result) -> None:
    annual = pd.read_csv(FLUX_ATTRIBUTION_TABLE_DIR / "annual_flux_attribution_by_river_year.csv")
    assert "annual_Q_volume_km3" in annual.columns
    assert annual["annual_Q_volume_km3"].notna().any()
    assert (annual["annual_Q_volume_km3"].dropna() > 0).all()


def test_flow_weighted_doc_computed(flux_attribution_result) -> None:
    annual = pd.read_csv(FLUX_ATTRIBUTION_TABLE_DIR / "annual_flux_attribution_by_river_year.csv")
    assert "flow_weighted_DOC_mgC_L" in annual.columns
    assert annual["flow_weighted_DOC_mgC_L"].notna().any()
    assert (annual["flow_weighted_DOC_mgC_L"].dropna() > 0).all()


def test_yukon_driver_classification_exists(flux_attribution_result) -> None:
    drivers = pd.read_csv(FLUX_ATTRIBUTION_TABLE_DIR / "flux_driver_classification.csv")
    yukon = drivers[drivers["river"].astype(str).eq("Yukon")]
    assert not yukon.empty
    assert yukon["driver_classification"].isin(
        {
            "discharge_volume_dominated",
            "concentration_dominated",
            "combined_Q_and_DOC",
            "seasonal_redistribution_or_unresolved",
            "not_applicable_no_detectable_annual_trend",
        }
    ).all()


def test_monthly_decomposition_has_valid_month_fields(flux_attribution_result) -> None:
    monthly = pd.read_csv(FLUX_ATTRIBUTION_TABLE_DIR / "monthly_flux_by_river_year.csv")
    assert {"month", "monthly_flux_TgC", "monthly_fraction_of_annual", "monthly_Q_volume_km3", "monthly_flow_weighted_DOC"}.issubset(monthly.columns)
    assert monthly["month"].between(1, 12).all()


def test_seasonal_decomposition_exists(flux_attribution_result) -> None:
    seasonal = pd.read_csv(FLUX_ATTRIBUTION_TABLE_DIR / "seasonal_flux_decomposition_by_river_year.csv")
    required = {"winter", "spring_transition", "may_july", "late_summer", "fall_winter"}
    assert required.issubset(set(seasonal["season_window"].astype(str)))


def test_export_phenology_dates_are_monotonic(flux_attribution_result) -> None:
    phenology = pd.read_csv(FLUX_ATTRIBUTION_TABLE_DIR / "export_phenology_by_river_year.csv")
    ordered = phenology[["flux_10pct_doy", "flux_25pct_doy", "flux_50pct_doy", "flux_75pct_doy", "flux_90pct_doy"]].dropna()
    assert not ordered.empty
    assert (ordered["flux_10pct_doy"] <= ordered["flux_25pct_doy"]).all()
    assert (ordered["flux_25pct_doy"] <= ordered["flux_50pct_doy"]).all()
    assert (ordered["flux_50pct_doy"] <= ordered["flux_75pct_doy"]).all()
    assert (ordered["flux_75pct_doy"] <= ordered["flux_90pct_doy"]).all()


def test_no_model_retraining(flux_attribution_result) -> None:
    models_dir = project_root() / "outputs" / "models"
    allowed = {"production_candidate_r4_daily_doc_model.joblib", "production_candidate_r4_daily_doc_model_metadata.json"}
    model_files = {item.name for item in models_dir.glob("*") if item.is_file()} if models_dir.exists() else set()
    assert model_files.issubset(allowed)


def test_no_new_doc_prediction() -> None:
    daily_prediction = REQUIRED_ATTRIBUTION_INPUTS["daily_doc_prediction"]
    before = sha256_file(daily_prediction)
    run_flux_attribution()
    assert sha256_file(daily_prediction) == before
    assert not (project_root() / "outputs" / "predictions").exists()


def test_no_flux_recalculation() -> None:
    input_hashes = {name: sha256_file(path) for name, path in REQUIRED_ATTRIBUTION_INPUTS.items()}
    daily_flux = REQUIRED_ATTRIBUTION_INPUTS["daily_doc_flux"]
    daily_before = sha256_file(daily_flux)
    run_flux_attribution()
    assert {name: sha256_file(path) for name, path in REQUIRED_ATTRIBUTION_INPUTS.items()} == input_hashes
    assert sha256_file(daily_flux) == daily_before


def test_report_includes_discharge_uncertainty_caveat(flux_attribution_result) -> None:
    text = FLUX_ATTRIBUTION_REPORT_PATH.read_text(encoding="utf-8").lower()
    assert "discharge uncertainty was not propagated" in text


def test_report_says_exploratory_not_causal_proof(flux_attribution_result) -> None:
    text = FLUX_ATTRIBUTION_REPORT_PATH.read_text(encoding="utf-8").lower()
    assert "exploratory mechanism analysis, not causal proof" in text


def test_gold_data_unchanged_by_flux_attribution() -> None:
    gold_dir = require_gold_data_dir()
    contract = load_contract()
    before = {table_name: sha256_file(table_path(table_name, gold_dir=gold_dir)) for table_name in contract["expected_tables"]}
    run_flux_attribution()
    after = {table_name: sha256_file(table_path(table_name, gold_dir=gold_dir)) for table_name in contract["expected_tables"]}
    assert after == before
