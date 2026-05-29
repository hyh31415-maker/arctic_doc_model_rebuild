from __future__ import annotations

import pandas as pd
import pytest

from arctic_doc_model_rebuild.flux.freshet_control import (
    FRESHET_REPORT_PATH,
    FRESHET_TABLE_DIR,
    REQUIRED_FRESHET_INPUTS,
    synthesize_freshet_control,
)
from arctic_doc_model_rebuild.gold_contract import sha256_file
from arctic_doc_model_rebuild.paths import project_root


@pytest.fixture(scope="session")
def freshet_result():
    return synthesize_freshet_control()


def test_freshet_outputs_exist(freshet_result) -> None:
    assert FRESHET_REPORT_PATH.exists()
    required = [
        "freshet_control_summary_by_river.csv",
        "annual_flux_vs_window_flux_coupling.csv",
        "export_regime_classification.csv",
        "yukon_extended_season_diagnosis.csv",
    ]
    for name in required:
        assert (FRESHET_TABLE_DIR / name).exists()


def test_regime_classification_exists_for_all_six_rivers(freshet_result) -> None:
    regimes = pd.read_csv(FRESHET_TABLE_DIR / "export_regime_classification.csv")
    assert set(regimes["river"]) == {"Kolyma", "Lena", "Mackenzie", "Ob", "Yenisey", "Yukon"}
    assert regimes["assigned_regime"].notna().all()


def test_yukon_classified_as_extended_season(freshet_result) -> None:
    regimes = pd.read_csv(FRESHET_TABLE_DIR / "export_regime_classification.csv")
    yukon = regimes[regimes["river"].eq("Yukon")]
    assert not yukon.empty
    assert yukon.iloc[0]["assigned_regime"] == "discharge_volume_extended_season"
    diagnosis = pd.read_csv(FRESHET_TABLE_DIR / "yukon_extended_season_diagnosis.csv")
    final = diagnosis[diagnosis["diagnostic_item"].eq("final_interpretation")].iloc[0]
    assert "discharge-volume-driven extended-season export expansion" in final["evidence"]


def test_annual_flux_vs_window_flux_coupling_exists(freshet_result) -> None:
    coupling = pd.read_csv(FRESHET_TABLE_DIR / "annual_flux_vs_window_flux_coupling.csv")
    assert not coupling.empty
    assert {"annual_flux_window_flux_correlation", "ols_r2_annual_flux_vs_window_flux", "annual_variability_explained_category"}.issubset(coupling.columns)
    assert coupling["window_id"].astype(str).str.contains("q75_peak_contiguous").any()


def test_report_includes_not_causal_proof(freshet_result) -> None:
    text = FRESHET_REPORT_PATH.read_text(encoding="utf-8").lower()
    assert "exploratory mechanism analysis, not causal proof" in text


def test_no_model_retraining(freshet_result) -> None:
    models_dir = project_root() / "outputs" / "models"
    allowed = {"production_candidate_r4_daily_doc_model.joblib", "production_candidate_r4_daily_doc_model_metadata.json"}
    model_files = {item.name for item in models_dir.glob("*") if item.is_file()} if models_dir.exists() else set()
    assert model_files.issubset(allowed)


def test_no_new_prediction() -> None:
    daily_prediction = project_root() / "outputs" / "tables" / "daily_doc_prediction" / "daily_doc_prediction.csv"
    before = sha256_file(daily_prediction)
    synthesize_freshet_control()
    assert sha256_file(daily_prediction) == before
    assert not (project_root() / "outputs" / "predictions").exists()


def test_no_flux_recalculation() -> None:
    before = {name: sha256_file(path) for name, path in REQUIRED_FRESHET_INPUTS.items()}
    daily_flux = REQUIRED_FRESHET_INPUTS["daily_doc_flux"]
    daily_before = sha256_file(daily_flux)
    synthesize_freshet_control()
    assert {name: sha256_file(path) for name, path in REQUIRED_FRESHET_INPUTS.items()} == before
    assert sha256_file(daily_flux) == daily_before


def test_report_states_operational_windows_and_uncertainty(freshet_result) -> None:
    text = FRESHET_REPORT_PATH.read_text(encoding="utf-8").lower()
    assert "snowmelt/freshet windows are operational definitions" in text
    assert "discharge uncertainty was not propagated" in text
