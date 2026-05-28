from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from arctic_doc_model_rebuild.flux.flux_cohorts import (
    FLUX_COHORT_REPORT_PATH,
    FLUX_INTERPRETATION_TABLE_DIR,
    REQUIRED_FLUX_INPUTS,
    select_flux_analysis_cohorts,
)
from arctic_doc_model_rebuild.gold_contract import load_contract, require_gold_data_dir, sha256_file, table_path
from arctic_doc_model_rebuild.paths import project_root


@pytest.fixture(scope="session")
def flux_cohort_result():
    return select_flux_analysis_cohorts()


def test_cohort_table_exists(flux_cohort_result) -> None:
    path = FLUX_INTERPRETATION_TABLE_DIR / "annual_flux_analysis_cohorts.csv"
    assert path.exists()
    cohorts = pd.read_csv(path)
    assert not cohorts.empty
    assert {"cohort_core_2003_2024", "cohort_sensitivity_only", "cohort_exclude_from_trend"}.issubset(cohorts.columns)


def test_all_annual_flux_rows_assigned_a_cohort_status(flux_cohort_result) -> None:
    cohorts = pd.read_csv(FLUX_INTERPRETATION_TABLE_DIR / "annual_flux_analysis_cohorts.csv")
    status_cols = [
        "cohort_core_2003_2024",
        "cohort_full_2000_2025",
        "cohort_high_confidence_only",
        "cohort_sensitivity_only",
        "cohort_exclude_from_trend",
    ]
    statuses = cohorts[status_cols].astype(str).apply(lambda column: column.str.lower().isin({"true", "1"}))
    assert statuses.any(axis=1).all()
    assert cohorts["cohort_full_2000_2025"].astype(str).str.lower().isin({"true", "1"}).all()


def test_core_cohort_excludes_low_coverage_rows(flux_cohort_result) -> None:
    cohorts = pd.read_csv(FLUX_INTERPRETATION_TABLE_DIR / "annual_flux_analysis_cohorts.csv")
    low_coverage_core = cohorts[(cohorts["coverage_rate"] < 0.95) & cohorts["cohort_core_2003_2024"].astype(str).str.lower().isin({"true", "1"})]
    assert low_coverage_core.empty


def test_core_cohort_excludes_high_low_confidence_fraction_rows(flux_cohort_result) -> None:
    cohorts = pd.read_csv(FLUX_INTERPRETATION_TABLE_DIR / "annual_flux_analysis_cohorts.csv")
    bad = cohorts[
        (cohorts["fraction_flux_from_low_confidence_days"] >= 0.25)
        & cohorts["cohort_core_2003_2024"].astype(str).str.lower().isin({"true", "1"})
    ]
    assert bad.empty


def test_may_july_report_does_not_call_it_final_snowmelt(flux_cohort_result) -> None:
    text = FLUX_COHORT_REPORT_PATH.read_text(encoding="utf-8").lower()
    assert "provisional" in text
    assert "may-july is provisional, not final snowmelt" in text
    assert "final snowmelt contribution estimate" not in text


def test_no_model_prediction_or_flux_recomputation_outputs(flux_cohort_result) -> None:
    root = project_root()
    assert not (root / "outputs" / "predictions").exists()
    assert not (root / "outputs" / "flux").exists()
    forbidden_legacy = [
        item
        for item in (root / "outputs").rglob("*")
        if item.is_file() and item.name.lower() in {"daily_flux.csv", "annual_flux.csv", "snowmelt_flux.csv"}
    ]
    assert forbidden_legacy == []


def test_doc_flux_inputs_unchanged_by_cohort_selection() -> None:
    before = {name: sha256_file(path) for name, path in REQUIRED_FLUX_INPUTS.items() if path.suffix.lower() in {".csv", ".md"}}
    select_flux_analysis_cohorts()
    after = {name: sha256_file(path) for name, path in REQUIRED_FLUX_INPUTS.items() if path.suffix.lower() in {".csv", ".md"}}
    assert before == after


def test_gold_data_unchanged_by_cohort_selection() -> None:
    gold_dir = require_gold_data_dir()
    contract = load_contract()
    before = {table_name: sha256_file(table_path(table_name, gold_dir=gold_dir)) for table_name in contract["expected_tables"]}
    select_flux_analysis_cohorts()
    after = {table_name: sha256_file(table_path(table_name, gold_dir=gold_dir)) for table_name in contract["expected_tables"]}
    assert before == after


def test_flux_cohort_summary_and_diagnostics_exist(flux_cohort_result) -> None:
    summary = pd.read_csv(FLUX_INTERPRETATION_TABLE_DIR / "flux_cohort_summary_by_river.csv")
    diagnostics = pd.read_csv(FLUX_INTERPRETATION_TABLE_DIR / "flux_confidence_diagnostics.csv")
    may_july = pd.read_csv(FLUX_INTERPRETATION_TABLE_DIR / "provisional_may_july_cohort_summary.csv")
    assert set(summary["river"]) == {"Kolyma", "Lena", "Mackenzie", "Ob", "Yenisey", "Yukon"}
    assert {"Yenisey_low_confidence_flux_issue", "Yukon_zero_or_near_zero_annual_flux_issue"}.issubset(set(diagnostics["diagnostic_item"]))
    assert "core_may_july_interpretation_allowed" in may_july.columns


def test_no_forbidden_inputs_loaded(monkeypatch) -> None:
    import pandas as pd_module

    original_read_csv = pd_module.read_csv
    read_paths: list[str] = []

    def spy_read_csv(filepath, *args, **kwargs):
        read_paths.append(str(filepath))
        return original_read_csv(filepath, *args, **kwargs)

    monkeypatch.setattr(pd_module, "read_csv", spy_read_csv)
    select_flux_analysis_cohorts()
    forbidden = [
        "training_matrix_hydrocore",
        "prediction_grid_daily",
        "training_matrix_optical_matched",
        "training_matrix_basin_context",
        "basin_attributes_curated",
        "lab_optical_proxy_gold",
    ]
    assert not [read_path for read_path in read_paths if any(token in Path(read_path).name for token in forbidden)]
