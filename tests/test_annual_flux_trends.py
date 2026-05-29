from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from arctic_doc_model_rebuild.flux.trend_analysis import (
    ANNUAL_TREND_REPORT_PATH,
    ANNUAL_TREND_TABLE_DIR,
    REQUIRED_TREND_INPUTS,
    run_annual_flux_trends,
)
from arctic_doc_model_rebuild.gold_contract import load_contract, require_gold_data_dir, sha256_file, table_path
from arctic_doc_model_rebuild.paths import project_root


@pytest.fixture(scope="session")
def annual_trend_result():
    return run_annual_flux_trends()


def test_annual_flux_trend_outputs_exist(annual_trend_result) -> None:
    assert ANNUAL_TREND_REPORT_PATH.exists()
    required = [
        "annual_flux_trend_input_audit.csv",
        "annual_flux_trends_by_river.csv",
        "annual_flux_trends_aggregate.csv",
        "annual_flux_trend_sensitivity_by_cohort.csv",
        "annual_flux_trend_uncertainty_sensitivity.csv",
        "annual_flux_trend_caveat_summary.csv",
    ]
    for name in required:
        assert (ANNUAL_TREND_TABLE_DIR / name).exists()


def test_core_cohort_used_for_primary_trend(annual_trend_result) -> None:
    trends = pd.read_csv(ANNUAL_TREND_TABLE_DIR / "annual_flux_trends_by_river.csv")
    core = trends[trends["analysis_cohort"].eq("core_2003_2024")]
    assert not core.empty
    assert core["year_min"].min() >= 2003
    assert core["year_max"].max() <= 2024


def test_excluded_rows_not_in_core_trend(annual_trend_result) -> None:
    cohorts = pd.read_csv(project_root() / "outputs" / "tables" / "flux_interpretation" / "annual_flux_analysis_cohorts.csv")
    trends = pd.read_csv(ANNUAL_TREND_TABLE_DIR / "annual_flux_trends_by_river.csv")
    core_counts = cohorts[cohorts["cohort_core_2003_2024"].astype(str).str.lower().isin({"true", "1"})].groupby("river").size()
    trend_core = trends[trends["analysis_cohort"].eq("core_2003_2024")].set_index("river")
    for river, count in core_counts.items():
        assert int(trend_core.loc[river, "n_years"]) == int(count)
    excluded_core = cohorts[
        cohorts["cohort_exclude_from_trend"].astype(str).str.lower().isin({"true", "1"})
        & cohorts["cohort_core_2003_2024"].astype(str).str.lower().isin({"true", "1"})
    ]
    assert excluded_core.empty


def test_full_cohort_marked_sensitivity(annual_trend_result) -> None:
    sensitivity = pd.read_csv(ANNUAL_TREND_TABLE_DIR / "annual_flux_trend_sensitivity_by_cohort.csv")
    full = sensitivity[sensitivity["analysis_cohort"].eq("full_2000_2025_sensitivity")]
    assert not full.empty
    assert full["primary_or_sensitivity"].eq("sensitivity").all()


def test_trend_tables_have_required_columns(annual_trend_result) -> None:
    by_river = pd.read_csv(ANNUAL_TREND_TABLE_DIR / "annual_flux_trends_by_river.csv")
    aggregate = pd.read_csv(ANNUAL_TREND_TABLE_DIR / "annual_flux_trends_aggregate.csv")
    required_river = {
        "analysis_cohort",
        "river",
        "n_years",
        "slope_ols_TgC_per_year",
        "slope_ols_p_value",
        "slope_theilsen_TgC_per_year",
        "kendall_tau",
        "kendall_p_value",
        "trend_direction",
        "trend_strength",
        "significant_at_0_05",
    }
    required_aggregate = {"analysis_cohort", "aggregate_type", "rivers_included", "slope_ols_TgC_per_year", "trend_direction"}
    assert required_river.issubset(by_river.columns)
    assert required_aggregate.issubset(aggregate.columns)


def test_uncertainty_sensitivity_uses_lower_upper_flux(annual_trend_result) -> None:
    uncertainty = pd.read_csv(ANNUAL_TREND_TABLE_DIR / "annual_flux_trend_uncertainty_sensitivity.csv")
    assert not uncertainty.empty
    assert {"slope_central", "slope_lower90", "slope_upper90", "trend_sign_robust_to_doc_uncertainty"}.issubset(uncertainty.columns)
    assert uncertainty["uncertainty_scope"].eq("DOC_concentration_empirical_residual_interval_only").all()
    assert uncertainty["discharge_uncertainty_included"].astype(str).str.lower().isin({"false", "0"}).all()


def test_no_model_retraining(annual_trend_result) -> None:
    audit = pd.read_csv(ANNUAL_TREND_TABLE_DIR / "annual_flux_trend_input_audit.csv")
    assert audit["trained_model"].astype(str).str.lower().isin({"false", "0"}).all()


def test_no_new_doc_prediction(annual_trend_result) -> None:
    root = project_root()
    assert not (root / "outputs" / "predictions").exists()
    audit = pd.read_csv(ANNUAL_TREND_TABLE_DIR / "annual_flux_trend_input_audit.csv")
    assert audit["regenerated_doc_prediction"].astype(str).str.lower().isin({"false", "0"}).all()


def test_no_flux_recalculation() -> None:
    before = {name: sha256_file(path) for name, path in REQUIRED_TREND_INPUTS.items()}
    daily_flux_path = project_root() / "outputs" / "tables" / "doc_flux" / "daily_doc_flux.csv"
    daily_before = sha256_file(daily_flux_path)
    run_annual_flux_trends()
    after = {name: sha256_file(path) for name, path in REQUIRED_TREND_INPUTS.items()}
    assert before == after
    assert daily_before == sha256_file(daily_flux_path)


def test_may_july_not_called_snowmelt(annual_trend_result) -> None:
    text = ANNUAL_TREND_REPORT_PATH.read_text(encoding="utf-8").lower()
    assert "annual may-july or snowmelt was not analyzed here" in text
    assert "final snowmelt" not in text


def test_report_mentions_discharge_uncertainty_not_propagated(annual_trend_result) -> None:
    text = ANNUAL_TREND_REPORT_PATH.read_text(encoding="utf-8").lower()
    assert "discharge uncertainty is not propagated" in text


def test_report_uses_detectable_trend_language(annual_trend_result) -> None:
    text = ANNUAL_TREND_REPORT_PATH.read_text(encoding="utf-8").lower()
    assert "no detectable trend" in text


def test_gold_data_unchanged_by_trend_analysis() -> None:
    gold_dir = require_gold_data_dir()
    contract = load_contract()
    before = {table_name: sha256_file(table_path(table_name, gold_dir=gold_dir)) for table_name in contract["expected_tables"]}
    run_annual_flux_trends()
    after = {table_name: sha256_file(table_path(table_name, gold_dir=gold_dir)) for table_name in contract["expected_tables"]}
    assert before == after


def test_no_forbidden_inputs_loaded(monkeypatch) -> None:
    import pandas as pd_module

    original_read_csv = pd_module.read_csv
    read_paths: list[str] = []

    def spy_read_csv(filepath, *args, **kwargs):
        read_paths.append(str(filepath))
        return original_read_csv(filepath, *args, **kwargs)

    monkeypatch.setattr(pd_module, "read_csv", spy_read_csv)
    run_annual_flux_trends()
    forbidden = [
        "daily_doc_flux",
        "training_matrix_hydrocore",
        "prediction_grid_daily",
        "training_matrix_optical_matched",
        "training_matrix_basin_context",
        "basin_attributes_curated",
        "lab_optical_proxy_gold",
    ]
    assert not [read_path for read_path in read_paths if any(token in Path(read_path).name for token in forbidden)]
