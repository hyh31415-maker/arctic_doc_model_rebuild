from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from arctic_doc_model_rebuild.flux.may_july_interpretation import (
    MAY_JULY_REPORT_PATH,
    MAY_JULY_TABLE_DIR,
    REQUIRED_MAY_JULY_INPUTS,
    run_may_july_flux_interpretation,
)
from arctic_doc_model_rebuild.gold_contract import load_contract, require_gold_data_dir, sha256_file, table_path
from arctic_doc_model_rebuild.paths import project_root


@pytest.fixture(scope="session")
def may_july_result():
    return run_may_july_flux_interpretation()


def test_may_july_outputs_exist(may_july_result) -> None:
    assert MAY_JULY_REPORT_PATH.exists()
    required = [
        "may_july_flux_interpretation_by_river_year.csv",
        "may_july_fraction_summary_by_river.csv",
        "may_july_flux_trends_by_river.csv",
        "may_july_vs_annual_trend_comparison.csv",
        "may_july_caveat_summary.csv",
    ]
    for name in required:
        assert (MAY_JULY_TABLE_DIR / name).exists()


def test_may_july_report_does_not_call_it_final_snowmelt(may_july_result) -> None:
    text = MAY_JULY_REPORT_PATH.read_text(encoding="utf-8").lower()
    assert "may-july is provisional, not final snowmelt" in text
    assert "is final snowmelt" not in text
    assert "final snowmelt flux" not in text


def test_no_model_retraining(may_july_result) -> None:
    root = project_root()
    model_artifacts = sorted((root / "outputs" / "models").glob("*")) if (root / "outputs" / "models").exists() else []
    allowed = {"production_candidate_r4_daily_doc_model.joblib", "production_candidate_r4_daily_doc_model_metadata.json"}
    assert {item.name for item in model_artifacts}.issubset(allowed)


def test_no_new_doc_prediction() -> None:
    daily_prediction = project_root() / "outputs" / "tables" / "daily_doc_prediction" / "daily_doc_prediction.csv"
    before = sha256_file(daily_prediction)
    run_may_july_flux_interpretation()
    assert sha256_file(daily_prediction) == before


def test_no_flux_recalculation() -> None:
    before = {name: sha256_file(path) for name, path in REQUIRED_MAY_JULY_INPUTS.items()}
    run_may_july_flux_interpretation()
    after = {name: sha256_file(path) for name, path in REQUIRED_MAY_JULY_INPUTS.items()}
    assert after == before


def test_core_cohort_used_for_primary_interpretation(may_july_result) -> None:
    rows = pd.read_csv(MAY_JULY_TABLE_DIR / "may_july_flux_interpretation_by_river_year.csv")
    trends = pd.read_csv(MAY_JULY_TABLE_DIR / "may_july_flux_trends_by_river.csv")
    core_rows = rows[rows["cohort_core_2003_2024"].astype(str).str.lower().isin({"true", "1"})]
    assert not core_rows.empty
    core_trends = trends[trends["analysis_cohort"].eq("core_2003_2024")]
    assert not core_trends.empty
    assert core_trends["year_min"].min() >= 2003
    assert core_trends["year_max"].max() <= 2024


def test_trend_language_uses_detectable_terms(may_july_result) -> None:
    trends = pd.read_csv(MAY_JULY_TABLE_DIR / "may_july_flux_trends_by_river.csv")
    assert trends["trend_language"].isin({"detectable trend", "no detectable trend"}).all()
    text = MAY_JULY_REPORT_PATH.read_text(encoding="utf-8").lower()
    assert "no detectable trend" in text
    assert "no trend" not in text


def test_yukon_annual_vs_may_july_comparison_exists(may_july_result) -> None:
    comparison = pd.read_csv(MAY_JULY_TABLE_DIR / "may_july_vs_annual_trend_comparison.csv")
    yukon = comparison[comparison["river"].astype(str).eq("Yukon")]
    assert len(yukon) == 1
    assert yukon.iloc[0]["does_may_july_explain_annual_signal"] in {"yes", "no", "partial", "uncertain", "not_applicable"}


def test_no_forbidden_inputs_loaded(monkeypatch) -> None:
    import pandas as pd_module

    original_read_csv = pd_module.read_csv
    read_paths: list[str] = []

    def spy_read_csv(filepath, *args, **kwargs):
        read_paths.append(str(filepath))
        return original_read_csv(filepath, *args, **kwargs)

    monkeypatch.setattr(pd_module, "read_csv", spy_read_csv)
    run_may_july_flux_interpretation()
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


def test_gold_data_unchanged_by_may_july_interpretation() -> None:
    gold_dir = require_gold_data_dir()
    contract = load_contract()
    before = {table_name: sha256_file(table_path(table_name, gold_dir=gold_dir)) for table_name in contract["expected_tables"]}
    run_may_july_flux_interpretation()
    after = {table_name: sha256_file(table_path(table_name, gold_dir=gold_dir)) for table_name in contract["expected_tables"]}
    assert after == before
