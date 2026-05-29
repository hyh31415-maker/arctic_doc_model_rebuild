from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from arctic_doc_model_rebuild.final_synthesis import (
    FINAL_REPORT_DIR,
    FINAL_SYNTHESIS_REPORT_PATH,
    FINAL_TABLE_DIR,
    synthesize_results,
    write_final_synthesis_report,
)
from arctic_doc_model_rebuild.gold_contract import sha256_file
from arctic_doc_model_rebuild.modeling.diagnostics import ALLOWED_DOC_MODEL_ARTIFACTS
from arctic_doc_model_rebuild.paths import project_root


@pytest.fixture(scope="session")
def final_synthesis_result():
    return synthesize_results()


def test_final_synthesis_report_exists(final_synthesis_result) -> None:
    assert FINAL_SYNTHESIS_REPORT_PATH.exists()
    assert FINAL_SYNTHESIS_REPORT_PATH.stat().st_size > 0


def test_core_findings_table_exists(final_synthesis_result) -> None:
    path = FINAL_TABLE_DIR / "core_findings.csv"
    assert path.exists()
    findings = pd.read_csv(path)
    required = {"finding_id", "finding_category", "finding_statement", "evidence_table_or_report", "strength", "caveat", "manuscript_ready"}
    assert required.issubset(findings.columns)
    assert len(findings) >= 12


def test_core_findings_include_yukon_increasing_annual_flux(final_synthesis_result) -> None:
    findings = pd.read_csv(FINAL_TABLE_DIR / "core_findings.csv")
    text = " ".join(findings["finding_statement"].astype(str)).lower()
    assert "yukon" in text
    assert "detectable increasing annual doc flux" in text


def test_core_findings_include_aggregate_no_detectable_trend(final_synthesis_result) -> None:
    findings = pd.read_csv(FINAL_TABLE_DIR / "core_findings.csv")
    text = " ".join(findings["finding_statement"].astype(str)).lower()
    assert "aggregate annual doc flux has no detectable trend" in text


def test_core_findings_include_optical_no_incremental_value(final_synthesis_result) -> None:
    findings = pd.read_csv(FINAL_TABLE_DIR / "core_findings.csv")
    optical = findings[findings["finding_category"].astype(str).str.lower().eq("optical")]
    assert not optical.empty
    text = " ".join(optical["finding_statement"].astype(str)).lower()
    assert "did not improve" in text
    assert "excluded from the primary model" in text


def test_core_findings_include_may_july_not_explaining_yukon(final_synthesis_result) -> None:
    findings = pd.read_csv(FINAL_TABLE_DIR / "core_findings.csv")
    text = " ".join(findings["finding_statement"].astype(str)).lower()
    assert "fixed may-july flux does not explain yukon annual increase" in text


def test_snowmelt_summary_marks_yukon_partial_not_yes(final_synthesis_result) -> None:
    summary = pd.read_csv(FINAL_TABLE_DIR / "snowmelt_interpretation_summary.csv")
    yukon = summary[summary["river"].astype(str).eq("Yukon")]
    assert not yukon.empty
    value = str(yukon.iloc[0]["does_snowmelt_window_explain_annual_signal"])
    assert value == "partial"
    assert value != "yes"
    assert "partial" in str(yukon.iloc[0]["interpretation"]).lower()


def test_caveat_register_includes_discharge_uncertainty_not_propagated(final_synthesis_result) -> None:
    caveats = pd.read_csv(FINAL_TABLE_DIR / "caveat_register.csv")
    text = " ".join(caveats.astype(str).agg(" ".join, axis=1).tolist()).lower()
    assert "discharge uncertainty" in text
    assert "not propagated" in text


def test_what_not_to_claim_section_exists(final_synthesis_result) -> None:
    text = FINAL_SYNTHESIS_REPORT_PATH.read_text(encoding="utf-8").lower()
    assert "## 17. what not to claim" in text
    assert "do not claim a pan-arctic doc flux increase" in text
    assert "do not claim yukon annual increase is definitively snowmelt-driven" in text


def test_final_synthesis_does_not_create_new_model_training_outputs() -> None:
    root = project_root()
    models_dir = root / "outputs" / "models"
    before = {item.name for item in models_dir.iterdir() if item.is_file()} if models_dir.exists() else set()
    synthesize_results()
    after = {item.name for item in models_dir.iterdir() if item.is_file()} if models_dir.exists() else set()
    assert after == before
    assert after.issubset(ALLOWED_DOC_MODEL_ARTIFACTS)


def test_final_synthesis_does_not_modify_doc_prediction_or_flux() -> None:
    root = project_root()
    guarded_outputs = [
        root / "outputs" / "tables" / "daily_doc_prediction" / "daily_doc_prediction.csv",
        root / "outputs" / "tables" / "doc_flux" / "daily_doc_flux.csv",
        root / "outputs" / "tables" / "doc_flux" / "annual_doc_flux_summary.csv",
    ]
    before = {path: sha256_file(path) for path in guarded_outputs}
    synthesize_results()
    write_final_synthesis_report()
    after = {path: sha256_file(path) for path in guarded_outputs}
    assert after == before


def test_synthesis_report_command_style_rewrites_report_only(final_synthesis_result) -> None:
    before_tables = {path.name: sha256_file(path) for path in FINAL_TABLE_DIR.glob("*.csv")}
    path = write_final_synthesis_report()
    after_tables = {path.name: sha256_file(path) for path in FINAL_TABLE_DIR.glob("*.csv")}
    assert path == FINAL_REPORT_DIR / "final_synthesis_report.md"
    assert after_tables == before_tables
