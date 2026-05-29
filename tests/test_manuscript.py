from __future__ import annotations

import pandas as pd
import pytest

from arctic_doc_model_rebuild.gold_contract import sha256_file
from arctic_doc_model_rebuild.manuscript import (
    ABSTRACT_DRAFT_PATH,
    CLAIMS_TO_AVOID_PATH,
    FIGURE_PLAN_PATH,
    KEY_CLAIMS_PATH,
    MANUSCRIPT_OUTLINE_PATH,
    RESULTS_NARRATIVE_PATH,
    TABLE_PLAN_PATH,
    draft_manuscript_outline,
)
from arctic_doc_model_rebuild.paths import project_root


@pytest.fixture(scope="session")
def manuscript_result():
    return draft_manuscript_outline()


def test_manuscript_outline_exists(manuscript_result) -> None:
    assert MANUSCRIPT_OUTLINE_PATH.exists()
    text = MANUSCRIPT_OUTLINE_PATH.read_text(encoding="utf-8")
    for section in ["Title options", "Abstract skeleton", "Introduction logic", "Methods outline", "Results outline", "Discussion outline", "Limitations", "Conclusion"]:
        assert section in text


def test_results_narrative_exists(manuscript_result) -> None:
    assert RESULTS_NARRATIVE_PATH.exists()
    text = RESULTS_NARRATIVE_PATH.read_text(encoding="utf-8")
    assert "Yukon is the only river" in text
    assert "discharge-volume dominated" in text


def test_abstract_draft_exists(manuscript_result) -> None:
    assert ABSTRACT_DRAFT_PATH.exists()
    text = ABSTRACT_DRAFT_PATH.read_text(encoding="utf-8")
    assert "background" not in text.lower()
    assert "discharge uncertainty is not propagated" in text


def test_key_claims_map_exists(manuscript_result) -> None:
    assert KEY_CLAIMS_PATH.exists()
    claims = pd.read_csv(KEY_CLAIMS_PATH)
    assert {"claim", "evidence_table", "figure_candidate", "strength", "caveat", "manuscript_section"}.issubset(claims.columns)
    assert len(claims) >= 8


def test_claims_to_avoid_exists(manuscript_result) -> None:
    assert CLAIMS_TO_AVOID_PATH.exists()
    text = CLAIMS_TO_AVOID_PATH.read_text(encoding="utf-8").lower()
    required = [
        "no pan-arctic doc increase",
        "no all-river increase",
        "no optical improvement claim",
        "no causality proof",
        "no discharge uncertainty propagation",
        "no final snowmelt attribution",
        "no extrapolation beyond six arcticgro rivers",
    ]
    for item in required:
        assert item in text


def test_figure_and_table_plans_exist(manuscript_result) -> None:
    figures = pd.read_csv(FIGURE_PLAN_PATH)
    tables = pd.read_csv(TABLE_PLAN_PATH)
    assert set(figures["figure_id"]) == {f"Figure {idx}" for idx in range(1, 8)}
    assert set(tables["table_id"]) == {f"Table {idx}" for idx in range(1, 6)}


def test_no_model_prediction_or_flux_generated() -> None:
    daily_prediction = project_root() / "outputs" / "tables" / "daily_doc_prediction" / "daily_doc_prediction.csv"
    daily_flux = project_root() / "outputs" / "tables" / "doc_flux" / "daily_doc_flux.csv"
    before_prediction = sha256_file(daily_prediction)
    before_flux = sha256_file(daily_flux)
    draft_manuscript_outline()
    assert sha256_file(daily_prediction) == before_prediction
    assert sha256_file(daily_flux) == before_flux
    assert not (project_root() / "outputs" / "predictions").exists()
    assert not (project_root() / "outputs" / "flux").exists()


def test_manuscript_tables_do_not_claim_causal_proof(manuscript_result) -> None:
    claims = pd.read_csv(KEY_CLAIMS_PATH)
    joined = " ".join(claims["caveat"].astype(str)).lower()
    assert "not causal proof" in joined
