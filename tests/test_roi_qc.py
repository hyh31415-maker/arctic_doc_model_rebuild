from __future__ import annotations

import pandas as pd
import pytest

from arctic_doc_model_rebuild.gold_contract import load_contract, require_gold_data_dir, sha256_file, table_path
from arctic_doc_model_rebuild.modeling.diagnostics import ALLOWED_DOC_MODEL_ARTIFACTS
from arctic_doc_model_rebuild.paths import project_root
from arctic_doc_model_rebuild.roi_qc import ROI_QC_REPORT_PATH, ROI_QC_TABLE_DIR, run_roi_final_qc


@pytest.fixture(scope="session")
def roi_qc_result():
    return run_roi_final_qc()


def test_roi_final_qc_summary_exists(roi_qc_result) -> None:
    assert ROI_QC_REPORT_PATH.exists()
    assert (ROI_QC_TABLE_DIR / "roi_final_qc_summary.csv").exists()


def test_all_six_rivers_present(roi_qc_result) -> None:
    summary = pd.read_csv(ROI_QC_TABLE_DIR / "roi_final_qc_summary.csv")
    assert set(summary["river"]) == {"Kolyma", "Lena", "Mackenzie", "Ob", "Yenisey", "Yukon"}


def test_roi_qc_no_gold_data_modified() -> None:
    gold_dir = require_gold_data_dir()
    contract = load_contract()
    before = {table_name: sha256_file(table_path(table_name, gold_dir=gold_dir)) for table_name in contract["expected_tables"]}
    run_roi_final_qc()
    after = {table_name: sha256_file(table_path(table_name, gold_dir=gold_dir)) for table_name in contract["expected_tables"]}
    assert before == after


def test_roi_qc_no_model_prediction_or_flux_outputs(roi_qc_result) -> None:
    root = project_root()
    model_dir = root / "outputs" / "models"
    if model_dir.exists():
        assert {item.name for item in model_dir.iterdir() if item.is_file()}.issubset(ALLOWED_DOC_MODEL_ARTIFACTS)
    assert not (root / "outputs" / "predictions").exists()
    assert not (root / "outputs" / "flux").exists()
    forbidden = [
        item
        for item in root.rglob("*")
        if item.is_file()
        and (
            item.name.lower() in {"daily_flux.csv", "annual_flux.csv", "snowmelt_flux.csv"}
            or (item.suffix.lower() in {".joblib", ".pkl", ".pickle"} and item.name not in ALLOWED_DOC_MODEL_ARTIFACTS)
        )
    ]
    assert forbidden == []


def test_roi_report_includes_reopen_freeze_recommendation(roi_qc_result) -> None:
    text = ROI_QC_REPORT_PATH.read_text(encoding="utf-8")
    assert "reopen_freeze_recommendation" in text


def test_optical_negative_result_interpretation_field_exists(roi_qc_result) -> None:
    summary = pd.read_csv(ROI_QC_TABLE_DIR / "roi_final_qc_summary.csv")
    assert "optical_negative_result_interpretation" in summary.columns
    assert "optical_negative_result_likely_roi_driven" in summary.columns
    assert summary["optical_negative_result_interpretation"].notna().all()


def test_roi_qc_does_not_load_forbidden_tables(monkeypatch) -> None:
    import pandas as pd_module

    original_read_csv = pd_module.read_csv
    read_paths: list[str] = []

    def spy_read_csv(filepath, *args, **kwargs):
        read_paths.append(str(filepath))
        return original_read_csv(filepath, *args, **kwargs)

    monkeypatch.setattr(pd_module, "read_csv", spy_read_csv)
    run_roi_final_qc()
    forbidden = [
        "prediction_grid_daily",
        "training_matrix_basin_context",
        "basin_attributes_curated",
        "lab_optical_proxy_gold",
        "training_matrix_hydrocore",
    ]
    assert not [path for path in read_paths if any(token in str(path) for token in forbidden)]
