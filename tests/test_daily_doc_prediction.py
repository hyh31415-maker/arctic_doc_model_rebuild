from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from arctic_doc_model_rebuild.gold_contract import load_contract, require_gold_data_dir, sha256_file, table_path
from arctic_doc_model_rebuild.modeling.daily_doc_prediction import (
    DAILY_DOC_REPORT_PATH,
    DAILY_DOC_TABLE_DIR,
    MODEL_ARTIFACT_PATH,
    MODEL_METADATA_PATH,
    PRODUCTION_CANDIDATE_TABLE_DIR,
    PRODUCTION_SPEC_PATH,
    freeze_production_candidate,
    run_daily_doc_prediction,
)
from arctic_doc_model_rebuild.paths import project_root


@pytest.fixture(scope="session")
def daily_doc_result():
    freeze_production_candidate()
    return run_daily_doc_prediction()


def test_daily_doc_prediction_output_exists(daily_doc_result) -> None:
    assert DAILY_DOC_REPORT_PATH.exists()
    path = DAILY_DOC_TABLE_DIR / "daily_doc_prediction.csv"
    assert path.exists()
    frame = pd.read_csv(path)
    assert not frame.empty
    assert frame["prediction_status"].eq("predicted").any()


def test_daily_doc_prediction_has_no_flux_columns(daily_doc_result) -> None:
    frame = pd.read_csv(DAILY_DOC_TABLE_DIR / "daily_doc_prediction.csv")
    forbidden = {"daily_flux", "Mg_day", "TgC", "kg_day", "DOC_flux", "annual_flux", "snowmelt_flux"}
    assert forbidden.isdisjoint(frame.columns)


def test_is_flux_false(daily_doc_result) -> None:
    frame = pd.read_csv(DAILY_DOC_TABLE_DIR / "daily_doc_prediction.csv")
    assert frame["is_flux"].astype(str).str.lower().isin({"false", "0"}).all()
    assert frame["is_production_daily_doc_prediction"].astype(str).str.lower().isin({"true", "1"}).all()


def test_prediction_grid_used_but_gold_not_modified() -> None:
    gold_dir = require_gold_data_dir()
    contract = load_contract()
    before = {table_name: sha256_file(table_path(table_name, gold_dir=gold_dir)) for table_name in contract["expected_tables"]}
    run_daily_doc_prediction()
    after = {table_name: sha256_file(table_path(table_name, gold_dir=gold_dir)) for table_name in contract["expected_tables"]}
    assert before == after


def test_model_spec_exists(daily_doc_result) -> None:
    assert PRODUCTION_SPEC_PATH.exists()
    spec = yaml.safe_load(PRODUCTION_SPEC_PATH.read_text(encoding="utf-8"))
    assert spec["model_spec_id"] == "production_candidate_r4_river_specific_q_and_season_linear"
    assert spec["production_daily_doc_prediction_allowed"] is True
    assert spec["flux_allowed"] is False
    assert spec["feature_set"] == "R4_river_specific_Q_and_season"


def test_model_fit_summary_exists(daily_doc_result) -> None:
    path = PRODUCTION_CANDIDATE_TABLE_DIR / "production_model_fit_summary.csv"
    assert path.exists()
    frame = pd.read_csv(path)
    assert int(frame["training_rows_used"].iloc[0]) > 0
    assert frame["production_candidate_not_flux_model"].astype(str).str.lower().iloc[0] in {"true", "1"}


def test_prediction_intervals_exist(daily_doc_result) -> None:
    frame = pd.read_csv(DAILY_DOC_TABLE_DIR / "daily_doc_prediction.csv")
    required = {
        "DOC_prediction_interval_80_lower",
        "DOC_prediction_interval_80_upper",
        "DOC_prediction_interval_90_lower",
        "DOC_prediction_interval_90_upper",
        "DOC_prediction_interval_95_lower",
        "DOC_prediction_interval_95_upper",
        "interval_source_scope",
        "interval_source_n",
    }
    assert required.issubset(frame.columns)
    predicted = frame[frame["prediction_status"].eq("predicted")]
    assert predicted["DOC_prediction_interval_95_lower"].notna().all()


def test_prediction_qc_summary_exists(daily_doc_result) -> None:
    path = DAILY_DOC_TABLE_DIR / "daily_doc_prediction_qc_summary.csv"
    assert path.exists()
    frame = pd.read_csv(path)
    assert {"n_grid_rows", "n_predicted_rows", "prediction_coverage_rate"}.issubset(frame.columns)
    assert set(frame["river"]) == {"Kolyma", "Lena", "Mackenzie", "Ob", "Yenisey", "Yukon"}


def test_flux_readiness_decision_exists(daily_doc_result) -> None:
    readiness = pd.read_csv(DAILY_DOC_TABLE_DIR / "flux_readiness_decision.csv")
    assert "ready_for_flux_calculation" in set(readiness["decision_item"])
    ready = readiness[readiness["decision_item"].eq("ready_for_flux_calculation")].iloc[0]
    assert ready["status"] in {"true", "true_with_caveats", "false"}


def test_no_flux_outputs(daily_doc_result) -> None:
    root = project_root()
    assert not (root / "outputs" / "flux").exists()
    forbidden = [
        item
        for item in root.rglob("*")
        if item.is_file() and item.name.lower() in {"daily_flux.csv", "annual_flux.csv", "snowmelt_flux.csv"}
    ]
    assert forbidden == []


def test_no_optical_basin_lab_features_used(monkeypatch) -> None:
    import pandas as pd_module

    original_read_csv = pd_module.read_csv
    read_paths: list[str] = []

    def spy_read_csv(filepath, *args, **kwargs):
        read_paths.append(str(filepath))
        return original_read_csv(filepath, *args, **kwargs)

    monkeypatch.setattr(pd_module, "read_csv", spy_read_csv)
    run_daily_doc_prediction()
    forbidden = [
        "training_matrix_optical_matched",
        "training_matrix_basin_context",
        "basin_attributes_curated",
        "lab_optical_proxy_gold",
    ]
    assert not [path for path in read_paths if any(token in Path(path).name for token in forbidden)]


def test_missing_predictor_rows_flagged(daily_doc_result) -> None:
    missing = pd.read_csv(DAILY_DOC_TABLE_DIR / "daily_doc_prediction_missing_predictor_rows.csv")
    assert "missing_predictor_reason" in missing.columns
    assert missing["prediction_status"].eq("missing_predictor").all() if not missing.empty else True


def test_prediction_rows_only_known_rivers(daily_doc_result) -> None:
    frame = pd.read_csv(DAILY_DOC_TABLE_DIR / "daily_doc_prediction.csv")
    known = {"Kolyma", "Lena", "Mackenzie", "Ob", "Yenisey", "Yukon"}
    predicted = frame[frame["prediction_status"].eq("predicted")]
    assert set(predicted["river"]).issubset(known)


def test_joblib_allowed_only_for_doc_model_not_flux(daily_doc_result) -> None:
    assert MODEL_ARTIFACT_PATH.exists()
    assert MODEL_METADATA_PATH.exists()
    metadata = pd.read_json(MODEL_METADATA_PATH, typ="series")
    assert bool(metadata["production_candidate_not_flux_model"]) is True
    assert bool(metadata["is_flux_model"]) is False
    root = project_root()
    joblibs = [item for item in root.rglob("*.joblib")]
    assert joblibs == [MODEL_ARTIFACT_PATH]
