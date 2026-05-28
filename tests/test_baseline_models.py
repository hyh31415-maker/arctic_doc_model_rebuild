from __future__ import annotations

import pandas as pd
import pytest

from arctic_doc_model_rebuild.gold_contract import load_contract, require_gold_data_dir, sha256_file, table_path
from arctic_doc_model_rebuild.modeling.baseline_models import run_baseline_models
from arctic_doc_model_rebuild.modeling.reports import BASELINE_REPORT_PATH, BASELINE_TABLE_DIR
from arctic_doc_model_rebuild.paths import project_root


@pytest.fixture(scope="session")
def baseline_result():
    return run_baseline_models()


def test_baseline_outputs_exist(baseline_result) -> None:
    assert BASELINE_REPORT_PATH.exists()
    required = [
        "baseline_metrics_overall.csv",
        "baseline_cv_predictions.csv",
        "baseline_model_ranking.csv",
        "baseline_residuals.csv",
    ]
    for name in required:
        assert (BASELINE_TABLE_DIR / name).exists()


def test_no_gold_data_modified() -> None:
    gold_dir = require_gold_data_dir()
    contract = load_contract()
    before = {
        table_name: sha256_file(table_path(table_name, gold_dir=gold_dir))
        for table_name in contract["expected_tables"]
    }
    run_baseline_models()
    after = {
        table_name: sha256_file(table_path(table_name, gold_dir=gold_dir))
        for table_name in contract["expected_tables"]
    }
    assert before == after


def test_no_production_predictions(baseline_result) -> None:
    root = project_root()
    assert not (root / "outputs" / "predictions").exists()
    forbidden = [item for item in root.rglob("*daily_doc_prediction*") if item.is_file()]
    assert forbidden == []


def test_no_flux_outputs(baseline_result) -> None:
    root = project_root()
    assert not (root / "outputs" / "flux").exists()
    forbidden_names = [
        item
        for item in (root / "outputs").rglob("*")
        if item.is_file() and ("flux" in item.name.lower() or item.name.lower().endswith("_flux.csv"))
    ]
    assert forbidden_names == []
    non_report_flux_mentions = []
    baseline_outputs = [root / "outputs" / "tables" / "baseline", root / "outputs" / "figures" / "baseline"]
    for base in baseline_outputs:
        if not base.exists():
            continue
        for item in base.rglob("*"):
            if not item.is_file() or item.suffix.lower() not in {".csv", ".md"}:
                continue
            text = item.read_text(encoding="utf-8", errors="ignore").lower()
            if "flux" in text:
                non_report_flux_mentions.append(item)
    assert non_report_flux_mentions == []


def test_cv_predictions_are_labeled_validation_only(baseline_result) -> None:
    frame = pd.read_csv(BASELINE_TABLE_DIR / "baseline_cv_predictions.csv")
    assert not frame.empty
    assert frame["is_cv_prediction"].astype(str).str.lower().isin({"true", "1"}).all()
    assert frame["is_production_prediction"].astype(str).str.lower().isin({"false", "0"}).all()


def test_feature_sets_no_forbidden_columns(baseline_result) -> None:
    registry = pd.read_csv(BASELINE_TABLE_DIR / "feature_set_registry.csv").fillna("")
    forbidden = {
        "A254",
        "A375",
        "A440",
        "SUVA254",
        "spectral_slope_275_295",
        "spectral_slope_350_400",
        "blue",
        "green",
        "red",
        "nir",
        "swir1",
        "swir2",
        "ndwi",
        "mndwi",
        "basin_SUB_AREA",
        "HYBAS_ID_mean",
        "NEXT_DOWN_mean",
        "PFAF_ID_mean",
    }
    used = set()
    for text in registry["all_features"].astype(str):
        used.update(item for item in text.split(";") if item)
    assert forbidden.isdisjoint(used)


def test_validation_has_leave_one_year_out(baseline_result) -> None:
    registry = pd.read_csv(BASELINE_TABLE_DIR / "validation_scheme_registry.csv")
    assert "leave_one_year_out" in set(registry["validation_scheme"])
    row = registry[registry["validation_scheme"].eq("leave_one_year_out")].iloc[0]
    assert str(row["primary_for_model_selection"]).lower() in {"true", "1"}


def test_leave_one_river_out_marked_stress_test(baseline_result) -> None:
    registry = pd.read_csv(BASELINE_TABLE_DIR / "validation_scheme_registry.csv")
    row = registry[registry["validation_scheme"].eq("leave_one_river_out")].iloc[0]
    assert str(row["stress_test"]).lower() in {"true", "1"}
    assert row["validation_role"] == "stress_test"


def test_model_registry_simple_models_only(baseline_result) -> None:
    registry = pd.read_csv(BASELINE_TABLE_DIR / "model_registry.csv").fillna("")
    forbidden = registry["model_class"].astype(str).str.contains("RandomForest|XGBoost|PyMC|Neural|Tensor|Torch|LightGBM", case=False, regex=True)
    assert not forbidden.any()
    assert set(registry["model_family"]).issubset({"mean", "linear", "ridge"})


def test_metrics_nonempty(baseline_result) -> None:
    metrics = pd.read_csv(BASELINE_TABLE_DIR / "baseline_metrics_overall.csv")
    assert not metrics.empty
    assert metrics["validation_scheme"].isin(["leave_one_year_out", "river_year_groupkfold", "leave_one_river_out"]).all()
