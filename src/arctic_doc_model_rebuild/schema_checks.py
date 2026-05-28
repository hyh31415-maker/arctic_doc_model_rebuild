from __future__ import annotations

import re
from typing import Any

import pandas as pd

from .data_loader import load_basin_attributes, load_hydrocore, load_optical_matched, load_prediction_grid
from .gold_contract import load_contract


HYDROCORE_REQUIRED = {
    "label_id",
    "river",
    "date",
    "DOC_mgC_L",
    "Q_m3s",
    "doy",
    "sin_doy",
    "cos_doy",
    "temperature_2m_C",
    "positive_degree_day_Cday",
    "snow_cover_fraction",
    "snow_depletion_rate_7d",
    "surface_runoff_m",
}
OPTICAL_REQUIRED = {
    "label_id",
    "river",
    "date",
    "DOC_mgC_L",
    "sensor",
    "optical_date",
    "days_offset",
    "blue",
    "green",
    "red",
    "nir",
    "ndwi",
    "mndwi",
}
PREDICTION_GRID_REQUIRED = {"river", "date", "Q_m3s", "doy", "sin_doy", "cos_doy"}
ID_MEANS = {"HYBAS_ID_mean", "NEXT_DOWN_mean", "PFAF_ID_mean"}
SIX_RIVER_MINIMUM = 6


def _forbidden_predictor_columns() -> set[str]:
    return set(load_contract().get("forbidden_predictor_columns", []))


def _prediction_or_flux_columns(columns: list[str] | pd.Index) -> list[str]:
    pattern = re.compile(r"prediction|pred_doc|predicted_doc|flux|TgC|Mg_day", re.IGNORECASE)
    return [column for column in columns if pattern.search(str(column))]


def _row(check_name: str, table_name: str, passed: bool, message: str, severity: str = "error") -> dict[str, Any]:
    return {
        "check_name": check_name,
        "table_name": table_name,
        "passed": bool(passed),
        "status": "ok" if passed else "failed",
        "severity": severity,
        "message": message,
    }


def check_hydrocore_matrix(frame: pd.DataFrame) -> list[dict[str, Any]]:
    table = "training_matrix_hydrocore.csv"
    missing = sorted(HYDROCORE_REQUIRED.difference(frame.columns))
    forbidden = sorted(_forbidden_predictor_columns().intersection(frame.columns))
    leakage = _prediction_or_flux_columns(frame.columns)
    return [
        _row("hydrocore_required_columns", table, not missing, f"Missing columns: {missing}" if missing else "Required columns present."),
        _row("hydrocore_response_column", table, "DOC_mgC_L" in frame.columns, "DOC_mgC_L exists."),
        _row("hydrocore_no_lab_optical", table, not forbidden, f"Forbidden lab optical columns: {forbidden}" if forbidden else "No lab optical/CDOM predictor columns."),
        _row("hydrocore_no_prediction_flux", table, not leakage, f"Prediction/flux leakage columns: {leakage}" if leakage else "No prediction or flux columns."),
        _row("hydrocore_min_rows", table, len(frame) >= 500, f"Rows: {len(frame)}"),
    ]


def check_optical_matched_matrix(frame: pd.DataFrame, table_name: str = "training_matrix_optical_matched_3d.csv") -> list[dict[str, Any]]:
    missing = sorted(OPTICAL_REQUIRED.difference(frame.columns))
    forbidden = sorted(_forbidden_predictor_columns().intersection(frame.columns))
    leakage = _prediction_or_flux_columns(frame.columns)
    band_columns = [column for column in ["blue", "green", "red", "nir"] if column in frame.columns]
    has_band_values = bool(band_columns) and frame[band_columns].replace("", pd.NA).notna().any().any()
    return [
        _row("optical_required_columns", table_name, not missing, f"Missing columns: {missing}" if missing else "Required columns present."),
        _row("optical_has_bands", table_name, has_band_values, "Optical band values found." if has_band_values else "No actual optical band values found."),
        _row("optical_has_sensor", table_name, "sensor" in frame.columns, "Sensor column exists."),
        _row("optical_has_days_offset", table_name, "days_offset" in frame.columns, "days_offset column exists."),
        _row("optical_no_lab_optical", table_name, not forbidden, f"Forbidden lab optical columns: {forbidden}" if forbidden else "No lab optical/CDOM predictor columns."),
        _row("optical_no_prediction_flux", table_name, not leakage, f"Prediction/flux leakage columns: {leakage}" if leakage else "No prediction or flux columns."),
    ]


def check_prediction_grid(frame: pd.DataFrame, table_name: str = "prediction_grid_daily_hydrocore.csv") -> list[dict[str, Any]]:
    missing = sorted(PREDICTION_GRID_REQUIRED.difference(frame.columns))
    forbidden_exact = [column for column in ["DOC_mgC_L", "predicted_DOC", "pred_DOC"] if column in frame.columns]
    leakage = sorted(set(forbidden_exact + _prediction_or_flux_columns(frame.columns)))
    return [
        _row("prediction_grid_required_columns", table_name, not missing, f"Missing columns: {missing}" if missing else "Required columns present."),
        _row("prediction_grid_no_doc", table_name, "DOC_mgC_L" not in frame.columns, "DOC_mgC_L absent."),
        _row("prediction_grid_no_prediction_flux", table_name, not leakage, f"Forbidden prediction/flux columns: {leakage}" if leakage else "No prediction or flux columns."),
        _row("prediction_grid_min_rows", table_name, len(frame) >= 50000, f"Rows: {len(frame)}"),
    ]


def check_basin_attributes(long_frame: pd.DataFrame, wide_frame: pd.DataFrame) -> list[dict[str, Any]]:
    table = "basin_attributes_curated.csv"
    rivers = set(long_frame.get("river", pd.Series(dtype=str)).astype(str))
    source_fields = long_frame.get("source_field", pd.Series(dtype=str)).astype(str)
    model_use = long_frame.get("model_use", pd.Series(dtype=str)).astype(str).str.lower().isin({"true", "1", "yes"})
    id_rows = long_frame[source_fields.isin(ID_MEANS)] if "source_field" in long_frame.columns else pd.DataFrame()
    id_model_use = id_rows.get("model_use", pd.Series(dtype=str)).astype(str).str.lower().isin({"true", "1", "yes"}).any() if not id_rows.empty else False
    wide_id_leakage = sorted(ID_MEANS.intersection(wide_frame.columns))
    has_upstream_area = "upstream_area_km2" in long_frame.columns and pd.to_numeric(long_frame["upstream_area_km2"], errors="coerce").notna().any()
    usable = long_frame[model_use] if "model_use" in long_frame.columns else pd.DataFrame()
    return [
        _row("basin_six_rivers", table, len(rivers) == SIX_RIVER_MINIMUM, f"Rivers found: {sorted(rivers)}"),
        _row("basin_id_means_not_predictors", table, not id_model_use, "ID/topology means are not marked model_use=True."),
        _row("basin_wide_no_id_means", "basin_attributes_curated_wide.csv", not wide_id_leakage, f"Wide ID mean leakage: {wide_id_leakage}" if wide_id_leakage else "No ID/topology means in wide predictor table."),
        _row("basin_attributes_exist", table, not usable.empty, "Usable basin attributes exist."),
        _row("basin_upstream_area_present", table, has_upstream_area, "upstream_area_km2 present."),
    ]


def run_all_schema_checks() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.extend(check_hydrocore_matrix(load_hydrocore()))
    rows.extend(check_optical_matched_matrix(load_optical_matched("3d")))
    rows.extend(check_prediction_grid(load_prediction_grid(False), "prediction_grid_daily_hydrocore.csv"))
    rows.extend(check_prediction_grid(load_prediction_grid(True), "prediction_grid_daily_with_basin_context.csv"))
    rows.extend(check_basin_attributes(load_basin_attributes(True), load_basin_attributes(False)))
    return pd.DataFrame(rows)


def issue_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    return int((~frame["passed"].astype(bool)).sum())
