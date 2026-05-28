from __future__ import annotations

import pandas as pd


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def daily_confidence_tier(row: pd.Series) -> str:
    outside_logq = bool(row.get("outside_training_logQ_range", False))
    outside_doy = bool(row.get("outside_training_doy_range", False))
    outside_year = bool(row.get("outside_training_year_range", False))
    point_clipped = bool(row.get("point_prediction_clipped_at_zero", False))
    interval_clipped = bool(row.get("interval_lower_clipped_at_zero", False))
    q_valid = pd.notna(row.get("Q_m3s")) and float(row.get("Q_m3s")) > 0
    doc_valid = pd.notna(row.get("DOC_predicted_mgC_L"))
    if outside_logq or outside_doy or point_clipped:
        return "low"
    if not q_valid or not doc_valid:
        return "missing"
    if outside_year or interval_clipped:
        return "medium"
    return "high"


def annual_confidence_tier(coverage_rate: float, fraction_flux_from_low_confidence_days: float) -> str:
    if pd.isna(coverage_rate):
        return "low"
    low_fraction = 1.0 if pd.isna(fraction_flux_from_low_confidence_days) else float(fraction_flux_from_low_confidence_days)
    if float(coverage_rate) >= 0.98 and low_fraction < 0.10:
        return "high"
    if float(coverage_rate) >= 0.95 and low_fraction < 0.25:
        return "medium"
    return "low"


def analysis_readiness_status(annual_summary: pd.DataFrame, range_flags: pd.DataFrame) -> str:
    if annual_summary.empty:
        return "false"
    severe_flags = range_flags[
        range_flags["flag_type"].astype(str).isin(
            {"daily_flux_negative", "daily_flux_nan", "annual_flux_gt_river_median_3x", "annual_flux_lt_river_median_div_3"}
        )
    ]
    if not severe_flags.empty:
        return "true_with_caveats"
    if (annual_summary["annual_confidence_tier"].astype(str) == "low").any():
        return "true_with_caveats"
    return "true"
