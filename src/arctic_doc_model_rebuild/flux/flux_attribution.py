from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..gold_contract import sha256_file
from ..paths import REPORT_DIR, TABLE_DIR, path
from ..reports import _md_table, utc_now
from .flux_qc import as_bool
from .trend_statistics import trend_for_series


FLUX_ATTRIBUTION_TABLE_DIR = TABLE_DIR / "flux_attribution"
FLUX_ATTRIBUTION_REPORT_DIR = REPORT_DIR / "flux_attribution"
FLUX_ATTRIBUTION_FIGURE_DIR = path("outputs", "figures", "flux_attribution")
FLUX_ATTRIBUTION_REPORT_PATH = FLUX_ATTRIBUTION_REPORT_DIR / "flux_attribution_report.md"
YUKON_ATTRIBUTION_REPORT_PATH = FLUX_ATTRIBUTION_REPORT_DIR / "yukon_flux_attribution_report.md"

DOC_FLUX_TABLE_DIR = TABLE_DIR / "doc_flux"
DAILY_DOC_PREDICTION_TABLE_DIR = TABLE_DIR / "daily_doc_prediction"
FLUX_INTERPRETATION_TABLE_DIR = TABLE_DIR / "flux_interpretation"
ANNUAL_TREND_TABLE_DIR = TABLE_DIR / "annual_flux_trends"
MAY_JULY_TABLE_DIR = TABLE_DIR / "may_july_flux"
SNOWMELT_TABLE_DIR = TABLE_DIR / "snowmelt_windows"
FINAL_SYNTHESIS_REPORT_PATH = REPORT_DIR / "final_synthesis" / "final_synthesis_report.md"

REQUIRED_ATTRIBUTION_INPUTS = {
    "daily_doc_flux": DOC_FLUX_TABLE_DIR / "daily_doc_flux.csv",
    "annual_doc_flux_summary": DOC_FLUX_TABLE_DIR / "annual_doc_flux_summary.csv",
    "daily_doc_prediction": DAILY_DOC_PREDICTION_TABLE_DIR / "daily_doc_prediction.csv",
    "annual_flux_analysis_cohorts": FLUX_INTERPRETATION_TABLE_DIR / "annual_flux_analysis_cohorts.csv",
    "annual_flux_trends_by_river": ANNUAL_TREND_TABLE_DIR / "annual_flux_trends_by_river.csv",
    "may_july_interpretation": MAY_JULY_TABLE_DIR / "may_july_flux_interpretation_by_river_year.csv",
    "snowmelt_window_flux_summary": SNOWMELT_TABLE_DIR / "snowmelt_window_flux_summary.csv",
    "annual_vs_snowmelt_signal_comparison": SNOWMELT_TABLE_DIR / "annual_vs_snowmelt_signal_comparison.csv",
    "final_synthesis_report": FINAL_SYNTHESIS_REPORT_PATH,
}

SEASON_ORDER = ["winter", "spring_transition", "may_july", "late_summer", "fall_winter"]
SEASON_MONTHS = {
    "winter": [1, 2, 3],
    "spring_transition": [4],
    "may_july": [5, 6, 7],
    "late_summer": [8, 9],
    "fall_winter": [10, 11, 12],
}


def _ensure_dirs() -> None:
    for directory in [FLUX_ATTRIBUTION_TABLE_DIR, FLUX_ATTRIBUTION_REPORT_DIR, FLUX_ATTRIBUTION_FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def _write_csv(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8")
    return destination


def _read_csv(destination: Path) -> pd.DataFrame:
    if not destination.exists():
        raise FileNotFoundError(f"Required flux attribution input is missing: {destination}")
    return pd.read_csv(destination, low_memory=False)


def _read_text(destination: Path) -> str:
    if not destination.exists():
        raise FileNotFoundError(f"Required flux attribution input is missing: {destination}")
    return destination.read_text(encoding="utf-8")


def _input_hashes() -> dict[Path, str]:
    return {destination: sha256_file(destination) for destination in REQUIRED_ATTRIBUTION_INPUTS.values()}


def _verify_inputs_unchanged(before: dict[Path, str]) -> None:
    after = _input_hashes()
    changed = [str(destination) for destination, before_hash in before.items() if after.get(destination) != before_hash]
    if changed:
        raise RuntimeError(f"Flux attribution inputs changed during analysis: {changed}")


def _load_inputs() -> dict[str, Any]:
    return {
        "daily_flux": _read_csv(REQUIRED_ATTRIBUTION_INPUTS["daily_doc_flux"]),
        "annual_flux": _read_csv(REQUIRED_ATTRIBUTION_INPUTS["annual_doc_flux_summary"]),
        "daily_prediction": _read_csv(REQUIRED_ATTRIBUTION_INPUTS["daily_doc_prediction"]),
        "cohorts": _read_csv(REQUIRED_ATTRIBUTION_INPUTS["annual_flux_analysis_cohorts"]),
        "annual_trends": _read_csv(REQUIRED_ATTRIBUTION_INPUTS["annual_flux_trends_by_river"]),
        "may_july": _read_csv(REQUIRED_ATTRIBUTION_INPUTS["may_july_interpretation"]),
        "snowmelt": _read_csv(REQUIRED_ATTRIBUTION_INPUTS["snowmelt_window_flux_summary"]),
        "snowmelt_comparison": _read_csv(REQUIRED_ATTRIBUTION_INPUTS["annual_vs_snowmelt_signal_comparison"]),
        "final_synthesis_text": _read_text(REQUIRED_ATTRIBUTION_INPUTS["final_synthesis_report"]),
    }


def _to_bool(series: pd.Series) -> pd.Series:
    return as_bool(series)


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _prepare_daily_flux(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for column in [
        "year",
        "month",
        "doy",
        "Q_m3s",
        "DOC_predicted_mgC_L",
        "daily_flux_TgC_day",
        "daily_flux_90_lower_TgC_day",
        "daily_flux_90_upper_TgC_day",
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in [
        "outside_training_logQ_range",
        "outside_training_doy_range",
        "outside_training_year_range",
        "point_prediction_clipped_at_zero",
        "interval_lower_clipped_at_zero",
        "is_doc_prediction",
        "is_flux",
    ]:
        if column in out.columns:
            out[column] = _to_bool(out[column])
    return out[out["flux_status"].astype(str).eq("calculated")].copy()


def _prepare_annual(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    for column in [
        "annual_flux_TgC",
        "annual_flux_90_lower_TgC",
        "annual_flux_90_upper_TgC",
        "coverage_rate",
        "fraction_flux_from_low_confidence_days",
        "n_days_with_flux",
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _prepare_cohorts(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    for column in [
        "annual_flux_TgC",
        "coverage_rate",
        "fraction_flux_from_low_confidence_days",
        "n_low_confidence_days",
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in [
        "cohort_core_2003_2024",
        "cohort_high_confidence_only",
        "cohort_sensitivity_only",
        "cohort_exclude_from_trend",
        "cohort_full_2000_2025",
    ]:
        out[column] = _to_bool(out[column])
    return out


def _season_from_month(month: int | float | None) -> str:
    if pd.isna(month):
        return "unknown"
    month_int = int(month)
    for season, months in SEASON_MONTHS.items():
        if month_int in months:
            return season
    return "unknown"


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


def _annual_attribution(daily_flux: pd.DataFrame, annual: pd.DataFrame, cohorts: pd.DataFrame) -> pd.DataFrame:
    daily = daily_flux.copy()
    daily["q_volume_km3_day"] = daily["Q_m3s"] * 86400.0 / 1e9
    daily["doc_q_product"] = daily["DOC_predicted_mgC_L"] * daily["Q_m3s"]
    grouped = daily.groupby(["river", "year"], dropna=False)
    derived_rows = []
    for (river, year), group in grouped:
        q_sum = float(group["Q_m3s"].sum())
        derived_rows.append(
            {
                "river": river,
                "year": int(year),
                "annual_Q_volume_km3": float(group["q_volume_km3_day"].sum()),
                "annual_mean_DOC_mgC_L": float(group["DOC_predicted_mgC_L"].mean()),
                "flow_weighted_DOC_mgC_L": float(group["doc_q_product"].sum() / q_sum) if q_sum > 0 else np.nan,
                "n_days_with_flux_from_daily": int(len(group)),
            }
        )
    derived = pd.DataFrame(derived_rows)
    annual_keep = annual[
        [
            "river",
            "year",
            "annual_flux_TgC",
            "annual_flux_90_lower_TgC",
            "annual_flux_90_upper_TgC",
            "n_days_with_flux",
            "coverage_rate",
            "annual_confidence_tier",
        ]
    ].copy()
    cohorts_keep = cohorts[
        [
            "river",
            "year",
            "cohort_core_2003_2024",
            "cohort_high_confidence_only",
            "cohort_sensitivity_only",
            "cohort_exclude_from_trend",
            "exclusion_or_caveat_reason",
        ]
    ].copy()
    out = annual_keep.merge(derived, on=["river", "year"], how="left").merge(cohorts_keep, on=["river", "year"], how="left")
    out["annual_flux_yield_gC_m2_yr"] = np.nan
    out["caveat_reason"] = out["exclusion_or_caveat_reason"].fillna("")
    out.loc[:, "caveat_reason"] = out["caveat_reason"].where(
        out["caveat_reason"].astype(str).str.len() > 0,
        "upstream_area_not_available_in_allowed_inputs_for_yield",
    )
    return out[
        [
            "river",
            "year",
            "annual_flux_TgC",
            "annual_Q_volume_km3",
            "annual_mean_DOC_mgC_L",
            "flow_weighted_DOC_mgC_L",
            "annual_flux_yield_gC_m2_yr",
            "n_days_with_flux",
            "coverage_rate",
            "annual_confidence_tier",
            "cohort_core_2003_2024",
            "cohort_high_confidence_only",
            "caveat_reason",
        ]
    ].sort_values(["river", "year"])


def _trend_record(
    group: pd.DataFrame,
    value_column: str,
    *,
    river: str,
    metric: str,
    analysis_cohort: str,
    unit: str,
) -> dict[str, Any]:
    result = trend_for_series(group.rename(columns={value_column: "trend_value"}), "trend_value")
    detectable = bool(result.significant_at_0_05 and result.trend_direction in {"increasing", "decreasing"})
    trend_label = f"detectable {result.trend_direction} trend" if detectable else "no detectable trend"
    return {
        "river": river,
        "analysis_cohort": analysis_cohort,
        "metric": metric,
        "unit": unit,
        "n_years": result.n_years,
        "year_min": result.year_min,
        "year_max": result.year_max,
        "mean_value": result.value_mean,
        "median_value": result.value_median,
        "slope_ols_per_year": result.slope_ols,
        "intercept_ols": result.intercept_ols,
        "slope_ols_p_value": result.slope_ols_p_value,
        "slope_ols_r2": result.slope_ols_r2,
        "slope_theilsen_per_year": result.slope_theilsen,
        "kendall_tau": result.kendall_tau,
        "kendall_p_value": result.kendall_p_value,
        "slope_percent_per_year": result.slope_percent_per_year,
        "total_change_percent_over_period": result.total_change_percent_over_period,
        "trend_direction": result.trend_direction,
        "detectable_trend": detectable,
        "trend_language": trend_label,
    }


def _component_trends(annual_attr: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("annual_flux_TgC", "annual_flux_TgC", "Tg C/year"),
        ("annual_Q_volume_km3", "annual_Q_volume_km3", "km3/year"),
        ("annual_mean_DOC_mgC_L", "annual_mean_DOC_mgC_L", "mg C/L/year"),
        ("flow_weighted_DOC_mgC_L", "flow_weighted_DOC_mgC_L", "mg C/L/year"),
    ]
    rows: list[dict[str, Any]] = []
    core = annual_attr[annual_attr["cohort_core_2003_2024"].astype(bool)].copy()
    for river, group in core.groupby("river", dropna=False):
        for column, metric, unit in metrics:
            if pd.to_numeric(group[column], errors="coerce").notna().sum() >= 3:
                rows.append(_trend_record(group, column, river=str(river), metric=metric, analysis_cohort="core_2003_2024", unit=unit))
    return pd.DataFrame(rows)


def _trend_lookup(trends: pd.DataFrame, river: str, metric: str) -> pd.Series:
    subset = trends[trends["river"].astype(str).eq(river) & trends["metric"].astype(str).eq(metric)]
    return subset.iloc[0] if not subset.empty else pd.Series(dtype=object)


def _annual_trend_lookup(annual_trends: pd.DataFrame, river: str) -> pd.Series:
    subset = annual_trends[
        annual_trends["analysis_cohort"].astype(str).eq("core_2003_2024")
        & annual_trends["river"].astype(str).eq(river)
    ]
    return subset.iloc[0] if not subset.empty else pd.Series(dtype=object)


def _is_detectable_increasing(row: pd.Series) -> bool:
    if row.empty:
        return False
    if "detectable_trend" in row.index:
        detectable = bool(row.get("detectable_trend", False))
    else:
        detectable = str(row.get("significant_at_0_05", "")).strip().lower() in {"true", "1"}
    return detectable and str(row.get("trend_direction", "")) == "increasing"


def _driver_classification(annual_trends: pd.DataFrame, component_trends: pd.DataFrame) -> pd.DataFrame:
    rivers = sorted(set(annual_trends["river"].dropna().astype(str)) | set(component_trends["river"].dropna().astype(str)))
    rows: list[dict[str, Any]] = []
    for river in rivers:
        annual = _annual_trend_lookup(annual_trends, river)
        q = _trend_lookup(component_trends, river, "annual_Q_volume_km3")
        fw_doc = _trend_lookup(component_trends, river, "flow_weighted_DOC_mgC_L")
        annual_inc = _is_detectable_increasing(annual)
        q_inc = _is_detectable_increasing(q)
        doc_inc = _is_detectable_increasing(fw_doc)
        if annual_inc and q_inc and not doc_inc:
            driver = "discharge_volume_dominated"
        elif annual_inc and doc_inc and not q_inc:
            driver = "concentration_dominated"
        elif annual_inc and q_inc and doc_inc:
            driver = "combined_Q_and_DOC"
        elif annual_inc and not q_inc and not doc_inc:
            driver = "seasonal_redistribution_or_unresolved"
        else:
            driver = "not_applicable_no_detectable_annual_trend"
        rows.append(
            {
                "river": river,
                "annual_flux_trend_direction": annual.get("trend_direction", ""),
                "annual_flux_detectable": bool(str(annual.get("significant_at_0_05", "")).strip().lower() in {"true", "1"}),
                "annual_flux_slope_TgC_per_year": annual.get("slope_ols_TgC_per_year", np.nan),
                "annual_flux_p_value": annual.get("slope_ols_p_value", np.nan),
                "Q_volume_trend_direction": q.get("trend_direction", ""),
                "Q_volume_detectable": bool(q.get("detectable_trend", False)),
                "Q_volume_slope_km3_per_year": q.get("slope_ols_per_year", np.nan),
                "Q_volume_p_value": q.get("slope_ols_p_value", np.nan),
                "flow_weighted_DOC_trend_direction": fw_doc.get("trend_direction", ""),
                "flow_weighted_DOC_detectable": bool(fw_doc.get("detectable_trend", False)),
                "flow_weighted_DOC_slope_mgC_L_per_year": fw_doc.get("slope_ols_per_year", np.nan),
                "flow_weighted_DOC_p_value": fw_doc.get("slope_ols_p_value", np.nan),
                "driver_classification": driver,
                "interpretation": _driver_interpretation(driver, river),
            }
        )
    return pd.DataFrame(rows)


def _driver_interpretation(driver: str, river: str) -> str:
    if driver == "discharge_volume_dominated":
        return f"{river} annual flux increase aligns with increasing annual discharge volume while flow-weighted DOC has no detectable trend."
    if driver == "concentration_dominated":
        return f"{river} annual flux increase aligns with increasing flow-weighted DOC while annual discharge volume has no detectable trend."
    if driver == "combined_Q_and_DOC":
        return f"{river} annual flux increase aligns with detectable increases in both discharge volume and flow-weighted DOC."
    if driver == "seasonal_redistribution_or_unresolved":
        return f"{river} annual flux increase is not explained by detectable annual Q-volume or flow-weighted DOC trends; seasonal redistribution or unresolved structure remains plausible."
    return f"{river} has no detectable annual flux trend in the core cohort, so driver attribution is not applicable."


def _monthly_decomposition(daily_flux: pd.DataFrame, annual_attr: pd.DataFrame) -> pd.DataFrame:
    daily = daily_flux.copy()
    daily["q_volume_km3_day"] = daily["Q_m3s"] * 86400.0 / 1e9
    daily["doc_q_product"] = daily["DOC_predicted_mgC_L"] * daily["Q_m3s"]
    rows: list[dict[str, Any]] = []
    annual_key = annual_attr.set_index(["river", "year"])
    for (river, year, month), group in daily.groupby(["river", "year", "month"], dropna=False):
        q_sum = float(group["Q_m3s"].sum())
        annual_row = annual_key.loc[(river, int(year))] if (river, int(year)) in annual_key.index else pd.Series(dtype=object)
        annual_flux = float(annual_row.get("annual_flux_TgC", np.nan))
        monthly_flux = float(group["daily_flux_TgC_day"].sum())
        low_flux = float(group.loc[group["daily_confidence_tier"].astype(str).eq("low"), "daily_flux_TgC_day"].sum())
        rows.append(
            {
                "river": river,
                "year": int(year),
                "month": int(month),
                "monthly_flux_TgC": monthly_flux,
                "monthly_fraction_of_annual": monthly_flux / annual_flux if pd.notna(annual_flux) and annual_flux > 0 else np.nan,
                "monthly_Q_volume_km3": float(group["q_volume_km3_day"].sum()),
                "monthly_flow_weighted_DOC": float(group["doc_q_product"].sum() / q_sum) if q_sum > 0 else np.nan,
                "n_days_with_flux": int(len(group)),
                "n_low_confidence_days": int(group["daily_confidence_tier"].astype(str).eq("low").sum()),
                "fraction_flux_from_low_confidence_days": low_flux / monthly_flux if monthly_flux > 0 else np.nan,
                "annual_confidence_tier": annual_row.get("annual_confidence_tier", ""),
                "cohort_core_2003_2024": bool(annual_row.get("cohort_core_2003_2024", False)),
                "cohort_high_confidence_only": bool(annual_row.get("cohort_high_confidence_only", False)),
                "cohort_sensitivity_only": bool(annual_row.get("cohort_sensitivity_only", False)),
                "cohort_exclude_from_trend": bool(annual_row.get("cohort_exclude_from_trend", False)),
                "caveat_reason": annual_row.get("caveat_reason", ""),
            }
        )
    return pd.DataFrame(rows).sort_values(["river", "year", "month"])


def _seasonal_decomposition(monthly: pd.DataFrame, annual_attr: pd.DataFrame) -> pd.DataFrame:
    month_to_season = {month: season for season, months in SEASON_MONTHS.items() for month in months}
    work = monthly.copy()
    work["season_window"] = work["month"].map(month_to_season).fillna("unknown")
    rows: list[dict[str, Any]] = []
    annual_key = annual_attr.set_index(["river", "year"])
    for (river, year, season), group in work.groupby(["river", "year", "season_window"], dropna=False):
        annual_row = annual_key.loc[(river, int(year))] if (river, int(year)) in annual_key.index else pd.Series(dtype=object)
        annual_flux = float(annual_row.get("annual_flux_TgC", np.nan))
        seasonal_flux = float(group["monthly_flux_TgC"].sum())
        q_volume = float(group["monthly_Q_volume_km3"].sum())
        weighted_doc = _weighted_mean(group["monthly_flow_weighted_DOC"], group["monthly_Q_volume_km3"])
        low_flux_fraction = _weighted_mean(group["fraction_flux_from_low_confidence_days"], group["monthly_flux_TgC"])
        rows.append(
            {
                "river": river,
                "year": int(year),
                "season_window": season,
                "seasonal_flux_TgC": seasonal_flux,
                "seasonal_fraction_of_annual": seasonal_flux / annual_flux if pd.notna(annual_flux) and annual_flux > 0 else np.nan,
                "seasonal_Q_volume_km3": q_volume,
                "seasonal_flow_weighted_DOC": weighted_doc,
                "n_days_with_flux": int(group["n_days_with_flux"].sum()),
                "fraction_flux_from_low_confidence_days": low_flux_fraction,
                "annual_confidence_tier": annual_row.get("annual_confidence_tier", ""),
                "cohort_core_2003_2024": bool(annual_row.get("cohort_core_2003_2024", False)),
                "cohort_high_confidence_only": bool(annual_row.get("cohort_high_confidence_only", False)),
                "cohort_sensitivity_only": bool(annual_row.get("cohort_sensitivity_only", False)),
                "cohort_exclude_from_trend": bool(annual_row.get("cohort_exclude_from_trend", False)),
                "caveat_reason": annual_row.get("caveat_reason", ""),
            }
        )
    out = pd.DataFrame(rows)
    out["season_order"] = out["season_window"].map({season: i for i, season in enumerate(SEASON_ORDER)}).fillna(99).astype(int)
    return out.sort_values(["river", "year", "season_order"]).drop(columns=["season_order"])


def _grouped_trends(frame: pd.DataFrame, group_columns: list[str], metrics: list[tuple[str, str]]) -> pd.DataFrame:
    core = frame[frame["cohort_core_2003_2024"].astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    for keys, group in core.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group_columns, keys))
        for column, unit in metrics:
            if column in group.columns and pd.to_numeric(group[column], errors="coerce").notna().sum() >= 3:
                row = _trend_record(
                    group,
                    column,
                    river=str(key_values.get("river", "")),
                    metric=column,
                    analysis_cohort="core_2003_2024",
                    unit=unit,
                )
                for column_name, value in key_values.items():
                    row[column_name] = value
                rows.append(row)
    return pd.DataFrame(rows)


def _quantile_date(group: pd.DataFrame, threshold: float) -> pd.Timestamp | pd.NaT:
    total = float(group["daily_flux_TgC_day"].sum())
    if total <= 0 or pd.isna(total):
        return pd.NaT
    cumulative = group["daily_flux_TgC_day"].cumsum() / total
    reached = group.loc[cumulative >= threshold, "date"]
    return reached.iloc[0] if not reached.empty else pd.NaT


def _phenology(daily_flux: pd.DataFrame, annual_attr: pd.DataFrame) -> pd.DataFrame:
    annual_key = annual_attr.set_index(["river", "year"])
    rows: list[dict[str, Any]] = []
    for (river, year), group in daily_flux.sort_values("date").groupby(["river", "year"], dropna=False):
        group = group.copy().sort_values("date")
        annual_row = annual_key.loc[(river, int(year))] if (river, int(year)) in annual_key.index else pd.Series(dtype=object)
        annual_flux = float(group["daily_flux_TgC_day"].sum())
        q10 = _quantile_date(group, 0.10)
        q25 = _quantile_date(group, 0.25)
        q50 = _quantile_date(group, 0.50)
        q75 = _quantile_date(group, 0.75)
        q90 = _quantile_date(group, 0.90)
        peak_idx = group["daily_flux_TgC_day"].idxmax() if not group.empty else None
        peak = group.loc[peak_idx] if peak_idx is not None else pd.Series(dtype=object)
        weights = pd.to_numeric(group["daily_flux_TgC_day"], errors="coerce")
        centroid = _weighted_mean(group["doy"], weights)
        active_length = (q90 - q10).days + 1 if pd.notna(q10) and pd.notna(q90) else np.nan
        rows.append(
            {
                "river": river,
                "year": int(year),
                "flux_10pct_date": q10.date().isoformat() if pd.notna(q10) else "",
                "flux_25pct_date": q25.date().isoformat() if pd.notna(q25) else "",
                "flux_50pct_date": q50.date().isoformat() if pd.notna(q50) else "",
                "flux_75pct_date": q75.date().isoformat() if pd.notna(q75) else "",
                "flux_90pct_date": q90.date().isoformat() if pd.notna(q90) else "",
                "flux_10pct_doy": q10.dayofyear if pd.notna(q10) else np.nan,
                "flux_25pct_doy": q25.dayofyear if pd.notna(q25) else np.nan,
                "flux_50pct_doy": q50.dayofyear if pd.notna(q50) else np.nan,
                "flux_75pct_doy": q75.dayofyear if pd.notna(q75) else np.nan,
                "flux_90pct_doy": q90.dayofyear if pd.notna(q90) else np.nan,
                "peak_daily_flux_date": pd.to_datetime(peak.get("date"), errors="coerce").date().isoformat() if not peak.empty and pd.notna(peak.get("date")) else "",
                "peak_daily_flux_doy": int(peak.get("doy")) if not peak.empty and pd.notna(peak.get("doy")) else np.nan,
                "peak_daily_flux_TgC_day": float(peak.get("daily_flux_TgC_day", np.nan)) if not peak.empty else np.nan,
                "flux_centroid_doy": centroid,
                "active_flux_season_length": active_length,
                "fraction_flux_before_may": float(group.loc[group["month"] < 5, "daily_flux_TgC_day"].sum() / annual_flux) if annual_flux > 0 else np.nan,
                "fraction_flux_after_july": float(group.loc[group["month"] > 7, "daily_flux_TgC_day"].sum() / annual_flux) if annual_flux > 0 else np.nan,
                "fraction_flux_after_september": float(group.loc[group["month"] > 9, "daily_flux_TgC_day"].sum() / annual_flux) if annual_flux > 0 else np.nan,
                "cohort_core_2003_2024": bool(annual_row.get("cohort_core_2003_2024", False)),
                "cohort_high_confidence_only": bool(annual_row.get("cohort_high_confidence_only", False)),
                "caveat_reason": annual_row.get("caveat_reason", ""),
            }
        )
    return pd.DataFrame(rows).sort_values(["river", "year"])


def _phenology_trends(phenology: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("flux_centroid_doy", "day/year"),
        ("flux_25pct_doy", "day/year"),
        ("flux_50pct_doy", "day/year"),
        ("flux_75pct_doy", "day/year"),
        ("active_flux_season_length", "days/year"),
        ("fraction_flux_after_july", "fraction/year"),
    ]
    return _grouped_trends(phenology, ["river"], metrics)


def _monthly_and_seasonal_drivers(monthly_trends: pd.DataFrame, seasonal_trends: pd.DataFrame, river: str = "Yukon") -> str:
    monthly = monthly_trends[
        monthly_trends["river"].astype(str).eq(river)
        & monthly_trends["metric"].eq("monthly_flux_TgC")
        & monthly_trends["detectable_trend"].astype(bool)
    ].sort_values("slope_ols_per_year", ascending=False)
    seasonal = seasonal_trends[
        seasonal_trends["river"].astype(str).eq(river)
        & seasonal_trends["metric"].eq("seasonal_flux_TgC")
        & seasonal_trends["detectable_trend"].astype(bool)
    ].sort_values("slope_ols_per_year", ascending=False)
    parts: list[str] = []
    if not monthly.empty:
        parts.append("detectable_months=" + ",".join(monthly["month"].astype(int).astype(str).head(4)))
    else:
        top_months = monthly_trends[
            monthly_trends["river"].astype(str).eq(river) & monthly_trends["metric"].eq("monthly_flux_TgC")
        ].sort_values("slope_ols_per_year", ascending=False)
        parts.append("no_detectable_monthly_flux_trend")
        if not top_months.empty:
            parts.append("largest_positive_month_slopes=" + ",".join(top_months["month"].astype(int).astype(str).head(3)))
    if not seasonal.empty:
        parts.append("detectable_seasons=" + ",".join(seasonal["season_window"].astype(str).head(4)))
    else:
        top_seasons = seasonal_trends[
            seasonal_trends["river"].astype(str).eq(river) & seasonal_trends["metric"].eq("seasonal_flux_TgC")
        ].sort_values("slope_ols_per_year", ascending=False)
        parts.append("no_detectable_seasonal_flux_trend")
        if not top_seasons.empty:
            parts.append("largest_positive_season_slopes=" + ",".join(top_seasons["season_window"].astype(str).head(3)))
    return "; ".join(parts)


def _timing_shift_summary(phenology_trends: pd.DataFrame, river: str = "Yukon") -> str:
    subset = phenology_trends[phenology_trends["river"].astype(str).eq(river)].copy()
    detectable = subset[subset["detectable_trend"].astype(bool)]
    if detectable.empty:
        pieces = ["no_detectable_export_timing_shift"]
        for metric in ["flux_centroid_doy", "flux_50pct_doy", "fraction_flux_after_july"]:
            row = subset[subset["metric"].eq(metric)]
            if not row.empty:
                pieces.append(f"{metric}_slope={float(row.iloc[0]['slope_ols_per_year']):.3g}")
        return "; ".join(pieces)
    return "; ".join(
        f"{row.metric}:{row.trend_language}, slope={float(row.slope_ols_per_year):.3g}"
        for row in detectable.itertuples(index=False)
    )


def _snowmelt_yukon_summary(snowmelt_comparison: pd.DataFrame) -> str:
    subset = snowmelt_comparison[snowmelt_comparison["river"].astype(str).eq("Yukon")]
    if subset.empty:
        return "dynamic_snowmelt_comparison_unavailable"
    return "; ".join(
        f"{row.window_id}={row.does_window_explain_annual_signal}" for row in subset.itertuples(index=False)
    )


def _yukon_summary(
    driver: pd.DataFrame,
    component_trends: pd.DataFrame,
    monthly_trends: pd.DataFrame,
    seasonal_trends: pd.DataFrame,
    phenology_trends: pd.DataFrame,
    snowmelt_comparison: pd.DataFrame,
) -> pd.DataFrame:
    y_driver = driver[driver["river"].astype(str).eq("Yukon")]
    driver_row = y_driver.iloc[0] if not y_driver.empty else pd.Series(dtype=object)
    q = _trend_lookup(component_trends, "Yukon", "annual_Q_volume_km3")
    fw_doc = _trend_lookup(component_trends, "Yukon", "flow_weighted_DOC_mgC_L")
    monthly_seasonal = _monthly_and_seasonal_drivers(monthly_trends, seasonal_trends, "Yukon")
    timing = _timing_shift_summary(phenology_trends, "Yukon")
    dynamic = _snowmelt_yukon_summary(snowmelt_comparison)
    rows = [
        {
            "question": "Is Yukon annual increase Q-volume driven?",
            "answer": "yes" if bool(q.get("detectable_trend", False)) and str(q.get("trend_direction", "")) == "increasing" else "no_detectable_Q_volume_trend",
            "evidence": f"slope={q.get('slope_ols_per_year', np.nan)}, p={q.get('slope_ols_p_value', np.nan)}, direction={q.get('trend_direction', '')}",
            "driver_classification": driver_row.get("driver_classification", ""),
        },
        {
            "question": "Is Yukon flow-weighted DOC increasing?",
            "answer": "yes" if bool(fw_doc.get("detectable_trend", False)) and str(fw_doc.get("trend_direction", "")) == "increasing" else "no_detectable_flow_weighted_DOC_trend",
            "evidence": f"slope={fw_doc.get('slope_ols_per_year', np.nan)}, p={fw_doc.get('slope_ols_p_value', np.nan)}, direction={fw_doc.get('trend_direction', '')}",
            "driver_classification": driver_row.get("driver_classification", ""),
        },
        {
            "question": "Is Yukon annual increase concentrated in specific months/seasons?",
            "answer": monthly_seasonal,
            "evidence": "core_2003_2024 monthly and seasonal trend diagnostics",
            "driver_classification": driver_row.get("driver_classification", ""),
        },
        {
            "question": "Is Yukon export timing shifting?",
            "answer": timing,
            "evidence": "export phenology trend diagnostics",
            "driver_classification": driver_row.get("driver_classification", ""),
        },
        {
            "question": "Is the increase outside May-July?",
            "answer": "see_monthly_and_after_july_fraction_diagnostics",
            "evidence": monthly_seasonal + "; " + timing,
            "driver_classification": driver_row.get("driver_classification", ""),
        },
        {
            "question": "Does dynamic snowmelt window explain it?",
            "answer": dynamic,
            "evidence": "annual_vs_snowmelt_signal_comparison.csv",
            "driver_classification": driver_row.get("driver_classification", ""),
        },
        {
            "question": "Final classification",
            "answer": driver_row.get("driver_classification", "unresolved"),
            "evidence": driver_row.get("interpretation", ""),
            "driver_classification": driver_row.get("driver_classification", ""),
        },
    ]
    return pd.DataFrame(rows)


def _make_figures(
    annual_attr: pd.DataFrame,
    component_trends: pd.DataFrame,
    monthly: pd.DataFrame,
    monthly_trends: pd.DataFrame,
    seasonal: pd.DataFrame,
    phenology: pd.DataFrame,
    phenology_trends: pd.DataFrame,
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    FLUX_ATTRIBUTION_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def save(fig, name: str) -> None:
        fig.tight_layout()
        destination = FLUX_ATTRIBUTION_FIGURE_DIR / name
        fig.savefig(destination, dpi=120)
        plt.close(fig)
        paths.append(destination)

    core = annual_attr[annual_attr["cohort_core_2003_2024"].astype(bool)].copy()
    if not core.empty:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for river, group in core.groupby("river"):
            ax.scatter(group["annual_Q_volume_km3"], group["annual_flux_TgC"], s=20, label=str(river))
        ax.set_xlabel("Annual Q volume (km3)")
        ax.set_ylabel("Annual DOC flux (Tg C)")
        ax.set_title("Annual flux vs discharge volume")
        ax.legend(fontsize="x-small")
        save(fig, "annual_flux_vs_Q_volume_by_river.png")

        fig, ax = plt.subplots(figsize=(8.5, 5))
        for river, group in core.groupby("river"):
            group = group.sort_values("year")
            ax.plot(group["year"], group["flow_weighted_DOC_mgC_L"], marker="o", linewidth=1.0, label=str(river))
        ax.set_xlabel("Year")
        ax.set_ylabel("Flow-weighted DOC (mg C/L)")
        ax.set_title("Flow-weighted DOC trends by river")
        ax.legend(fontsize="x-small")
        save(fig, "flow_weighted_DOC_trends_by_river.png")

    yukon = core[core["river"].astype(str).eq("Yukon")]
    if not yukon.empty:
        fig, ax1 = plt.subplots(figsize=(8.5, 5))
        ax1.plot(yukon["year"], yukon["annual_flux_TgC"], marker="o", label="annual flux")
        ax1.set_ylabel("Annual DOC flux (Tg C)")
        ax2 = ax1.twinx()
        ax2.plot(yukon["year"], yukon["annual_Q_volume_km3"], marker="s", linestyle="--", label="Q volume")
        ax2.plot(yukon["year"], yukon["flow_weighted_DOC_mgC_L"], marker="^", linestyle=":", label="flow-weighted DOC")
        ax2.set_ylabel("Q volume (km3) / DOC (mg C/L)")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize="x-small")
        ax1.set_xlabel("Year")
        ax1.set_title("Yukon annual flux, Q volume, and flow-weighted DOC")
        save(fig, "yukon_flux_Q_DOC_decomposition.png")

    heat = monthly_trends[
        monthly_trends["metric"].eq("monthly_flux_TgC")
        & monthly_trends["analysis_cohort"].eq("core_2003_2024")
    ]
    if not heat.empty:
        pivot = heat.pivot_table(index="river", columns="month", values="slope_ols_per_year", aggfunc="first").sort_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        image = ax.imshow(pivot.fillna(0).to_numpy(), aspect="auto")
        ax.set_xticks(np.arange(len(pivot.columns)), [str(int(c)) for c in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
        ax.set_xlabel("Month")
        ax.set_title("Monthly DOC flux trend slopes by river")
        fig.colorbar(image, ax=ax, label="Tg C/year")
        save(fig, "monthly_flux_trend_heatmap_by_river.png")

    yukon_monthly = monthly[(monthly["river"].astype(str).eq("Yukon")) & monthly["cohort_core_2003_2024"].astype(bool)]
    if not yukon_monthly.empty:
        pivot = yukon_monthly.pivot_table(index="year", columns="month", values="monthly_fraction_of_annual", aggfunc="first")
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot.plot(ax=ax, linewidth=1.0)
        ax.set_ylabel("Monthly fraction of annual flux")
        ax.set_title("Yukon monthly flux contribution fractions")
        ax.legend(fontsize="xx-small", ncol=2)
        save(fig, "yukon_monthly_flux_contribution_trends.png")

    seasonal_core = seasonal[seasonal["cohort_core_2003_2024"].astype(bool)].copy()
    if not seasonal_core.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        seasonal_mean = seasonal_core.groupby(["river", "season_window"])["seasonal_fraction_of_annual"].mean().reset_index()
        pivot = seasonal_mean.pivot(index="river", columns="season_window", values="seasonal_fraction_of_annual").reindex(columns=SEASON_ORDER)
        pivot.plot(kind="bar", stacked=True, ax=ax)
        ax.set_ylabel("Mean fraction of annual flux")
        ax.set_title("Seasonal flux fractions by river")
        ax.legend(fontsize="x-small")
        save(fig, "seasonal_flux_fraction_by_river.png")

    phen_core = phenology[phenology["cohort_core_2003_2024"].astype(bool)]
    if not phen_core.empty:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for river, group in phen_core.groupby("river"):
            group = group.sort_values("year")
            ax.plot(group["year"], group["flux_centroid_doy"], marker="o", linewidth=1.0, label=str(river))
        ax.set_xlabel("Year")
        ax.set_ylabel("Flux centroid DOY")
        ax.set_title("Export centroid trends by river")
        ax.legend(fontsize="x-small")
        save(fig, "export_centroid_trends_by_river.png")

        yukon_phen = phen_core[phen_core["river"].astype(str).eq("Yukon")]
        if not yukon_phen.empty:
            fig, ax = plt.subplots(figsize=(8.5, 5))
            for column, label in [
                ("flux_25pct_doy", "25%"),
                ("flux_50pct_doy", "50%"),
                ("flux_75pct_doy", "75%"),
                ("flux_centroid_doy", "centroid"),
            ]:
                ax.plot(yukon_phen["year"], yukon_phen[column], marker="o", linewidth=1.0, label=label)
            ax.set_xlabel("Year")
            ax.set_ylabel("Day of year")
            ax.set_title("Yukon cumulative flux timing")
            ax.legend(fontsize="x-small")
            save(fig, "yukon_cumulative_flux_timing.png")
    return paths


def _trend_phrase(row: pd.Series, metric_label: str) -> str:
    if row.empty:
        return f"{metric_label}: unavailable"
    if bool(row.get("detectable_trend", False)):
        return f"{metric_label}: {row.get('trend_language')} (slope={row.get('slope_ols_per_year'):.4g}, p={row.get('slope_ols_p_value'):.3g})"
    return f"{metric_label}: no detectable trend (slope={row.get('slope_ols_per_year'):.4g}, p={row.get('slope_ols_p_value'):.3g})"


def write_yukon_flux_attribution_report() -> Path:
    FLUX_ATTRIBUTION_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = _read_csv(FLUX_ATTRIBUTION_TABLE_DIR / "yukon_flux_attribution_summary.csv")
    driver = _read_csv(FLUX_ATTRIBUTION_TABLE_DIR / "flux_driver_classification.csv")
    component = _read_csv(FLUX_ATTRIBUTION_TABLE_DIR / "q_doc_component_trends_by_river.csv")
    monthly = _read_csv(FLUX_ATTRIBUTION_TABLE_DIR / "monthly_flux_trends_by_river.csv")
    seasonal = _read_csv(FLUX_ATTRIBUTION_TABLE_DIR / "seasonal_flux_trends_by_river.csv")
    phenology = _read_csv(FLUX_ATTRIBUTION_TABLE_DIR / "export_phenology_trends_by_river.csv")
    y_driver = driver[driver["river"].astype(str).eq("Yukon")].iloc[0]
    q = _trend_lookup(component, "Yukon", "annual_Q_volume_km3")
    fw_doc = _trend_lookup(component, "Yukon", "flow_weighted_DOC_mgC_L")
    after_july = _trend_lookup(phenology, "Yukon", "fraction_flux_after_july")
    lines = [
        "# Yukon Flux Attribution Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Scope",
        "",
        "This Yukon focus report uses existing guarded daily flux, annual flux cohorts, seasonal summaries, and export timing diagnostics. It performs exploratory mechanism analysis only; it does not retrain a model, generate new DOC predictions, recalculate daily flux, or propagate discharge uncertainty.",
        "",
        "## Key answers",
        "",
        _md_table(summary, max_rows=20),
        "",
        "## Q-volume and DOC components",
        "",
        "- " + _trend_phrase(q, "Annual Q volume"),
        "- " + _trend_phrase(fw_doc, "Flow-weighted DOC"),
        "",
        "## Monthly and seasonal concentration of the signal",
        "",
        _md_table(monthly[(monthly["river"].astype(str).eq("Yukon")) & monthly["metric"].eq("monthly_flux_TgC")], max_rows=12),
        "",
        _md_table(seasonal[(seasonal["river"].astype(str).eq("Yukon")) & seasonal["metric"].eq("seasonal_flux_TgC")], max_rows=10),
        "",
        "## Export timing",
        "",
        "- " + _trend_phrase(after_july, "Fraction after July"),
        "",
        _md_table(phenology[phenology["river"].astype(str).eq("Yukon")], max_rows=20),
        "",
        "## Final classification",
        "",
        f"`{y_driver['driver_classification']}`. {y_driver['interpretation']}",
        "",
        "This classification is exploratory and should not be treated as causal proof.",
    ]
    YUKON_ATTRIBUTION_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return YUKON_ATTRIBUTION_REPORT_PATH


def write_flux_attribution_report() -> Path:
    FLUX_ATTRIBUTION_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    annual = _read_csv(FLUX_ATTRIBUTION_TABLE_DIR / "annual_flux_attribution_by_river_year.csv")
    component = _read_csv(FLUX_ATTRIBUTION_TABLE_DIR / "q_doc_component_trends_by_river.csv")
    driver = _read_csv(FLUX_ATTRIBUTION_TABLE_DIR / "flux_driver_classification.csv")
    monthly_trends = _read_csv(FLUX_ATTRIBUTION_TABLE_DIR / "monthly_flux_trends_by_river.csv")
    seasonal_trends = _read_csv(FLUX_ATTRIBUTION_TABLE_DIR / "seasonal_flux_trends_by_river.csv")
    phenology = _read_csv(FLUX_ATTRIBUTION_TABLE_DIR / "export_phenology_by_river_year.csv")
    phenology_trends = _read_csv(FLUX_ATTRIBUTION_TABLE_DIR / "export_phenology_trends_by_river.csv")
    yukon = _read_csv(FLUX_ATTRIBUTION_TABLE_DIR / "yukon_flux_attribution_summary.csv")
    core_driver = driver[driver["driver_classification"].ne("not_applicable_no_detectable_annual_trend")]
    lines = [
        "# Flux Attribution, Seasonal Redistribution, and Export Phenology Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "## 1. Scope and guardrails",
        "",
        "This phase explores annual DOC flux drivers using existing daily DOC flux, daily DOC predictions, cohort flags, annual trends, fixed May-July interpretation, dynamic snowmelt-window summaries, and final synthesis metadata. It does not train models, generate new DOC predictions, recalculate daily flux, read raw/interim/canonical data, or modify gold data.",
        "",
        "## 2. Why attribution is needed",
        "",
        "Yukon is the only core 2003-2024 river with a detectable increasing annual DOC flux trend, while the six-river aggregate has no detectable trend. Fixed May-July did not explain the Yukon signal, and dynamic snowmelt windows were partial but non-decisive. This analysis separates discharge volume, flow-weighted DOC, seasonal redistribution, and export timing diagnostics.",
        "",
        "## 3. Annual Q-volume and DOC decomposition",
        "",
        _md_table(annual.head(24), max_rows=24),
        "",
        "## 4. Component trends by river",
        "",
        _md_table(component, max_rows=40),
        "",
        "## 5. Flux driver classification",
        "",
        _md_table(driver, max_rows=20),
        "",
        "## 6. Monthly flux decomposition",
        "",
        _md_table(monthly_trends[monthly_trends["metric"].eq("monthly_flux_TgC")], max_rows=40),
        "",
        "## 7. Seasonal flux redistribution",
        "",
        _md_table(seasonal_trends[seasonal_trends["metric"].isin(["seasonal_flux_TgC", "seasonal_fraction_of_annual"])], max_rows=40),
        "",
        "## 8. Export phenology and cumulative timing",
        "",
        _md_table(phenology.head(24), max_rows=24),
        "",
        _md_table(phenology_trends, max_rows=40),
        "",
        "## 9. Yukon focus",
        "",
        _md_table(yukon, max_rows=20),
        "",
        "## 10. Relationship to May-July and dynamic snowmelt findings",
        "",
        "Fixed May-July remains provisional and did not explain the Yukon annual increase. Dynamic hydrologic windows provide exploratory context, but they do not make the attribution decisive. The attribution classification therefore treats seasonal redistribution and unresolved timing structure cautiously unless component trends are detectable.",
        "",
        "## 11. Caveats",
        "",
        "- Flux uncertainty includes DOC concentration empirical residual intervals only; discharge uncertainty is not propagated.",
        "- Attribution is based on modeled DOC concentration and observed Q.",
        "- The analysis is exploratory mechanism analysis, not causal proof.",
        "- Low-confidence flux fractions, extrapolation flags, and coverage caveats remain inherited from flux cohorts.",
        "",
        "## 12. Manuscript-ready interpretation",
        "",
        "The core annual trend signal remains river-specific: Yukon shows a detectable increase, while the aggregate does not. Driver attribution should be framed as exploratory evidence distinguishing discharge-volume, concentration, seasonal redistribution, and export-phenology hypotheses.",
        "",
        "Detected annual-driver classifications:",
        "",
        _md_table(core_driver, max_rows=10),
        "",
        "## 13. What not to claim",
        "",
        "- Do not claim formal causal proof.",
        "- Do not claim discharge uncertainty has been propagated.",
        "- Do not claim optical proxy explained the flux trend.",
        "- Do not call fixed May-July a final snowmelt flux window.",
        "",
        "## Explicit statements",
        "",
        "- No model retraining was performed.",
        "- No new DOC prediction was generated.",
        "- No flux recalculation was performed.",
        "- Discharge uncertainty was not propagated.",
        "- Attribution is based on modeled DOC concentration and observed Q.",
        "- Results are exploratory mechanism analysis, not causal proof.",
    ]
    FLUX_ATTRIBUTION_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_yukon_flux_attribution_report()
    return FLUX_ATTRIBUTION_REPORT_PATH


def run_flux_attribution() -> dict[str, Any]:
    _ensure_dirs()
    before_hashes = _input_hashes()
    inputs = _load_inputs()
    daily_flux = _prepare_daily_flux(inputs["daily_flux"])
    annual = _prepare_annual(inputs["annual_flux"])
    cohorts = _prepare_cohorts(inputs["cohorts"])
    annual_attr = _annual_attribution(daily_flux, annual, cohorts)
    component_trends = _component_trends(annual_attr)
    driver = _driver_classification(inputs["annual_trends"], component_trends)
    monthly = _monthly_decomposition(daily_flux, annual_attr)
    monthly_trends = _grouped_trends(
        monthly,
        ["river", "month"],
        [
            ("monthly_flux_TgC", "Tg C/year"),
            ("monthly_fraction_of_annual", "fraction/year"),
            ("monthly_Q_volume_km3", "km3/year"),
            ("monthly_flow_weighted_DOC", "mg C/L/year"),
        ],
    )
    seasonal = _seasonal_decomposition(monthly, annual_attr)
    seasonal_trends = _grouped_trends(
        seasonal,
        ["river", "season_window"],
        [
            ("seasonal_flux_TgC", "Tg C/year"),
            ("seasonal_fraction_of_annual", "fraction/year"),
            ("seasonal_Q_volume_km3", "km3/year"),
            ("seasonal_flow_weighted_DOC", "mg C/L/year"),
        ],
    )
    phenology = _phenology(daily_flux, annual_attr)
    phenology_trends = _phenology_trends(phenology)
    yukon = _yukon_summary(
        driver,
        component_trends,
        monthly_trends,
        seasonal_trends,
        phenology_trends,
        inputs["snowmelt_comparison"],
    )
    figures = _make_figures(annual_attr, component_trends, monthly, monthly_trends, seasonal, phenology, phenology_trends)
    table_paths = [
        _write_csv(annual_attr, FLUX_ATTRIBUTION_TABLE_DIR / "annual_flux_attribution_by_river_year.csv"),
        _write_csv(component_trends, FLUX_ATTRIBUTION_TABLE_DIR / "q_doc_component_trends_by_river.csv"),
        _write_csv(driver, FLUX_ATTRIBUTION_TABLE_DIR / "flux_driver_classification.csv"),
        _write_csv(monthly, FLUX_ATTRIBUTION_TABLE_DIR / "monthly_flux_by_river_year.csv"),
        _write_csv(monthly_trends, FLUX_ATTRIBUTION_TABLE_DIR / "monthly_flux_trends_by_river.csv"),
        _write_csv(seasonal, FLUX_ATTRIBUTION_TABLE_DIR / "seasonal_flux_decomposition_by_river_year.csv"),
        _write_csv(seasonal_trends, FLUX_ATTRIBUTION_TABLE_DIR / "seasonal_flux_trends_by_river.csv"),
        _write_csv(phenology, FLUX_ATTRIBUTION_TABLE_DIR / "export_phenology_by_river_year.csv"),
        _write_csv(phenology_trends, FLUX_ATTRIBUTION_TABLE_DIR / "export_phenology_trends_by_river.csv"),
        _write_csv(yukon, FLUX_ATTRIBUTION_TABLE_DIR / "yukon_flux_attribution_summary.csv"),
    ]
    report_path = write_flux_attribution_report()
    _verify_inputs_unchanged(before_hashes)
    return {
        "tables": table_paths,
        "figures": figures,
        "report": report_path,
        "yukon_report": YUKON_ATTRIBUTION_REPORT_PATH,
        "annual_attribution": annual_attr,
        "component_trends": component_trends,
        "driver_classification": driver,
        "monthly_trends": monthly_trends,
        "seasonal_trends": seasonal_trends,
        "phenology_trends": phenology_trends,
    }


__all__ = [
    "FLUX_ATTRIBUTION_TABLE_DIR",
    "FLUX_ATTRIBUTION_REPORT_DIR",
    "FLUX_ATTRIBUTION_FIGURE_DIR",
    "FLUX_ATTRIBUTION_REPORT_PATH",
    "YUKON_ATTRIBUTION_REPORT_PATH",
    "REQUIRED_ATTRIBUTION_INPUTS",
    "run_flux_attribution",
    "write_flux_attribution_report",
    "write_yukon_flux_attribution_report",
]
