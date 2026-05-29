from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .trend_statistics import TrendResult, trend_for_series


TREND_METRICS = [
    "window_flux_TgC",
    "window_fraction_of_annual",
    "window_start_doy",
    "window_end_doy",
    "window_peak_q_doy",
    "window_length_days",
]


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _analysis_sets(summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = summary.copy()
    out["core_2003_2024"] = _as_bool(out["core_2003_2024"])
    out["sensitivity_only"] = _as_bool(out["sensitivity_only"])
    out["exclude_from_trend"] = _as_bool(out["exclude_from_trend"])
    return {
        "core_2003_2024": out[out["core_2003_2024"] & ~out["exclude_from_trend"]].copy(),
        "full_2000_2025_sensitivity": out[~out["exclude_from_trend"]].copy(),
        "high_confidence_only_sensitivity": out[
            ~out["exclude_from_trend"]
            & out["window_confidence_tier"].astype(str).eq("high")
            & out["annual_confidence_tier"].astype(str).eq("high")
        ].copy(),
    }


def _trend_row(window_id: str, metric: str, river: str, analysis_cohort: str, group: pd.DataFrame, result: TrendResult) -> dict[str, Any]:
    confidence_bits = ["DOC_concentration_uncertainty_only", "discharge_uncertainty_not_propagated"]
    if "sensitivity" in analysis_cohort:
        confidence_bits.append("sensitivity_not_primary")
    if group["window_confidence_tier"].astype(str).eq("low").any():
        confidence_bits.append("contains_low_confidence_window_years")
    if group["caveat_reason"].astype(str).str.contains("fallback", case=False, na=False).any():
        confidence_bits.append("contains_fallback_windows")
    return {
        "window_id": window_id,
        "metric": metric,
        "river": river,
        "analysis_cohort": analysis_cohort,
        "n_years": result.n_years,
        "year_min": result.year_min,
        "year_max": result.year_max,
        "value_mean": result.value_mean,
        "value_median": result.value_median,
        "slope_ols_per_year": result.slope_ols,
        "slope_ols_p_value": result.slope_ols_p_value,
        "slope_ols_r2": result.slope_ols_r2,
        "slope_theilsen_per_year": result.slope_theilsen,
        "kendall_tau": result.kendall_tau,
        "kendall_p_value": result.kendall_p_value,
        "trend_direction": result.trend_direction,
        "trend_strength": result.trend_strength,
        "detectable_trend": result.significant_at_0_05,
        "confidence_caveat": ";".join(confidence_bits),
    }


def build_snowmelt_window_trends(summary: pd.DataFrame) -> pd.DataFrame:
    sets = _analysis_sets(summary)
    rows: list[dict[str, Any]] = []
    for analysis_cohort, frame in sets.items():
        for (window_id, river), group in frame.groupby(["window_id", "river"], dropna=False):
            if len(group) < 3:
                continue
            for metric in TREND_METRICS:
                result = trend_for_series(group, metric)
                rows.append(_trend_row(str(window_id), metric, str(river), analysis_cohort, group, result))
    return pd.DataFrame(rows)


def _bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def build_annual_vs_snowmelt_comparison(annual_trends: pd.DataFrame, window_trends: pd.DataFrame) -> pd.DataFrame:
    annual_core = annual_trends[annual_trends["analysis_cohort"].astype(str).eq("core_2003_2024")].set_index("river")
    core = window_trends[window_trends["analysis_cohort"].astype(str).eq("core_2003_2024")].copy()
    flux = core[core["metric"].eq("window_flux_TgC")].set_index(["river", "window_id"])
    fraction = core[core["metric"].eq("window_fraction_of_annual")].set_index(["river", "window_id"])
    keys = sorted(set(flux.index).union(fraction.index))
    rows: list[dict[str, Any]] = []
    for river, window_id in keys:
        annual = annual_core.loc[river] if river in annual_core.index else pd.Series(dtype=object)
        flux_row = flux.loc[(river, window_id)] if (river, window_id) in flux.index else pd.Series(dtype=object)
        fraction_row = fraction.loc[(river, window_id)] if (river, window_id) in fraction.index else pd.Series(dtype=object)
        annual_direction = str(annual.get("trend_direction", ""))
        annual_slope = float(annual.get("slope_ols_TgC_per_year", np.nan))
        annual_p = float(annual.get("slope_ols_p_value", np.nan))
        annual_detectable = _bool_value(annual.get("significant_at_0_05", False))
        window_direction = str(flux_row.get("trend_direction", ""))
        window_slope = float(flux_row.get("slope_ols_per_year", np.nan))
        window_p = float(flux_row.get("slope_ols_p_value", np.nan))
        window_detectable = _bool_value(flux_row.get("detectable_trend", False))
        fraction_direction = str(fraction_row.get("trend_direction", ""))
        fraction_slope = float(fraction_row.get("slope_ols_per_year", np.nan))
        fraction_p = float(fraction_row.get("slope_ols_p_value", np.nan))
        fraction_detectable = _bool_value(fraction_row.get("detectable_trend", False))
        if not annual_detectable or annual_direction != "increasing":
            decision = "not_applicable"
            interpretation = "No detectable increasing annual flux signal in the core cohort."
        elif window_detectable and window_direction == "increasing" and fraction_direction in {"increasing", "flat_or_uncertain"}:
            decision = "yes"
            interpretation = "Window flux increases detectably and the window fraction is stable or increasing."
        elif window_slope > 0 and (not fraction_detectable or fraction_direction in {"increasing", "flat_or_uncertain"}):
            decision = "partial"
            interpretation = "Window flux slope is positive but not detectably increasing, or support is weak."
        elif (not window_detectable and window_direction == "flat_or_uncertain") or fraction_direction == "decreasing":
            decision = "no"
            interpretation = "Annual increase is not matched by a detectable window-flux increase or the window fraction decreases."
        else:
            decision = "uncertain"
            interpretation = "Window trend diagnostics are inconsistent or underpowered."
        if river == "Yukon" and annual_detectable and annual_direction == "increasing":
            if decision == "yes":
                interpretation = "Yukon annual flux increase is consistent with a detectable dynamic window flux increase."
            elif decision == "partial":
                interpretation = "Yukon annual flux increase has a positive but not decisive dynamic window signal."
            elif decision == "no":
                interpretation = "Yukon annual flux increase is not explained by this dynamic window under the core trend criteria."
        rows.append(
            {
                "river": river,
                "window_id": window_id,
                "annual_trend_direction": annual_direction,
                "annual_slope_TgC_per_year": annual_slope,
                "annual_p_value": annual_p,
                "window_flux_trend_direction": window_direction,
                "window_flux_slope_TgC_per_year": window_slope,
                "window_flux_p_value": window_p,
                "window_fraction_trend_direction": fraction_direction,
                "window_fraction_slope_per_year": fraction_slope,
                "window_fraction_p_value": fraction_p,
                "does_window_explain_annual_signal": decision,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


__all__ = ["TREND_METRICS", "build_snowmelt_window_trends", "build_annual_vs_snowmelt_comparison"]
