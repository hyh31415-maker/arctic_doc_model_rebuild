from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class TrendResult:
    n_years: int
    year_min: int | None
    year_max: int | None
    value_mean: float
    value_median: float
    slope_ols: float
    intercept_ols: float
    slope_ols_p_value: float
    slope_ols_r2: float
    slope_ols_stderr: float
    slope_theilsen: float
    intercept_theilsen: float
    theilsen_ci_lower: float
    theilsen_ci_upper: float
    kendall_tau: float
    kendall_p_value: float
    slope_percent_per_year: float
    total_change_percent_over_period: float
    trend_direction: str
    trend_strength: str
    significant_at_0_05: bool


def _clean_xy(frame: pd.DataFrame, value_column: str) -> tuple[np.ndarray, np.ndarray]:
    work = frame[["year", value_column]].copy()
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    work = work.dropna().sort_values("year")
    return work["year"].to_numpy(dtype=float), work[value_column].to_numpy(dtype=float)


def classify_trend(slope: float, p_value: float, slope_percent: float) -> tuple[str, str, bool]:
    significant = bool(pd.notna(p_value) and p_value < 0.05)
    if not significant or pd.isna(slope) or abs(float(slope)) < 1e-12:
        return "flat_or_uncertain", "not_detected", significant
    direction = "increasing" if slope > 0 else "decreasing"
    magnitude = abs(float(slope_percent))
    if magnitude >= 2.0:
        strength = "strong"
    elif magnitude >= 1.0:
        strength = "moderate"
    else:
        strength = "weak"
    return direction, strength, significant


def trend_for_series(frame: pd.DataFrame, value_column: str = "annual_flux_TgC") -> TrendResult:
    years, values = _clean_xy(frame, value_column)
    n = int(len(values))
    if n < 3:
        nan = float("nan")
        return TrendResult(n, None, None, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, "flat_or_uncertain", "not_detected", False)

    ols = stats.linregress(years, values)
    try:
        ts_slope, ts_intercept, ts_low, ts_high = stats.theilslopes(values, years, alpha=0.90)
    except Exception:
        ts_slope, ts_intercept, ts_low, ts_high = (float("nan"), float("nan"), float("nan"), float("nan"))
    kendall = stats.kendalltau(years, values, nan_policy="omit")
    mean_value = float(np.nanmean(values))
    median_value = float(np.nanmedian(values))
    year_span = float(np.nanmax(years) - np.nanmin(years))
    slope_percent = float(ols.slope / mean_value * 100.0) if mean_value else float("nan")
    total_change = float((ols.slope * year_span) / mean_value * 100.0) if mean_value else float("nan")
    direction, strength, significant = classify_trend(float(ols.slope), float(ols.pvalue), slope_percent)
    return TrendResult(
        n_years=n,
        year_min=int(np.nanmin(years)),
        year_max=int(np.nanmax(years)),
        value_mean=mean_value,
        value_median=median_value,
        slope_ols=float(ols.slope),
        intercept_ols=float(ols.intercept),
        slope_ols_p_value=float(ols.pvalue),
        slope_ols_r2=float(ols.rvalue**2),
        slope_ols_stderr=float(ols.stderr),
        slope_theilsen=float(ts_slope),
        intercept_theilsen=float(ts_intercept),
        theilsen_ci_lower=float(ts_low),
        theilsen_ci_upper=float(ts_high),
        kendall_tau=float(kendall.statistic) if pd.notna(kendall.statistic) else float("nan"),
        kendall_p_value=float(kendall.pvalue) if pd.notna(kendall.pvalue) else float("nan"),
        slope_percent_per_year=slope_percent,
        total_change_percent_over_period=total_change,
        trend_direction=direction,
        trend_strength=strength,
        significant_at_0_05=significant,
    )
