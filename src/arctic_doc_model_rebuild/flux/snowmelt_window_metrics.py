from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..gold_contract import sha256_file
from .flux_qc import as_bool
from .snowmelt_window_reports import SNOWMELT_REPORT_PATH, SNOWMELT_TABLE_DIR, write_snowmelt_window_report
from .snowmelt_window_trends import build_annual_vs_snowmelt_comparison, build_snowmelt_window_trends
from .snowmelt_windows import (
    SNOWMELT_FIGURE_DIR,
    _read_csv,
    _write_csv,
    required_snowmelt_input_paths,
)


DEFINITION_PATH = SNOWMELT_TABLE_DIR / "snowmelt_window_definitions_by_river_year.csv"


def _input_hashes() -> dict[Path, str]:
    paths = required_snowmelt_input_paths()
    paths["snowmelt_window_definitions"] = DEFINITION_PATH
    return {destination: sha256_file(destination) for destination in paths.values()}


def _verify_metric_inputs_unchanged(before: dict[Path, str]) -> None:
    after = _input_hashes()
    changed = [str(destination) for destination, before_hash in before.items() if after.get(destination) != before_hash]
    if changed:
        raise RuntimeError(f"Snowmelt window flux inputs changed during summarization: {changed}")


def _prepare_daily_flux(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    for column in out.columns:
        if column.endswith("_TgC_day") or column in ["daily_flux_TgC_day", "Q_m3s"]:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in [
        "outside_training_logQ_range",
        "outside_training_doy_range",
        "outside_training_year_range",
        "point_prediction_clipped_at_zero",
        "interval_lower_clipped_at_zero",
    ]:
        if column in out.columns:
            out[column] = as_bool(out[column])
    return out


def _date_or_nat(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce")


def _doy(value: Any) -> float:
    dt = pd.to_datetime(value, errors="coerce")
    return float(dt.dayofyear) if pd.notna(dt) else np.nan


def _window_daily_subset(daily_flux: pd.DataFrame, river: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if pd.isna(start) or pd.isna(end) or end < start:
        return daily_flux.iloc[0:0].copy()
    return daily_flux[
        daily_flux["river"].astype(str).eq(str(river))
        & daily_flux["date"].between(start, end, inclusive="both")
        & daily_flux["flux_status"].astype(str).eq("calculated")
    ].copy()


def _sum_column(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").sum()) if column in frame.columns and not frame.empty else np.nan


def _flux_summary_rows(definitions: pd.DataFrame, daily_flux: pd.DataFrame, annual: pd.DataFrame, cohorts: pd.DataFrame) -> pd.DataFrame:
    annual_key = annual.copy()
    annual_key["year"] = pd.to_numeric(annual_key["year"], errors="coerce").astype("Int64")
    annual_key = annual_key.set_index(["river", "year"])
    cohorts_key = cohorts.copy()
    cohorts_key["year"] = pd.to_numeric(cohorts_key["year"], errors="coerce").astype("Int64")
    for column in ["cohort_core_2003_2024", "cohort_sensitivity_only", "cohort_exclude_from_trend"]:
        cohorts_key[column] = as_bool(cohorts_key[column])
    cohorts_key = cohorts_key.set_index(["river", "year"])
    rows: list[dict[str, Any]] = []
    for definition in definitions.itertuples(index=False):
        river = str(definition.river)
        year = int(definition.year)
        start = _date_or_nat(definition.start_date)
        end = _date_or_nat(definition.end_date)
        subset = _window_daily_subset(daily_flux, river, start, end)
        annual_row = annual_key.loc[(river, year)] if (river, year) in annual_key.index else pd.Series(dtype=object)
        cohort_row = cohorts_key.loc[(river, year)] if (river, year) in cohorts_key.index else pd.Series(dtype=object)
        window_flux = _sum_column(subset, "daily_flux_TgC_day")
        low_flux = float(subset.loc[subset["daily_confidence_tier"].astype(str).eq("low"), "daily_flux_TgC_day"].sum()) if not subset.empty else np.nan
        fraction_low = float(low_flux / window_flux) if pd.notna(window_flux) and window_flux > 0 else np.nan
        annual_flux = float(annual_row.get("annual_flux_TgC", np.nan))
        rows.append(
            {
                "river": river,
                "year": year,
                "window_id": definition.window_id,
                "start_date": definition.start_date,
                "end_date": definition.end_date,
                "window_start_doy": _doy(definition.start_date),
                "window_end_doy": _doy(definition.end_date),
                "window_peak_q_doy": _doy(definition.peak_q_date),
                "peak_q_date": definition.peak_q_date,
                "peak_Q_m3s": definition.peak_Q_m3s,
                "window_length_days": definition.window_length_days,
                "window_flux_TgC": window_flux,
                "window_flux_80_lower_TgC": _sum_column(subset, "daily_flux_80_lower_TgC_day"),
                "window_flux_80_upper_TgC": _sum_column(subset, "daily_flux_80_upper_TgC_day"),
                "window_flux_90_lower_TgC": _sum_column(subset, "daily_flux_90_lower_TgC_day"),
                "window_flux_90_upper_TgC": _sum_column(subset, "daily_flux_90_upper_TgC_day"),
                "window_flux_95_lower_TgC": _sum_column(subset, "daily_flux_95_lower_TgC_day"),
                "window_flux_95_upper_TgC": _sum_column(subset, "daily_flux_95_upper_TgC_day"),
                "annual_flux_TgC": annual_flux,
                "window_fraction_of_annual": float(window_flux / annual_flux) if pd.notna(window_flux) and pd.notna(annual_flux) and annual_flux > 0 else np.nan,
                "n_days_with_flux": int(len(subset)),
                "coverage_rate": definition.coverage_rate,
                "n_low_confidence_days": int(subset["daily_confidence_tier"].astype(str).eq("low").sum()) if not subset.empty else 0,
                "fraction_flux_from_low_confidence_days": fraction_low,
                "window_confidence_tier": definition.window_confidence_tier,
                "annual_confidence_tier": annual_row.get("annual_confidence_tier", ""),
                "core_2003_2024": bool(cohort_row.get("cohort_core_2003_2024", False)),
                "sensitivity_only": bool(cohort_row.get("cohort_sensitivity_only", False)),
                "exclude_from_trend": bool(cohort_row.get("cohort_exclude_from_trend", False)) or str(definition.window_confidence_tier) == "low",
                "caveat_reason": definition.caveat_reason,
            }
        )
    return pd.DataFrame(rows)


def _qc_summary(definitions: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window_id, group in summary.groupby("window_id", dropna=False):
        definitions_group = definitions[definitions["window_id"].astype(str).eq(str(window_id))]
        rows.append(
            {
                "window_id": window_id,
                "n_river_years": int(len(group)),
                "n_high_confidence": int(group["window_confidence_tier"].astype(str).eq("high").sum()),
                "n_medium_confidence": int(group["window_confidence_tier"].astype(str).eq("medium").sum()),
                "n_low_confidence": int(group["window_confidence_tier"].astype(str).eq("low").sum()),
                "n_fallback_used": int(definitions_group["fallback_used"].astype(str).str.lower().isin({"true", "1"}).sum()),
                "median_window_length_days": float(pd.to_numeric(group["window_length_days"], errors="coerce").median()),
                "median_window_fraction_of_annual": float(pd.to_numeric(group["window_fraction_of_annual"], errors="coerce").median()),
                "notes": "fixed_may_july_reference_is_reference_only" if window_id == "fixed_may_july_reference" else "dynamic_hydrologic_window",
            }
        )
    return pd.DataFrame(rows)


def _make_figures(summary: pd.DataFrame, trends: pd.DataFrame, comparison: pd.DataFrame) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    SNOWMELT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def save(fig, name: str) -> None:
        fig.tight_layout()
        destination = SNOWMELT_FIGURE_DIR / name
        fig.savefig(destination, dpi=120)
        plt.close(fig)
        paths.append(destination)

    core = summary[summary["core_2003_2024"].astype(bool)].copy()
    primary = core[core["window_id"].eq("discharge_centered_freshet")]
    if not primary.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        for river, group in primary.groupby("river"):
            ax.plot(group["year"], group["window_start_doy"], marker="o", linewidth=1.0, label=f"{river} start")
            ax.plot(group["year"], group["window_end_doy"], marker="x", linewidth=1.0, alpha=0.7)
        ax.set_xlabel("Year")
        ax.set_ylabel("Day of year")
        ax.set_title("Discharge-centered window timing by river-year")
        ax.legend(fontsize="xx-small", ncol=2)
        save(fig, "window_timing_by_river_year.png")

        fig, ax = plt.subplots(figsize=(9, 5))
        for river, group in primary.groupby("river"):
            ax.plot(group["year"], group["window_flux_TgC"], marker="o", linewidth=1.0, label=str(river))
        ax.set_xlabel("Year")
        ax.set_ylabel("Window DOC flux (Tg C)")
        ax.set_title("Discharge-centered window flux by river")
        ax.legend(fontsize="x-small")
        save(fig, "window_flux_timeseries_by_river.png")

        fig, ax = plt.subplots(figsize=(9, 5))
        for river, group in primary.groupby("river"):
            ax.plot(group["year"], group["window_fraction_of_annual"], marker="o", linewidth=1.0, label=str(river))
        ax.set_xlabel("Year")
        ax.set_ylabel("Window fraction of annual flux")
        ax.set_title("Discharge-centered window fraction by river")
        ax.legend(fontsize="x-small")
        save(fig, "window_fraction_timeseries_by_river.png")

        yukon = primary[primary["river"].astype(str).eq("Yukon")]
        if not yukon.empty:
            fig, ax1 = plt.subplots(figsize=(8, 4.8))
            ax1.plot(yukon["year"], yukon["annual_flux_TgC"], marker="o", label="annual flux")
            ax1.plot(yukon["year"], yukon["window_flux_TgC"], marker="o", label="dynamic window flux")
            ax1.set_xlabel("Year")
            ax1.set_ylabel("DOC flux (Tg C)")
            ax2 = ax1.twinx()
            ax2.plot(yukon["year"], yukon["window_fraction_of_annual"], marker="s", linestyle="--", label="window fraction")
            ax2.set_ylabel("Window fraction")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, fontsize="x-small")
            ax1.set_title("Yukon annual vs dynamic snowmelt window")
            save(fig, "yukon_annual_vs_snowmelt_window.png")

    for metric, name, title in [
        ("window_start_doy", "window_start_doy_trends.png", "Core start-DOY trend slopes"),
        ("window_length_days", "window_length_trends.png", "Core window-length trend slopes"),
    ]:
        subset = trends[trends["analysis_cohort"].eq("core_2003_2024") & trends["metric"].eq(metric) & trends["window_id"].eq("discharge_centered_freshet")]
        if not subset.empty:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            ax.barh(subset["river"], subset["slope_ols_per_year"])
            ax.axvline(0, linestyle="--")
            ax.set_xlabel("Slope per year")
            ax.set_title(title)
            save(fig, name)

    compare = core[core["window_id"].isin(["fixed_may_july_reference", "discharge_centered_freshet", "q75_peak_contiguous", "snow_depletion_assisted"])]
    if not compare.empty:
        pivot_data = [group["window_fraction_of_annual"].dropna().to_numpy() for _, group in compare.groupby("window_id")]
        labels = [str(window_id) for window_id, _ in compare.groupby("window_id")]
        fig, ax = plt.subplots(figsize=(10, 5))
        if pivot_data:
            try:
                ax.boxplot(pivot_data, tick_labels=labels, showfliers=False)
            except TypeError:
                ax.boxplot(pivot_data, labels=labels, showfliers=False)
        ax.set_ylabel("Window fraction of annual flux")
        ax.set_title("Fixed May-July vs dynamic window fractions")
        ax.tick_params(axis="x", rotation=25)
        save(fig, "may_july_vs_dynamic_window_fraction.png")
    return paths


def run_snowmelt_window_flux() -> dict[str, Any]:
    if not DEFINITION_PATH.exists():
        raise FileNotFoundError("Run `python -m arctic_doc_model_rebuild.cli define-snowmelt-windows` before window flux summarization.")
    before_hashes = _input_hashes()
    paths = required_snowmelt_input_paths()
    definitions = _read_csv(DEFINITION_PATH)
    daily_flux = _prepare_daily_flux(_read_csv(paths["daily_doc_flux"]))
    annual = _read_csv(paths["annual_doc_flux_summary"])
    cohorts = _read_csv(paths["annual_flux_analysis_cohorts"])
    annual_trends = _read_csv(paths["annual_flux_trends_by_river"])
    summary = _flux_summary_rows(definitions, daily_flux, annual, cohorts)
    trends = build_snowmelt_window_trends(summary)
    comparison = build_annual_vs_snowmelt_comparison(annual_trends, trends)
    qc = _qc_summary(definitions, summary)
    figures = _make_figures(summary, trends, comparison)
    table_paths = [
        _write_csv(summary, SNOWMELT_TABLE_DIR / "snowmelt_window_flux_summary.csv"),
        _write_csv(trends, SNOWMELT_TABLE_DIR / "snowmelt_window_trends_by_river.csv"),
        _write_csv(comparison, SNOWMELT_TABLE_DIR / "annual_vs_snowmelt_signal_comparison.csv"),
        _write_csv(qc, SNOWMELT_TABLE_DIR / "snowmelt_window_qc_summary.csv"),
    ]
    report_path = write_snowmelt_window_report()
    _verify_metric_inputs_unchanged(before_hashes)
    return {
        "tables": table_paths,
        "figures": figures,
        "report": report_path,
        "summary": summary,
        "trends": trends,
        "comparison": comparison,
        "qc": qc,
    }


__all__ = ["DEFINITION_PATH", "run_snowmelt_window_flux"]
