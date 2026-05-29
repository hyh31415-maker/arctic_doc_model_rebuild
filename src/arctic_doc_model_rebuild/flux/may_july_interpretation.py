from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..gold_contract import sha256_file
from ..paths import REPORT_DIR, TABLE_DIR, path
from .may_july_reports import MAY_JULY_REPORT_DIR, MAY_JULY_REPORT_PATH, MAY_JULY_TABLE_DIR, write_may_july_flux_report
from .trend_statistics import TrendResult, trend_for_series


MAY_JULY_FIGURE_DIR = path("outputs", "figures", "may_july_flux")
DOC_FLUX_TABLE_DIR = TABLE_DIR / "doc_flux"
FLUX_INTERPRETATION_TABLE_DIR = TABLE_DIR / "flux_interpretation"
ANNUAL_TREND_TABLE_DIR = TABLE_DIR / "annual_flux_trends"

REQUIRED_MAY_JULY_INPUTS = {
    "provisional_may_july_flux_summary": DOC_FLUX_TABLE_DIR / "provisional_may_july_flux_summary.csv",
    "annual_doc_flux_summary": DOC_FLUX_TABLE_DIR / "annual_doc_flux_summary.csv",
    "annual_flux_analysis_cohorts": FLUX_INTERPRETATION_TABLE_DIR / "annual_flux_analysis_cohorts.csv",
    "provisional_may_july_cohort_summary": FLUX_INTERPRETATION_TABLE_DIR / "provisional_may_july_cohort_summary.csv",
    "annual_flux_trends_by_river": ANNUAL_TREND_TABLE_DIR / "annual_flux_trends_by_river.csv",
    "annual_flux_trends_aggregate": ANNUAL_TREND_TABLE_DIR / "annual_flux_trends_aggregate.csv",
    "annual_flux_trend_report": REPORT_DIR / "annual_flux_trends" / "annual_flux_trend_report.md",
    "doc_flux_report": REPORT_DIR / "doc_flux" / "doc_flux_report.md",
}


def _ensure_dirs() -> None:
    for directory in [MAY_JULY_TABLE_DIR, MAY_JULY_REPORT_DIR, MAY_JULY_FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def _read_csv(destination: Path) -> pd.DataFrame:
    if not destination.exists():
        raise FileNotFoundError(f"Required May-July flux interpretation input is missing: {destination}")
    return pd.read_csv(destination, low_memory=False)


def _read_text(destination: Path) -> str:
    if not destination.exists():
        raise FileNotFoundError(f"Required May-July flux interpretation input is missing: {destination}")
    return destination.read_text(encoding="utf-8")


def _write_csv(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8")
    return destination


def _input_hashes() -> dict[Path, str]:
    return {destination: sha256_file(destination) for destination in REQUIRED_MAY_JULY_INPUTS.values()}


def _verify_inputs_unchanged(before: dict[Path, str]) -> None:
    after = _input_hashes()
    changed = [str(destination) for destination, before_hash in before.items() if after.get(destination) != before_hash]
    if changed:
        raise RuntimeError(f"May-July flux interpretation inputs changed during analysis: {changed}")


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _load_inputs() -> dict[str, Any]:
    return {
        "may_july_flux": _read_csv(REQUIRED_MAY_JULY_INPUTS["provisional_may_july_flux_summary"]),
        "annual_flux": _read_csv(REQUIRED_MAY_JULY_INPUTS["annual_doc_flux_summary"]),
        "annual_cohorts": _read_csv(REQUIRED_MAY_JULY_INPUTS["annual_flux_analysis_cohorts"]),
        "may_july_cohorts": _read_csv(REQUIRED_MAY_JULY_INPUTS["provisional_may_july_cohort_summary"]),
        "annual_trends": _read_csv(REQUIRED_MAY_JULY_INPUTS["annual_flux_trends_by_river"]),
        "annual_aggregate": _read_csv(REQUIRED_MAY_JULY_INPUTS["annual_flux_trends_aggregate"]),
        "annual_trend_report_text": _read_text(REQUIRED_MAY_JULY_INPUTS["annual_flux_trend_report"]),
        "doc_flux_report_text": _read_text(REQUIRED_MAY_JULY_INPUTS["doc_flux_report"]),
    }


def _prepare_interpretation_rows(inputs: dict[str, Any]) -> pd.DataFrame:
    may_july = inputs["may_july_flux"].copy()
    annual_cohorts = inputs["annual_cohorts"].copy()
    may_july_cohorts = inputs["may_july_cohorts"].copy()
    for frame in [may_july, annual_cohorts, may_july_cohorts]:
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    numeric_columns = [
        "may_july_flux_TgC",
        "may_july_flux_90_lower_TgC",
        "may_july_flux_90_upper_TgC",
        "may_july_flux_fraction_of_annual",
        "coverage_rate",
        "fraction_flux_from_low_confidence_days",
    ]
    for column in numeric_columns:
        if column in may_july.columns:
            may_july[column] = pd.to_numeric(may_july[column], errors="coerce")
    for column in [
        "annual_flux_TgC",
        "annual_flux_90_lower_TgC",
        "annual_flux_90_upper_TgC",
        "coverage_rate",
        "fraction_flux_from_low_confidence_days",
    ]:
        annual_cohorts[column] = pd.to_numeric(annual_cohorts[column], errors="coerce")
    for column in [
        "cohort_core_2003_2024",
        "cohort_full_2000_2025",
        "cohort_high_confidence_only",
        "cohort_sensitivity_only",
        "cohort_exclude_from_trend",
    ]:
        annual_cohorts[column] = _as_bool(annual_cohorts[column])
    may_july_cohorts["core_may_july_interpretation_allowed"] = _as_bool(may_july_cohorts["core_may_july_interpretation_allowed"])
    annual_keep = [
        "river",
        "year",
        "annual_flux_TgC",
        "annual_flux_90_lower_TgC",
        "annual_flux_90_upper_TgC",
        "annual_confidence_tier",
        "cohort_core_2003_2024",
        "cohort_full_2000_2025",
        "cohort_high_confidence_only",
        "cohort_sensitivity_only",
        "cohort_exclude_from_trend",
        "exclusion_or_caveat_reason",
    ]
    may_keep = [
        "river",
        "year",
        "core_may_july_interpretation_allowed",
        "caveat_reason",
    ]
    out = may_july.merge(annual_cohorts[annual_keep], on=["river", "year"], how="left")
    out = out.merge(may_july_cohorts[may_keep], on=["river", "year"], how="left", suffixes=("", "_may_july_cohort"))
    out["may_july_confidence_tier"] = out["may_july_confidence_tier"].astype(str)

    def caveat(row: pd.Series) -> str:
        reasons = []
        annual_reason = str(row.get("exclusion_or_caveat_reason", "") or "")
        may_reason = str(row.get("caveat_reason", "") or "")
        if annual_reason and annual_reason != "meets_core_annual_criteria":
            reasons.append(f"annual:{annual_reason}")
        if may_reason and may_reason != "provisional_may_july_core_interpretation_allowed":
            reasons.append(f"may_july:{may_reason}")
        if not reasons:
            reasons.append("provisional_may_july_core_interpretation_allowed")
        return ";".join(dict.fromkeys(reasons))

    out["caveat_reason"] = out.apply(caveat, axis=1)
    keep = [
        "river",
        "year",
        "annual_flux_TgC",
        "annual_flux_90_lower_TgC",
        "annual_flux_90_upper_TgC",
        "may_july_flux_TgC",
        "may_july_flux_90_lower_TgC",
        "may_july_flux_90_upper_TgC",
        "may_july_flux_fraction_of_annual",
        "annual_confidence_tier",
        "may_july_confidence_tier",
        "cohort_core_2003_2024",
        "cohort_high_confidence_only",
        "cohort_sensitivity_only",
        "cohort_exclude_from_trend",
        "caveat_reason",
    ]
    return out[keep].sort_values(["river", "year"]).reset_index(drop=True)


def _analysis_sets(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "core_2003_2024": rows[rows["cohort_core_2003_2024"]].copy(),
        "full_2000_2025_sensitivity": rows.copy(),
        "high_confidence_only_sensitivity": rows[rows["cohort_high_confidence_only"] & rows["may_july_confidence_tier"].eq("high")].copy(),
        "core_excluding_yenisey_sensitivity": rows[rows["cohort_core_2003_2024"] & ~rows["river"].astype(str).eq("Yenisey")].copy(),
    }


def _trend_record(analysis_cohort: str, river: str, metric: str, result: TrendResult) -> dict[str, Any]:
    unit = "TgC_per_year" if metric == "may_july_flux_TgC" else "fraction_per_year"
    return {
        "analysis_cohort": analysis_cohort,
        "river": river,
        "metric": metric,
        "metric_unit": unit,
        "n_years": result.n_years,
        "year_min": result.year_min,
        "year_max": result.year_max,
        "value_mean": result.value_mean,
        "value_median": result.value_median,
        "slope_ols_per_year": result.slope_ols,
        "intercept_ols": result.intercept_ols,
        "slope_ols_p_value": result.slope_ols_p_value,
        "slope_ols_r2": result.slope_ols_r2,
        "slope_ols_standard_error": result.slope_ols_stderr,
        "slope_theilsen_per_year": result.slope_theilsen,
        "theilsen_ci_lower": result.theilsen_ci_lower,
        "theilsen_ci_upper": result.theilsen_ci_upper,
        "kendall_tau": result.kendall_tau,
        "kendall_p_value": result.kendall_p_value,
        "slope_percent_per_year": result.slope_percent_per_year,
        "total_change_percent_over_period": result.total_change_percent_over_period,
        "trend_direction": result.trend_direction,
        "trend_strength": result.trend_strength,
        "detectable_trend": result.significant_at_0_05,
        "trend_language": "detectable trend" if result.significant_at_0_05 else "no detectable trend",
    }


def _trend_table(sets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for analysis_cohort, frame in sets.items():
        for river, group in frame.groupby("river", dropna=False):
            if len(group) < 3:
                continue
            for metric in ["may_july_flux_TgC", "may_july_flux_fraction_of_annual"]:
                rows.append(_trend_record(analysis_cohort, str(river), metric, trend_for_series(group, metric)))
    return pd.DataFrame(rows)


def _fraction_summary(rows: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    for river, group in rows.groupby("river", dropna=False):
        core = group[group["cohort_core_2003_2024"]].copy()
        caveats = []
        if group["cohort_exclude_from_trend"].any():
            caveats.append(f"excluded_years:{int(group['cohort_exclude_from_trend'].sum())}")
        if group["cohort_sensitivity_only"].any():
            caveats.append(f"sensitivity_only_years:{int(group['cohort_sensitivity_only'].sum())}")
        if group["caveat_reason"].astype(str).str.contains("low_confidence", case=False, na=False).any():
            caveats.append("low_confidence_may_july_or_annual_years")
        summary_rows.append(
            {
                "river": river,
                "n_core_years": int(len(core)),
                "mean_may_july_fraction_core": float(core["may_july_flux_fraction_of_annual"].mean()) if not core.empty else np.nan,
                "median_may_july_fraction_core": float(core["may_july_flux_fraction_of_annual"].median()) if not core.empty else np.nan,
                "min_may_july_fraction_core": float(core["may_july_flux_fraction_of_annual"].min()) if not core.empty else np.nan,
                "max_may_july_fraction_core": float(core["may_july_flux_fraction_of_annual"].max()) if not core.empty else np.nan,
                "mean_may_july_flux_TgC": float(core["may_july_flux_TgC"].mean()) if not core.empty else np.nan,
                "mean_annual_flux_TgC": float(core["annual_flux_TgC"].mean()) if not core.empty else np.nan,
                "caveat_summary": ";".join(caveats) if caveats else "core_years_available_with_standard_caveats",
            }
        )
    return pd.DataFrame(summary_rows)


def _coerce_bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _annual_vs_may_july_comparison(annual_trends: pd.DataFrame, may_july_trends: pd.DataFrame) -> pd.DataFrame:
    annual_core = annual_trends[annual_trends["analysis_cohort"].eq("core_2003_2024")].set_index("river")
    mj_core = may_july_trends[may_july_trends["analysis_cohort"].eq("core_2003_2024")]
    flux = mj_core[mj_core["metric"].eq("may_july_flux_TgC")].set_index("river")
    fraction = mj_core[mj_core["metric"].eq("may_july_flux_fraction_of_annual")].set_index("river")
    rivers = sorted(set(annual_core.index).union(flux.index).union(fraction.index))
    rows: list[dict[str, Any]] = []
    for river in rivers:
        annual = annual_core.loc[river] if river in annual_core.index else pd.Series(dtype=object)
        mj_flux = flux.loc[river] if river in flux.index else pd.Series(dtype=object)
        mj_fraction = fraction.loc[river] if river in fraction.index else pd.Series(dtype=object)
        annual_direction = str(annual.get("trend_direction", ""))
        annual_detectable = _coerce_bool_value(annual.get("significant_at_0_05", False))
        flux_direction = str(mj_flux.get("trend_direction", ""))
        flux_detectable = _coerce_bool_value(mj_flux.get("detectable_trend", False))
        fraction_direction = str(mj_fraction.get("trend_direction", ""))
        fraction_detectable = _coerce_bool_value(mj_fraction.get("detectable_trend", False))
        if not annual_detectable:
            explain = "not_applicable"
            interpretation = "No detectable annual flux trend in the core cohort, so May-July does not need to explain an annual signal."
        elif flux_detectable and flux_direction == annual_direction and fraction_detectable and fraction_direction == annual_direction:
            explain = "yes"
            interpretation = "May-July flux and its annual share both move with the detectable annual signal."
        elif flux_detectable and flux_direction == annual_direction:
            explain = "partial"
            interpretation = "May-July flux moves with the annual signal, but the May-July annual share has no detectable matching trend."
        elif not flux_detectable:
            explain = "no"
            interpretation = "The annual flux signal is not mirrored by a detectable May-July flux trend."
        else:
            explain = "uncertain"
            interpretation = "May-July and annual trend diagnostics are directionally inconsistent or underpowered."
        if river == "Yukon" and annual_detectable and annual_direction == "increasing" and flux_detectable and flux_direction == "increasing" and not fraction_detectable:
            interpretation = "Yukon annual flux increases and provisional May-July flux also increases, but the May-July fraction is flat or uncertain; the signal may reflect broader whole-year or discharge-volume changes rather than a higher May-July share."
        rows.append(
            {
                "river": river,
                "annual_flux_trend_direction": annual_direction,
                "annual_flux_detectable_trend": annual_detectable,
                "may_july_flux_trend_direction": flux_direction,
                "may_july_flux_detectable_trend": flux_detectable,
                "may_july_fraction_trend_direction": fraction_direction,
                "may_july_fraction_detectable_trend": fraction_detectable,
                "does_may_july_explain_annual_signal": explain,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def _caveat_summary(rows: pd.DataFrame) -> pd.DataFrame:
    caveat_rows: list[dict[str, Any]] = []
    for river, group in rows.groupby("river", dropna=False):
        low_mj = group[group["may_july_confidence_tier"].astype(str).eq("low")]
        annual_low = group[group["annual_confidence_tier"].astype(str).eq("low")]
        excluded = group[group["cohort_exclude_from_trend"]]
        caveat_rows.append(
            {
                "river": river,
                "n_low_confidence_may_july_years": int(len(low_mj)),
                "low_confidence_may_july_years": _year_list(low_mj),
                "n_annual_low_confidence_overlap_years": int(len(annual_low)),
                "annual_low_confidence_overlap_years": _year_list(annual_low),
                "n_excluded_years": int(len(excluded)),
                "excluded_years": _year_list(excluded),
                "caveats_by_river": ";".join(sorted(set(";".join(group["caveat_reason"].dropna().astype(str)).split(";")))),
            }
        )
    caveat_rows.append(
        {
            "river": "all",
            "n_low_confidence_may_july_years": int(rows["may_july_confidence_tier"].astype(str).eq("low").sum()),
            "low_confidence_may_july_years": "see_river_rows",
            "n_annual_low_confidence_overlap_years": int(rows["annual_confidence_tier"].astype(str).eq("low").sum()),
            "annual_low_confidence_overlap_years": "see_river_rows",
            "n_excluded_years": int(rows["cohort_exclude_from_trend"].sum()),
            "excluded_years": "see_river_rows",
            "caveats_by_river": "May-July is provisional; not a hydrologic snowmelt-window refinement; discharge uncertainty not propagated.",
        }
    )
    return pd.DataFrame(caveat_rows)


def _year_list(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    return ";".join(str(int(year)) for year in sorted(pd.to_numeric(frame["year"], errors="coerce").dropna().unique()))


def _make_figures(rows: pd.DataFrame, trends: pd.DataFrame, comparison: pd.DataFrame) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    MAY_JULY_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def save(fig, name: str) -> None:
        fig.tight_layout()
        destination = MAY_JULY_FIGURE_DIR / name
        fig.savefig(destination, dpi=120)
        plt.close(fig)
        paths.append(destination)

    core = rows[rows["cohort_core_2003_2024"]].copy()
    if not core.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        for river, group in core.groupby("river"):
            ax.plot(group["year"], group["may_july_flux_TgC"], marker="o", linewidth=1.0, label=str(river))
        ax.set_xlabel("Year")
        ax.set_ylabel("Provisional May-July DOC flux (Tg C)")
        ax.set_title("Provisional May-July DOC flux by river")
        ax.legend(fontsize="x-small")
        save(fig, "may_july_flux_timeseries_by_river.png")

        fig, ax = plt.subplots(figsize=(9, 5))
        for river, group in core.groupby("river"):
            ax.plot(group["year"], group["may_july_flux_fraction_of_annual"], marker="o", linewidth=1.0, label=str(river))
        ax.set_xlabel("Year")
        ax.set_ylabel("Provisional May-July fraction of annual flux")
        ax.set_title("Provisional May-July fraction by river")
        ax.legend(fontsize="x-small")
        save(fig, "may_july_fraction_timeseries_by_river.png")

        fig, ax = plt.subplots(figsize=(7, 5))
        for river, group in core.groupby("river"):
            ax.scatter(group["annual_flux_TgC"], group["may_july_flux_TgC"], s=22, label=str(river))
        ax.set_xlabel("Annual DOC flux (Tg C)")
        ax.set_ylabel("Provisional May-July DOC flux (Tg C)")
        ax.set_title("Provisional May-July vs annual DOC flux")
        ax.legend(fontsize="x-small")
        save(fig, "may_july_vs_annual_flux_scatter.png")

        fig, ax = plt.subplots(figsize=(8, 4.8))
        groups = list(core.groupby("river"))
        data = [group["may_july_flux_fraction_of_annual"].dropna().to_numpy() for _, group in groups]
        labels = [str(river) for river, _ in groups]
        if data:
            try:
                ax.boxplot(data, tick_labels=labels, showfliers=False)
            except TypeError:
                ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_ylabel("Provisional May-July fraction of annual flux")
        ax.set_title("Provisional May-July fraction by river")
        save(fig, "may_july_fraction_boxplot_by_river.png")

    yukon = core[core["river"].astype(str).eq("Yukon")]
    if not yukon.empty:
        fig, ax1 = plt.subplots(figsize=(8, 4.8))
        ax1.plot(yukon["year"], yukon["annual_flux_TgC"], marker="o", label="annual flux")
        ax1.plot(yukon["year"], yukon["may_july_flux_TgC"], marker="o", label="provisional May-July flux")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("DOC flux (Tg C)")
        ax2 = ax1.twinx()
        ax2.plot(yukon["year"], yukon["may_july_flux_fraction_of_annual"], marker="s", linestyle="--", label="May-July fraction")
        ax2.set_ylabel("May-July fraction")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize="x-small")
        ax1.set_title("Yukon annual vs provisional May-July signal")
        save(fig, "yukon_annual_vs_may_july_signal.png")
    return paths


def run_may_july_flux_interpretation() -> dict[str, Any]:
    _ensure_dirs()
    before_hashes = _input_hashes()
    inputs = _load_inputs()
    rows = _prepare_interpretation_rows(inputs)
    sets = _analysis_sets(rows)
    fraction_summary = _fraction_summary(rows)
    trends = _trend_table(sets)
    comparison = _annual_vs_may_july_comparison(inputs["annual_trends"], trends)
    caveats = _caveat_summary(rows)
    figures = _make_figures(rows, trends, comparison)
    table_paths = [
        _write_csv(rows, MAY_JULY_TABLE_DIR / "may_july_flux_interpretation_by_river_year.csv"),
        _write_csv(fraction_summary, MAY_JULY_TABLE_DIR / "may_july_fraction_summary_by_river.csv"),
        _write_csv(trends, MAY_JULY_TABLE_DIR / "may_july_flux_trends_by_river.csv"),
        _write_csv(comparison, MAY_JULY_TABLE_DIR / "may_july_vs_annual_trend_comparison.csv"),
        _write_csv(caveats, MAY_JULY_TABLE_DIR / "may_july_caveat_summary.csv"),
    ]
    report_path = write_may_july_flux_report()
    _verify_inputs_unchanged(before_hashes)
    return {
        "tables": table_paths,
        "figures": figures,
        "report": report_path,
        "rows": rows,
        "trends": trends,
        "comparison": comparison,
        "caveats": caveats,
    }


__all__ = [
    "MAY_JULY_TABLE_DIR",
    "MAY_JULY_REPORT_DIR",
    "MAY_JULY_FIGURE_DIR",
    "MAY_JULY_REPORT_PATH",
    "REQUIRED_MAY_JULY_INPUTS",
    "run_may_july_flux_interpretation",
    "write_may_july_flux_report",
]
