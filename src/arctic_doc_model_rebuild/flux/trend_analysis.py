from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..gold_contract import sha256_file
from ..paths import REPORT_DIR, TABLE_DIR, path
from .trend_reports import ANNUAL_TREND_REPORT_PATH, ANNUAL_TREND_REPORT_DIR, ANNUAL_TREND_TABLE_DIR, write_annual_flux_trend_report
from .trend_statistics import trend_for_series


ANNUAL_TREND_FIGURE_DIR = path("outputs", "figures", "annual_flux_trends")
FLUX_INTERPRETATION_TABLE_DIR = TABLE_DIR / "flux_interpretation"
DOC_FLUX_TABLE_DIR = TABLE_DIR / "doc_flux"
FLUX_INTERPRETATION_REPORT_PATH = REPORT_DIR / "flux_interpretation" / "flux_cohort_report.md"
DOC_FLUX_REPORT_PATH = REPORT_DIR / "doc_flux" / "doc_flux_report.md"

REQUIRED_TREND_INPUTS = {
    "annual_flux_analysis_cohorts": FLUX_INTERPRETATION_TABLE_DIR / "annual_flux_analysis_cohorts.csv",
    "flux_cohort_summary_by_river": FLUX_INTERPRETATION_TABLE_DIR / "flux_cohort_summary_by_river.csv",
    "flux_confidence_diagnostics": FLUX_INTERPRETATION_TABLE_DIR / "flux_confidence_diagnostics.csv",
    "annual_doc_flux_summary": DOC_FLUX_TABLE_DIR / "annual_doc_flux_summary.csv",
    "river_period_flux_summary": DOC_FLUX_TABLE_DIR / "river_period_flux_summary.csv",
    "flux_cohort_report": FLUX_INTERPRETATION_REPORT_PATH,
    "doc_flux_report": DOC_FLUX_REPORT_PATH,
}


def _ensure_dirs() -> None:
    for directory in [ANNUAL_TREND_TABLE_DIR, ANNUAL_TREND_REPORT_DIR, ANNUAL_TREND_FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def _write_csv(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8")
    return destination


def _read_csv(destination: Path) -> pd.DataFrame:
    if not destination.exists():
        raise FileNotFoundError(f"Required annual flux trend input is missing: {destination}")
    return pd.read_csv(destination, low_memory=False)


def _read_text(destination: Path) -> str:
    if not destination.exists():
        raise FileNotFoundError(f"Required annual flux trend input is missing: {destination}")
    return destination.read_text(encoding="utf-8")


def _input_hashes() -> dict[Path, str]:
    return {destination: sha256_file(destination) for destination in REQUIRED_TREND_INPUTS.values()}


def _verify_inputs_unchanged(before: dict[Path, str]) -> None:
    after = _input_hashes()
    changed = [str(destination) for destination, before_hash in before.items() if after.get(destination) != before_hash]
    if changed:
        raise RuntimeError(f"Annual flux trend inputs changed during analysis: {changed}")


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _load_inputs() -> dict[str, Any]:
    return {
        "cohorts": _read_csv(REQUIRED_TREND_INPUTS["annual_flux_analysis_cohorts"]),
        "cohort_summary": _read_csv(REQUIRED_TREND_INPUTS["flux_cohort_summary_by_river"]),
        "confidence_diagnostics": _read_csv(REQUIRED_TREND_INPUTS["flux_confidence_diagnostics"]),
        "annual_flux": _read_csv(REQUIRED_TREND_INPUTS["annual_doc_flux_summary"]),
        "period_summary": _read_csv(REQUIRED_TREND_INPUTS["river_period_flux_summary"]),
        "cohort_report_text": _read_text(REQUIRED_TREND_INPUTS["flux_cohort_report"]),
        "doc_flux_report_text": _read_text(REQUIRED_TREND_INPUTS["doc_flux_report"]),
    }


def _prepare_cohorts(cohorts: pd.DataFrame) -> pd.DataFrame:
    out = cohorts.copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    for column in [
        "annual_flux_TgC",
        "annual_flux_90_lower_TgC",
        "annual_flux_90_upper_TgC",
        "coverage_rate",
        "fraction_flux_from_low_confidence_days",
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in [
        "cohort_core_2003_2024",
        "cohort_full_2000_2025",
        "cohort_high_confidence_only",
        "cohort_sensitivity_only",
        "cohort_exclude_from_trend",
    ]:
        out[column] = _as_bool(out[column])
    return out


def _analysis_sets(cohorts: pd.DataFrame) -> dict[str, pd.DataFrame]:
    prepared = _prepare_cohorts(cohorts)
    return {
        "core_2003_2024": prepared[prepared["cohort_core_2003_2024"]].copy(),
        "full_2000_2025_sensitivity": prepared[prepared["cohort_full_2000_2025"]].copy(),
        "high_confidence_only_sensitivity": prepared[prepared["cohort_high_confidence_only"]].copy(),
        "core_excluding_yenisey_sensitivity": prepared[prepared["cohort_core_2003_2024"] & ~prepared["river"].astype(str).eq("Yenisey")].copy(),
    }


def _confidence_caveat(analysis_cohort: str, group: pd.DataFrame) -> str:
    caveats = ["DOC_concentration_uncertainty_only", "discharge_uncertainty_not_propagated"]
    if "sensitivity" in analysis_cohort:
        caveats.append("sensitivity_not_primary")
    if group["cohort_exclude_from_trend"].any():
        caveats.append("contains_rows_excluded_from_core")
    if group["fraction_flux_from_low_confidence_days"].ge(0.25).any():
        caveats.append("contains_high_low_confidence_flux_fraction_years")
    if group["coverage_rate"].lt(0.95).any():
        caveats.append("contains_coverage_lt_0_95_years")
    return ";".join(caveats)


def _trend_row(analysis_cohort: str, river: str, group: pd.DataFrame, value_column: str = "annual_flux_TgC") -> dict[str, Any]:
    result = trend_for_series(group, value_column)
    return {
        "analysis_cohort": analysis_cohort,
        "river": river,
        "n_years": result.n_years,
        "year_min": result.year_min,
        "year_max": result.year_max,
        "annual_flux_mean_TgC": result.value_mean,
        "annual_flux_median_TgC": result.value_median,
        "slope_ols_TgC_per_year": result.slope_ols,
        "intercept_ols": result.intercept_ols,
        "slope_ols_p_value": result.slope_ols_p_value,
        "slope_ols_r2": result.slope_ols_r2,
        "slope_ols_standard_error": result.slope_ols_stderr,
        "slope_theilsen_TgC_per_year": result.slope_theilsen,
        "intercept_theilsen": result.intercept_theilsen,
        "theilsen_ci_lower": result.theilsen_ci_lower,
        "theilsen_ci_upper": result.theilsen_ci_upper,
        "kendall_tau": result.kendall_tau,
        "kendall_p_value": result.kendall_p_value,
        "slope_percent_per_year": result.slope_percent_per_year,
        "total_change_percent_over_period": result.total_change_percent_over_period,
        "trend_direction": result.trend_direction,
        "trend_strength": result.trend_strength,
        "significant_at_0_05": result.significant_at_0_05,
        "confidence_caveat": _confidence_caveat(analysis_cohort, group),
    }


def _by_river_trends(sets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for analysis_cohort, frame in sets.items():
        for river, group in frame.groupby("river", dropna=False):
            if len(group) >= 3:
                rows.append(_trend_row(analysis_cohort, str(river), group))
    return pd.DataFrame(rows)


def _aggregate_series(frame: pd.DataFrame, common_only: bool) -> tuple[pd.DataFrame, str, str]:
    if frame.empty:
        return pd.DataFrame(), "", "empty_cohort"
    by_year_rivers = frame.groupby("year")["river"].apply(lambda values: set(values.astype(str))).to_dict()
    all_rivers = sorted(set().union(*by_year_rivers.values()))
    if common_only:
        common = sorted(set.intersection(*by_year_rivers.values())) if by_year_rivers else []
        if not common:
            return pd.DataFrame(), "", "no_common_river_set"
        work = frame[frame["river"].astype(str).isin(common)].copy()
        rivers_included = ";".join(common)
        caveat = "common_river_set_only;DOC_concentration_uncertainty_only;discharge_uncertainty_not_propagated"
    else:
        work = frame.copy()
        rivers_included = ";".join(all_rivers)
        varying = len({tuple(sorted(v)) for v in by_year_rivers.values()}) > 1
        caveat = "aggregate_all_available_rivers;varying_river_set" if varying else "aggregate_all_available_rivers"
        caveat += ";DOC_concentration_uncertainty_only;discharge_uncertainty_not_propagated"
    aggregate = (
        work.groupby("year", dropna=False)
        .agg(annual_flux_sum_TgC=("annual_flux_TgC", "sum"), rivers_in_year=("river", lambda values: ";".join(sorted(values.astype(str).unique()))))
        .reset_index()
        .rename(columns={"annual_flux_sum_TgC": "annual_flux_TgC"})
    )
    return aggregate, rivers_included, caveat


def _aggregate_trends(sets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for analysis_cohort, frame in sets.items():
        for aggregate_type, common_only in [
            ("aggregate_all_available_rivers", False),
            ("aggregate_common_river_set_only", True),
        ]:
            aggregate, rivers_included, caveat = _aggregate_series(frame, common_only)
            if len(aggregate) < 3:
                continue
            result = trend_for_series(aggregate, "annual_flux_TgC")
            rows.append(
                {
                    "analysis_cohort": analysis_cohort,
                    "aggregate_type": aggregate_type,
                    "rivers_included": rivers_included,
                    "n_years": result.n_years,
                    "year_min": result.year_min,
                    "year_max": result.year_max,
                    "annual_flux_sum_mean_TgC": result.value_mean,
                    "slope_ols_TgC_per_year": result.slope_ols,
                    "slope_ols_p_value": result.slope_ols_p_value,
                    "slope_ols_r2": result.slope_ols_r2,
                    "slope_theilsen_TgC_per_year": result.slope_theilsen,
                    "kendall_tau": result.kendall_tau,
                    "kendall_p_value": result.kendall_p_value,
                    "trend_direction": result.trend_direction,
                    "confidence_caveat": caveat,
                }
            )
    return pd.DataFrame(rows)


def _cohort_sensitivity(by_river: pd.DataFrame) -> pd.DataFrame:
    core = by_river[by_river["analysis_cohort"].eq("core_2003_2024")].set_index("river")
    rows: list[dict[str, Any]] = []
    for row in by_river.itertuples(index=False):
        core_direction = core.loc[row.river, "trend_direction"] if row.river in core.index else ""
        core_sig = bool(core.loc[row.river, "significant_at_0_05"]) if row.river in core.index else False
        rows.append(
            {
                "analysis_cohort": row.analysis_cohort,
                "river": row.river,
                "n_years": row.n_years,
                "slope_ols_TgC_per_year": row.slope_ols_TgC_per_year,
                "slope_ols_p_value": row.slope_ols_p_value,
                "slope_theilsen_TgC_per_year": row.slope_theilsen_TgC_per_year,
                "kendall_tau": row.kendall_tau,
                "trend_direction": row.trend_direction,
                "significant_at_0_05": row.significant_at_0_05,
                "primary_or_sensitivity": "primary" if row.analysis_cohort == "core_2003_2024" else "sensitivity",
                "conclusion_changed_vs_core": bool(
                    row.analysis_cohort != "core_2003_2024"
                    and core_direction
                    and (row.trend_direction != core_direction or bool(row.significant_at_0_05) != core_sig)
                ),
                "confidence_caveat": row.confidence_caveat,
            }
        )
    return pd.DataFrame(rows)


def _uncertainty_sensitivity(sets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for analysis_cohort, frame in sets.items():
        for river, group in frame.groupby("river", dropna=False):
            if len(group) < 3:
                continue
            central = trend_for_series(group, "annual_flux_TgC")
            lower = trend_for_series(group, "annual_flux_90_lower_TgC")
            upper = trend_for_series(group, "annual_flux_90_upper_TgC")
            signs = [np.sign(central.slope_ols), np.sign(lower.slope_ols), np.sign(upper.slope_ols)]
            robust = bool(all(sign > 0 for sign in signs) or all(sign < 0 for sign in signs))
            rows.append(
                {
                    "analysis_cohort": analysis_cohort,
                    "river": str(river),
                    "n_years": central.n_years,
                    "slope_central": central.slope_ols,
                    "slope_lower90": lower.slope_ols,
                    "slope_upper90": upper.slope_ols,
                    "trend_sign_robust_to_doc_uncertainty": robust,
                    "central_p_value": central.slope_ols_p_value,
                    "lower90_p_value": lower.slope_ols_p_value,
                    "upper90_p_value": upper.slope_ols_p_value,
                    "uncertainty_scope": "DOC_concentration_empirical_residual_interval_only",
                    "discharge_uncertainty_included": False,
                }
            )
    return pd.DataFrame(rows)


def _input_audit(inputs: dict[str, Any], sets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, destination in REQUIRED_TREND_INPUTS.items():
        rows.append(
            {
                "input_name": name,
                "file_path": str(destination),
                "sha256": sha256_file(destination),
                "row_count": len(_read_csv(destination)) if destination.suffix.lower() == ".csv" else np.nan,
                "used_for": "annual_flux_trend_metadata_or_cohort",
                "recomputed_flux": False,
                "regenerated_doc_prediction": False,
                "trained_model": False,
            }
        )
    for name, frame in sets.items():
        rows.append(
            {
                "input_name": f"analysis_set_{name}",
                "file_path": "derived_from_annual_flux_analysis_cohorts.csv",
                "sha256": "",
                "row_count": len(frame),
                "used_for": "trend_analysis_cohort",
                "recomputed_flux": False,
                "regenerated_doc_prediction": False,
                "trained_model": False,
            }
        )
    return pd.DataFrame(rows)


def _caveat_summary(inputs: dict[str, Any], cohorts: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "caveat_item": "Yenisey_high_low_confidence_flux_fraction",
            "river": "Yenisey",
            "status": "carry_forward",
            "evidence": "Yenisey has 11 years with fraction_flux_from_low_confidence_days >= 0.25.",
            "trend_implication": "Emphasize core cohort and inspect sensitivity; avoid overclaiming full-period Yenisey trend.",
        },
        {
            "caveat_item": "Yukon_zero_or_near_zero_2000",
            "river": "Yukon",
            "status": "carry_forward",
            "evidence": "Yukon 2000 annual flux is zero or near-zero in cohort diagnostics.",
            "trend_implication": "Do not let 2000 drive full-period sensitivity; primary core excludes 2000.",
        },
        {
            "caveat_item": "Kolyma_excluded_and_low_confidence_years",
            "river": "Kolyma",
            "status": "carry_forward",
            "evidence": "Kolyma has excluded years and high low-confidence-fraction years.",
            "trend_implication": "Use core cohort as primary and report excluded years.",
        },
        {
            "caveat_item": "Ob_excluded_and_low_confidence_year",
            "river": "Ob",
            "status": "carry_forward",
            "evidence": "Ob has one excluded year and one high low-confidence-fraction year.",
            "trend_implication": "Core result is primary; full-period sensitivity is caveated.",
        },
        {
            "caveat_item": "Mackenzie_2025_coverage",
            "river": "Mackenzie",
            "status": "carry_forward",
            "evidence": "Mackenzie 2025 is sensitivity due coverage <0.95.",
            "trend_implication": "Primary core excludes 2025.",
        },
        {
            "caveat_item": "May_July_not_analyzed",
            "river": "all",
            "status": "not_in_scope",
            "evidence": "Annual May-July and snowmelt windows are not analyzed in this phase.",
            "trend_implication": "Do not interpret annual trends as snowmelt-window trends.",
        },
        {
            "caveat_item": "uncertainty_scope",
            "river": "all",
            "status": "carry_forward",
            "evidence": "Annual flux intervals carry DOC concentration uncertainty only.",
            "trend_implication": "Discharge uncertainty is not propagated in trend sensitivity.",
        },
    ]
    return pd.DataFrame(rows)


def _make_figures(sets: dict[str, pd.DataFrame], by_river: pd.DataFrame, aggregate: pd.DataFrame, uncertainty: pd.DataFrame) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    ANNUAL_TREND_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def save(fig, name: str) -> None:
        fig.tight_layout()
        destination = ANNUAL_TREND_FIGURE_DIR / name
        fig.savefig(destination, dpi=120)
        plt.close(fig)
        paths.append(destination)

    core = sets["core_2003_2024"]
    if not core.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        for river, group in core.groupby("river"):
            ax.plot(group["year"], group["annual_flux_TgC"], marker="o", linewidth=1.0, label=str(river))
        ax.set_xlabel("Year")
        ax.set_ylabel("Annual DOC flux (Tg C)")
        ax.set_title("Core annual DOC flux time series by river")
        ax.legend(fontsize="x-small")
        save(fig, "annual_flux_timeseries_core_by_river.png")

        fig, ax = plt.subplots(figsize=(9, 5))
        for river, group in core.groupby("river"):
            group = group.sort_values("year")
            ax.scatter(group["year"], group["annual_flux_TgC"], s=18, label=str(river))
            result = trend_for_series(group, "annual_flux_TgC")
            x = np.array([group["year"].min(), group["year"].max()], dtype=float)
            y = result.intercept_ols + result.slope_ols * x
            ax.plot(x, y, linewidth=1.0)
        ax.set_xlabel("Year")
        ax.set_ylabel("Annual DOC flux (Tg C)")
        ax.set_title("Core annual DOC flux OLS trend lines")
        ax.legend(fontsize="x-small")
        save(fig, "annual_flux_trend_lines_core_by_river.png")

    core_slopes = by_river[by_river["analysis_cohort"].eq("core_2003_2024")].sort_values("slope_ols_TgC_per_year")
    if not core_slopes.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.barh(core_slopes["river"], core_slopes["slope_ols_TgC_per_year"])
        ax.axvline(0, linestyle="--")
        ax.set_xlabel("OLS slope (Tg C/year)")
        ax.set_title("Core annual DOC flux slope by river")
        save(fig, "annual_flux_slope_by_river_core.png")

    if not core.empty:
        agg, rivers, _ = _aggregate_series(core, common_only=True)
        if not agg.empty:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            ax.plot(agg["year"], agg["annual_flux_TgC"], marker="o")
            ax.set_xlabel("Year")
            ax.set_ylabel("Aggregate annual DOC flux (Tg C)")
            ax.set_title(f"Core aggregate flux time series ({rivers})")
            save(fig, "aggregate_flux_timeseries_core.png")

    sensitivity = by_river.pivot_table(index="river", columns="analysis_cohort", values="slope_ols_TgC_per_year", aggfunc="first")
    if not sensitivity.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        sensitivity.plot(kind="bar", ax=ax)
        ax.axhline(0, linestyle="--")
        ax.set_ylabel("OLS slope (Tg C/year)")
        ax.set_title("Trend sensitivity by cohort")
        save(fig, "trend_sensitivity_by_cohort.png")

    core_unc = uncertainty[uncertainty["analysis_cohort"].eq("core_2003_2024")].copy()
    if not core_unc.empty:
        x = np.arange(len(core_unc))
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x, core_unc["slope_central"], marker="o", label="central")
        ax.plot(x, core_unc["slope_lower90"], marker="o", label="lower90")
        ax.plot(x, core_unc["slope_upper90"], marker="o", label="upper90")
        ax.axhline(0, linestyle="--")
        ax.set_xticks(x, core_unc["river"], rotation=30)
        ax.set_ylabel("OLS slope (Tg C/year)")
        ax.set_title("DOC uncertainty sensitivity for core slopes")
        ax.legend()
        save(fig, "trend_uncertainty_sensitivity.png")
    return paths


def run_annual_flux_trends() -> dict[str, Any]:
    _ensure_dirs()
    before_hashes = _input_hashes()
    inputs = _load_inputs()
    sets = _analysis_sets(inputs["cohorts"])
    input_audit = _input_audit(inputs, sets)
    by_river = _by_river_trends(sets)
    aggregate = _aggregate_trends(sets)
    sensitivity = _cohort_sensitivity(by_river)
    uncertainty = _uncertainty_sensitivity(sets)
    caveats = _caveat_summary(inputs, _prepare_cohorts(inputs["cohorts"]))
    figures = _make_figures(sets, by_river, aggregate, uncertainty)
    table_paths = [
        _write_csv(input_audit, ANNUAL_TREND_TABLE_DIR / "annual_flux_trend_input_audit.csv"),
        _write_csv(by_river, ANNUAL_TREND_TABLE_DIR / "annual_flux_trends_by_river.csv"),
        _write_csv(aggregate, ANNUAL_TREND_TABLE_DIR / "annual_flux_trends_aggregate.csv"),
        _write_csv(sensitivity, ANNUAL_TREND_TABLE_DIR / "annual_flux_trend_sensitivity_by_cohort.csv"),
        _write_csv(uncertainty, ANNUAL_TREND_TABLE_DIR / "annual_flux_trend_uncertainty_sensitivity.csv"),
        _write_csv(caveats, ANNUAL_TREND_TABLE_DIR / "annual_flux_trend_caveat_summary.csv"),
    ]
    report_path = write_annual_flux_trend_report()
    _verify_inputs_unchanged(before_hashes)
    return {
        "tables": table_paths,
        "figures": figures,
        "report": report_path,
        "by_river": by_river,
        "aggregate": aggregate,
        "sensitivity": sensitivity,
        "uncertainty": uncertainty,
    }


__all__ = [
    "ANNUAL_TREND_TABLE_DIR",
    "ANNUAL_TREND_REPORT_DIR",
    "ANNUAL_TREND_FIGURE_DIR",
    "ANNUAL_TREND_REPORT_PATH",
    "REQUIRED_TREND_INPUTS",
    "run_annual_flux_trends",
    "write_annual_flux_trend_report",
]
