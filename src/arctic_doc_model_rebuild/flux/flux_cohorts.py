from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..gold_contract import sha256_file
from ..paths import REPORT_DIR, TABLE_DIR, path
from ..reports import _md_table, utc_now


FLUX_INPUT_TABLE_DIR = TABLE_DIR / "doc_flux"
FLUX_INTERPRETATION_TABLE_DIR = TABLE_DIR / "flux_interpretation"
FLUX_INTERPRETATION_REPORT_DIR = REPORT_DIR / "flux_interpretation"
FLUX_INTERPRETATION_FIGURE_DIR = path("outputs", "figures", "flux_interpretation")
FLUX_COHORT_REPORT_PATH = FLUX_INTERPRETATION_REPORT_DIR / "flux_cohort_report.md"

REQUIRED_FLUX_INPUTS = {
    "daily_doc_flux": FLUX_INPUT_TABLE_DIR / "daily_doc_flux.csv",
    "annual": FLUX_INPUT_TABLE_DIR / "annual_doc_flux_summary.csv",
    "may_july": FLUX_INPUT_TABLE_DIR / "provisional_may_july_flux_summary.csv",
    "period": FLUX_INPUT_TABLE_DIR / "river_period_flux_summary.csv",
    "qc": FLUX_INPUT_TABLE_DIR / "doc_flux_qc_summary.csv",
    "confidence": FLUX_INPUT_TABLE_DIR / "doc_flux_confidence_tier_summary.csv",
    "range_flags": FLUX_INPUT_TABLE_DIR / "doc_flux_range_flags.csv",
    "report": REPORT_DIR / "doc_flux" / "doc_flux_report.md",
}


def _ensure_dirs() -> None:
    for directory in [FLUX_INTERPRETATION_TABLE_DIR, FLUX_INTERPRETATION_REPORT_DIR, FLUX_INTERPRETATION_FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def _write_csv(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8")
    return destination


def _read_csv(destination: Path) -> pd.DataFrame:
    if not destination.exists():
        raise FileNotFoundError(f"Required flux interpretation input is missing: {destination}")
    return pd.read_csv(destination, low_memory=False)


def _read_text(destination: Path) -> str:
    if not destination.exists():
        raise FileNotFoundError(f"Required flux interpretation input is missing: {destination}")
    return destination.read_text(encoding="utf-8")


def _input_hashes() -> dict[Path, str]:
    hashes: dict[Path, str] = {}
    for destination in REQUIRED_FLUX_INPUTS.values():
        if destination.suffix.lower() in {".csv", ".md"}:
            hashes[destination] = sha256_file(destination)
    return hashes


def _verify_inputs_unchanged(before: dict[Path, str]) -> None:
    after = _input_hashes()
    changed = [str(destination) for destination, before_hash in before.items() if after.get(destination) != before_hash]
    if changed:
        raise RuntimeError(f"Flux interpretation inputs changed during cohort selection: {changed}")


def _load_inputs() -> dict[str, Any]:
    return {
        "daily_doc_flux": _read_csv(REQUIRED_FLUX_INPUTS["daily_doc_flux"]),
        "annual": _read_csv(REQUIRED_FLUX_INPUTS["annual"]),
        "may_july": _read_csv(REQUIRED_FLUX_INPUTS["may_july"]),
        "period": _read_csv(REQUIRED_FLUX_INPUTS["period"]),
        "qc": _read_csv(REQUIRED_FLUX_INPUTS["qc"]),
        "confidence": _read_csv(REQUIRED_FLUX_INPUTS["confidence"]),
        "range_flags": _read_csv(REQUIRED_FLUX_INPUTS["range_flags"]),
        "report_text": _read_text(REQUIRED_FLUX_INPUTS["report"]),
    }


def _critical_missing(row: pd.Series) -> bool:
    required = [
        "annual_flux_TgC",
        "annual_flux_90_lower_TgC",
        "annual_flux_90_upper_TgC",
        "coverage_rate",
        "fraction_flux_from_low_confidence_days",
        "annual_confidence_tier",
    ]
    return any(pd.isna(row.get(column)) for column in required)


def _cohort_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if bool(row["cohort_exclude_from_trend"]):
        if _critical_missing(row):
            reasons.append("missing_critical_metrics")
        if pd.notna(row.get("coverage_rate")) and float(row["coverage_rate"]) < 0.90:
            reasons.append("coverage_lt_0_90")
        if pd.notna(row.get("annual_flux_TgC")) and float(row["annual_flux_TgC"]) <= 0:
            reasons.append("annual_flux_nonpositive")
    else:
        if not bool(row["cohort_core_2003_2024"]):
            if int(row["year"]) < 2003 or int(row["year"]) > 2024:
                reasons.append("outside_core_2003_2024_period")
            if pd.notna(row.get("coverage_rate")) and float(row["coverage_rate"]) < 0.95:
                reasons.append("coverage_lt_0_95")
            if pd.notna(row.get("fraction_flux_from_low_confidence_days")) and float(row["fraction_flux_from_low_confidence_days"]) >= 0.25:
                reasons.append("low_confidence_flux_fraction_ge_0_25")
            if str(row.get("annual_confidence_tier")) not in {"high", "medium"}:
                reasons.append("annual_confidence_tier_low")
        extrap_days = float(row.get("n_outside_training_logQ_days", 0) or 0) + float(row.get("n_outside_training_doy_days", 0) or 0)
        if extrap_days >= 30:
            reasons.append("large_logQ_or_doy_extrapolation_day_count")
        if float(row.get("n_outside_training_year_days", 0) or 0) > 0:
            reasons.append("outside_training_year_range")
        if float(row.get("n_point_prediction_clipped_days", 0) or 0) > 0:
            reasons.append("point_prediction_clipped_days_present")
    return ";".join(dict.fromkeys(reasons)) if reasons else "meets_core_annual_criteria"


def _annual_cohorts(annual: pd.DataFrame) -> pd.DataFrame:
    out = annual.copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    numeric_cols = [
        "annual_flux_TgC",
        "annual_flux_90_lower_TgC",
        "annual_flux_90_upper_TgC",
        "coverage_rate",
        "fraction_flux_from_low_confidence_days",
        "n_low_confidence_days",
        "n_outside_training_logQ_days",
        "n_outside_training_doy_days",
        "n_outside_training_year_days",
        "n_point_prediction_clipped_days",
    ]
    for column in numeric_cols:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    critical_missing = out.apply(_critical_missing, axis=1)
    valid_positive = out["annual_flux_TgC"].gt(0) & ~critical_missing
    out["cohort_full_2000_2025"] = out["year"].between(2000, 2025, inclusive="both")
    out["cohort_core_2003_2024"] = (
        out["year"].between(2003, 2024, inclusive="both")
        & out["coverage_rate"].ge(0.95)
        & out["fraction_flux_from_low_confidence_days"].lt(0.25)
        & out["annual_confidence_tier"].astype(str).isin({"high", "medium"})
        & valid_positive
    )
    out["cohort_high_confidence_only"] = out["annual_confidence_tier"].astype(str).eq("high") & out["coverage_rate"].ge(0.98) & valid_positive
    out["cohort_exclude_from_trend"] = out["coverage_rate"].lt(0.90) | out["annual_flux_TgC"].le(0) | critical_missing
    out["cohort_sensitivity_only"] = out["coverage_rate"].ge(0.90) & ~out["cohort_core_2003_2024"] & ~out["cohort_exclude_from_trend"]
    out["exclusion_or_caveat_reason"] = out.apply(_cohort_reason, axis=1)
    keep = [
        "river",
        "year",
        "annual_flux_TgC",
        "annual_flux_90_lower_TgC",
        "annual_flux_90_upper_TgC",
        "coverage_rate",
        "annual_confidence_tier",
        "fraction_flux_from_low_confidence_days",
        "n_low_confidence_days",
        "n_outside_training_logQ_days",
        "n_outside_training_doy_days",
        "n_outside_training_year_days",
        "n_point_prediction_clipped_days",
        "cohort_core_2003_2024",
        "cohort_full_2000_2025",
        "cohort_high_confidence_only",
        "cohort_sensitivity_only",
        "cohort_exclude_from_trend",
        "exclusion_or_caveat_reason",
    ]
    return out[keep].sort_values(["river", "year"]).reset_index(drop=True)


def _river_summary(cohorts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for river, group in cohorts.groupby("river", dropna=False):
        core = group[group["cohort_core_2003_2024"]]
        caveats = {
            "outside_core": int(group["exclusion_or_caveat_reason"].astype(str).str.contains("outside_core").sum()),
            "coverage_lt_0_95": int(group["exclusion_or_caveat_reason"].astype(str).str.contains("coverage_lt_0_95").sum()),
            "low_conf_flux_ge_0_25": int(group["exclusion_or_caveat_reason"].astype(str).str.contains("low_confidence_flux_fraction_ge_0_25").sum()),
            "excluded": int(group["cohort_exclude_from_trend"].sum()),
        }
        rows.append(
            {
                "river": river,
                "n_years_total": int(len(group)),
                "n_core_years": int(group["cohort_core_2003_2024"].sum()),
                "n_high_conf_years": int(group["cohort_high_confidence_only"].sum()),
                "n_sensitivity_years": int(group["cohort_sensitivity_only"].sum()),
                "n_excluded_years": int(group["cohort_exclude_from_trend"].sum()),
                "annual_flux_min_core": float(core["annual_flux_TgC"].min()) if not core.empty else np.nan,
                "annual_flux_median_core": float(core["annual_flux_TgC"].median()) if not core.empty else np.nan,
                "annual_flux_max_core": float(core["annual_flux_TgC"].max()) if not core.empty else np.nan,
                "caveat_summary": ";".join(f"{key}:{value}" for key, value in caveats.items()),
            }
        )
    return pd.DataFrame(rows)


def _year_list(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    return ";".join(f"{row.river}:{int(row.year)}" for row in frame.itertuples(index=False))


def _confidence_diagnostics(cohorts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for river, group in cohorts.groupby("river", dropna=False):
        high_low_conf = group[group["fraction_flux_from_low_confidence_days"].ge(0.25)]
        low_coverage = group[group["coverage_rate"].lt(0.95)]
        extrap = group[(group["n_outside_training_logQ_days"] + group["n_outside_training_doy_days"]).ge(30)]
        rows.append(
            {
                "diagnostic_item": "river_confidence_summary",
                "river": river,
                "status": "caveated" if not high_low_conf.empty or not low_coverage.empty or not extrap.empty else "ok",
                "value": float(group["fraction_flux_from_low_confidence_days"].max()),
                "affected_years": _year_list(high_low_conf),
                "notes": "Maximum fraction of flux from low-confidence days; affected_years list years >=0.25.",
            }
        )
        rows.append(
            {
                "diagnostic_item": "coverage_lt_0_95_years",
                "river": river,
                "status": "caveated" if not low_coverage.empty else "ok",
                "value": int(len(low_coverage)),
                "affected_years": _year_list(low_coverage),
                "notes": "Years below 0.95 annual coverage are sensitivity or excluded depending on severity.",
            }
        )
        rows.append(
            {
                "diagnostic_item": "large_extrapolation_flag_years",
                "river": river,
                "status": "caveated" if not extrap.empty else "ok",
                "value": int(len(extrap)),
                "affected_years": _year_list(extrap),
                "notes": "Large extrapolation defined as >=30 logQ+doy extrapolation days in the year.",
            }
        )
    yenisey = cohorts[cohorts["river"].astype(str).eq("Yenisey")]
    yenisey_issue = yenisey[yenisey["fraction_flux_from_low_confidence_days"].ge(0.25)]
    rows.append(
        {
            "diagnostic_item": "Yenisey_low_confidence_flux_issue",
            "river": "Yenisey",
            "status": "caveated" if not yenisey_issue.empty else "ok",
            "value": float(yenisey["fraction_flux_from_low_confidence_days"].max()) if not yenisey.empty else np.nan,
            "affected_years": _year_list(yenisey_issue),
            "notes": "Tracks the Yenisey low-confidence flux issue before trend analysis.",
        }
    )
    yukon = cohorts[cohorts["river"].astype(str).eq("Yukon")]
    yukon_zero = yukon[yukon["annual_flux_TgC"].le(0.01)]
    rows.append(
        {
            "diagnostic_item": "Yukon_zero_or_near_zero_annual_flux_issue",
            "river": "Yukon",
            "status": "caveated" if not yukon_zero.empty else "ok",
            "value": float(yukon["annual_flux_TgC"].min()) if not yukon.empty else np.nan,
            "affected_years": _year_list(yukon_zero),
            "notes": "Tracks zero or near-zero Yukon annual flux before trend analysis.",
        }
    )
    return pd.DataFrame(rows)


def _may_july_cohorts(may_july: pd.DataFrame, annual_cohorts: pd.DataFrame) -> pd.DataFrame:
    out = may_july.copy()
    for column in ["year", "may_july_flux_TgC", "annual_total_flux_TgC", "may_july_flux_fraction_of_annual", "coverage_rate"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    annual_flags = annual_cohorts[
        ["river", "year", "cohort_core_2003_2024", "cohort_exclude_from_trend", "annual_flux_TgC"]
    ].rename(columns={"annual_flux_TgC": "annual_flux_TgC"})
    out = out.merge(annual_flags, on=["river", "year"], how="left")
    out["core_may_july_interpretation_allowed"] = (
        out["cohort_core_2003_2024"].fillna(False)
        & out["coverage_rate"].ge(0.95)
        & out["fraction_flux_from_low_confidence_days"].lt(0.25)
        & out["may_july_confidence_tier"].astype(str).isin({"high", "medium"})
        & out["may_july_flux_fraction_of_annual"].notna()
    )

    def reason(row: pd.Series) -> str:
        reasons: list[str] = []
        if not bool(row["core_may_july_interpretation_allowed"]):
            if not bool(row.get("cohort_core_2003_2024", False)):
                reasons.append("annual_year_not_in_core_cohort")
            if pd.notna(row.get("coverage_rate")) and float(row["coverage_rate"]) < 0.95:
                reasons.append("may_july_coverage_lt_0_95")
            if pd.notna(row.get("fraction_flux_from_low_confidence_days")) and float(row["fraction_flux_from_low_confidence_days"]) >= 0.25:
                reasons.append("may_july_low_confidence_flux_fraction_ge_0_25")
            if str(row.get("may_july_confidence_tier")) not in {"high", "medium"}:
                reasons.append("may_july_confidence_tier_low")
            if pd.isna(row.get("may_july_flux_fraction_of_annual")):
                reasons.append("annual_fraction_not_available")
        return ";".join(dict.fromkeys(reasons)) if reasons else "provisional_may_july_core_interpretation_allowed"

    out["caveat_reason"] = out.apply(reason, axis=1)
    out["window_label"] = "provisional May-July flux window"
    keep = [
        "river",
        "year",
        "may_july_flux_TgC",
        "annual_flux_TgC",
        "may_july_flux_fraction_of_annual",
        "coverage_rate",
        "may_july_confidence_tier",
        "core_may_july_interpretation_allowed",
        "caveat_reason",
        "window_label",
    ]
    return out[keep].rename(columns={"coverage_rate": "may_july_coverage_rate", "may_july_confidence_tier": "confidence_tier"})


def _make_figures(cohorts: pd.DataFrame, may_july: pd.DataFrame) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    FLUX_INTERPRETATION_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def save(fig, name: str) -> None:
        fig.tight_layout()
        destination = FLUX_INTERPRETATION_FIGURE_DIR / name
        fig.savefig(destination, dpi=120)
        plt.close(fig)
        paths.append(destination)

    fig, ax = plt.subplots(figsize=(9, 5))
    for river, group in cohorts.groupby("river"):
        core = group[group["cohort_core_2003_2024"]]
        other = group[~group["cohort_core_2003_2024"]]
        ax.scatter(core["year"], core["annual_flux_TgC"], s=24, label=f"{river} core")
        ax.scatter(other["year"], other["annual_flux_TgC"], s=16, marker="x", alpha=0.65)
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual DOC flux (Tg C)")
    ax.set_title("Annual flux core vs excluded/caveated years")
    ax.legend(fontsize="xx-small", ncol=2)
    save(fig, "annual_flux_core_vs_excluded_by_river.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    tier_order = {"high": 3, "medium": 2, "low": 1}
    for river, group in cohorts.groupby("river"):
        ax.plot(group["year"], group["annual_confidence_tier"].map(tier_order), marker="o", linewidth=1.0, label=str(river))
    ax.set_yticks([1, 2, 3], ["low", "medium", "high"])
    ax.set_xlabel("Year")
    ax.set_title("Annual confidence tier timeline")
    ax.legend(fontsize="x-small")
    save(fig, "annual_confidence_tier_timeline.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    for river, group in cohorts.groupby("river"):
        ax.plot(group["year"], group["fraction_flux_from_low_confidence_days"], marker="o", linewidth=1.0, label=str(river))
    ax.axhline(0.25, linestyle="--")
    ax.set_xlabel("Year")
    ax.set_ylabel("Fraction flux from low-confidence days")
    ax.set_title("Low-confidence flux fraction by year")
    ax.legend(fontsize="x-small")
    save(fig, "fraction_low_confidence_flux_by_year.png")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    data = [group["may_july_flux_fraction_of_annual"].dropna().to_numpy() for _, group in may_july.groupby("river")]
    labels = [str(river) for river, _ in may_july.groupby("river")]
    if data:
        try:
            ax.boxplot(data, tick_labels=labels, showfliers=False)
        except TypeError:
            ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("Provisional May-July fraction of annual flux")
    ax.set_title("Provisional May-July flux fraction by river")
    save(fig, "provisional_may_july_fraction_by_river.png")
    return paths


def _ready_status(summary: pd.DataFrame, cohorts: pd.DataFrame) -> tuple[str, str]:
    full_core_rivers = summary[summary["n_core_years"].ge(22)]["river"].astype(str).tolist()
    excluded = int(cohorts["cohort_exclude_from_trend"].sum())
    sensitivity = int(cohorts["cohort_sensitivity_only"].sum())
    if not full_core_rivers:
        return "false", "No river has complete 2003-2024 core coverage."
    if excluded or sensitivity:
        return "true_with_caveats", f"Core cohorts are available, but sensitivity_only={sensitivity} and excluded={excluded} rows require caveats."
    return "true", "All rows meet core interpretation criteria."


def write_flux_cohort_report() -> Path:
    FLUX_INTERPRETATION_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cohorts = _read_csv(FLUX_INTERPRETATION_TABLE_DIR / "annual_flux_analysis_cohorts.csv")
    summary = _read_csv(FLUX_INTERPRETATION_TABLE_DIR / "flux_cohort_summary_by_river.csv")
    diagnostics = _read_csv(FLUX_INTERPRETATION_TABLE_DIR / "flux_confidence_diagnostics.csv")
    may_july = _read_csv(FLUX_INTERPRETATION_TABLE_DIR / "provisional_may_july_cohort_summary.csv")
    ready, ready_note = _ready_status(summary, cohorts)
    may_july_ready = "true_with_caveats" if may_july["core_may_july_interpretation_allowed"].any() else "false"
    lines = [
        "# Flux Interpretation Cohort Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "## 1. Scope and guardrails",
        "",
        "This phase selects analysis cohorts from existing guarded DOC flux products. It does not retrain a model, does not regenerate DOC predictions, does not recalculate flux, does not read raw/interim/canonical data, and does not perform a formal trend test.",
        "",
        "## 2. Inputs",
        "",
        "- `outputs/tables/doc_flux/daily_doc_flux.csv`",
        "- `outputs/tables/doc_flux/annual_doc_flux_summary.csv`",
        "- `outputs/tables/doc_flux/provisional_may_july_flux_summary.csv`",
        "- `outputs/tables/doc_flux/doc_flux_qc_summary.csv`",
        "- `outputs/reports/doc_flux/doc_flux_report.md`",
        "",
        "## 3. Why cohort selection is needed",
        "",
        "Flux products carry coverage, extrapolation, confidence-tier, and interval caveats. Cohort selection separates core annual trend candidates from sensitivity-only and excluded river-years before any trend analysis.",
        "",
        "## 4. Core annual trend cohort definition",
        "",
        "`core_2003_2024` requires year 2003-2024, coverage_rate >= 0.95, fraction_flux_from_low_confidence_days < 0.25, confidence tier high or medium, positive annual flux, and no missing critical metrics.",
        "",
        _md_table(summary, max_rows=20),
        "",
        "## 5. Full hindcast sensitivity cohort",
        "",
        "`cohort_full_2000_2025` keeps all annual rows visible, with confidence flags preserved. Rows outside the 2003-2024 core period are treated as sensitivity context.",
        "",
        "## 6. High-confidence-only cohort",
        "",
        "`cohort_high_confidence_only` requires annual confidence tier high and coverage_rate >= 0.98.",
        "",
        "## 7. Excluded/caveated years",
        "",
        _md_table(cohorts[cohorts["cohort_exclude_from_trend"] | cohorts["cohort_sensitivity_only"]].head(30), max_rows=30),
        "",
        "## 8. May-July provisional window cohort",
        "",
        "May-July is treated as a provisional flux window only. Hydrologic snowmelt-window attribution is deferred.",
        "",
        _md_table(may_july.head(30), max_rows=30),
        "",
        "## 9. River-specific caveats",
        "",
        _md_table(diagnostics, max_rows=40),
        "",
        "## 10. Recommendation for next phase",
        "",
        f"ready_for_trend_analysis: `{ready}`. {ready_note}",
        "",
        f"ready_for_snowmelt_window_refinement: `{may_july_ready}`. Use May-July only as a provisional screening window before refining hydrologic windows.",
        "",
        "## 11. Explicit statements",
        "",
        "- No model retraining was performed.",
        "- No new DOC prediction was generated.",
        "- No flux recalculation was performed.",
        "- No formal trend test was run.",
        "- May-July is provisional, not final snowmelt.",
    ]
    FLUX_COHORT_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return FLUX_COHORT_REPORT_PATH


def select_flux_analysis_cohorts() -> dict[str, Any]:
    _ensure_dirs()
    before_hashes = _input_hashes()
    inputs = _load_inputs()
    annual_cohorts = _annual_cohorts(inputs["annual"])
    river_summary = _river_summary(annual_cohorts)
    diagnostics = _confidence_diagnostics(annual_cohorts)
    may_july = _may_july_cohorts(inputs["may_july"], annual_cohorts)
    figure_paths = _make_figures(annual_cohorts, may_july)
    table_paths = [
        _write_csv(annual_cohorts, FLUX_INTERPRETATION_TABLE_DIR / "annual_flux_analysis_cohorts.csv"),
        _write_csv(river_summary, FLUX_INTERPRETATION_TABLE_DIR / "flux_cohort_summary_by_river.csv"),
        _write_csv(diagnostics, FLUX_INTERPRETATION_TABLE_DIR / "flux_confidence_diagnostics.csv"),
        _write_csv(may_july, FLUX_INTERPRETATION_TABLE_DIR / "provisional_may_july_cohort_summary.csv"),
    ]
    report_path = write_flux_cohort_report()
    _verify_inputs_unchanged(before_hashes)
    ready, _ = _ready_status(river_summary, annual_cohorts)
    return {
        "tables": table_paths,
        "figures": figure_paths,
        "report": report_path,
        "annual_cohorts": annual_cohorts,
        "river_summary": river_summary,
        "diagnostics": diagnostics,
        "may_july": may_july,
        "ready_for_trend_analysis": ready,
    }


__all__ = [
    "FLUX_INTERPRETATION_TABLE_DIR",
    "FLUX_INTERPRETATION_REPORT_DIR",
    "FLUX_INTERPRETATION_FIGURE_DIR",
    "FLUX_COHORT_REPORT_PATH",
    "select_flux_analysis_cohorts",
    "write_flux_cohort_report",
]
