from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import REPORT_DIR, TABLE_DIR
from ..reports import _md_table, utc_now


MAY_JULY_TABLE_DIR = TABLE_DIR / "may_july_flux"
MAY_JULY_REPORT_DIR = REPORT_DIR / "may_july_flux"
MAY_JULY_REPORT_PATH = MAY_JULY_REPORT_DIR / "may_july_flux_interpretation_report.md"


def _read_csv(name: str) -> pd.DataFrame:
    destination = MAY_JULY_TABLE_DIR / name
    if not destination.exists():
        raise FileNotFoundError(f"Required May-July flux interpretation table is missing: {destination}")
    return pd.read_csv(destination, low_memory=False)


def _trend_language(trends: pd.DataFrame, metric: str) -> str:
    core = trends[trends["analysis_cohort"].eq("core_2003_2024") & trends["metric"].eq(metric)]
    detectable = core[core["detectable_trend"].astype(str).str.lower().isin({"true", "1"})]
    if detectable.empty:
        return "No river has a detectable trend for this metric in the core 2003-2024 cohort."
    parts = [f"{row.river}: {row.trend_direction}" for row in detectable.itertuples(index=False)]
    return "Detectable core trends: " + "; ".join(parts)


def _yukon_sentence(comparison: pd.DataFrame) -> str:
    yukon = comparison[comparison["river"].astype(str).eq("Yukon")]
    if yukon.empty:
        return "Yukon comparison was not available."
    row = yukon.iloc[0]
    return (
        f"Yukon does_may_july_explain_annual_signal: `{row['does_may_july_explain_annual_signal']}`. "
        f"{row['interpretation']}"
    )


def write_may_july_flux_report() -> Path:
    MAY_JULY_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _read_csv("may_july_flux_interpretation_by_river_year.csv")
    summary = _read_csv("may_july_fraction_summary_by_river.csv")
    trends = _read_csv("may_july_flux_trends_by_river.csv")
    comparison = _read_csv("may_july_vs_annual_trend_comparison.csv")
    caveats = _read_csv("may_july_caveat_summary.csv")
    core_rows = rows[rows["cohort_core_2003_2024"].astype(str).str.lower().isin({"true", "1"})]
    full_rows = rows
    high_rows = rows[rows["cohort_high_confidence_only"].astype(str).str.lower().isin({"true", "1"})]
    lines = [
        "# Provisional May-July DOC Flux Interpretation Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "## 1. Scope and guardrails",
        "",
        "This phase interprets existing provisional May-July DOC flux summaries and existing annual flux cohorts. It does not retrain models, does not generate new DOC predictions, does not recalculate flux, does not read raw/interim/canonical data, and does not refine hydrologic snowmelt windows.",
        "",
        "## 2. Why May-July is provisional",
        "",
        "May-July is a fixed screening window used for preliminary seasonal interpretation. It is not a hydrologically refined snowmelt window and should not be treated as a completed snowmelt attribution.",
        "",
        "## 3. Input cohorts",
        "",
        f"- Core 2003-2024 provisional May-July rows: `{len(core_rows)}`",
        f"- Full 2000-2025 provisional May-July rows: `{len(full_rows)}`",
        f"- High-confidence-only provisional May-July rows: `{len(high_rows)}`",
        "",
        "The core 2003-2024 cohort is the primary interpretation set. Full-period and high-confidence-only sets are sensitivity context.",
        "",
        "## 4. May-July fraction by river",
        "",
        _md_table(summary, max_rows=20),
        "",
        "## 5. May-July flux trends",
        "",
        _trend_language(trends, "may_july_flux_TgC"),
        "",
        _md_table(trends[trends["metric"].eq("may_july_flux_TgC")], max_rows=30),
        "",
        "## 6. May-July fraction trends",
        "",
        _trend_language(trends, "may_july_flux_fraction_of_annual"),
        "",
        _md_table(trends[trends["metric"].eq("may_july_flux_fraction_of_annual")], max_rows=30),
        "",
        "Trend language is intentionally conservative: non-significant results use no detectable trend wording.",
        "",
        "## 7. Relationship to annual flux trends",
        "",
        _md_table(comparison, max_rows=20),
        "",
        "## 8. Yukon-specific interpretation",
        "",
        _yukon_sentence(comparison),
        "",
        "## 9. River-specific caveats",
        "",
        _md_table(caveats, max_rows=30),
        "",
        "## 10. Recommended next step",
        "",
        "Recommended next step: hydrologic snowmelt window refinement or final interpretation, keeping this fixed May-July analysis as provisional context.",
        "",
        "## 11. Explicit statements",
        "",
        "- No model retraining was performed.",
        "- No new DOC prediction was generated.",
        "- No flux recalculation was performed.",
        "- May-July is provisional, not final snowmelt.",
        "- Discharge uncertainty is not propagated.",
    ]
    MAY_JULY_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return MAY_JULY_REPORT_PATH
