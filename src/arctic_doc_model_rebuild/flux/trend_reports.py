from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import REPORT_DIR, TABLE_DIR
from ..reports import _md_table, utc_now


ANNUAL_TREND_TABLE_DIR = TABLE_DIR / "annual_flux_trends"
ANNUAL_TREND_REPORT_DIR = REPORT_DIR / "annual_flux_trends"
ANNUAL_TREND_REPORT_PATH = ANNUAL_TREND_REPORT_DIR / "annual_flux_trend_report.md"


def _read_csv(name: str) -> pd.DataFrame:
    destination = ANNUAL_TREND_TABLE_DIR / name
    if not destination.exists():
        raise FileNotFoundError(f"Required annual flux trend table is missing: {destination}")
    return pd.read_csv(destination, low_memory=False)


def _detectable_rivers(core: pd.DataFrame) -> str:
    detectable = core[
        core["analysis_cohort"].eq("core_2003_2024")
        & core["significant_at_0_05"].astype(str).str.lower().isin({"true", "1"})
        & core["trend_direction"].isin(["increasing", "decreasing"])
    ]["river"].astype(str).tolist()
    return ", ".join(detectable) if detectable else "None; no detectable trend at p < 0.05."


def write_annual_flux_trend_report() -> Path:
    ANNUAL_TREND_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    audit = _read_csv("annual_flux_trend_input_audit.csv")
    by_river = _read_csv("annual_flux_trends_by_river.csv")
    aggregate = _read_csv("annual_flux_trends_aggregate.csv")
    sensitivity = _read_csv("annual_flux_trend_sensitivity_by_cohort.csv")
    uncertainty = _read_csv("annual_flux_trend_uncertainty_sensitivity.csv")
    caveats = _read_csv("annual_flux_trend_caveat_summary.csv")
    core = by_river[by_river["analysis_cohort"].eq("core_2003_2024")]
    aggregate_core = aggregate[aggregate["analysis_cohort"].eq("core_2003_2024")]
    full = by_river[by_river["analysis_cohort"].eq("full_2000_2025_sensitivity")]
    high = by_river[by_river["analysis_cohort"].eq("high_confidence_only_sensitivity")]
    lines = [
        "# Annual DOC Flux Trend Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "## 1. Scope and guardrails",
        "",
        "This phase analyzes annual DOC flux trends from preselected flux interpretation cohorts. It does not retrain models, regenerate DOC predictions, recalculate flux, read raw/interim/canonical data, or refine snowmelt windows.",
        "",
        "## 2. Input cohorts",
        "",
        _md_table(audit, max_rows=20),
        "",
        "## 3. Why trend analysis uses cohorts",
        "",
        "The primary analysis uses the core 2003-2024 cohort so low coverage rows, excluded rows, and full-hindcast extension years do not drive the main result. Full-period and high-confidence-only cohorts are sensitivity analyses.",
        "",
        "## 4. Core 2003-2024 trend results by river",
        "",
        f"Rivers with detectable core trend: {_detectable_rivers(core)}",
        "",
        _md_table(core, max_rows=20),
        "",
        "Use no detectable trend rather than no trend when p-values are non-significant.",
        "",
        "## 5. Aggregate trend results",
        "",
        _md_table(aggregate_core, max_rows=20),
        "",
        "## 6. Full 2000-2025 sensitivity",
        "",
        "The full hindcast sensitivity includes extension years and caveated rows. It is not the primary result.",
        "",
        _md_table(full, max_rows=20),
        "",
        "## 7. High-confidence-only sensitivity",
        "",
        _md_table(high, max_rows=20),
        "",
        "## 8. DOC uncertainty interval sensitivity",
        "",
        "The lower/upper sensitivity uses annual flux intervals derived from DOC concentration empirical residual intervals only. Discharge uncertainty is not propagated.",
        "",
        _md_table(uncertainty, max_rows=30),
        "",
        "## 9. River-specific caveats",
        "",
        _md_table(caveats, max_rows=30),
        "",
        "Required caveats carried forward: Yenisey has many years with high low-confidence flux fraction. Yukon has a near-zero/zero issue in 2000 and that year should not drive full-period trends. Kolyma has excluded years and high low-confidence-fraction years. Ob has one excluded year and one high low-confidence-fraction year. Mackenzie 2025 is sensitivity due coverage <0.95. May-July remains provisional and is not analyzed as snowmelt in this phase.",
        "",
        "## 10. Interpretation boundaries",
        "",
        "Trend p-values describe detectable monotonic or linear evidence within the selected annual cohorts. Non-significant results are reported as no detectable trend, not proof of no trend. Core trends should be emphasized over full hindcast if sensitivity differs.",
        "",
        "## 11. Recommended next phase",
        "",
        "Recommended next step: provisional May-July flux interpretation or snowmelt window refinement.",
        "",
        "## 12. Explicit statements",
        "",
        "- No model retraining was performed.",
        "- No new DOC prediction was generated.",
        "- No flux recalculation was performed.",
        "- Annual May-July or snowmelt was not analyzed here.",
        "- Discharge uncertainty is not propagated.",
        "",
        "## Sensitivity overview",
        "",
        _md_table(sensitivity, max_rows=30),
    ]
    ANNUAL_TREND_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ANNUAL_TREND_REPORT_PATH
