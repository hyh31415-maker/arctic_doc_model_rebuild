from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import REPORT_DIR, TABLE_DIR
from ..reports import _md_table, utc_now


SNOWMELT_TABLE_DIR = TABLE_DIR / "snowmelt_windows"
SNOWMELT_REPORT_DIR = REPORT_DIR / "snowmelt_windows"
SNOWMELT_REPORT_PATH = SNOWMELT_REPORT_DIR / "snowmelt_window_report.md"


def _read_csv(name: str) -> pd.DataFrame:
    destination = SNOWMELT_TABLE_DIR / name
    if not destination.exists():
        raise FileNotFoundError(f"Required snowmelt window table is missing: {destination}")
    return pd.read_csv(destination, low_memory=False)


def _detectable(trends: pd.DataFrame, metric: str) -> str:
    subset = trends[
        trends["analysis_cohort"].eq("core_2003_2024")
        & trends["metric"].eq(metric)
        & trends["detectable_trend"].astype(str).str.lower().isin({"true", "1"})
    ]
    if subset.empty:
        return "No detectable trend in the core cohort."
    parts = [f"{row.river}/{row.window_id}: {row.trend_direction}" for row in subset.itertuples(index=False)]
    return "; ".join(parts)


def _yukon_summary(comparison: pd.DataFrame) -> str:
    yukon = comparison[comparison["river"].astype(str).eq("Yukon")]
    if yukon.empty:
        return "Yukon comparison was unavailable."
    rows = []
    for row in yukon.itertuples(index=False):
        rows.append(f"{row.window_id}: `{row.does_window_explain_annual_signal}` ({row.interpretation})")
    return " ".join(rows)


def write_snowmelt_window_report() -> Path:
    SNOWMELT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    definitions = _read_csv("snowmelt_window_definitions_by_river_year.csv")
    summary = _read_csv("snowmelt_window_flux_summary.csv")
    trends = _read_csv("snowmelt_window_trends_by_river.csv")
    comparison = _read_csv("annual_vs_snowmelt_signal_comparison.csv")
    qc = _read_csv("snowmelt_window_qc_summary.csv")
    lines = [
        "# Hydrologic Snowmelt Window Refinement Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "## 1. Scope and guardrails",
        "",
        "This phase defines dynamic hydrologic freshet windows from existing daily Q and hydroclimate variables, then sums existing guarded daily DOC flux into those windows. It does not retrain DOC models, regenerate DOC predictions, recalculate daily flux, modify gold data, read raw/interim/canonical data, or propagate discharge uncertainty.",
        "",
        "## 2. Why dynamic hydrologic windows were needed",
        "",
        "The fixed May-July screening window did not explain the Yukon annual flux signal. Dynamic windows test whether year-specific discharge and snow timing better isolate the hydrologic freshet period.",
        "",
        "## 3. Window definitions W0-W4",
        "",
        "- `fixed_may_july_reference`: fixed May-July reference only.",
        "- `discharge_centered_freshet`: peak-Q centered melt-season window using rise and recession thresholds.",
        "- `q75_peak_contiguous`: contiguous melt-season days above the river-year Q75 threshold around peak Q.",
        "- `snow_depletion_assisted`: PDD/snow-depletion assisted start with Q recession fallback.",
        "- `common_overlap_w1_w2`: overlap of the discharge-centered and Q75 peak windows.",
        "",
        _md_table(definitions.head(30), max_rows=30),
        "",
        "## 4. Window QC and confidence tiers",
        "",
        _md_table(qc, max_rows=30),
        "",
        "## 5. Dynamic window timing",
        "",
        "Timing diagnostics include start DOY, end DOY, peak-Q DOY, and window length. Implausibly short or long windows are retained with low confidence instead of being silently edited.",
        "",
        "## 6. Window flux and fraction summaries",
        "",
        _md_table(summary.head(30), max_rows=30),
        "",
        "## 7. Window flux trends",
        "",
        _detectable(trends, "window_flux_TgC"),
        "",
        _md_table(trends[trends["metric"].eq("window_flux_TgC")], max_rows=30),
        "",
        "## 8. Window timing trends",
        "",
        _md_table(trends[trends["metric"].isin(["window_start_doy", "window_end_doy", "window_peak_q_doy", "window_length_days"])], max_rows=30),
        "",
        "## 9. Annual vs snowmelt-window comparison",
        "",
        _md_table(comparison, max_rows=40),
        "",
        "## 10. Yukon-specific interpretation",
        "",
        _yukon_summary(comparison),
        "",
        "## 11. Comparison with fixed May-July",
        "",
        "The fixed May-July row is retained as `fixed_may_july_reference` so dynamic windows can be compared against the previous provisional screening window. It remains a reference window, not a completed hydrologic attribution.",
        "",
        "## 12. Caveats",
        "",
        "Window-level flux uncertainty continues to include DOC concentration uncertainty only. Discharge uncertainty is not propagated. Snow-cover fields are incomplete in some river-years, so snow-assisted windows carry confidence flags and fallback notes.",
        "",
        "## 13. Recommended next phase",
        "",
        "Recommended next step: final synthesis / manuscript-ready results summary.",
        "",
        "## 14. Explicit statements",
        "",
        "- No model retraining was performed.",
        "- No new DOC prediction was generated.",
        "- No daily flux recalculation was performed.",
        "- Existing flux was summed into windows.",
        "- Discharge uncertainty was not propagated.",
        "- Fixed May-July remains a provisional reference.",
    ]
    SNOWMELT_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return SNOWMELT_REPORT_PATH
