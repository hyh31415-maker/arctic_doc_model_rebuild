from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..gold_contract import sha256_file
from ..paths import REPORT_DIR, TABLE_DIR
from ..reports import _md_table, utc_now
from .flux_qc import analysis_readiness_status


DOC_FLUX_REPORT_DIR = REPORT_DIR / "doc_flux"
DOC_FLUX_TABLE_DIR = TABLE_DIR / "doc_flux"
DOC_FLUX_REPORT_PATH = DOC_FLUX_REPORT_DIR / "doc_flux_report.md"


def _read_csv(name: str) -> pd.DataFrame:
    destination = DOC_FLUX_TABLE_DIR / name
    if not destination.exists():
        raise FileNotFoundError(f"Required DOC flux table is missing: {destination}")
    return pd.read_csv(destination, low_memory=False)


def write_doc_flux_report() -> Path:
    DOC_FLUX_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    daily_path = DOC_FLUX_TABLE_DIR / "daily_doc_flux.csv"
    daily = _read_csv("daily_doc_flux.csv")
    annual = _read_csv("annual_doc_flux_summary.csv")
    may_july = _read_csv("provisional_may_july_flux_summary.csv")
    period = _read_csv("river_period_flux_summary.csv")
    qc = _read_csv("doc_flux_qc_summary.csv")
    range_flags = _read_csv("doc_flux_range_flags.csv")
    confidence = _read_csv("doc_flux_confidence_tier_summary.csv")
    readiness = analysis_readiness_status(annual, range_flags)
    daily_hash = sha256_file(daily_path) if daily_path.exists() else ""
    lines = [
        "# Guarded DOC Flux Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "## 1. Scope and guardrails",
        "",
        "This phase calculates guarded DOC flux from the existing guarded daily DOC concentration prediction table. It does not retrain a DOC model, does not regenerate daily DOC predictions, does not read raw/interim/canonical data, and does not use optical, basin, or lab optical/CDOM features.",
        "",
        "## 2. Inputs",
        "",
        "- `outputs/tables/daily_doc_prediction/daily_doc_prediction.csv`",
        "- `outputs/tables/daily_doc_prediction/flux_readiness_decision.csv`",
        "- `configs/model_specs/production_candidate_r4_river_specific_q_and_season_linear.yaml`",
        f"- daily_doc_flux_sha256: `{daily_hash}`",
        "",
        "## 3. Flux formula and unit conversion",
        "",
        "`daily_flux_kgC_day = DOC_mgC_L * Q_m3s * 86.4`.",
        "",
        "This follows 1 mg/L * 1 m3/s = 1 g/s, and 1 g/s = 86.4 kg/day. Mg C/day is kg C/day divided by 1000; Tg C/day is kg C/day divided by 1e9.",
        "",
        "## 4. Daily flux output",
        "",
        f"- path: `{daily_path}`",
        f"- daily_flux_rows: `{len(daily)}`",
        "",
        _md_table(confidence[confidence["scope"].eq("overall")], max_rows=10),
        "",
        "## 5. Annual flux summary",
        "",
        _md_table(annual.head(20), max_rows=20),
        "",
        "## 6. Provisional May-July flux summary",
        "",
        "May-July is a provisional freshet window for screening only. It is not a final snowmelt contribution estimate.",
        "",
        _md_table(may_july.head(20), max_rows=20),
        "",
        "## 7. Prediction interval propagation",
        "",
        "Flux intervals propagate DOC concentration empirical residual intervals only. Discharge uncertainty was not propagated in this phase.",
        "",
        "## 8. Confidence tiers and extrapolation flags",
        "",
        _md_table(confidence.head(30), max_rows=30),
        "",
        "## 9. Range and coverage QC",
        "",
        _md_table(qc, max_rows=30),
        "",
        _md_table(range_flags.head(30), max_rows=30),
        "",
        "## 10. Caveats inherited from concentration model",
        "",
        "- within six ArcticGRO rivers only",
        "- no cross-river extrapolation",
        "- high-DOC caveat carried forward",
        "- fold stability caveat carried forward",
        "- empirical intervals are validation residual intervals",
        "- optical was excluded from the primary concentration model",
        "- ROI caveats are carried as metadata but do not drive hydrocore flux calculation",
        "",
        "## 11. What this flux can and cannot claim",
        "",
        "These tables provide guarded daily and annual DOC flux products from guarded DOC concentration predictions and Q. They do not include discharge uncertainty, do not establish a final snowmelt flux window, and should be interpreted with extrapolation and confidence-tier flags.",
        "",
        "## 12. Recommended next phase",
        "",
        f"ready_for_trend_or_snowmelt_analysis: `{readiness}`",
        "",
        "Recommended next step: flux interpretation / trend analysis / snowmelt window refinement.",
        "",
        "## 13. Explicit statements",
        "",
        "- Daily DOC flux was calculated.",
        "- Flux intervals include DOC concentration uncertainty only.",
        "- Discharge uncertainty was not propagated.",
        "- May-July flux is provisional, not final snowmelt contribution.",
        "- Gold data were not modified.",
        "- No new DOC model was trained.",
        "",
        "## Multi-year summary",
        "",
        _md_table(period.head(30), max_rows=30),
    ]
    DOC_FLUX_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return DOC_FLUX_REPORT_PATH
