from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..gold_contract import load_contract, resolve_gold_data_location, verification_problem_counts, verify_all_gold_tables
from ..paths import REPORT_DIR, TABLE_DIR
from ..reports import _md_table


BASELINE_TABLE_DIR = TABLE_DIR / "baseline"
BASELINE_REPORT_DIR = REPORT_DIR / "baseline"
BASELINE_REPORT_PATH = BASELINE_REPORT_DIR / "baseline_model_report.md"


def _read(name: str) -> pd.DataFrame:
    destination = BASELINE_TABLE_DIR / name
    if not destination.exists():
        return pd.DataFrame()
    return pd.read_csv(destination)


def write_baseline_report_from_tables() -> Path:
    BASELINE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    contract = load_contract()
    location = resolve_gold_data_location()
    verification = verify_all_gold_tables()
    counts = verification_problem_counts(verification)
    feature_sets = _read("feature_set_registry.csv")
    validations = _read("validation_scheme_registry.csv")
    overall = _read("baseline_metrics_overall.csv")
    by_river = _read("baseline_metrics_by_river.csv")
    by_year = _read("baseline_metrics_by_year.csv")
    by_season = _read("baseline_metrics_by_season_window.csv")
    residuals = _read("baseline_residuals.csv")
    ranking = _read("baseline_model_ranking.csv")
    missingness = _read("baseline_missingness_used_by_model.csv")
    best = ranking.head(8) if not ranking.empty else pd.DataFrame()
    lines = [
        "# Baseline DOC Concentration Model Report",
        "",
        "## 1. Scope and guardrails",
        "",
        "This phase trains simple DOC concentration baseline models for cross-validation diagnostics only. It does not generate production daily DOC predictions and does not compute flux.",
        "",
        "- model input table: `training_matrix_hydrocore.csv`",
        "- excluded in this phase: prediction grid, optical matrices, basin context matrices, lab optical/CDOM, flux products",
        "- model artifact policy: no final production model artifact is saved",
        "",
        "## 2. Data contract status",
        "",
        f"- freeze_id: `{contract['freeze_id']}`",
        f"- source_tag: `{contract['source_tag']}`",
        f"- gold_data_dir: `{location.data_dir}`",
        f"- contract_tables_ok: `{int(verification['status'].eq('ok').sum())}/{len(verification)}`",
        f"- hash_mismatches: `{counts['sha256_mismatch']}`",
        f"- row_count_mismatches: `{counts['row_count_mismatch']}`",
        "",
        "## 3. Input matrix summary",
        "",
        _md_table(missingness, max_rows=10),
        "",
        "## 4. Feature sets",
        "",
        _md_table(feature_sets, max_rows=20),
        "",
        "## 5. Validation schemes",
        "",
        _md_table(validations, max_rows=10),
        "",
        "## 6. Overall metrics",
        "",
        _md_table(overall.sort_values(["validation_scheme", "rmse"]).head(30) if not overall.empty else overall, max_rows=30),
        "",
        "## 7. Metrics by river",
        "",
        _md_table(by_river[by_river.get("validation_scheme", pd.Series(dtype=str)).eq("leave_one_year_out")].head(30) if not by_river.empty else by_river, max_rows=30),
        "",
        "## 8. Metrics by year",
        "",
        _md_table(by_year[by_year.get("validation_scheme", pd.Series(dtype=str)).eq("leave_one_year_out")].head(30) if not by_year.empty else by_year, max_rows=30),
        "",
        "## 9. Metrics by season window",
        "",
        "Season windows are provisional descriptive windows, not final hydrologic freshet definitions.",
        "",
        _md_table(by_season[by_season.get("validation_scheme", pd.Series(dtype=str)).eq("leave_one_year_out")].head(30) if not by_season.empty else by_season, max_rows=30),
        "",
        "## 10. Residual diagnostics",
        "",
        _md_table(residuals[["model_id", "feature_set", "validation_scheme", "river", "month", "residual_mgC_L", "abs_residual_mgC_L"]].head(30) if not residuals.empty else residuals, max_rows=30),
        "",
        "## 11. Best baseline candidates",
        "",
        "Primary ranking uses leave-one-year-out RMSE and MAE with penalties for low sample count and snow complete-case sample loss.",
        "",
        _md_table(best, max_rows=12),
        "",
        "Do not use leave-one-river-out as the primary winner selection because it is a high-risk extrapolation stress test with only six rivers.",
        "",
        "## 12. Why optical/basin/flux are excluded in this phase",
        "",
        "- Optical matrices are reserved for a later optical sensitivity phase.",
        "- Basin context matrices are reserved for a later basin sensitivity phase because only six river-level units exist.",
        "- Prediction grid outputs are x-only and are not used to generate daily DOC predictions in this phase.",
        "- Flux requires production DOC predictions and discharge integration, so it is explicitly out of scope.",
        "",
        "## 13. Recommended next phase",
        "",
        "Model refinement or optical sensitivity, after deciding whether the reduced hydroclimate LOYO candidate is stable enough for a baseline reference.",
        "",
        "## 14. Explicit statement",
        "",
        "- DOC concentration models were trained for validation only.",
        "- No production daily DOC prediction was generated.",
        "- No DOC flux was generated.",
        "- Gold data were not modified.",
    ]
    BASELINE_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return BASELINE_REPORT_PATH
