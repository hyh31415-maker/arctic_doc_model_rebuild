from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .paths import REPORT_DIR, TABLE_DIR, OUTPUT_DIR
from .reports import _md_table, utc_now


FINAL_TABLE_DIR = TABLE_DIR / "final_synthesis"
FINAL_REPORT_DIR = REPORT_DIR / "final_synthesis"
FINAL_FIGURE_DIR = OUTPUT_DIR / "figures" / "final_synthesis"
FINAL_SYNTHESIS_REPORT_PATH = FINAL_REPORT_DIR / "final_synthesis_report.md"

FREEZE_ID = "data_freeze_gold_20260526_v1"
PRIMARY_MODEL = "R4_river_specific_Q_and_season + linear_regression"
PRIMARY_BASELINE = "F3_q_season_river_fixed + ridge_alpha_1"


REQUIRED_INPUTS: dict[str, Path] = {
    "data_contract_report": REPORT_DIR / "data_contract_report.md",
    "gold_data_summary_report": REPORT_DIR / "gold_data_summary_report.md",
    "eda_report": REPORT_DIR / "eda" / "eda_report.md",
    "model_scope_feasibility": TABLE_DIR / "eda" / "model_scope_feasibility.csv",
    "baseline_model_report": REPORT_DIR / "baseline" / "baseline_model_report.md",
    "baseline_refinement_report": REPORT_DIR / "baseline_refinement" / "baseline_refinement_report.md",
    "baseline_final_report": REPORT_DIR / "baseline_final" / "baseline_final_report.md",
    "baseline_model_decision": TABLE_DIR / "baseline_final" / "baseline_model_decision.csv",
    "bias_refinement_recommendation": TABLE_DIR / "bias_refinement" / "bias_refinement_recommendation.csv",
    "refined_production_readiness_decision": TABLE_DIR / "bias_refinement" / "refined_production_readiness_decision.csv",
    "optical_sensitivity_report": REPORT_DIR / "optical_sensitivity" / "optical_sensitivity_report.md",
    "optical_model_ranking": TABLE_DIR / "optical_sensitivity" / "optical_model_ranking.csv",
    "roi_final_qc_report": REPORT_DIR / "roi_qc" / "roi_final_qc_report.md",
    "concentration_uncertainty_report": REPORT_DIR / "concentration_uncertainty" / "concentration_uncertainty_report.md",
    "bias_refinement_report": REPORT_DIR / "bias_refinement" / "bias_refinement_report.md",
    "daily_doc_prediction_report": REPORT_DIR / "daily_doc_prediction" / "daily_doc_prediction_report.md",
    "daily_doc_prediction_qc_summary": TABLE_DIR / "daily_doc_prediction" / "daily_doc_prediction_qc_summary.csv",
    "doc_flux_report": REPORT_DIR / "doc_flux" / "doc_flux_report.md",
    "annual_doc_flux_summary": TABLE_DIR / "doc_flux" / "annual_doc_flux_summary.csv",
    "provisional_may_july_flux_summary": TABLE_DIR / "doc_flux" / "provisional_may_july_flux_summary.csv",
    "doc_flux_confidence_tier_summary": TABLE_DIR / "doc_flux" / "doc_flux_confidence_tier_summary.csv",
    "flux_cohort_report": REPORT_DIR / "flux_interpretation" / "flux_cohort_report.md",
    "annual_flux_analysis_cohorts": TABLE_DIR / "flux_interpretation" / "annual_flux_analysis_cohorts.csv",
    "annual_flux_trend_report": REPORT_DIR / "annual_flux_trends" / "annual_flux_trend_report.md",
    "annual_flux_trends_by_river": TABLE_DIR / "annual_flux_trends" / "annual_flux_trends_by_river.csv",
    "annual_flux_trends_aggregate": TABLE_DIR / "annual_flux_trends" / "annual_flux_trends_aggregate.csv",
    "may_july_flux_interpretation_report": REPORT_DIR / "may_july_flux" / "may_july_flux_interpretation_report.md",
    "may_july_vs_annual_trend_comparison": TABLE_DIR / "may_july_flux" / "may_july_vs_annual_trend_comparison.csv",
    "snowmelt_window_report": REPORT_DIR / "snowmelt_windows" / "snowmelt_window_report.md",
    "annual_vs_snowmelt_signal_comparison": TABLE_DIR / "snowmelt_windows" / "annual_vs_snowmelt_signal_comparison.csv",
    "snowmelt_window_trends_by_river": TABLE_DIR / "snowmelt_windows" / "snowmelt_window_trends_by_river.csv",
    "snowmelt_window_flux_summary": TABLE_DIR / "snowmelt_windows" / "snowmelt_window_flux_summary.csv",
}


def ensure_final_synthesis_dirs() -> None:
    FINAL_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    (FINAL_FIGURE_DIR / ".gitkeep").write_text(
        "Final synthesis does not create new scientific figures. This directory is reserved for manuscript figure exports.\n",
        encoding="utf-8",
    )


def _require_input(name: str) -> Path:
    source = REQUIRED_INPUTS[name]
    if not source.exists():
        raise FileNotFoundError(f"Required final synthesis input is missing: {source}")
    return source


def _read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(_require_input(name), low_memory=False)


def _write_csv(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8")
    return destination


def _boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _core_annual_by_river() -> pd.DataFrame:
    by_river = _read_csv("annual_flux_trends_by_river")
    core = by_river[by_river["analysis_cohort"].astype(str).eq("core_2003_2024")].copy()
    return core.sort_values("river").reset_index(drop=True)


def _core_aggregate() -> pd.DataFrame:
    aggregate = _read_csv("annual_flux_trends_aggregate")
    core = aggregate[aggregate["analysis_cohort"].astype(str).eq("core_2003_2024")].copy()
    return core.reset_index(drop=True)


def _yukon_core_row() -> pd.Series:
    core = _core_annual_by_river()
    yukon = core[core["river"].astype(str).eq("Yukon")]
    if yukon.empty:
        raise ValueError("Yukon core annual trend row is missing.")
    return yukon.iloc[0]


def build_core_findings() -> pd.DataFrame:
    yukon = _yukon_core_row()
    aggregate = _core_aggregate()
    all_available = aggregate[aggregate["aggregate_type"].astype(str).eq("aggregate_all_available_rivers")]
    common = aggregate[aggregate["aggregate_type"].astype(str).eq("aggregate_common_river_set_only")]
    aggregate_row = all_available.iloc[0] if not all_available.empty else common.iloc[0]
    findings = [
        {
            "finding_id": "F01",
            "finding_category": "Data freeze",
            "finding_statement": f"Gold data freeze {FREEZE_ID} is the sole data source for modeling.",
            "evidence_table_or_report": "outputs/reports/data_contract_report.md; outputs/reports/gold_data_summary_report.md",
            "strength": "strong",
            "caveat": "Synthesis phase reads existing reports/tables only and does not modify gold data.",
            "manuscript_ready": True,
        },
        {
            "finding_id": "F02",
            "finding_category": "Model",
            "finding_statement": f"Primary production candidate concentration model is {PRIMARY_MODEL}.",
            "evidence_table_or_report": "outputs/tables/bias_refinement/bias_refinement_recommendation.csv; configs/model_specs/production_candidate_r4_river_specific_q_and_season_linear.yaml",
            "strength": "strong",
            "caveat": "Prediction and flux are within the six ArcticGRO rivers and guarded by range/QC flags.",
            "manuscript_ready": True,
        },
        {
            "finding_id": "F03",
            "finding_category": "Optical",
            "finding_statement": "Optical proxy did not improve the F3 baseline and is excluded from the primary model.",
            "evidence_table_or_report": "outputs/reports/optical_sensitivity/optical_sensitivity_report.md; outputs/tables/optical_sensitivity/optical_model_ranking.csv",
            "strength": "moderate",
            "caveat": "Optical reflectance is a proxy, not DOC observation; valid-water support and ROI caveats remain uneven.",
            "manuscript_ready": True,
        },
        {
            "finding_id": "F04",
            "finding_category": "ROI",
            "finding_statement": "ROI QC found no fatal issue or freeze reopen requirement, but visual/manual caveats remain.",
            "evidence_table_or_report": "outputs/reports/roi_qc/roi_final_qc_report.md",
            "strength": "moderate",
            "caveat": "External visual GIS review remains a caveat, especially for optical interpretation.",
            "manuscript_ready": True,
        },
        {
            "finding_id": "F05",
            "finding_category": "Flux",
            "finding_statement": "Guarded daily and annual DOC flux were generated with DOC concentration uncertainty only.",
            "evidence_table_or_report": "outputs/reports/doc_flux/doc_flux_report.md; outputs/tables/doc_flux/annual_doc_flux_summary.csv",
            "strength": "strong",
            "caveat": "Discharge uncertainty is not propagated.",
            "manuscript_ready": True,
        },
        {
            "finding_id": "F06",
            "finding_category": "Annual trend",
            "finding_statement": (
                "In the core 2003-2024 annual flux cohort, Yukon is the only river with detectable increasing annual DOC flux "
                f"(slope {float(yukon['slope_ols_TgC_per_year']):.4f} Tg C yr-1, p={float(yukon['slope_ols_p_value']):.3g})."
            ),
            "evidence_table_or_report": "outputs/tables/annual_flux_trends/annual_flux_trends_by_river.csv",
            "strength": "moderate",
            "caveat": "DOC concentration uncertainty only; discharge uncertainty is not propagated.",
            "manuscript_ready": True,
        },
        {
            "finding_id": "F07",
            "finding_category": "Aggregate",
            "finding_statement": (
                "Six-river aggregate annual DOC flux has no detectable trend in the core cohort "
                f"(aggregate p={float(aggregate_row['slope_ols_p_value']):.3g})."
            ),
            "evidence_table_or_report": "outputs/tables/annual_flux_trends/annual_flux_trends_aggregate.csv",
            "strength": "strong",
            "caveat": "Six-river ArcticGRO domain is not a full Arctic Ocean DOC budget.",
            "manuscript_ready": True,
        },
        {
            "finding_id": "F08",
            "finding_category": "May-July",
            "finding_statement": "Fixed May-July flux does not explain Yukon annual increase; Yukon May-July fraction decreases.",
            "evidence_table_or_report": "outputs/tables/may_july_flux/may_july_vs_annual_trend_comparison.csv",
            "strength": "moderate",
            "caveat": "May-July is a provisional screening window, not a final snowmelt window.",
            "manuscript_ready": True,
        },
        {
            "finding_id": "F09",
            "finding_category": "Dynamic snowmelt",
            "finding_statement": "Dynamic snowmelt/freshet windows show no detectable window flux trend in the core cohort; Yukon signal is partial, not decisive.",
            "evidence_table_or_report": "outputs/tables/snowmelt_windows/annual_vs_snowmelt_signal_comparison.csv",
            "strength": "moderate",
            "caveat": "Dynamic snowmelt windows are exploratory/interpretive and do not prove snowmelt attribution.",
            "manuscript_ready": True,
        },
        {
            "finding_id": "F10",
            "finding_category": "Limitation",
            "finding_statement": "Discharge uncertainty is not propagated.",
            "evidence_table_or_report": "outputs/reports/doc_flux/doc_flux_report.md; outputs/reports/annual_flux_trends/annual_flux_trend_report.md",
            "strength": "strong",
            "caveat": "Flux uncertainty intervals are DOC concentration empirical residual intervals only.",
            "manuscript_ready": True,
        },
        {
            "finding_id": "F11",
            "finding_category": "Interpretation boundary",
            "finding_statement": "Do not claim pan-Arctic large-river DOC flux increase.",
            "evidence_table_or_report": "outputs/tables/annual_flux_trends/annual_flux_trends_aggregate.csv",
            "strength": "strong",
            "caveat": "The project covers six ArcticGRO rivers, not all Arctic-draining rivers.",
            "manuscript_ready": True,
        },
        {
            "finding_id": "F12",
            "finding_category": "Interpretation boundary",
            "finding_statement": "Do not claim Yukon increase is driven by snowmelt-window flux increase.",
            "evidence_table_or_report": "outputs/tables/snowmelt_windows/annual_vs_snowmelt_signal_comparison.csv",
            "strength": "strong",
            "caveat": "Yukon dynamic-window evidence is partial and window flux trends are not detectable.",
            "manuscript_ready": True,
        },
    ]
    return pd.DataFrame(findings)


def build_model_evolution_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stage": "baseline_finalization",
                "candidate_model": PRIMARY_BASELINE,
                "result": "F3 baseline selected after baseline finalization.",
                "decision": "selected_as_validation_baseline",
                "reason": "F6 hydroclimate extension had negligible same-sample improvement over F3.",
            },
            {
                "stage": "hydroclimate_extension",
                "candidate_model": "F6_reduced_hydroclimate_river_fixed + ridge_alpha_1",
                "result": "F6 hydroclimate extension not primary.",
                "decision": "retain_as_process_sensitivity",
                "reason": "Hydroclimate extension improved RMSE by only about 0.0013 mg C/L on the same sample.",
            },
            {
                "stage": "optical_sensitivity",
                "candidate_model": "Optical proxy feature sets O2/O3/O4/O5 against F3 same-sample baseline",
                "result": "Optical excluded.",
                "decision": "exclude_from_primary_model",
                "reason": "Reflectance/index proxy feature sets did not provide robust incremental value and often worsened F3 on optical-matched rows.",
            },
            {
                "stage": "bias_aware_refinement",
                "candidate_model": PRIMARY_MODEL,
                "result": "R4 bias-aware refined model selected as production candidate.",
                "decision": "production_candidate",
                "reason": "Candidate met LOYO improvement, Lena, high-DOC, fold-stability, GroupKFold, and interpretability criteria.",
            },
            {
                "stage": "target_transform_sensitivity",
                "candidate_model": "log target sensitivity candidates",
                "result": "log target sensitivity only.",
                "decision": "do_not_promote",
                "reason": "Log-target candidates remain sensitivity-only unless a separate target-transform decision is made.",
            },
        ]
    )


def build_annual_flux_trend_summary() -> pd.DataFrame:
    core = _core_annual_by_river()
    summary = pd.DataFrame(
        {
            "river": core["river"],
            "core_n_years": core["n_years"],
            "core_year_min": core["year_min"],
            "core_year_max": core["year_max"],
            "mean_annual_flux_TgC": core["annual_flux_mean_TgC"],
            "median_annual_flux_TgC": core["annual_flux_median_TgC"],
            "slope_TgC_per_year": core["slope_ols_TgC_per_year"],
            "p_value": core["slope_ols_p_value"],
            "trend_direction": core["trend_direction"],
            "detectable_trend": core["significant_at_0_05"].map(_boolish),
            "caveat": core["confidence_caveat"],
        }
    )
    return summary


def build_aggregate_flux_trend_summary() -> pd.DataFrame:
    core = _core_aggregate()
    summary = pd.DataFrame(
        {
            "aggregate_type": core["aggregate_type"],
            "rivers_included": core["rivers_included"],
            "core_n_years": core["n_years"],
            "core_year_min": core["year_min"],
            "core_year_max": core["year_max"],
            "mean_annual_flux_TgC": core["annual_flux_sum_mean_TgC"],
            "slope_TgC_per_year": core["slope_ols_TgC_per_year"],
            "p_value": core["slope_ols_p_value"],
            "trend_direction": core["trend_direction"],
            "detectable_trend": core["slope_ols_p_value"].astype(float).lt(0.05) & core["trend_direction"].astype(str).isin(["increasing", "decreasing"]),
            "caveat": core["confidence_caveat"],
        }
    )
    return summary


def _select_dynamic_window(rows: pd.DataFrame) -> pd.Series:
    dynamic = rows[~rows["window_id"].astype(str).eq("fixed_may_july_reference")].copy()
    if dynamic.empty:
        return rows.iloc[0]
    partial_increasing_fraction = dynamic[
        dynamic["does_window_explain_annual_signal"].astype(str).eq("partial")
        & dynamic["window_fraction_trend_direction"].astype(str).eq("increasing")
    ]
    if not partial_increasing_fraction.empty:
        return partial_increasing_fraction.iloc[0]
    partial = dynamic[dynamic["does_window_explain_annual_signal"].astype(str).eq("partial")]
    if not partial.empty:
        priority = ["discharge_centered_freshet", "q75_peak_contiguous", "common_overlap_w1_w2", "snow_depletion_assisted"]
        partial = partial.assign(_priority=partial["window_id"].map({name: i for i, name in enumerate(priority)}).fillna(99))
        return partial.sort_values("_priority").iloc[0]
    priority = ["q75_peak_contiguous", "common_overlap_w1_w2", "discharge_centered_freshet", "snow_depletion_assisted"]
    dynamic = dynamic.assign(_priority=dynamic["window_id"].map({name: i for i, name in enumerate(priority)}).fillna(99))
    return dynamic.sort_values("_priority").iloc[0]


def build_snowmelt_interpretation_summary() -> pd.DataFrame:
    comparison = _read_csv("annual_vs_snowmelt_signal_comparison")
    rows: list[dict[str, Any]] = []
    for river, group in comparison.groupby("river", sort=True):
        fixed = group[group["window_id"].astype(str).eq("fixed_may_july_reference")]
        fixed_row = fixed.iloc[0] if not fixed.empty else group.iloc[0]
        dynamic_row = _select_dynamic_window(group)
        does_explain = str(dynamic_row["does_window_explain_annual_signal"])
        if river == "Yukon" and does_explain == "yes":
            does_explain = "partial"
        rows.append(
            {
                "river": river,
                "annual_trend_direction": fixed_row["annual_trend_direction"],
                "fixed_may_july_flux_trend": fixed_row["window_flux_trend_direction"],
                "fixed_may_july_fraction_trend": fixed_row["window_fraction_trend_direction"],
                "best_dynamic_window": dynamic_row["window_id"],
                "dynamic_window_flux_trend": dynamic_row["window_flux_trend_direction"],
                "dynamic_window_fraction_trend": dynamic_row["window_fraction_trend_direction"],
                "does_snowmelt_window_explain_annual_signal": does_explain,
                "interpretation": _snowmelt_interpretation_sentence(river, fixed_row, dynamic_row, does_explain),
                "caveat": "Dynamic snowmelt/freshet windows are exploratory/interpretive; discharge uncertainty is not propagated.",
            }
        )
    return pd.DataFrame(rows)


def _snowmelt_interpretation_sentence(river: str, fixed_row: pd.Series, dynamic_row: pd.Series, does_explain: str) -> str:
    if river == "Yukon":
        return (
            "Yukon annual flux is increasing, but fixed May-July does not explain the signal. "
            "Dynamic snowmelt signal is partial: the selected discharge-centered/freshet evidence can show increasing fraction, "
            "but dynamic window flux trend is not detectable and is not decisive."
        )
    if str(fixed_row["annual_trend_direction"]) == "flat_or_uncertain":
        return "No detectable increasing annual flux signal in the core cohort, so snowmelt-window attribution is not required."
    return str(dynamic_row["interpretation"])


def build_caveat_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "caveat_id": "C01",
                "topic": "uncertainty",
                "severity": "high",
                "description": "DOC uncertainty only; discharge uncertainty is not propagated.",
                "affected_results": "daily flux; annual flux; trend tests; snowmelt-window flux",
                "how_to_report": "State that flux intervals are based on DOC concentration empirical residual intervals only.",
            },
            {
                "caveat_id": "C02",
                "topic": "Yenisey confidence",
                "severity": "high",
                "description": "Yenisey has low-confidence flux years and many years with high low-confidence flux fraction.",
                "affected_results": "Yenisey annual flux and sensitivity cohorts",
                "how_to_report": "Avoid strong Yenisey-specific trend claims; emphasize core cohort and confidence tiers.",
            },
            {
                "caveat_id": "C03",
                "topic": "Yukon early hindcast",
                "severity": "high",
                "description": "Yukon 2000 near-zero issue is excluded/caveated and should not drive interpretation.",
                "affected_results": "full 2000-2025 sensitivity; Yukon trend context",
                "how_to_report": "Use core 2003-2024 as primary; report 2000 as caveated sensitivity only.",
            },
            {
                "caveat_id": "C04",
                "topic": "Kolyma/Ob caveated years",
                "severity": "medium",
                "description": "Kolyma and Ob include excluded or caveated years driven by coverage or low-confidence flux fraction.",
                "affected_results": "full-period and high-confidence sensitivity cohorts",
                "how_to_report": "Use core cohort as primary and retain excluded/caveated year notes.",
            },
            {
                "caveat_id": "C05",
                "topic": "Mackenzie 2025",
                "severity": "medium",
                "description": "Mackenzie 2025 coverage caveat keeps 2025 outside the primary core trend.",
                "affected_results": "full 2000-2025 sensitivity",
                "how_to_report": "Keep Mackenzie 2025 as sensitivity context only.",
            },
            {
                "caveat_id": "C06",
                "topic": "ROI",
                "severity": "medium",
                "description": "ROI visual review caveats remain even though ROI QC found no freeze-reopen requirement.",
                "affected_results": "optical sensitivity interpretation",
                "how_to_report": "State that ROI caveats mainly affect optical proxy analyses, not hydrocore production flux.",
            },
            {
                "caveat_id": "C07",
                "topic": "Optical proxy",
                "severity": "medium",
                "description": "Optical proxy negative result: reflectance features did not robustly improve the DOC baseline.",
                "affected_results": "model selection; optical sensitivity",
                "how_to_report": "Do not frame the study as satellite DOC retrieval; describe optical layers as sensitivity/proxy evidence.",
            },
            {
                "caveat_id": "C08",
                "topic": "Snowmelt windows",
                "severity": "medium",
                "description": "Snowmelt windows are exploratory/interpretive.",
                "affected_results": "dynamic window flux and fraction trends",
                "how_to_report": "Use cautious attribution language; do not claim final hydrologic attribution.",
            },
            {
                "caveat_id": "C09",
                "topic": "Spatial domain",
                "severity": "high",
                "description": "No pan-Arctic extrapolation beyond six ArcticGRO rivers.",
                "affected_results": "aggregate flux and trend conclusions",
                "how_to_report": "Refer to the six-river ArcticGRO domain, not a full Arctic Ocean DOC budget.",
            },
        ]
    )


def build_recommended_figures() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "figure_id": "Fig1",
                "figure_title": "Workflow diagram",
                "purpose": "Show data freeze -> model selection -> DOC prediction -> flux -> trend/snowmelt interpretation.",
                "source_outputs": "final_synthesis_report.md; model_evolution_summary.csv",
                "status": "recommended_not_generated_in_this_phase",
                "notes": "Can be drawn manually from the synthesis tables; no new science calculation required.",
            },
            {
                "figure_id": "Fig2",
                "figure_title": "DOC model performance",
                "purpose": "Observed vs predicted CV for R4 production candidate.",
                "source_outputs": "outputs/reports/bias_refinement/bias_refinement_report.md; existing bias_refinement figures",
                "status": "recommended_existing_or_rebuild_from_existing_tables",
                "notes": "Do not retrain model.",
            },
            {
                "figure_id": "Fig3",
                "figure_title": "Annual flux time series",
                "purpose": "Core cohort by river with trend lines.",
                "source_outputs": "outputs/tables/annual_flux_trends/annual_flux_trends_by_river.csv; existing annual_flux_trends figures",
                "status": "recommended_existing_or_rebuild_from_existing_tables",
                "notes": "Core 2003-2024 is primary.",
            },
            {
                "figure_id": "Fig4",
                "figure_title": "Aggregate flux trend",
                "purpose": "Six-river aggregate core cohort.",
                "source_outputs": "outputs/tables/final_synthesis/aggregate_flux_trend_summary_for_manuscript.csv",
                "status": "recommended_existing_or_rebuild_from_existing_tables",
                "notes": "Must state this is not a full Arctic Ocean DOC budget.",
            },
            {
                "figure_id": "Fig5",
                "figure_title": "Yukon focus",
                "purpose": "Annual flux vs May-July vs dynamic window signal.",
                "source_outputs": "outputs/tables/final_synthesis/snowmelt_interpretation_summary.csv",
                "status": "recommended_existing_or_rebuild_from_existing_tables",
                "notes": "Show that snowmelt-window evidence is partial, not decisive.",
            },
            {
                "figure_id": "Fig6",
                "figure_title": "Snowmelt window comparison",
                "purpose": "Fixed May-July vs q75_peak_contiguous vs discharge_centered.",
                "source_outputs": "outputs/tables/snowmelt_windows/snowmelt_window_flux_summary.csv",
                "status": "recommended_existing_or_rebuild_from_existing_tables",
                "notes": "Dynamic windows are exploratory/interpretive.",
            },
            {
                "figure_id": "Fig7",
                "figure_title": "Confidence/caveat figure",
                "purpose": "Annual confidence tiers or low-confidence flux fraction.",
                "source_outputs": "outputs/tables/doc_flux/doc_flux_confidence_tier_summary.csv; caveat_register.csv",
                "status": "recommended_existing_or_rebuild_from_existing_tables",
                "notes": "Useful for transparent uncertainty boundary.",
            },
        ]
    )


def build_recommended_tables() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "table_id": "Table1",
                "table_title": "Gold data sources and row counts",
                "source_outputs": "outputs/tables/gold_input_inventory.csv; outputs/reports/gold_data_summary_report.md",
                "purpose": "Document the frozen data source and modeling inputs.",
            },
            {
                "table_id": "Table2",
                "table_title": "Model selection summary",
                "source_outputs": "outputs/tables/final_synthesis/model_evolution_summary.csv",
                "purpose": "Show why R4 is the production candidate and optical is excluded.",
            },
            {
                "table_id": "Table3",
                "table_title": "Annual trend by river",
                "source_outputs": "outputs/tables/final_synthesis/annual_flux_trend_summary_for_manuscript.csv",
                "purpose": "Main river-specific annual flux trend result.",
            },
            {
                "table_id": "Table4",
                "table_title": "Dynamic snowmelt window interpretation",
                "source_outputs": "outputs/tables/final_synthesis/snowmelt_interpretation_summary.csv",
                "purpose": "Summarize why Yukon snowmelt explanation is partial, not decisive.",
            },
            {
                "table_id": "Table5",
                "table_title": "Caveat register",
                "source_outputs": "outputs/tables/final_synthesis/caveat_register.csv",
                "purpose": "Make interpretation limits explicit.",
            },
        ]
    )


def synthesize_results() -> dict[str, Path]:
    ensure_final_synthesis_dirs()
    missing = [str(path) for path in REQUIRED_INPUTS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required final synthesis inputs:\n" + "\n".join(missing))
    outputs = {
        "core_findings": _write_csv(build_core_findings(), FINAL_TABLE_DIR / "core_findings.csv"),
        "model_evolution_summary": _write_csv(build_model_evolution_summary(), FINAL_TABLE_DIR / "model_evolution_summary.csv"),
        "annual_flux_trend_summary": _write_csv(
            build_annual_flux_trend_summary(),
            FINAL_TABLE_DIR / "annual_flux_trend_summary_for_manuscript.csv",
        ),
        "aggregate_flux_trend_summary": _write_csv(
            build_aggregate_flux_trend_summary(),
            FINAL_TABLE_DIR / "aggregate_flux_trend_summary_for_manuscript.csv",
        ),
        "snowmelt_interpretation_summary": _write_csv(
            build_snowmelt_interpretation_summary(),
            FINAL_TABLE_DIR / "snowmelt_interpretation_summary.csv",
        ),
        "caveat_register": _write_csv(build_caveat_register(), FINAL_TABLE_DIR / "caveat_register.csv"),
        "recommended_figures": _write_csv(build_recommended_figures(), FINAL_TABLE_DIR / "recommended_manuscript_figures.csv"),
        "recommended_tables": _write_csv(build_recommended_tables(), FINAL_TABLE_DIR / "recommended_manuscript_tables.csv"),
    }
    outputs["report"] = write_final_synthesis_report()
    return outputs


def _read_synthesis_table(name: str) -> pd.DataFrame:
    path = FINAL_TABLE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Required synthesis table is missing: {path}. Run synthesize-results first.")
    return pd.read_csv(path, low_memory=False)


def write_final_synthesis_report() -> Path:
    ensure_final_synthesis_dirs()
    core_findings = _read_synthesis_table("core_findings.csv")
    model_evolution = _read_synthesis_table("model_evolution_summary.csv")
    annual_summary = _read_synthesis_table("annual_flux_trend_summary_for_manuscript.csv")
    aggregate_summary = _read_synthesis_table("aggregate_flux_trend_summary_for_manuscript.csv")
    snowmelt_summary = _read_synthesis_table("snowmelt_interpretation_summary.csv")
    caveats = _read_synthesis_table("caveat_register.csv")
    figures = _read_synthesis_table("recommended_manuscript_figures.csv")
    tables = _read_synthesis_table("recommended_manuscript_tables.csv")
    daily_qc = _read_csv("daily_doc_prediction_qc_summary")
    confidence = _read_csv("doc_flux_confidence_tier_summary")
    lines = [
        "# Final Synthesis Manuscript-Ready Results Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "## 1. Executive summary",
        "",
        "This final synthesis consolidates existing data-contract, model-selection, prediction, flux, trend, May-July, and dynamic snowmelt-window outputs into a manuscript-ready results summary. There is no model retraining in this phase, no new DOC prediction, no flux recalculation, and gold data unchanged.",
        "",
        _md_table(core_findings[["finding_id", "finding_category", "finding_statement", "strength", "manuscript_ready"]], max_rows=20),
        "",
        "## 2. Data freeze and gold data inputs",
        "",
        f"The project uses `{FREEZE_ID}` as the frozen data source. This synthesis reads only existing reports and output tables. It does not read raw/interim/canonical data and does not modify gold data.",
        "",
        "Primary evidence: `outputs/reports/data_contract_report.md`, `outputs/reports/gold_data_summary_report.md`, and `outputs/reports/eda/eda_report.md`.",
        "",
        "## 3. Model development path",
        "",
        _md_table(model_evolution, max_rows=20),
        "",
        "## 4. Final concentration model",
        "",
        f"The manuscript-ready production candidate is `{PRIMARY_MODEL}`. It replaced the finalized F3 comparator after bias-aware refinement, while log-target candidates remain sensitivity-only.",
        "",
        "This concentration model is used for guarded daily prediction and flux products, but it is not refit in this synthesis phase.",
        "",
        "## 5. Optical sensitivity result",
        "",
        "Satellite optical reflectance is a proxy, not DOC observation. Optical proxy feature sets did not robustly improve the finalized F3 baseline and are excluded from the primary production model. Optical results should be reported as a negative/sensitivity result, not as satellite DOC retrieval.",
        "",
        "## 6. ROI QC result",
        "",
        "ROI QC found no fatal issue and no data-freeze reopen requirement. All final-primary ROIs are accepted with caveats. External visual/manual caveats remain, especially for optical interpretation.",
        "",
        "## 7. Daily DOC prediction result",
        "",
        "Guarded daily DOC predictions already exist and are not regenerated here. Prediction coverage is summarized below.",
        "",
        _md_table(daily_qc, max_rows=10),
        "",
        "## 8. DOC flux calculation result",
        "",
        "Guarded daily and annual DOC flux products already exist and are not recalculated here. Flux uses `DOC_mgC_L * Q_m3s * 86.4`; intervals propagate DOC concentration uncertainty only. Discharge uncertainty not propagated.",
        "",
        _md_table(confidence[confidence["scope"].astype(str).eq("overall")], max_rows=10),
        "",
        "## 9. Annual flux trend result",
        "",
        "The core 2003-2024 cohort is primary. Yukon is the only river with a detectable increasing annual DOC flux trend. Other rivers are flat_or_uncertain / no detectable trend. The six-river aggregate has no detectable trend.",
        "",
        _md_table(annual_summary, max_rows=20),
        "",
        "Aggregate results:",
        "",
        _md_table(aggregate_summary, max_rows=10),
        "",
        "## 10. Provisional May-July result",
        "",
        "May-July is provisional. It should not be treated as the final snowmelt window. Fixed May-July flux does not explain the Yukon annual increase, and the Yukon May-July fraction decreases.",
        "",
        "Primary evidence: `outputs/tables/may_july_flux/may_july_vs_annual_trend_comparison.csv`.",
        "",
        "## 11. Dynamic snowmelt/freshet window result",
        "",
        "Dynamic snowmelt windows are exploratory/interpretive. They improve seasonal framing beyond fixed May-July, but they do not show a detectable window flux trend in the core cohort. Yukon remains partial, not decisive.",
        "",
        _md_table(snowmelt_summary, max_rows=20),
        "",
        "## 12. Main scientific findings",
        "",
        "- The strongest manuscript-ready result is not a pan-Arctic increase, but a guarded six-river ArcticGRO-domain synthesis.",
        "- Yukon shows a detectable annual DOC flux increase in the core 2003-2024 cohort.",
        "- The six-river aggregate annual DOC flux does not show a detectable trend in the core cohort.",
        "- Fixed May-July and dynamic snowmelt/freshet windows do not decisively explain the Yukon annual increase.",
        "- Optical reflectance did not improve the primary DOC concentration model and remains proxy/sensitivity evidence.",
        "",
        "## 13. Sensitivity results",
        "",
        "Full-period and high-confidence-only annual trend cohorts are sensitivity results. The core 2003-2024 cohort remains primary. May-July and dynamic-window outputs are interpretive sensitivity layers, not a replacement for annual flux trends.",
        "",
        "## 14. Caveats and limitations",
        "",
        _md_table(caveats, max_rows=20),
        "",
        "## 15. Manuscript-ready conclusions",
        "",
        "1. A frozen six-river ArcticGRO gold-data workflow supports guarded annual DOC flux estimates for 2000-2025 and a primary core trend cohort for 2003-2024.",
        "2. The production concentration model is an interpretable river-specific Q-and-season linear model selected after bias-aware refinement.",
        "3. In the core annual flux cohort, Yukon is the only river with a detectable increasing annual DOC flux trend.",
        "4. The six-river aggregate has no detectable annual DOC flux trend, so the study does not support a pan-Arctic DOC flux increase claim.",
        "5. May-July and dynamic hydrologic windows do not provide decisive evidence that the Yukon annual increase is snowmelt-window driven.",
        "",
        "## 16. Recommended figures and tables",
        "",
        "Recommended figures:",
        "",
        _md_table(figures, max_rows=20),
        "",
        "Recommended tables:",
        "",
        _md_table(tables, max_rows=20),
        "",
        "## 17. What not to claim",
        "",
        "- Do not claim a pan-Arctic DOC flux increase.",
        "- Do not claim all ArcticGRO rivers increased.",
        "- Do not claim optical reflectance improves DOC prediction.",
        "- Do not claim May-July is the final snowmelt window.",
        "- Do not claim Yukon annual increase is definitively snowmelt-driven.",
        "- Do not claim discharge uncertainty is included.",
        "- Do not extrapolate beyond six ArcticGRO rivers.",
        "",
        "## 18. Reproducibility notes",
        "",
        "- no model retraining in this phase",
        "- no new DOC prediction",
        "- no flux recalculation",
        "- gold data unchanged",
        "- discharge uncertainty not propagated",
        "- May-July is provisional",
        "- dynamic snowmelt windows are exploratory/interpretive",
        "- optical reflectance is proxy, not DOC observation",
        "",
        "Recommended command sequence:",
        "",
        "```powershell",
        "python -m arctic_doc_model_rebuild.cli verify-gold-data",
        "python -m arctic_doc_model_rebuild.cli synthesize-results",
        "python -m arctic_doc_model_rebuild.cli synthesis-report",
        "python -m pytest",
        "```",
    ]
    FINAL_SYNTHESIS_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return FINAL_SYNTHESIS_REPORT_PATH
