from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .gold_contract import sha256_file
from .paths import REPORT_DIR, TABLE_DIR
from .reports import _md_table, utc_now


MANUSCRIPT_REPORT_DIR = REPORT_DIR / "manuscript"
MANUSCRIPT_TABLE_DIR = TABLE_DIR / "manuscript"

MANUSCRIPT_OUTLINE_PATH = MANUSCRIPT_REPORT_DIR / "manuscript_outline.md"
RESULTS_NARRATIVE_PATH = MANUSCRIPT_REPORT_DIR / "results_narrative_draft.md"
ABSTRACT_DRAFT_PATH = MANUSCRIPT_REPORT_DIR / "abstract_draft.md"
CLAIMS_TO_AVOID_PATH = MANUSCRIPT_REPORT_DIR / "claims_to_avoid.md"

KEY_CLAIMS_PATH = MANUSCRIPT_TABLE_DIR / "key_claims_to_evidence_map.csv"
FIGURE_PLAN_PATH = MANUSCRIPT_TABLE_DIR / "figure_plan.csv"
TABLE_PLAN_PATH = MANUSCRIPT_TABLE_DIR / "table_plan.csv"

REQUIRED_MANUSCRIPT_INPUTS = {
    "final_synthesis_report": REPORT_DIR / "final_synthesis" / "final_synthesis_report.md",
    "flux_attribution_report": REPORT_DIR / "flux_attribution" / "flux_attribution_report.md",
    "yukon_flux_attribution_report": REPORT_DIR / "flux_attribution" / "yukon_flux_attribution_report.md",
    "freshet_control_report": REPORT_DIR / "freshet_control" / "freshet_control_synthesis_report.md",
    "annual_flux_trend_report": REPORT_DIR / "annual_flux_trends" / "annual_flux_trend_report.md",
    "snowmelt_window_report": REPORT_DIR / "snowmelt_windows" / "snowmelt_window_report.md",
    "export_regime_classification": TABLE_DIR / "freshet_control" / "export_regime_classification.csv",
    "flux_driver_classification": TABLE_DIR / "flux_attribution" / "flux_driver_classification.csv",
    "core_findings": TABLE_DIR / "final_synthesis" / "core_findings.csv",
    "caveat_register": TABLE_DIR / "final_synthesis" / "caveat_register.csv",
    "recommended_manuscript_figures": TABLE_DIR / "final_synthesis" / "recommended_manuscript_figures.csv",
    "recommended_manuscript_tables": TABLE_DIR / "final_synthesis" / "recommended_manuscript_tables.csv",
}

TITLE_OPTIONS = [
    "River-Specific Arctic DOC Export Change Revealed by Guarded Daily Concentration and Flux Reconstruction",
    "Discharge-Driven Extended-Season DOC Export in the Yukon River Within a Six-River ArcticGRO Synthesis",
    "Freshet Control, Export Phenology, and River-Specific DOC Flux Trends Across Six Arctic Rivers",
    "A Frozen-Data Rebuild of Arctic River DOC Flux Points to Yukon-Specific Extended-Season Export",
]

CENTRAL_STORY = (
    "Using a frozen, audited ArcticGRO gold dataset and a guarded daily DOC concentration model, the analysis finds a "
    "detectable annual DOC flux increase only for Yukon, not for the six-river aggregate. The Yukon signal is best framed "
    "as discharge-volume-driven extended-season export expansion, while several other rivers remain freshet-dominated but stable."
)


def _ensure_dirs() -> None:
    MANUSCRIPT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _read_csv(name: str) -> pd.DataFrame:
    source = REQUIRED_MANUSCRIPT_INPUTS[name]
    if not source.exists():
        raise FileNotFoundError(f"Required manuscript input is missing: {source}")
    return pd.read_csv(source, low_memory=False)


def _read_text(name: str) -> str:
    source = REQUIRED_MANUSCRIPT_INPUTS[name]
    if not source.exists():
        raise FileNotFoundError(f"Required manuscript input is missing: {source}")
    return source.read_text(encoding="utf-8")


def _write_csv(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8")
    return destination


def _input_hashes() -> dict[Path, str]:
    return {destination: sha256_file(destination) for destination in REQUIRED_MANUSCRIPT_INPUTS.values()}


def _verify_inputs_unchanged(before: dict[Path, str]) -> None:
    after = _input_hashes()
    changed = [str(destination) for destination, before_hash in before.items() if after.get(destination) != before_hash]
    if changed:
        raise RuntimeError(f"Manuscript outline inputs changed during drafting: {changed}")


def _load_inputs() -> dict[str, Any]:
    return {
        "core_findings": _read_csv("core_findings"),
        "caveats": _read_csv("caveat_register"),
        "figure_recommendations": _read_csv("recommended_manuscript_figures"),
        "table_recommendations": _read_csv("recommended_manuscript_tables"),
        "regimes": _read_csv("export_regime_classification"),
        "drivers": _read_csv("flux_driver_classification"),
        "final_synthesis_text": _read_text("final_synthesis_report"),
        "flux_attribution_text": _read_text("flux_attribution_report"),
        "yukon_attribution_text": _read_text("yukon_flux_attribution_report"),
        "freshet_control_text": _read_text("freshet_control_report"),
        "annual_trend_text": _read_text("annual_flux_trend_report"),
        "snowmelt_window_text": _read_text("snowmelt_window_report"),
    }


def build_key_claims_to_evidence_map(inputs: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "claim": "The project uses a frozen audited gold data freeze as the sole modeling data source.",
            "evidence_table": "outputs/tables/final_synthesis/core_findings.csv; outputs/reports/data_contract_report.md",
            "figure_candidate": "Figure 1 workflow/data/model",
            "strength": "strong",
            "caveat": "Do not imply the model repository owns or modifies the gold data.",
            "manuscript_section": "Methods",
        },
        {
            "claim": "The final concentration model is a guarded R4 river-specific Q and season linear model.",
            "evidence_table": "outputs/tables/final_synthesis/model_evolution_summary.csv; configs/model_specs/production_candidate_r4_river_specific_q_and_season_linear.yaml",
            "figure_candidate": "Figure 2 model performance",
            "strength": "strong",
            "caveat": "Within-six-river ArcticGRO domain only; validation and range flags remain relevant.",
            "manuscript_section": "Methods / Results",
        },
        {
            "claim": "Yukon is the only core 2003-2024 river with a detectable increasing annual DOC flux trend.",
            "evidence_table": "outputs/tables/final_synthesis/core_findings.csv; outputs/tables/annual_flux_trends/annual_flux_trends_by_river.csv",
            "figure_candidate": "Figure 3 annual flux trends",
            "strength": "moderate",
            "caveat": "DOC concentration uncertainty only; discharge uncertainty not propagated.",
            "manuscript_section": "Results",
        },
        {
            "claim": "The six-river aggregate core annual DOC flux has no detectable trend.",
            "evidence_table": "outputs/tables/final_synthesis/aggregate_flux_trend_summary_for_manuscript.csv",
            "figure_candidate": "Figure 3 annual flux trends",
            "strength": "strong",
            "caveat": "This is not a pan-Arctic DOC budget or all-river statement.",
            "manuscript_section": "Results",
        },
        {
            "claim": "Yukon annual increase is best interpreted as discharge-volume-driven extended-season export expansion.",
            "evidence_table": "outputs/tables/flux_attribution/flux_driver_classification.csv; outputs/tables/freshet_control/yukon_extended_season_diagnosis.csv",
            "figure_candidate": "Figure 4 Yukon attribution; Figure 5 export phenology",
            "strength": "moderate",
            "caveat": "Exploratory mechanism analysis, not causal proof.",
            "manuscript_section": "Results / Discussion",
        },
        {
            "claim": "Fixed May-July does not explain the Yukon annual increase, and May-July remains provisional.",
            "evidence_table": "outputs/tables/may_july_flux/may_july_vs_annual_trend_comparison.csv; outputs/reports/snowmelt_windows/snowmelt_window_report.md",
            "figure_candidate": "Figure 4 Yukon attribution",
            "strength": "moderate",
            "caveat": "Do not call fixed May-July final snowmelt flux.",
            "manuscript_section": "Results",
        },
        {
            "claim": "Several rivers are freshet-dominated but stable, while Yukon is classified as extended-season.",
            "evidence_table": "outputs/tables/freshet_control/export_regime_classification.csv",
            "figure_candidate": "Figure 6 regime classification",
            "strength": "moderate",
            "caveat": "Regime labels are operational synthesis categories.",
            "manuscript_section": "Discussion",
        },
        {
            "claim": "Optical proxy features did not improve the F3 baseline and are excluded from the primary model.",
            "evidence_table": "outputs/reports/optical_sensitivity/optical_sensitivity_report.md; outputs/tables/final_synthesis/core_findings.csv",
            "figure_candidate": "Figure 7 caveats/sensitivity",
            "strength": "moderate",
            "caveat": "Optical reflectance is a proxy, not DOC observation; ROI caveats remain.",
            "manuscript_section": "Discussion / Limitations",
        },
        {
            "claim": "Flux intervals propagate DOC concentration uncertainty only.",
            "evidence_table": "outputs/tables/final_synthesis/caveat_register.csv; outputs/reports/doc_flux/doc_flux_report.md",
            "figure_candidate": "Figure 7 caveats/sensitivity",
            "strength": "strong",
            "caveat": "Discharge uncertainty is not propagated.",
            "manuscript_section": "Limitations",
        },
    ]
    return pd.DataFrame(rows)


def build_figure_plan(inputs: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "figure_id": "Figure 1",
            "figure_title": "Workflow, frozen data, and guarded modeling chain",
            "source_outputs": "outputs/reports/final_synthesis/final_synthesis_report.md; outputs/tables/final_synthesis/model_evolution_summary.csv",
            "main_message": "Data freeze to model selection to daily DOC prediction to guarded flux and interpretation.",
            "panel_suggestions": "A data freeze schematic; B model evolution; C prediction/flux guardrails.",
            "status": "draft_from_existing_outputs",
        },
        {
            "figure_id": "Figure 2",
            "figure_title": "DOC concentration model performance and selection",
            "source_outputs": "outputs/reports/bias_refinement/bias_refinement_report.md; existing bias_refinement figures",
            "main_message": "R4 improves validation diagnostics enough to serve as guarded production candidate.",
            "panel_suggestions": "Observed-vs-CV predicted; river residuals; high-DOC residual review; readiness decision.",
            "status": "use_existing_or_rebuild_from_existing_tables",
        },
        {
            "figure_id": "Figure 3",
            "figure_title": "Annual DOC flux trends across six ArcticGRO rivers",
            "source_outputs": "outputs/tables/annual_flux_trends/annual_flux_trends_by_river.csv; outputs/figures/annual_flux_trends/",
            "main_message": "Yukon increases; aggregate and other rivers show no detectable trend in the core cohort.",
            "panel_suggestions": "River time series; slope by river; aggregate core trend; uncertainty sensitivity.",
            "status": "use_existing_or_rebuild_from_existing_tables",
        },
        {
            "figure_id": "Figure 4",
            "figure_title": "Yukon attribution: discharge volume, DOC, and seasonal export",
            "source_outputs": "outputs/tables/flux_attribution/flux_driver_classification.csv; outputs/figures/flux_attribution/yukon_flux_Q_DOC_decomposition.png",
            "main_message": "Yukon signal is discharge-volume dominated, not flow-weighted DOC dominated.",
            "panel_suggestions": "Annual flux/Q/DOC decomposition; monthly/seasonal contribution trends; May-July comparison.",
            "status": "use_existing_flux_attribution_outputs",
        },
        {
            "figure_id": "Figure 5",
            "figure_title": "Export phenology and extended-season signal",
            "source_outputs": "outputs/tables/flux_attribution/export_phenology_trends_by_river.csv; outputs/figures/freshet_control/",
            "main_message": "Yukon export timing shifts later and after-July export fraction increases.",
            "panel_suggestions": "Centroid slopes; after-July fraction slopes; Yukon cumulative timing dashboard.",
            "status": "use_existing_freshet_control_outputs",
        },
        {
            "figure_id": "Figure 6",
            "figure_title": "DOC export regime classification",
            "source_outputs": "outputs/tables/freshet_control/export_regime_classification.csv; outputs/figures/freshet_control/export_regime_summary.png",
            "main_message": "Most rivers are freshet-dominated stable; Yukon is extended-season discharge-volume export.",
            "panel_suggestions": "Regime map/table; freshet fraction by river; annual-window coupling categories.",
            "status": "use_existing_freshet_control_outputs",
        },
        {
            "figure_id": "Figure 7",
            "figure_title": "Caveats, sensitivity, and interpretation boundaries",
            "source_outputs": "outputs/tables/final_synthesis/caveat_register.csv; outputs/tables/manuscript/key_claims_to_evidence_map.csv",
            "main_message": "Results are guarded by uncertainty, coverage, cohort, and domain boundaries.",
            "panel_suggestions": "Caveat register; confidence tiers; claims-to-avoid callouts.",
            "status": "draft_from_existing_outputs",
        },
    ]
    return pd.DataFrame(rows)


def build_table_plan(inputs: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "table_id": "Table 1",
            "table_title": "Frozen data sources and model inputs",
            "source_outputs": "outputs/tables/gold_input_inventory.csv; outputs/reports/data_contract_report.md",
            "purpose": "Document gold freeze provenance, tables, row counts, and hashes.",
        },
        {
            "table_id": "Table 2",
            "table_title": "DOC concentration model selection and guardrails",
            "source_outputs": "outputs/tables/final_synthesis/model_evolution_summary.csv; outputs/tables/bias_refinement/bias_refinement_recommendation.csv",
            "purpose": "Show progression from F3 baseline to R4 production candidate and optical exclusion.",
        },
        {
            "table_id": "Table 3",
            "table_title": "Annual DOC flux trend by river",
            "source_outputs": "outputs/tables/final_synthesis/annual_flux_trend_summary_for_manuscript.csv",
            "purpose": "Primary trend evidence for Yukon-only detectable annual increase.",
        },
        {
            "table_id": "Table 4",
            "table_title": "Flux attribution and DOC export regime classification",
            "source_outputs": "outputs/tables/flux_attribution/flux_driver_classification.csv; outputs/tables/freshet_control/export_regime_classification.csv",
            "purpose": "Connect annual trends to Q volume, flow-weighted DOC, export timing, and regime labels.",
        },
        {
            "table_id": "Table 5",
            "table_title": "Caveats and claims boundary register",
            "source_outputs": "outputs/tables/final_synthesis/caveat_register.csv; outputs/reports/manuscript/claims_to_avoid.md",
            "purpose": "Make uncertainty, domain, optical, and snowmelt interpretation limits explicit.",
        },
    ]
    return pd.DataFrame(rows)


def _regime_sentence(regimes: pd.DataFrame) -> str:
    labels = [f"{row.river}: `{row.assigned_regime}`" for row in regimes.itertuples(index=False)]
    return "; ".join(labels)


def write_manuscript_outline(inputs: dict[str, Any], figure_plan: pd.DataFrame, table_plan: pd.DataFrame) -> Path:
    core = inputs["core_findings"]
    caveats = inputs["caveats"]
    regimes = inputs["regimes"]
    lines = [
        "# Manuscript Outline",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Title options",
        "",
        *[f"{idx}. {title}" for idx, title in enumerate(TITLE_OPTIONS, start=1)],
        "",
        "## Abstract skeleton",
        "",
        "- Background: Arctic river DOC export is sensitive to changing discharge, seasonality, and snowmelt/freshet timing, but trend attribution is limited by data integration and model uncertainty.",
        "- Data/model: Use frozen gold data freeze `data_freeze_gold_20260526_v1`, guarded DOC concentration modeling, daily DOC prediction, and DOC-concentration-only flux intervals.",
        "- Main result: Yukon has a detectable increasing annual DOC flux trend in the core 2003-2024 cohort; the six-river aggregate has no detectable trend.",
        "- Mechanism result: Yukon is best framed as discharge-volume-driven extended-season export expansion, not fixed May-July intensification.",
        "- Limitation: Discharge uncertainty is not propagated and results are limited to six ArcticGRO rivers.",
        "",
        "## Introduction logic",
        "",
        "1. Arctic river DOC export matters for carbon delivery to coastal Arctic systems.",
        "2. Warming can alter discharge volume, freshet timing, season length, and concentration dynamics.",
        "3. Existing work often emphasizes freshet/snowmelt, but annual export may also shift through shoulder-season discharge.",
        "4. A guarded frozen-data rebuild can separate model, prediction, flux, trend, and attribution decisions.",
        "5. The manuscript asks whether annual DOC flux change is aggregate-wide, river-specific, freshet-controlled, or extended-season.",
        "",
        "## Methods outline",
        "",
        "- Frozen data contract and gold data provenance.",
        "- DOC concentration model selection: F3 baseline, bias-aware R4 production candidate, optical sensitivity exclusion.",
        "- Guarded daily DOC prediction and flux calculation with empirical DOC residual intervals.",
        "- Cohort selection for annual trends and sensitivity checks.",
        "- Annual trend methods and uncertainty sensitivity.",
        "- Flux attribution: Q volume, annual mean DOC, flow-weighted DOC, monthly/seasonal decomposition.",
        "- Snowmelt/freshet operational windows and export phenology diagnostics.",
        "- Claims boundary and caveat register.",
        "",
        "## Results outline",
        "",
        "1. The frozen-data and model rebuild produced guarded daily DOC and flux products.",
        "2. Yukon is the only core river with detectable increasing annual DOC flux; aggregate flux has no detectable trend.",
        "3. Fixed May-July does not explain Yukon annual increase.",
        "4. Flux attribution identifies Yukon as discharge-volume dominated with no detectable flow-weighted DOC trend.",
        "5. Export phenology indicates later and extended-season Yukon export.",
        "6. Regime classification separates freshet-dominated stable rivers from Yukon extended-season export.",
        "7. Optical proxy evidence is negative for primary model improvement.",
        "",
        "## Discussion outline",
        "",
        "- The strongest story is river-specific change, not pan-Arctic or aggregate increase.",
        "- Yukon may represent discharge-driven expansion of the export season rather than a simple freshet intensification.",
        "- Freshet dominance can remain important without producing detectable annual trend.",
        "- Optical non-improvement constrains remote-sensing interpretation and supports hydrocore-first modeling.",
        "- Guardrail-first analysis improves reproducibility but narrows the inferential domain.",
        "",
        "## Limitations",
        "",
        _md_table(caveats, max_rows=20),
        "",
        "## Conclusion",
        "",
        CENTRAL_STORY,
        "",
        "## Proposed figures",
        "",
        _md_table(figure_plan, max_rows=20),
        "",
        "## Proposed tables",
        "",
        _md_table(table_plan, max_rows=20),
        "",
        "## Core evidence snapshot",
        "",
        _md_table(core[["finding_id", "finding_category", "finding_statement", "strength"]], max_rows=20),
        "",
        "## Regime labels",
        "",
        _regime_sentence(regimes),
    ]
    MANUSCRIPT_OUTLINE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return MANUSCRIPT_OUTLINE_PATH


def write_results_narrative(inputs: dict[str, Any]) -> Path:
    regimes = inputs["regimes"]
    drivers = inputs["drivers"]
    yukon = drivers[drivers["river"].astype(str).eq("Yukon")].iloc[0]
    lines = [
        "# Results Narrative Draft",
        "",
        f"Generated: {utc_now()}",
        "",
        "The analysis begins from a frozen gold data release, `data_freeze_gold_20260526_v1`, and treats all subsequent modeling and flux products as reproducible derivatives of that sealed input. The final DOC concentration prediction workflow uses the bias-aware R4 river-specific Q-and-season linear model, selected after comparison with the original F3 baseline, hydroclimate extensions, log-target sensitivities, and optical proxy experiments. Optical features did not provide robust incremental skill over the F3 comparator and were excluded from the primary production pathway.",
        "",
        "The annual DOC flux result is river-specific rather than aggregate-wide. In the core 2003-2024 cohort, Yukon is the only river with a detectable increasing annual DOC flux trend. Other individual rivers show no detectable annual trend, and the six-river aggregate also has no detectable trend. This means the manuscript should frame the result as a Yukon-specific signal within a six-river ArcticGRO reconstruction, not as a pan-Arctic DOC flux increase.",
        "",
        "Attribution diagnostics indicate that the Yukon increase is discharge-volume dominated. Annual Q volume increases detectably, while flow-weighted DOC does not show a detectable increasing trend. This points away from a simple concentration-intensification story and toward volume and timing as the first-order interpretation. The result remains exploratory mechanism analysis rather than causal proof because discharge uncertainty is not propagated and attribution uses modeled DOC concentration with observed Q.",
        "",
        "Seasonal and phenology diagnostics further refine the Yukon story. Fixed May-July flux does not explain the annual increase, and the May-July fraction decreases. In contrast, Yukon shows evidence of later and extended-season export: after-July fraction increases, flux centroid shifts later, and the active export season length increases. Dynamic snowmelt and high-flow windows provide useful operational context, but they do not turn the Yukon increase into a decisive freshet-window flux trend.",
        "",
        "The river regime synthesis separates Yukon from the other rivers. Kolyma, Lena, Ob, and Yenisey are classified as freshet-dominated stable, Mackenzie as no detectable change, and Yukon as discharge-volume extended-season export. These labels should be described as operational synthesis categories built from existing guarded flux and phenology outputs, not as universal process types.",
        "",
        "The key limitations should be carried into every results paragraph. Flux intervals include DOC concentration uncertainty only, not discharge uncertainty. The domain is six ArcticGRO rivers and should not be generalized to all Arctic rivers. May-July remains provisional, dynamic snowmelt windows are operational definitions, optical reflectance is a proxy rather than DOC observation, and the attribution analysis is hypothesis-generating rather than causal proof.",
        "",
        "Manuscript-ready central story:",
        "",
        CENTRAL_STORY,
        "",
        "Yukon attribution evidence:",
        "",
        f"- annual flux trend: `{yukon['annual_flux_trend_direction']}`",
        f"- Q-volume trend: `{yukon['Q_volume_trend_direction']}`",
        f"- flow-weighted DOC trend: `{yukon['flow_weighted_DOC_trend_direction']}`",
        f"- driver classification: `{yukon['driver_classification']}`",
        "",
        "Regime classification:",
        "",
        _md_table(regimes, max_rows=10),
    ]
    RESULTS_NARRATIVE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return RESULTS_NARRATIVE_PATH


def write_abstract_draft(inputs: dict[str, Any]) -> Path:
    lines = [
        "# Abstract Draft",
        "",
        f"Generated: {utc_now()}",
        "",
        "Arctic river dissolved organic carbon (DOC) export is expected to respond to changing discharge volume, seasonality, and snowmelt/freshet dynamics, but separating concentration, discharge, and timing effects requires traceable data and model decisions. We rebuilt a six-river ArcticGRO DOC workflow from the frozen gold data release `data_freeze_gold_20260526_v1`, selected a guarded river-specific Q-and-season DOC concentration model, generated daily DOC concentration and flux products, and evaluated annual trends, flux attribution, and export phenology. In the core 2003-2024 cohort, Yukon was the only river with a detectable increasing annual DOC flux trend, while the six-river aggregate showed no detectable trend. Attribution diagnostics indicate that the Yukon increase is discharge-volume dominated: annual Q volume increased detectably, whereas flow-weighted DOC did not. Fixed May-July flux did not explain the Yukon signal, and export phenology instead points to later and extended-season export, including increasing after-July fraction and later flux centroid. Several other rivers remain freshet-dominated but stable in this operational synthesis. These results support a guarded interpretation of Yukon-specific extended-season DOC export expansion rather than a pan-Arctic or all-river increase. Flux intervals propagate DOC concentration uncertainty only, discharge uncertainty is not propagated, and the mechanism analysis is exploratory rather than causal proof.",
    ]
    ABSTRACT_DRAFT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ABSTRACT_DRAFT_PATH


def write_claims_to_avoid() -> Path:
    lines = [
        "# Claims To Avoid",
        "",
        f"Generated: {utc_now()}",
        "",
        "- No pan-Arctic DOC increase: this is a six-river ArcticGRO-domain analysis, not a complete Arctic Ocean DOC budget.",
        "- No all-river increase: Yukon is the only core river with a detectable increasing annual DOC flux trend.",
        "- No optical improvement claim: optical proxy features did not robustly improve the F3 baseline and are excluded from the primary model.",
        "- No causality proof: attribution and freshet-control synthesis are exploratory mechanism analyses, not causal proof.",
        "- No discharge uncertainty propagation: flux intervals include DOC concentration empirical residual uncertainty only.",
        "- No final snowmelt attribution: fixed May-July is provisional and dynamic snowmelt/freshet windows are operational definitions.",
        "- No extrapolation beyond six ArcticGRO rivers: river fixed effects and validation boundaries limit the domain.",
        "- No claim that Yukon increase is caused by DOC concentration intensification: flow-weighted DOC has no detectable increasing trend.",
        "- No claim that snowmelt-window flux alone explains Yukon annual increase: dynamic windows are partial and non-decisive.",
    ]
    CLAIMS_TO_AVOID_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return CLAIMS_TO_AVOID_PATH


def draft_manuscript_outline() -> dict[str, Any]:
    _ensure_dirs()
    before_hashes = _input_hashes()
    inputs = _load_inputs()
    claims = build_key_claims_to_evidence_map(inputs)
    figures = build_figure_plan(inputs)
    tables = build_table_plan(inputs)
    table_paths = [
        _write_csv(claims, KEY_CLAIMS_PATH),
        _write_csv(figures, FIGURE_PLAN_PATH),
        _write_csv(tables, TABLE_PLAN_PATH),
    ]
    report_paths = [
        write_manuscript_outline(inputs, figures, tables),
        write_results_narrative(inputs),
        write_abstract_draft(inputs),
        write_claims_to_avoid(),
    ]
    _verify_inputs_unchanged(before_hashes)
    return {
        "reports": report_paths,
        "tables": table_paths,
        "title_options": TITLE_OPTIONS,
        "central_story": CENTRAL_STORY,
        "claims": claims,
        "figures": figures,
        "manuscript_tables": tables,
    }


def write_manuscript_outline_report() -> Path:
    inputs = _load_inputs()
    figures = _read_csv("recommended_manuscript_figures")
    tables = _read_csv("recommended_manuscript_tables")
    if FIGURE_PLAN_PATH.exists():
        figures = pd.read_csv(FIGURE_PLAN_PATH, low_memory=False)
    if TABLE_PLAN_PATH.exists():
        tables = pd.read_csv(TABLE_PLAN_PATH, low_memory=False)
    return write_manuscript_outline(inputs, figures, tables)


__all__ = [
    "MANUSCRIPT_REPORT_DIR",
    "MANUSCRIPT_TABLE_DIR",
    "MANUSCRIPT_OUTLINE_PATH",
    "RESULTS_NARRATIVE_PATH",
    "ABSTRACT_DRAFT_PATH",
    "CLAIMS_TO_AVOID_PATH",
    "KEY_CLAIMS_PATH",
    "FIGURE_PLAN_PATH",
    "TABLE_PLAN_PATH",
    "REQUIRED_MANUSCRIPT_INPUTS",
    "TITLE_OPTIONS",
    "CENTRAL_STORY",
    "draft_manuscript_outline",
    "write_manuscript_outline_report",
]
