# Manuscript Outline

Generated: 2026-05-29T20:56:51Z

## Title options

1. River-Specific Arctic DOC Export Change Revealed by Guarded Daily Concentration and Flux Reconstruction
2. Discharge-Driven Extended-Season DOC Export in the Yukon River Within a Six-River ArcticGRO Synthesis
3. Freshet Control, Export Phenology, and River-Specific DOC Flux Trends Across Six Arctic Rivers
4. A Frozen-Data Rebuild of Arctic River DOC Flux Points to Yukon-Specific Extended-Season Export

## Abstract skeleton

- Background: Arctic river DOC export is sensitive to changing discharge, seasonality, and snowmelt/freshet timing, but trend attribution is limited by data integration and model uncertainty.
- Data/model: Use frozen gold data freeze `data_freeze_gold_20260526_v1`, guarded DOC concentration modeling, daily DOC prediction, and DOC-concentration-only flux intervals.
- Main result: Yukon has a detectable increasing annual DOC flux trend in the core 2003-2024 cohort; the six-river aggregate has no detectable trend.
- Mechanism result: Yukon is best framed as discharge-volume-driven extended-season export expansion, not fixed May-July intensification.
- Limitation: Discharge uncertainty is not propagated and results are limited to six ArcticGRO rivers.

## Introduction logic

1. Arctic river DOC export matters for carbon delivery to coastal Arctic systems.
2. Warming can alter discharge volume, freshet timing, season length, and concentration dynamics.
3. Existing work often emphasizes freshet/snowmelt, but annual export may also shift through shoulder-season discharge.
4. A guarded frozen-data rebuild can separate model, prediction, flux, trend, and attribution decisions.
5. The manuscript asks whether annual DOC flux change is aggregate-wide, river-specific, freshet-controlled, or extended-season.

## Methods outline

- Frozen data contract and gold data provenance.
- DOC concentration model selection: F3 baseline, bias-aware R4 production candidate, optical sensitivity exclusion.
- Guarded daily DOC prediction and flux calculation with empirical DOC residual intervals.
- Cohort selection for annual trends and sensitivity checks.
- Annual trend methods and uncertainty sensitivity.
- Flux attribution: Q volume, annual mean DOC, flow-weighted DOC, monthly/seasonal decomposition.
- Snowmelt/freshet operational windows and export phenology diagnostics.
- Claims boundary and caveat register.

## Results outline

1. The frozen-data and model rebuild produced guarded daily DOC and flux products.
2. Yukon is the only core river with detectable increasing annual DOC flux; aggregate flux has no detectable trend.
3. Fixed May-July does not explain Yukon annual increase.
4. Flux attribution identifies Yukon as discharge-volume dominated with no detectable flow-weighted DOC trend.
5. Export phenology indicates later and extended-season Yukon export.
6. Regime classification separates freshet-dominated stable rivers from Yukon extended-season export.
7. Optical proxy evidence is negative for primary model improvement.

## Discussion outline

- The strongest story is river-specific change, not pan-Arctic or aggregate increase.
- Yukon may represent discharge-driven expansion of the export season rather than a simple freshet intensification.
- Freshet dominance can remain important without producing detectable annual trend.
- Optical non-improvement constrains remote-sensing interpretation and supports hydrocore-first modeling.
- Guardrail-first analysis improves reproducibility but narrows the inferential domain.

## Limitations

| caveat_id   | topic                    | severity   | description                                                                                          | affected_results                                           | how_to_report                                                                                             |
|:------------|:-------------------------|:-----------|:-----------------------------------------------------------------------------------------------------|:-----------------------------------------------------------|:----------------------------------------------------------------------------------------------------------|
| C01         | uncertainty              | high       | DOC uncertainty only; discharge uncertainty is not propagated.                                       | daily flux; annual flux; trend tests; snowmelt-window flux | State that flux intervals are based on DOC concentration empirical residual intervals only.               |
| C02         | Yenisey confidence       | high       | Yenisey has low-confidence flux years and many years with high low-confidence flux fraction.         | Yenisey annual flux and sensitivity cohorts                | Avoid strong Yenisey-specific trend claims; emphasize core cohort and confidence tiers.                   |
| C03         | Yukon early hindcast     | high       | Yukon 2000 near-zero issue is excluded/caveated and should not drive interpretation.                 | full 2000-2025 sensitivity; Yukon trend context            | Use core 2003-2024 as primary; report 2000 as caveated sensitivity only.                                  |
| C04         | Kolyma/Ob caveated years | medium     | Kolyma and Ob include excluded or caveated years driven by coverage or low-confidence flux fraction. | full-period and high-confidence sensitivity cohorts        | Use core cohort as primary and retain excluded/caveated year notes.                                       |
| C05         | Mackenzie 2025           | medium     | Mackenzie 2025 coverage caveat keeps 2025 outside the primary core trend.                            | full 2000-2025 sensitivity                                 | Keep Mackenzie 2025 as sensitivity context only.                                                          |
| C06         | ROI                      | medium     | ROI visual review caveats remain even though ROI QC found no freeze-reopen requirement.              | optical sensitivity interpretation                         | State that ROI caveats mainly affect optical proxy analyses, not hydrocore production flux.               |
| C07         | Optical proxy            | medium     | Optical proxy negative result: reflectance features did not robustly improve the DOC baseline.       | model selection; optical sensitivity                       | Do not frame the study as satellite DOC retrieval; describe optical layers as sensitivity/proxy evidence. |
| C08         | Snowmelt windows         | medium     | Snowmelt windows are exploratory/interpretive.                                                       | dynamic window flux and fraction trends                    | Use cautious attribution language; do not claim final hydrologic attribution.                             |
| C09         | Spatial domain           | high       | No pan-Arctic extrapolation beyond six ArcticGRO rivers.                                             | aggregate flux and trend conclusions                       | Refer to the six-river ArcticGRO domain, not a full Arctic Ocean DOC budget.                              |

## Conclusion

Using a frozen, audited ArcticGRO gold dataset and a guarded daily DOC concentration model, the analysis finds a detectable annual DOC flux increase only for Yukon, not for the six-river aggregate. The Yukon signal is best framed as discharge-volume-driven extended-season export expansion, while several other rivers remain freshet-dominated but stable.

## Proposed figures

| figure_id   | figure_title                                                  | source_outputs                                                                                                                      | main_message                                                                                | panel_suggestions                                                                           | status                                       |
|:------------|:--------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------|:---------------------------------------------|
| Figure 1    | Workflow, frozen data, and guarded modeling chain             | outputs/reports/final_synthesis/final_synthesis_report.md; outputs/tables/final_synthesis/model_evolution_summary.csv               | Data freeze to model selection to daily DOC prediction to guarded flux and interpretation.  | A data freeze schematic; B model evolution; C prediction/flux guardrails.                   | draft_from_existing_outputs                  |
| Figure 2    | DOC concentration model performance and selection             | outputs/reports/bias_refinement/bias_refinement_report.md; existing bias_refinement figures                                         | R4 improves validation diagnostics enough to serve as guarded production candidate.         | Observed-vs-CV predicted; river residuals; high-DOC residual review; readiness decision.    | use_existing_or_rebuild_from_existing_tables |
| Figure 3    | Annual DOC flux trends across six ArcticGRO rivers            | outputs/tables/annual_flux_trends/annual_flux_trends_by_river.csv; outputs/figures/annual_flux_trends/                              | Yukon increases; aggregate and other rivers show no detectable trend in the core cohort.    | River time series; slope by river; aggregate core trend; uncertainty sensitivity.           | use_existing_or_rebuild_from_existing_tables |
| Figure 4    | Yukon attribution: discharge volume, DOC, and seasonal export | outputs/tables/flux_attribution/flux_driver_classification.csv; outputs/figures/flux_attribution/yukon_flux_Q_DOC_decomposition.png | Yukon signal is discharge-volume dominated, not flow-weighted DOC dominated.                | Annual flux/Q/DOC decomposition; monthly/seasonal contribution trends; May-July comparison. | use_existing_flux_attribution_outputs        |
| Figure 5    | Export phenology and extended-season signal                   | outputs/tables/flux_attribution/export_phenology_trends_by_river.csv; outputs/figures/freshet_control/                              | Yukon export timing shifts later and after-July export fraction increases.                  | Centroid slopes; after-July fraction slopes; Yukon cumulative timing dashboard.             | use_existing_freshet_control_outputs         |
| Figure 6    | DOC export regime classification                              | outputs/tables/freshet_control/export_regime_classification.csv; outputs/figures/freshet_control/export_regime_summary.png          | Most rivers are freshet-dominated stable; Yukon is extended-season discharge-volume export. | Regime map/table; freshet fraction by river; annual-window coupling categories.             | use_existing_freshet_control_outputs         |
| Figure 7    | Caveats, sensitivity, and interpretation boundaries           | outputs/tables/final_synthesis/caveat_register.csv; outputs/tables/manuscript/key_claims_to_evidence_map.csv                        | Results are guarded by uncertainty, coverage, cohort, and domain boundaries.                | Caveat register; confidence tiers; claims-to-avoid callouts.                                | draft_from_existing_outputs                  |

## Proposed tables

| table_id   | table_title                                           | source_outputs                                                                                                                  | purpose                                                                                 |
|:-----------|:------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------|
| Table 1    | Frozen data sources and model inputs                  | outputs/tables/gold_input_inventory.csv; outputs/reports/data_contract_report.md                                                | Document gold freeze provenance, tables, row counts, and hashes.                        |
| Table 2    | DOC concentration model selection and guardrails      | outputs/tables/final_synthesis/model_evolution_summary.csv; outputs/tables/bias_refinement/bias_refinement_recommendation.csv   | Show progression from F3 baseline to R4 production candidate and optical exclusion.     |
| Table 3    | Annual DOC flux trend by river                        | outputs/tables/final_synthesis/annual_flux_trend_summary_for_manuscript.csv                                                     | Primary trend evidence for Yukon-only detectable annual increase.                       |
| Table 4    | Flux attribution and DOC export regime classification | outputs/tables/flux_attribution/flux_driver_classification.csv; outputs/tables/freshet_control/export_regime_classification.csv | Connect annual trends to Q volume, flow-weighted DOC, export timing, and regime labels. |
| Table 5    | Caveats and claims boundary register                  | outputs/tables/final_synthesis/caveat_register.csv; outputs/reports/manuscript/claims_to_avoid.md                               | Make uncertainty, domain, optical, and snowmelt interpretation limits explicit.         |

## Core evidence snapshot

| finding_id   | finding_category        | finding_statement                                                                                                                                | strength   |
|:-------------|:------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------|:-----------|
| F01          | Data freeze             | Gold data freeze data_freeze_gold_20260526_v1 is the sole data source for modeling.                                                              | strong     |
| F02          | Model                   | Primary production candidate concentration model is R4_river_specific_Q_and_season + linear_regression.                                          | strong     |
| F03          | Optical                 | Optical proxy did not improve the F3 baseline and is excluded from the primary model.                                                            | moderate   |
| F04          | ROI                     | ROI QC found no fatal issue or freeze reopen requirement, but visual/manual caveats remain.                                                      | moderate   |
| F05          | Flux                    | Guarded daily and annual DOC flux were generated with DOC concentration uncertainty only.                                                        | strong     |
| F06          | Annual trend            | In the core 2003-2024 annual flux cohort, Yukon is the only river with detectable increasing annual DOC flux (slope 0.0218 Tg C yr-1, p=0.0151). | moderate   |
| F07          | Aggregate               | Six-river aggregate annual DOC flux has no detectable trend in the core cohort (aggregate p=0.199).                                              | strong     |
| F08          | May-July                | Fixed May-July flux does not explain Yukon annual increase; Yukon May-July fraction decreases.                                                   | moderate   |
| F09          | Dynamic snowmelt        | Dynamic snowmelt/freshet windows show no detectable window flux trend in the core cohort; Yukon signal is partial, not decisive.                 | moderate   |
| F10          | Limitation              | Discharge uncertainty is not propagated.                                                                                                         | strong     |
| F11          | Interpretation boundary | Do not claim pan-Arctic large-river DOC flux increase.                                                                                           | strong     |
| F12          | Interpretation boundary | Do not claim Yukon increase is driven by snowmelt-window flux increase.                                                                          | strong     |

## Regime labels

Kolyma: `freshet_dominated_stable`; Lena: `freshet_dominated_stable`; Mackenzie: `no_detectable_change`; Ob: `freshet_dominated_stable`; Yenisey: `freshet_dominated_stable`; Yukon: `discharge_volume_extended_season`
