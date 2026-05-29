# Final Synthesis Manuscript-Ready Results Report

Generated: 2026-05-29T17:47:00Z

## 1. Executive summary

This final synthesis consolidates existing data-contract, model-selection, prediction, flux, trend, May-July, and dynamic snowmelt-window outputs into a manuscript-ready results summary. There is no model retraining in this phase, no new DOC prediction, no flux recalculation, and gold data unchanged.

| finding_id   | finding_category        | finding_statement                                                                                                                                | strength   | manuscript_ready   |
|:-------------|:------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------|:-----------|:-------------------|
| F01          | Data freeze             | Gold data freeze data_freeze_gold_20260526_v1 is the sole data source for modeling.                                                              | strong     | True               |
| F02          | Model                   | Primary production candidate concentration model is R4_river_specific_Q_and_season + linear_regression.                                          | strong     | True               |
| F03          | Optical                 | Optical proxy did not improve the F3 baseline and is excluded from the primary model.                                                            | moderate   | True               |
| F04          | ROI                     | ROI QC found no fatal issue or freeze reopen requirement, but visual/manual caveats remain.                                                      | moderate   | True               |
| F05          | Flux                    | Guarded daily and annual DOC flux were generated with DOC concentration uncertainty only.                                                        | strong     | True               |
| F06          | Annual trend            | In the core 2003-2024 annual flux cohort, Yukon is the only river with detectable increasing annual DOC flux (slope 0.0218 Tg C yr-1, p=0.0151). | moderate   | True               |
| F07          | Aggregate               | Six-river aggregate annual DOC flux has no detectable trend in the core cohort (aggregate p=0.199).                                              | strong     | True               |
| F08          | May-July                | Fixed May-July flux does not explain Yukon annual increase; Yukon May-July fraction decreases.                                                   | moderate   | True               |
| F09          | Dynamic snowmelt        | Dynamic snowmelt/freshet windows show no detectable window flux trend in the core cohort; Yukon signal is partial, not decisive.                 | moderate   | True               |
| F10          | Limitation              | Discharge uncertainty is not propagated.                                                                                                         | strong     | True               |
| F11          | Interpretation boundary | Do not claim pan-Arctic large-river DOC flux increase.                                                                                           | strong     | True               |
| F12          | Interpretation boundary | Do not claim Yukon increase is driven by snowmelt-window flux increase.                                                                          | strong     | True               |

## 2. Data freeze and gold data inputs

The project uses `data_freeze_gold_20260526_v1` as the frozen data source. This synthesis reads only existing reports and output tables. It does not read raw/interim/canonical data and does not modify gold data.

Primary evidence: `outputs/reports/data_contract_report.md`, `outputs/reports/gold_data_summary_report.md`, and `outputs/reports/eda/eda_report.md`.

## 3. Model development path

| stage                        | candidate_model                                                        | result                                                        | decision                        | reason                                                                                                                       |
|:-----------------------------|:-----------------------------------------------------------------------|:--------------------------------------------------------------|:--------------------------------|:-----------------------------------------------------------------------------------------------------------------------------|
| baseline_finalization        | F3_q_season_river_fixed + ridge_alpha_1                                | F3 baseline selected after baseline finalization.             | selected_as_validation_baseline | F6 hydroclimate extension had negligible same-sample improvement over F3.                                                    |
| hydroclimate_extension       | F6_reduced_hydroclimate_river_fixed + ridge_alpha_1                    | F6 hydroclimate extension not primary.                        | retain_as_process_sensitivity   | Hydroclimate extension improved RMSE by only about 0.0013 mg C/L on the same sample.                                         |
| optical_sensitivity          | Optical proxy feature sets O2/O3/O4/O5 against F3 same-sample baseline | Optical excluded.                                             | exclude_from_primary_model      | Reflectance/index proxy feature sets did not provide robust incremental value and often worsened F3 on optical-matched rows. |
| bias_aware_refinement        | R4_river_specific_Q_and_season + linear_regression                     | R4 bias-aware refined model selected as production candidate. | production_candidate            | Candidate met LOYO improvement, Lena, high-DOC, fold-stability, GroupKFold, and interpretability criteria.                   |
| target_transform_sensitivity | log target sensitivity candidates                                      | log target sensitivity only.                                  | do_not_promote                  | Log-target candidates remain sensitivity-only unless a separate target-transform decision is made.                           |

## 4. Final concentration model

The manuscript-ready production candidate is `R4_river_specific_Q_and_season + linear_regression`. It replaced the finalized F3 comparator after bias-aware refinement, while log-target candidates remain sensitivity-only.

This concentration model is used for guarded daily prediction and flux products, but it is not refit in this synthesis phase.

## 5. Optical sensitivity result

Satellite optical reflectance is a proxy, not DOC observation. Optical proxy feature sets did not robustly improve the finalized F3 baseline and are excluded from the primary production model. Optical results should be reported as a negative/sensitivity result, not as satellite DOC retrieval.

## 6. ROI QC result

ROI QC found no fatal issue and no data-freeze reopen requirement. All final-primary ROIs are accepted with caveats. External visual/manual caveats remain, especially for optical interpretation.

## 7. Daily DOC prediction result

Guarded daily DOC predictions already exist and are not regenerated here. Prediction coverage is summarized below.

| river     |   n_grid_rows |   n_predicted_rows | date_min   | date_max   |   n_years |   outside_training_logQ_rows |   outside_training_doy_rows |   outside_training_year_rows |   prediction_coverage_rate |
|:----------|--------------:|-------------------:|:-----------|:-----------|----------:|-----------------------------:|----------------------------:|-----------------------------:|---------------------------:|
| Kolyma    |          9497 |               9301 | 2000-01-01 | 2025-12-31 |        26 |                          211 |                         766 |                         2557 |                   0.979362 |
| Lena      |          9497 |               9351 | 2000-01-01 | 2025-12-31 |        26 |                           23 |                         617 |                         2411 |                   0.984627 |
| Mackenzie |          9472 |               9439 | 2000-01-01 | 2025-12-06 |        26 |                          160 |                         520 |                         1405 |                   0.996516 |
| Ob        |          9476 |               9416 | 2000-01-01 | 2025-12-10 |        26 |                          238 |                         482 |                         2476 |                   0.993668 |
| Yenisey   |          5241 |               5241 | 2000-03-03 | 2025-11-03 |        26 |                          188 |                           0 |                         1475 |                   1        |
| Yukon     |          9411 |               8955 | 2000-01-01 | 2025-10-06 |        26 |                          154 |                         294 |                          919 |                   0.951546 |

## 8. DOC flux calculation result

Guarded daily and annual DOC flux products already exist and are not recalculated here. Flux uses `DOC_mgC_L * Q_m3s * 86.4`; intervals propagate DOC concentration uncertainty only. Discharge uncertainty not propagated.

| scope   | group_value   | daily_confidence_tier   |   n_days |   fraction_days |   flux_TgC |   fraction_flux |
|:--------|:--------------|:------------------------|---------:|----------------:|-----------:|----------------:|
| overall | overall       | high                    |    34556 |        0.668356 |   326.101  |       0.69161   |
| overall | overall       | medium                  |    13484 |        0.260797 |   110.237  |       0.233797  |
| overall | overall       | low                     |     3663 |        0.070847 |    35.1716 |       0.0745935 |

## 9. Annual flux trend result

The core 2003-2024 cohort is primary. Yukon is the only river with a detectable increasing annual DOC flux trend. Other rivers are flat_or_uncertain / no detectable trend. The six-river aggregate has no detectable trend.

| river     |   core_n_years |   core_year_min |   core_year_max |   mean_annual_flux_TgC |   median_annual_flux_TgC |   slope_TgC_per_year |   p_value | trend_direction   | detectable_trend   | caveat                                                                  |
|:----------|---------------:|----------------:|----------------:|-----------------------:|-------------------------:|---------------------:|----------:|:------------------|:-------------------|:------------------------------------------------------------------------|
| Kolyma    |             19 |            2003 |            2024 |               0.787139 |                 0.808398 |         -0.000530137 | 0.945309  | flat_or_uncertain | False              | DOC_concentration_uncertainty_only;discharge_uncertainty_not_propagated |
| Lena      |             22 |            2003 |            2024 |               6.30245  |                 6.55356  |         -0.0287218   | 0.358626  | flat_or_uncertain | False              | DOC_concentration_uncertainty_only;discharge_uncertainty_not_propagated |
| Mackenzie |             22 |            2003 |            2024 |               1.50957  |                 1.49942  |         -0.01052     | 0.345737  | flat_or_uncertain | False              | DOC_concentration_uncertainty_only;discharge_uncertainty_not_propagated |
| Ob        |             22 |            2003 |            2024 |               4.13136  |                 3.94997  |          0.0264945   | 0.289588  | flat_or_uncertain | False              | DOC_concentration_uncertainty_only;discharge_uncertainty_not_propagated |
| Yenisey   |             14 |            2007 |            2024 |               3.94035  |                 3.94579  |          0.0030672   | 0.939428  | flat_or_uncertain | False              | DOC_concentration_uncertainty_only;discharge_uncertainty_not_propagated |
| Yukon     |             22 |            2003 |            2024 |               1.40435  |                 1.37424  |          0.0217924   | 0.0150804 | increasing        | True               | DOC_concentration_uncertainty_only;discharge_uncertainty_not_propagated |

Aggregate results:

| aggregate_type                  | rivers_included                        |   core_n_years |   core_year_min |   core_year_max |   mean_annual_flux_TgC |   slope_TgC_per_year |   p_value | trend_direction   | detectable_trend   | caveat                                                                                                                   |
|:--------------------------------|:---------------------------------------|---------------:|----------------:|----------------:|-----------------------:|---------------------:|----------:|:------------------|:-------------------|:-------------------------------------------------------------------------------------------------------------------------|
| aggregate_all_available_rivers  | Kolyma;Lena;Mackenzie;Ob;Yenisey;Yukon |             22 |            2003 |            2024 |                16.535  |           0.124272   |  0.199212 | flat_or_uncertain | False              | aggregate_all_available_rivers;varying_river_set;DOC_concentration_uncertainty_only;discharge_uncertainty_not_propagated |
| aggregate_common_river_set_only | Lena;Mackenzie;Ob;Yukon                |             22 |            2003 |            2024 |                13.3477 |           0.00904513 |  0.843999 | flat_or_uncertain | False              | common_river_set_only;DOC_concentration_uncertainty_only;discharge_uncertainty_not_propagated                            |

## 10. Provisional May-July result

May-July is provisional. It should not be treated as the final snowmelt window. Fixed May-July flux does not explain the Yukon annual increase, and the Yukon May-July fraction decreases.

Primary evidence: `outputs/tables/may_july_flux/may_july_vs_annual_trend_comparison.csv`.

## 11. Dynamic snowmelt/freshet window result

Dynamic snowmelt windows are exploratory/interpretive. They improve seasonal framing beyond fixed May-July, but they do not show a detectable window flux trend in the core cohort. Yukon remains partial, not decisive.

| river     | annual_trend_direction   | fixed_may_july_flux_trend   | fixed_may_july_fraction_trend   | best_dynamic_window        | dynamic_window_flux_trend   | dynamic_window_fraction_trend   | does_snowmelt_window_explain_annual_signal   | interpretation                                                                                                                                                                                                                                                           | caveat                                                                                                  |
|:----------|:-------------------------|:----------------------------|:--------------------------------|:---------------------------|:----------------------------|:--------------------------------|:---------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------|
| Kolyma    | flat_or_uncertain        | flat_or_uncertain           | flat_or_uncertain               | q75_peak_contiguous        | flat_or_uncertain           | flat_or_uncertain               | not_applicable                               | No detectable increasing annual flux signal in the core cohort, so snowmelt-window attribution is not required.                                                                                                                                                          | Dynamic snowmelt/freshet windows are exploratory/interpretive; discharge uncertainty is not propagated. |
| Lena      | flat_or_uncertain        | flat_or_uncertain           | flat_or_uncertain               | q75_peak_contiguous        | flat_or_uncertain           | flat_or_uncertain               | not_applicable                               | No detectable increasing annual flux signal in the core cohort, so snowmelt-window attribution is not required.                                                                                                                                                          | Dynamic snowmelt/freshet windows are exploratory/interpretive; discharge uncertainty is not propagated. |
| Mackenzie | flat_or_uncertain        | flat_or_uncertain           | flat_or_uncertain               | q75_peak_contiguous        | flat_or_uncertain           | flat_or_uncertain               | not_applicable                               | No detectable increasing annual flux signal in the core cohort, so snowmelt-window attribution is not required.                                                                                                                                                          | Dynamic snowmelt/freshet windows are exploratory/interpretive; discharge uncertainty is not propagated. |
| Ob        | flat_or_uncertain        | flat_or_uncertain           | flat_or_uncertain               | q75_peak_contiguous        | flat_or_uncertain           | flat_or_uncertain               | not_applicable                               | No detectable increasing annual flux signal in the core cohort, so snowmelt-window attribution is not required.                                                                                                                                                          | Dynamic snowmelt/freshet windows are exploratory/interpretive; discharge uncertainty is not propagated. |
| Yenisey   | flat_or_uncertain        | flat_or_uncertain           | flat_or_uncertain               | q75_peak_contiguous        | flat_or_uncertain           | flat_or_uncertain               | not_applicable                               | No detectable increasing annual flux signal in the core cohort, so snowmelt-window attribution is not required.                                                                                                                                                          | Dynamic snowmelt/freshet windows are exploratory/interpretive; discharge uncertainty is not propagated. |
| Yukon     | increasing               | flat_or_uncertain           | decreasing                      | discharge_centered_freshet | flat_or_uncertain           | increasing                      | partial                                      | Yukon annual flux is increasing, but fixed May-July does not explain the signal. Dynamic snowmelt signal is partial: the selected discharge-centered/freshet evidence can show increasing fraction, but dynamic window flux trend is not detectable and is not decisive. | Dynamic snowmelt/freshet windows are exploratory/interpretive; discharge uncertainty is not propagated. |

## 12. Main scientific findings

- The strongest manuscript-ready result is not a pan-Arctic increase, but a guarded six-river ArcticGRO-domain synthesis.
- Yukon shows a detectable annual DOC flux increase in the core 2003-2024 cohort.
- The six-river aggregate annual DOC flux does not show a detectable trend in the core cohort.
- Fixed May-July and dynamic snowmelt/freshet windows do not decisively explain the Yukon annual increase.
- Optical reflectance did not improve the primary DOC concentration model and remains proxy/sensitivity evidence.

## 13. Sensitivity results

Full-period and high-confidence-only annual trend cohorts are sensitivity results. The core 2003-2024 cohort remains primary. May-July and dynamic-window outputs are interpretive sensitivity layers, not a replacement for annual flux trends.

## 14. Caveats and limitations

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

## 15. Manuscript-ready conclusions

1. A frozen six-river ArcticGRO gold-data workflow supports guarded annual DOC flux estimates for 2000-2025 and a primary core trend cohort for 2003-2024.
2. The production concentration model is an interpretable river-specific Q-and-season linear model selected after bias-aware refinement.
3. In the core annual flux cohort, Yukon is the only river with a detectable increasing annual DOC flux trend.
4. The six-river aggregate has no detectable annual DOC flux trend, so the study does not support a pan-Arctic DOC flux increase claim.
5. May-July and dynamic hydrologic windows do not provide decisive evidence that the Yukon annual increase is snowmelt-window driven.

## 16. Recommended figures and tables

Recommended figures:

| figure_id   | figure_title               | purpose                                                                                         | source_outputs                                                                                         | status                                               | notes                                                                                 |
|:------------|:---------------------------|:------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------|:-----------------------------------------------------|:--------------------------------------------------------------------------------------|
| Fig1        | Workflow diagram           | Show data freeze -> model selection -> DOC prediction -> flux -> trend/snowmelt interpretation. | final_synthesis_report.md; model_evolution_summary.csv                                                 | recommended_not_generated_in_this_phase              | Can be drawn manually from the synthesis tables; no new science calculation required. |
| Fig2        | DOC model performance      | Observed vs predicted CV for R4 production candidate.                                           | outputs/reports/bias_refinement/bias_refinement_report.md; existing bias_refinement figures            | recommended_existing_or_rebuild_from_existing_tables | Do not retrain model.                                                                 |
| Fig3        | Annual flux time series    | Core cohort by river with trend lines.                                                          | outputs/tables/annual_flux_trends/annual_flux_trends_by_river.csv; existing annual_flux_trends figures | recommended_existing_or_rebuild_from_existing_tables | Core 2003-2024 is primary.                                                            |
| Fig4        | Aggregate flux trend       | Six-river aggregate core cohort.                                                                | outputs/tables/final_synthesis/aggregate_flux_trend_summary_for_manuscript.csv                         | recommended_existing_or_rebuild_from_existing_tables | Must state this is not a full Arctic Ocean DOC budget.                                |
| Fig5        | Yukon focus                | Annual flux vs May-July vs dynamic window signal.                                               | outputs/tables/final_synthesis/snowmelt_interpretation_summary.csv                                     | recommended_existing_or_rebuild_from_existing_tables | Show that snowmelt-window evidence is partial, not decisive.                          |
| Fig6        | Snowmelt window comparison | Fixed May-July vs q75_peak_contiguous vs discharge_centered.                                    | outputs/tables/snowmelt_windows/snowmelt_window_flux_summary.csv                                       | recommended_existing_or_rebuild_from_existing_tables | Dynamic windows are exploratory/interpretive.                                         |
| Fig7        | Confidence/caveat figure   | Annual confidence tiers or low-confidence flux fraction.                                        | outputs/tables/doc_flux/doc_flux_confidence_tier_summary.csv; caveat_register.csv                      | recommended_existing_or_rebuild_from_existing_tables | Useful for transparent uncertainty boundary.                                          |

Recommended tables:

| table_id   | table_title                            | source_outputs                                                                       | purpose                                                            |
|:-----------|:---------------------------------------|:-------------------------------------------------------------------------------------|:-------------------------------------------------------------------|
| Table1     | Gold data sources and row counts       | outputs/tables/gold_input_inventory.csv; outputs/reports/gold_data_summary_report.md | Document the frozen data source and modeling inputs.               |
| Table2     | Model selection summary                | outputs/tables/final_synthesis/model_evolution_summary.csv                           | Show why R4 is the production candidate and optical is excluded.   |
| Table3     | Annual trend by river                  | outputs/tables/final_synthesis/annual_flux_trend_summary_for_manuscript.csv          | Main river-specific annual flux trend result.                      |
| Table4     | Dynamic snowmelt window interpretation | outputs/tables/final_synthesis/snowmelt_interpretation_summary.csv                   | Summarize why Yukon snowmelt explanation is partial, not decisive. |
| Table5     | Caveat register                        | outputs/tables/final_synthesis/caveat_register.csv                                   | Make interpretation limits explicit.                               |

## 17. What not to claim

- Do not claim a pan-Arctic DOC flux increase.
- Do not claim all ArcticGRO rivers increased.
- Do not claim optical reflectance improves DOC prediction.
- Do not claim May-July is the final snowmelt window.
- Do not claim Yukon annual increase is definitively snowmelt-driven.
- Do not claim discharge uncertainty is included.
- Do not extrapolate beyond six ArcticGRO rivers.

## 18. Reproducibility notes

- no model retraining in this phase
- no new DOC prediction
- no flux recalculation
- gold data unchanged
- discharge uncertainty not propagated
- May-July is provisional
- dynamic snowmelt windows are exploratory/interpretive
- optical reflectance is proxy, not DOC observation

Recommended command sequence:

```powershell
python -m arctic_doc_model_rebuild.cli verify-gold-data
python -m arctic_doc_model_rebuild.cli synthesize-results
python -m arctic_doc_model_rebuild.cli synthesis-report
python -m pytest
```
