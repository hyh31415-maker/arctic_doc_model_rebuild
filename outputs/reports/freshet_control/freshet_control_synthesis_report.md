# Freshet Control and Export Phenology Synthesis Report

Generated: 2026-05-29T19:59:31Z

## 1. Scope and guardrails

This synthesis integrates existing daily flux, annual flux, snowmelt-window, flux-attribution, export-phenology, and annual-trend outputs. It does not train models, generate new DOC predictions, recalculate flux, read raw/interim/canonical data, or modify gold data.

## 2. Literature-inspired framing: freshet control vs extended-season export

The process question is whether annual DOC export is primarily governed by operational snowmelt/freshet or high-flow windows, or whether late-season and shoulder-season discharge expansion contributes meaningfully to annual export. These are process hypotheses, not formal causal proof.

## 3. Freshet/high-flow contribution to annual DOC flux

| river     | best_dynamic_window        |   mean_window_fraction_of_annual |   annual_variability_explained_by_window_flux_r2 | regime_label                     |
|:----------|:---------------------------|---------------------------------:|-------------------------------------------------:|:---------------------------------|
| Kolyma    | discharge_centered_freshet |                         0.726254 |                                         0.855081 | freshet_dominated_stable         |
| Lena      | discharge_centered_freshet |                         0.690386 |                                         0.415938 | freshet_dominated_stable         |
| Mackenzie | discharge_centered_freshet |                         0.541963 |                                         0.382544 | no_detectable_change             |
| Ob        | discharge_centered_freshet |                         0.636914 |                                         0.187311 | freshet_dominated_stable         |
| Yenisey   | discharge_centered_freshet |                         0.739343 |                                         0.813091 | freshet_dominated_stable         |
| Yukon     | discharge_centered_freshet |                         0.584368 |                                         0.706636 | discharge_volume_extended_season |

## 4. Does freshet-window flux explain annual variability?

| river     | window_id                  | window_label               |   n_years |   mean_window_fraction_of_annual |   median_window_fraction_of_annual |   annual_flux_window_flux_correlation |   annual_flux_window_fraction_correlation |   ols_r2_annual_flux_vs_window_flux | annual_variability_explained_category   | is_dynamic_window   | notes                                          |
|:----------|:---------------------------|:---------------------------|----------:|---------------------------------:|-----------------------------------:|--------------------------------------:|------------------------------------------:|------------------------------------:|:----------------------------------------|:--------------------|:-----------------------------------------------|
| Kolyma    | discharge_centered_freshet | Discharge-centered freshet |        19 |                         0.726254 |                           0.72751  |                              0.924706 |                                0.404738   |                           0.855081  | strong                                  | True                | operational_window_definition;not_causal_proof |
| Kolyma    | common_overlap_w1_w2       | Common overlap W1/W2       |        19 |                         0.558829 |                           0.607884 |                              0.783072 |                                0.132419   |                           0.613201  | moderate                                | True                | operational_window_definition;not_causal_proof |
| Kolyma    | q75_peak_contiguous        | Q75 high-flow peak window  |        19 |                         0.558829 |                           0.607884 |                              0.783072 |                                0.132419   |                           0.613201  | moderate                                | True                | operational_window_definition;not_causal_proof |
| Kolyma    | snow_depletion_assisted    | Snow-depletion assisted    |        19 |                         0.677747 |                           0.714472 |                              0.38094  |                               -0.254048   |                           0.145115  | weak                                    | True                | operational_window_definition;not_causal_proof |
| Kolyma    | fixed_may_july_reference   | Fixed May-July reference   |        19 |                         0.701639 |                           0.693664 |                              0.925924 |                                0.10453    |                           0.857336  | strong                                  | False               | operational_window_definition;not_causal_proof |
| Lena      | discharge_centered_freshet | Discharge-centered freshet |        22 |                         0.690386 |                           0.717811 |                              0.644933 |                               -0.0508296  |                           0.415938  | moderate                                | True                | operational_window_definition;not_causal_proof |
| Lena      | snow_depletion_assisted    | Snow-depletion assisted    |        22 |                         0.674395 |                           0.703253 |                              0.636578 |                               -0.0433079  |                           0.405232  | moderate                                | True                | operational_window_definition;not_causal_proof |
| Lena      | common_overlap_w1_w2       | Common overlap W1/W2       |        22 |                         0.526196 |                           0.521509 |                              0.432497 |                               -0.320101   |                           0.187054  | weak                                    | True                | operational_window_definition;not_causal_proof |
| Lena      | q75_peak_contiguous        | Q75 high-flow peak window  |        22 |                         0.532145 |                           0.534202 |                              0.402166 |                               -0.355515   |                           0.161738  | weak                                    | True                | operational_window_definition;not_causal_proof |
| Lena      | fixed_may_july_reference   | Fixed May-July reference   |        22 |                         0.6606   |                           0.661513 |                              0.788072 |                               -0.308577   |                           0.621058  | moderate                                | False               | operational_window_definition;not_causal_proof |
| Mackenzie | discharge_centered_freshet | Discharge-centered freshet |        22 |                         0.541963 |                           0.583471 |                              0.618501 |                               -0.0811216  |                           0.382544  | weak                                    | True                | operational_window_definition;not_causal_proof |
| Mackenzie | snow_depletion_assisted    | Snow-depletion assisted    |        22 |                         0.352097 |                           0.423079 |                              0.441233 |                                0.214302   |                           0.194687  | weak                                    | True                | operational_window_definition;not_causal_proof |
| Mackenzie | common_overlap_w1_w2       | Common overlap W1/W2       |        22 |                         0.226952 |                           0.233972 |                              0.305181 |                                0.00691768 |                           0.0931353 | weak                                    | True                | operational_window_definition;not_causal_proof |
| Mackenzie | q75_peak_contiguous        | Q75 high-flow peak window  |        22 |                         0.226952 |                           0.233972 |                              0.305181 |                                0.00691768 |                           0.0931353 | weak                                    | True                | operational_window_definition;not_causal_proof |
| Mackenzie | fixed_may_july_reference   | Fixed May-July reference   |        22 |                         0.511922 |                           0.520083 |                              0.946611 |                                0.222907   |                           0.896072  | strong                                  | False               | operational_window_definition;not_causal_proof |
| Ob        | discharge_centered_freshet | Discharge-centered freshet |        22 |                         0.636914 |                           0.654534 |                              0.432794 |                               -0.361922   |                           0.187311  | weak                                    | True                | operational_window_definition;not_causal_proof |
| Ob        | common_overlap_w1_w2       | Common overlap W1/W2       |        22 |                         0.314879 |                           0.339699 |                             -0.139006 |                               -0.659642   |                           0.0193228 | weak                                    | True                | operational_window_definition;not_causal_proof |
| Ob        | snow_depletion_assisted    | Snow-depletion assisted    |        22 |                         0.300391 |                           0.137322 |                             -0.129148 |                               -0.26759    |                           0.0166793 | weak                                    | True                | operational_window_definition;not_causal_proof |
| Ob        | q75_peak_contiguous        | Q75 high-flow peak window  |        22 |                         0.318466 |                           0.346373 |                             -0.11209  |                               -0.640883   |                           0.0125642 | weak                                    | True                | operational_window_definition;not_causal_proof |
| Ob        | fixed_may_july_reference   | Fixed May-July reference   |        22 |                         0.544574 |                           0.549982 |                              0.810744 |                               -0.535988   |                           0.657306  | moderate                                | False               | operational_window_definition;not_causal_proof |
| Yenisey   | discharge_centered_freshet | Discharge-centered freshet |        14 |                         0.739343 |                           0.757957 |                              0.901716 |                               -0.35671    |                           0.813091  | strong                                  | True                | operational_window_definition;not_causal_proof |
| Yenisey   | common_overlap_w1_w2       | Common overlap W1/W2       |        14 |                         0.590497 |                           0.595367 |                              0.892699 |                               -0.557884   |                           0.796912  | strong                                  | True                | operational_window_definition;not_causal_proof |
| Yenisey   | q75_peak_contiguous        | Q75 high-flow peak window  |        14 |                         0.590497 |                           0.595367 |                              0.892699 |                               -0.557884   |                           0.796912  | strong                                  | True                | operational_window_definition;not_causal_proof |
| Yenisey   | snow_depletion_assisted    | Snow-depletion assisted    |        14 |                         0.46254  |                           0.48127  |                              0.707501 |                                0.362442   |                           0.500558  | moderate                                | True                | operational_window_definition;not_causal_proof |
| Yenisey   | fixed_may_july_reference   | Fixed May-July reference   |        14 |                         0.767448 |                           0.774766 |                              0.949481 |                               -0.448974   |                           0.901514  | strong                                  | False               | operational_window_definition;not_causal_proof |
| Yukon     | discharge_centered_freshet | Discharge-centered freshet |        22 |                         0.584368 |                           0.605336 |                              0.840616 |                                0.471336   |                           0.706636  | moderate                                | True                | operational_window_definition;not_causal_proof |
| Yukon     | snow_depletion_assisted    | Snow-depletion assisted    |        22 |                         0.567213 |                           0.595506 |                              0.825775 |                                0.47778    |                           0.681905  | moderate                                | True                | operational_window_definition;not_causal_proof |
| Yukon     | common_overlap_w1_w2       | Common overlap W1/W2       |        22 |                         0.353765 |                           0.398687 |                              0.601857 |                                0.244162   |                           0.362232  | weak                                    | True                | operational_window_definition;not_causal_proof |
| Yukon     | q75_peak_contiguous        | Q75 high-flow peak window  |        22 |                         0.357482 |                           0.398687 |                              0.590221 |                                0.231835   |                           0.348361  | weak                                    | True                | operational_window_definition;not_causal_proof |
| Yukon     | fixed_may_july_reference   | Fixed May-July reference   |        22 |                         0.603857 |                           0.598825 |                              0.833992 |                                0.0536636  |                           0.695542  | moderate                                | False               | operational_window_definition;not_causal_proof |

## 5. Yukon extended-season diagnosis

| diagnostic_item                        | status   | evidence                                                                                          |
|:---------------------------------------|:---------|:--------------------------------------------------------------------------------------------------|
| annual_flux_increasing                 | True     | slope=0.0217923713793453, p=0.0150804369764447                                                    |
| Q_volume_increasing                    | True     | slope=2.3798816090344443, p=0.0020853174903879                                                    |
| flow_weighted_DOC_no_detectable_trend  | True     | slope=0.0307664931021726, p=0.1064746142447846                                                    |
| May_July_flux_trend_no                 | True     | no detectable trend                                                                               |
| May_July_fraction_decreasing           | True     | detectable decreasing trend                                                                       |
| q75_window_flux_trend_no               | True     | no detectable trend                                                                               |
| discharge_centered_fraction_increasing | True     | detectable increasing trend                                                                       |
| after_July_fraction_increasing         | True     | detectable increasing trend                                                                       |
| centroid_later                         | True     | detectable increasing trend                                                                       |
| active_season_length_increasing        | True     | detectable increasing trend                                                                       |
| final_interpretation                   | True     | discharge-volume-driven extended-season export expansion; regime=discharge_volume_extended_season |

## 6. Export phenology shifts

Yukon shows later/export-expanded diagnostics: after-July fraction, flux centroid, and active export season length increase in the core cohort. This supports an extended-season export interpretation rather than a fixed May-July explanation.

## 7. River export regime classification

| river     |   total_flux_rank | yield_rank                                      |   mean_may_july_fraction |   mean_q75_window_fraction | after_july_fraction_trend   | flux_centroid_trend         | annual_flux_trend   | Q_volume_trend    | flow_weighted_DOC_trend   | assigned_regime                  | evidence_summary                                                                                                                                              |
|:----------|------------------:|:------------------------------------------------|-------------------------:|---------------------------:|:----------------------------|:----------------------------|:--------------------|:------------------|:--------------------------|:---------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Kolyma    |                 6 | not_available_allowed_inputs_lack_upstream_area |                 0.701639 |                   0.558829 | no detectable trend         | no detectable trend         | flat_or_uncertain   | flat_or_uncertain | flat_or_uncertain         | freshet_dominated_stable         | Kolyma exports a large stable share of annual DOC during Discharge-centered freshet; no detectable annual change is carried forward.                          |
| Lena      |                 1 | not_available_allowed_inputs_lack_upstream_area |                 0.6606   |                   0.532145 | no detectable trend         | no detectable trend         | flat_or_uncertain   | flat_or_uncertain | flat_or_uncertain         | freshet_dominated_stable         | Lena exports a large stable share of annual DOC during Discharge-centered freshet; no detectable annual change is carried forward.                            |
| Mackenzie |                 4 | not_available_allowed_inputs_lack_upstream_area |                 0.511922 |                   0.226952 | no detectable trend         | no detectable trend         | flat_or_uncertain   | flat_or_uncertain | flat_or_uncertain         | no_detectable_change             | Mackenzie has no detectable annual flux change and no strong evidence for changing export phenology.                                                          |
| Ob        |                 2 | not_available_allowed_inputs_lack_upstream_area |                 0.544574 |                   0.318466 | no detectable trend         | no detectable trend         | flat_or_uncertain   | flat_or_uncertain | flat_or_uncertain         | freshet_dominated_stable         | Ob exports a large stable share of annual DOC during Discharge-centered freshet; no detectable annual change is carried forward.                              |
| Yenisey   |                 3 | not_available_allowed_inputs_lack_upstream_area |                 0.767448 |                   0.590497 | no detectable trend         | no detectable trend         | flat_or_uncertain   | flat_or_uncertain | flat_or_uncertain         | freshet_dominated_stable         | Yenisey exports a large stable share of annual DOC during Discharge-centered freshet; no detectable annual change is carried forward.                         |
| Yukon     |                 5 | not_available_allowed_inputs_lack_upstream_area |                 0.603857 |                   0.357482 | detectable increasing trend | detectable increasing trend | increasing          | increasing        | flat_or_uncertain         | discharge_volume_extended_season | Yukon is best interpreted as discharge-volume-driven extended-season export expansion; high-flow windows matter but do not alone explain the annual increase. |

## 8. Mechanistic interpretation

Freshet or high-flow windows explain a large share of annual flux for several rivers, but the Yukon annual increase is best described as discharge-volume-driven extended-season export expansion. Fixed May-July remains a provisional reference, and dynamic windows are operational definitions rather than final hydrologic truth.

## 9. Caveats

- This is exploratory mechanism analysis, not causal proof.
- No new model, prediction, or flux product was generated.
- Discharge uncertainty was not propagated.
- Snowmelt/freshet windows are operational definitions.
- DOC concentration and flux results inherit the caveats from the guarded production and flux phases.

## 10. Manuscript-ready process hypotheses

1. ArcticGRO DOC export is often concentrated in operational high-flow/freshet windows, but the strength of annual coupling differs by river.
2. Yukon annual DOC flux increase is consistent with discharge-volume-driven extended-season export expansion rather than a stronger fixed May-July contribution.
3. Flow-weighted DOC does not show a detectable Yukon increase in the core cohort, suggesting volume and timing are more plausible first-order explanations than concentration intensification.

## 11. What not to claim

- Do not claim causal proof.
- Do not claim discharge uncertainty was propagated.
- Do not claim fixed May-July is final snowmelt flux.
- Do not claim new DOC model, prediction, or flux products were created in this phase.
