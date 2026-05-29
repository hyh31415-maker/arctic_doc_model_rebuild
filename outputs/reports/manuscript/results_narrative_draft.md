# Results Narrative Draft

Generated: 2026-05-29T20:56:51Z

The analysis begins from a frozen gold data release, `data_freeze_gold_20260526_v1`, and treats all subsequent modeling and flux products as reproducible derivatives of that sealed input. The final DOC concentration prediction workflow uses the bias-aware R4 river-specific Q-and-season linear model, selected after comparison with the original F3 baseline, hydroclimate extensions, log-target sensitivities, and optical proxy experiments. Optical features did not provide robust incremental skill over the F3 comparator and were excluded from the primary production pathway.

The annual DOC flux result is river-specific rather than aggregate-wide. In the core 2003-2024 cohort, Yukon is the only river with a detectable increasing annual DOC flux trend. Other individual rivers show no detectable annual trend, and the six-river aggregate also has no detectable trend. This means the manuscript should frame the result as a Yukon-specific signal within a six-river ArcticGRO reconstruction, not as a pan-Arctic DOC flux increase.

Attribution diagnostics indicate that the Yukon increase is discharge-volume dominated. Annual Q volume increases detectably, while flow-weighted DOC does not show a detectable increasing trend. This points away from a simple concentration-intensification story and toward volume and timing as the first-order interpretation. The result remains exploratory mechanism analysis rather than causal proof because discharge uncertainty is not propagated and attribution uses modeled DOC concentration with observed Q.

Seasonal and phenology diagnostics further refine the Yukon story. Fixed May-July flux does not explain the annual increase, and the May-July fraction decreases. In contrast, Yukon shows evidence of later and extended-season export: after-July fraction increases, flux centroid shifts later, and the active export season length increases. Dynamic snowmelt and high-flow windows provide useful operational context, but they do not turn the Yukon increase into a decisive freshet-window flux trend.

The river regime synthesis separates Yukon from the other rivers. Kolyma, Lena, Ob, and Yenisey are classified as freshet-dominated stable, Mackenzie as no detectable change, and Yukon as discharge-volume extended-season export. These labels should be described as operational synthesis categories built from existing guarded flux and phenology outputs, not as universal process types.

The key limitations should be carried into every results paragraph. Flux intervals include DOC concentration uncertainty only, not discharge uncertainty. The domain is six ArcticGRO rivers and should not be generalized to all Arctic rivers. May-July remains provisional, dynamic snowmelt windows are operational definitions, optical reflectance is a proxy rather than DOC observation, and the attribution analysis is hypothesis-generating rather than causal proof.

Manuscript-ready central story:

Using a frozen, audited ArcticGRO gold dataset and a guarded daily DOC concentration model, the analysis finds a detectable annual DOC flux increase only for Yukon, not for the six-river aggregate. The Yukon signal is best framed as discharge-volume-driven extended-season export expansion, while several other rivers remain freshet-dominated but stable.

Yukon attribution evidence:

- annual flux trend: `increasing`
- Q-volume trend: `increasing`
- flow-weighted DOC trend: `flat_or_uncertain`
- driver classification: `discharge_volume_dominated`

Regime classification:

| river     |   total_flux_rank | yield_rank                                      |   mean_may_july_fraction |   mean_q75_window_fraction | after_july_fraction_trend   | flux_centroid_trend         | annual_flux_trend   | Q_volume_trend    | flow_weighted_DOC_trend   | assigned_regime                  | evidence_summary                                                                                                                                              |
|:----------|------------------:|:------------------------------------------------|-------------------------:|---------------------------:|:----------------------------|:----------------------------|:--------------------|:------------------|:--------------------------|:---------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Kolyma    |                 6 | not_available_allowed_inputs_lack_upstream_area |                 0.701639 |                   0.558829 | no detectable trend         | no detectable trend         | flat_or_uncertain   | flat_or_uncertain | flat_or_uncertain         | freshet_dominated_stable         | Kolyma exports a large stable share of annual DOC during Discharge-centered freshet; no detectable annual change is carried forward.                          |
| Lena      |                 1 | not_available_allowed_inputs_lack_upstream_area |                 0.6606   |                   0.532145 | no detectable trend         | no detectable trend         | flat_or_uncertain   | flat_or_uncertain | flat_or_uncertain         | freshet_dominated_stable         | Lena exports a large stable share of annual DOC during Discharge-centered freshet; no detectable annual change is carried forward.                            |
| Mackenzie |                 4 | not_available_allowed_inputs_lack_upstream_area |                 0.511922 |                   0.226952 | no detectable trend         | no detectable trend         | flat_or_uncertain   | flat_or_uncertain | flat_or_uncertain         | no_detectable_change             | Mackenzie has no detectable annual flux change and no strong evidence for changing export phenology.                                                          |
| Ob        |                 2 | not_available_allowed_inputs_lack_upstream_area |                 0.544574 |                   0.318466 | no detectable trend         | no detectable trend         | flat_or_uncertain   | flat_or_uncertain | flat_or_uncertain         | freshet_dominated_stable         | Ob exports a large stable share of annual DOC during Discharge-centered freshet; no detectable annual change is carried forward.                              |
| Yenisey   |                 3 | not_available_allowed_inputs_lack_upstream_area |                 0.767448 |                   0.590497 | no detectable trend         | no detectable trend         | flat_or_uncertain   | flat_or_uncertain | flat_or_uncertain         | freshet_dominated_stable         | Yenisey exports a large stable share of annual DOC during Discharge-centered freshet; no detectable annual change is carried forward.                         |
| Yukon     |                 5 | not_available_allowed_inputs_lack_upstream_area |                 0.603857 |                   0.357482 | detectable increasing trend | detectable increasing trend | increasing          | increasing        | flat_or_uncertain         | discharge_volume_extended_season | Yukon is best interpreted as discharge-volume-driven extended-season export expansion; high-flow windows matter but do not alone explain the annual increase. |
