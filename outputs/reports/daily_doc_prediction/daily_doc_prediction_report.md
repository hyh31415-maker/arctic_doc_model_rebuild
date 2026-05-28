# Daily DOC Prediction Report

Generated: 2026-05-28T13:37:45Z

## 1. Scope and guardrails

This phase fits the frozen refined R4 DOC concentration candidate and generates guarded daily DOC concentration predictions. It does not calculate DOC flux.

## 2. Model spec and freeze ID

- freeze_id: `data_freeze_gold_20260526_v1`
- model_spec_id: `production_candidate_r4_river_specific_q_and_season_linear`
- prediction_output_sha256: `e7a6e4cb9fc9c4e5fe3671f7eade6c4f9ba44da1e16e9f078ffb8f13a98de8ba`

## 3. Training rows used

| model_fit_id                                                    | model_spec_id                                              | freeze_id                    |   training_rows_used | training_rivers                        | training_date_min   | training_date_max   | training_table_sha256                                            | prediction_grid_sha256                                           |   in_sample_rmse_diagnostic |   in_sample_mae_diagnostic | production_candidate_not_flux_model   | flux_allowed   |
|:----------------------------------------------------------------|:-----------------------------------------------------------|:-----------------------------|---------------------:|:---------------------------------------|:--------------------|:--------------------|:-----------------------------------------------------------------|:-----------------------------------------------------------------|----------------------------:|---------------------------:|:--------------------------------------|:---------------|
| production_candidate_r4_linear_fit_data_freeze_gold_20260526_v1 | production_candidate_r4_river_specific_q_and_season_linear | data_freeze_gold_20260526_v1 |                  545 | Kolyma;Lena;Mackenzie;Ob;Yenisey;Yukon | 2003-06-18          | 2024-08-24          | 0b4eaf0ebf716e23379f15d56cbff82c105251497346b29aa82c8adba95db8e1 | 3ba36c79ad53b14e6b9d08e699bb995a9f694d4ffe75d4565126fb84cec4767b |                     1.75801 |                    1.26589 | True                                  | False          |

## 4. Prediction grid coverage

| river     |   n_grid_rows |   n_predicted_rows | date_min   | date_max   |   n_years |   outside_training_logQ_rows |   outside_training_doy_rows |   outside_training_year_rows |   prediction_coverage_rate |
|:----------|--------------:|-------------------:|:-----------|:-----------|----------:|-----------------------------:|----------------------------:|-----------------------------:|---------------------------:|
| Kolyma    |          9497 |               9301 | 2000-01-01 | 2025-12-31 |        26 |                          211 |                         766 |                         2557 |                   0.979362 |
| Lena      |          9497 |               9351 | 2000-01-01 | 2025-12-31 |        26 |                           23 |                         617 |                         2411 |                   0.984627 |
| Mackenzie |          9472 |               9439 | 2000-01-01 | 2025-12-06 |        26 |                          160 |                         520 |                         1405 |                   0.996516 |
| Ob        |          9476 |               9416 | 2000-01-01 | 2025-12-10 |        26 |                          238 |                         482 |                         2476 |                   0.993668 |
| Yenisey   |          5241 |               5241 | 2000-03-03 | 2025-11-03 |        26 |                          188 |                           0 |                         1475 |                   1        |
| Yukon     |          9411 |               8955 | 2000-01-01 | 2025-10-06 |        26 |                          154 |                         294 |                          919 |                   0.951546 |

## 5. Daily DOC prediction output

- path: `D:\Hao\Desktop\冰冻圈水文\北极大河\arctic_doc_model_rebuild\outputs\tables\daily_doc_prediction\daily_doc_prediction.csv`
- output is DOC concentration only.

## 6. Prediction intervals

Intervals are empirical validation residual intervals attached to the daily predictions; they are not flux uncertainty estimates.

| scope   | group_value   |   n | interval_source                                 | is_production_prediction_interval   |    q02_5 |      q05 |      q10 |       q25 |        q50 |      q75 |     q90 |     q95 |   q97_5 |   empirical_80pct_interval_lower |   empirical_80pct_interval_upper |   empirical_90pct_interval_lower |   empirical_90pct_interval_upper |   empirical_95pct_interval_lower |   empirical_95pct_interval_upper | used_for_daily_doc_prediction   |
|:--------|:--------------|----:|:------------------------------------------------|:------------------------------------|---------:|---------:|---------:|----------:|-----------:|---------:|--------:|--------:|--------:|---------------------------------:|---------------------------------:|---------------------------------:|---------------------------------:|---------------------------------:|---------------------------------:|:--------------------------------|
| overall | overall       | 545 | bias_refinement_leave_one_year_out_cv_residuals | False                               | -3.2078  | -2.68538 | -2.06401 | -1.03282  | -0.156252  | 0.92366  | 2.10636 | 3.26372 | 4.30355 |                         -2.06401 |                          2.10636 |                         -2.68538 |                          3.26372 |                         -3.2078  |                          4.30355 | True                            |
| river   | Kolyma        |  85 | bias_refinement_leave_one_year_out_cv_residuals | False                               | -2.34934 | -2.27125 | -2.08717 | -1.01428  | -0.320145  | 1.00625  | 2.12181 | 3.20597 | 3.69194 |                         -2.08717 |                          2.12181 |                         -2.27125 |                          3.20597 |                         -2.34934 |                          3.69194 | True                            |
| river   | Lena          |  87 | bias_refinement_leave_one_year_out_cv_residuals | False                               | -4.55203 | -3.85339 | -2.91686 | -1.86254  | -0.730946  | 1.54634  | 3.71962 | 5.3095  | 6.68362 |                         -2.91686 |                          3.71962 |                         -3.85339 |                          5.3095  |                         -4.55203 |                          6.68362 | True                            |
| river   | Mackenzie     | 103 | bias_refinement_leave_one_year_out_cv_residuals | False                               | -1.71805 | -1.35315 | -1.13231 | -0.663345 | -0.0476913 | 0.608413 | 1.14826 | 1.37496 | 1.76952 |                         -1.13231 |                          1.14826 |                         -1.35315 |                          1.37496 |                         -1.71805 |                          1.76952 | True                            |
| river   | Ob            |  87 | bias_refinement_leave_one_year_out_cv_residuals | False                               | -3.10456 | -2.95981 | -2.44863 | -1.25811  | -0.4077    | 0.974956 | 2.18309 | 4.03588 | 4.92344 |                         -2.44863 |                          2.18309 |                         -2.95981 |                          4.03588 |                         -3.10456 |                          4.92344 | True                            |
| river   | Yenisey       |  87 | bias_refinement_leave_one_year_out_cv_residuals | False                               | -3.02329 | -1.79119 | -1.54506 | -0.87772  |  0.0210187 | 0.755127 | 1.73101 | 2.37356 | 2.75337 |                         -1.54506 |                          1.73101 |                         -1.79119 |                          2.37356 |                         -3.02329 |                          2.75337 | True                            |
| river   | Yukon         |  96 | bias_refinement_leave_one_year_out_cv_residuals | False                               | -3.09195 | -2.5843  | -1.74666 | -1.04273  | -0.166632  | 0.832818 | 2.34077 | 3.08591 | 4.20876 |                         -1.74666 |                          2.34077 |                         -2.5843  |                          3.08591 |                         -3.09195 |                          4.20876 | True                            |

## 7. Extrapolation flags

| flag_type                   |   n_rows | example_prediction_ids                                                                                                                                                                                                                                                                                                                                                                           |   train_DOC_min |   train_DOC_max |   train_DOC_p01 |   train_DOC_p99 |   high_doc_threshold_1_5x_train_max |   hard_high_doc_threshold |
|:----------------------------|---------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------:|----------------:|----------------:|----------------:|------------------------------------:|--------------------------:|
| outside_training_logQ_range |      974 | production_candidate_r4_river_specific_q_and_season_linear_Kolyma_2000-03-12;production_candidate_r4_river_specific_q_and_season_linear_Kolyma_2000-03-13;production_candidate_r4_river_specific_q_and_season_linear_Kolyma_2000-03-14;production_candidate_r4_river_specific_q_and_season_linear_Kolyma_2000-03-15;production_candidate_r4_river_specific_q_and_season_linear_Kolyma_2000-03-16 |             2.1 |            23.5 |         2.35868 |          17.936 |                               35.25 |                        30 |
| outside_training_doy_range  |     2679 | production_candidate_r4_river_specific_q_and_season_linear_Kolyma_2000-01-01;production_candidate_r4_river_specific_q_and_season_linear_Kolyma_2000-01-02;production_candidate_r4_river_specific_q_and_season_linear_Kolyma_2000-01-03;production_candidate_r4_river_specific_q_and_season_linear_Kolyma_2000-01-04;production_candidate_r4_river_specific_q_and_season_linear_Kolyma_2000-01-05 |             2.1 |            23.5 |         2.35868 |          17.936 |                               35.25 |                        30 |
| outside_training_year_range |    11243 | production_candidate_r4_river_specific_q_and_season_linear_Kolyma_2000-01-01;production_candidate_r4_river_specific_q_and_season_linear_Kolyma_2000-01-02;production_candidate_r4_river_specific_q_and_season_linear_Kolyma_2000-01-03;production_candidate_r4_river_specific_q_and_season_linear_Kolyma_2000-01-04;production_candidate_r4_river_specific_q_and_season_linear_Kolyma_2000-01-05 |             2.1 |            23.5 |         2.35868 |          17.936 |                               35.25 |                        30 |

## 8. Range flags

| flag_type                           |   n_rows | example_prediction_ids                                                                                                                                                                                                                                                                                                                                                                           |   train_DOC_min |   train_DOC_max |   train_DOC_p01 |   train_DOC_p99 |   high_doc_threshold_1_5x_train_max |   hard_high_doc_threshold |
|:------------------------------------|---------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------:|----------------:|----------------:|----------------:|------------------------------------:|--------------------------:|
| point_prediction_raw_lt_0           |       43 | production_candidate_r4_river_specific_q_and_season_linear_Kolyma_2000-05-01;production_candidate_r4_river_specific_q_and_season_linear_Kolyma_2000-05-02;production_candidate_r4_river_specific_q_and_season_linear_Kolyma_2000-05-03;production_candidate_r4_river_specific_q_and_season_linear_Kolyma_2000-05-04;production_candidate_r4_river_specific_q_and_season_linear_Kolyma_2000-05-05 |             2.1 |            23.5 |         2.35868 |          17.936 |                               35.25 |                        30 |
| point_prediction_gt_train_max_1_5x  |        0 | nan                                                                                                                                                                                                                                                                                                                                                                                              |             2.1 |            23.5 |         2.35868 |          17.936 |                               35.25 |                        30 |
| point_prediction_gt_30              |        0 | nan                                                                                                                                                                                                                                                                                                                                                                                              |             2.1 |            23.5 |         2.35868 |          17.936 |                               35.25 |                        30 |
| interval_lower_lt_0_before_clipping |     3998 | production_candidate_r4_river_specific_q_and_season_linear_Kolyma_2000-03-27;production_candidate_r4_river_specific_q_and_season_linear_Kolyma_2000-03-28;production_candidate_r4_river_specific_q_and_season_linear_Kolyma_2000-03-29;production_candidate_r4_river_specific_q_and_season_linear_Kolyma_2000-03-30;production_candidate_r4_river_specific_q_and_season_linear_Kolyma_2000-03-31 |             2.1 |            23.5 |         2.35868 |          17.936 |                               35.25 |                        30 |
| interval_width_95_gt_15             |        0 | nan                                                                                                                                                                                                                                                                                                                                                                                              |             2.1 |            23.5 |         2.35868 |          17.936 |                               35.25 |                        30 |

## 9. River/year coverage

See `outputs/tables/daily_doc_prediction/daily_doc_prediction_by_river_year.csv`.

## 10. Caveats carried forward

- within_six_arcticgro_rivers_only
- no_cross_river_extrapolation
- fold_stability_caveated
- high_doc_bias_caveated
- ROI caveat is not directly relevant to the hydrocore prediction path
- optical excluded

## 11. Readiness for flux phase

ready_for_flux_calculation: `true_with_caveats`

| decision_item                  | status            | evidence                              | recommendation                                                             | blocking_for_flux   |
|:-------------------------------|:------------------|:--------------------------------------|:---------------------------------------------------------------------------|:--------------------|
| daily_doc_prediction_generated | true              | predicted_rows=51703                  | Use only as DOC concentration input to a separate flux phase.              | False               |
| prediction_coverage_acceptable | true_with_caveats | minimum_river_coverage=0.9515         | Review missing predictor rows before flux.                                 | False               |
| range_flags_acceptable         | true              | severe_range_flag_rows=0              | No severe range flags.                                                     | False               |
| extrapolation_flags_acceptable | true_with_caveats | extrapolation_flag_rows=14896         | Carry extrapolation flags into flux uncertainty.                           | False               |
| intervals_available            | true              | interval_rows=7                       | Use empirical residual intervals as concentration uncertainty input.       | False               |
| ready_for_flux_calculation     | true_with_caveats | No flux was calculated in this phase. | Proceed to flux only with concentration caveats and uncertainty intervals. | False               |

## 12. Explicit statements

- Production daily DOC prediction was generated.
- No DOC flux was generated.
- Gold data were not modified.
- Optical/basin/lab features were not used.
