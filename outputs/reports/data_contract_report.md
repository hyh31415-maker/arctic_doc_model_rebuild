# Gold Data Contract Report

Generated: 2026-05-28T07:56:13Z

- freeze_id: `data_freeze_gold_20260526_v1`
- source_repo: `hyh31415-maker/arctic_doc_data_audit`
- source_tag: `data_freeze_gold_20260526_v1`
- data_dir: `D:\Hao\Desktop\冰冻圈水文\北极大河\arctic_doc_data_audit\data\processed\gold`
- data_dir_source: `built-in default fallback`
- expected_tables: `20`
- verified_tables_ok: `20`
- hash_mismatches: `0`
- row_count_mismatches: `0`
- missing_tables: `0`
- read_errors: `0`
- schema_issues: `0`
- forbidden_output_dirs_present: `0`
- model_binary_count: `0`

No model was trained.
No DOC prediction was generated.
No flux was generated.
Only gold freeze data were read.

## Table Statuses

| table_name                                       | role                                |   expected_row_count |   actual_row_count | row_count_ok   | sha256_ok   | status   |
|:-------------------------------------------------|:------------------------------------|---------------------:|-------------------:|:---------------|:------------|:---------|
| doc_labels_gold.csv                              | response_labels                     |                  547 |                547 | True           | True        | ok       |
| training_matrix_hydrocore.csv                    | main_training_matrix                |                  547 |                547 | True           | True        | ok       |
| training_matrix_basin_context.csv                | basin_context_training_matrix       |                  547 |                547 | True           | True        | ok       |
| training_matrix_optical_matched_0d.csv           | optical_sensitivity                 |                  188 |                188 | True           | True        | ok       |
| training_matrix_optical_matched_1d.csv           | optical_sensitivity                 |                  297 |                297 | True           | True        | ok       |
| training_matrix_optical_matched_3d.csv           | optical_sensitivity_main            |                  374 |                374 | True           | True        | ok       |
| training_matrix_optical_matched_3d_hls.csv       | optical_sensitivity_sensor_specific |                  144 |                144 | True           | True        | ok       |
| training_matrix_optical_matched_3d_landsat.csv   | optical_sensitivity_sensor_specific |                  184 |                184 | True           | True        | ok       |
| training_matrix_optical_matched_3d_sentinel2.csv | optical_sensitivity_sensor_specific |                   46 |                 46 | True           | True        | ok       |
| training_matrix_optical_matched_7d.csv           | optical_sensitivity_wide_window     |                  420 |                420 | True           | True        | ok       |
| prediction_grid_daily_hydrocore.csv              | future_prediction_grid_x_only       |                52594 |              52594 | True           | True        | ok       |
| prediction_grid_daily_with_basin_context.csv     | future_prediction_grid_x_only       |                52594 |              52594 | True           | True        | ok       |
| basin_attributes_curated.csv                     | basin_attribute_long                |                  240 |                240 | True           | True        | ok       |
| basin_attributes_curated_wide.csv                | basin_attribute_wide                |                    6 |                  6 | True           | True        | ok       |
| basin_context_gold.csv                           | basin_context                       |                    6 |                  6 | True           | True        | ok       |
| daily_discharge_gold.csv                         | daily_discharge                     |               154370 |             154370 | True           | True        | ok       |
| daily_hydroclimate_gold.csv                      | daily_hydroclimate                  |                52726 |              52726 | True           | True        | ok       |
| optical_timeseries_gold.csv                      | optical_time_series                 |               142058 |             142058 | True           | True        | ok       |
| lab_optical_proxy_gold.csv                       | lab_optical_mechanism_only          |                  882 |                882 | True           | True        | ok       |
| roi_catalog_gold.csv                             | roi_metadata                        |                   39 |                 39 | True           | True        | ok       |

## Schema And Leakage Checks

| check_name                         | table_name                                   | passed   | status   | message                                                                 |
|:-----------------------------------|:---------------------------------------------|:---------|:---------|:------------------------------------------------------------------------|
| hydrocore_required_columns         | training_matrix_hydrocore.csv                | True     | ok       | Required columns present.                                               |
| hydrocore_response_column          | training_matrix_hydrocore.csv                | True     | ok       | DOC_mgC_L exists.                                                       |
| hydrocore_no_lab_optical           | training_matrix_hydrocore.csv                | True     | ok       | No lab optical/CDOM predictor columns.                                  |
| hydrocore_no_prediction_flux       | training_matrix_hydrocore.csv                | True     | ok       | No prediction or flux columns.                                          |
| hydrocore_min_rows                 | training_matrix_hydrocore.csv                | True     | ok       | Rows: 547                                                               |
| optical_required_columns           | training_matrix_optical_matched_3d.csv       | True     | ok       | Required columns present.                                               |
| optical_has_bands                  | training_matrix_optical_matched_3d.csv       | True     | ok       | Optical band values found.                                              |
| optical_has_sensor                 | training_matrix_optical_matched_3d.csv       | True     | ok       | Sensor column exists.                                                   |
| optical_has_days_offset            | training_matrix_optical_matched_3d.csv       | True     | ok       | days_offset column exists.                                              |
| optical_no_lab_optical             | training_matrix_optical_matched_3d.csv       | True     | ok       | No lab optical/CDOM predictor columns.                                  |
| optical_no_prediction_flux         | training_matrix_optical_matched_3d.csv       | True     | ok       | No prediction or flux columns.                                          |
| prediction_grid_required_columns   | prediction_grid_daily_hydrocore.csv          | True     | ok       | Required columns present.                                               |
| prediction_grid_no_doc             | prediction_grid_daily_hydrocore.csv          | True     | ok       | DOC_mgC_L absent.                                                       |
| prediction_grid_no_prediction_flux | prediction_grid_daily_hydrocore.csv          | True     | ok       | No prediction or flux columns.                                          |
| prediction_grid_min_rows           | prediction_grid_daily_hydrocore.csv          | True     | ok       | Rows: 52594                                                             |
| prediction_grid_required_columns   | prediction_grid_daily_with_basin_context.csv | True     | ok       | Required columns present.                                               |
| prediction_grid_no_doc             | prediction_grid_daily_with_basin_context.csv | True     | ok       | DOC_mgC_L absent.                                                       |
| prediction_grid_no_prediction_flux | prediction_grid_daily_with_basin_context.csv | True     | ok       | No prediction or flux columns.                                          |
| prediction_grid_min_rows           | prediction_grid_daily_with_basin_context.csv | True     | ok       | Rows: 52594                                                             |
| basin_six_rivers                   | basin_attributes_curated.csv                 | True     | ok       | Rivers found: ['Kolyma', 'Lena', 'Mackenzie', 'Ob', 'Yenisey', 'Yukon'] |
| basin_id_means_not_predictors      | basin_attributes_curated.csv                 | True     | ok       | ID/topology means are not marked model_use=True.                        |
| basin_wide_no_id_means             | basin_attributes_curated_wide.csv            | True     | ok       | No ID/topology means in wide predictor table.                           |
| basin_attributes_exist             | basin_attributes_curated.csv                 | True     | ok       | Usable basin attributes exist.                                          |
| basin_upstream_area_present        | basin_attributes_curated.csv                 | True     | ok       | upstream_area_km2 present.                                              |

## Next Recommended Step

EDA phase, after the data contract remains fully passing.
