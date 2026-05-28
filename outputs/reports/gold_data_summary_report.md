# Gold Data Summary Report

Generated: 2026-05-28T06:06:08Z

No model was trained. No DOC prediction was generated. No flux was generated.

## Inventory

| table_name                                       | role                                |   rows |   columns |   file_size_bytes |
|:-------------------------------------------------|:------------------------------------|-------:|----------:|------------------:|
| doc_labels_gold.csv                              | response_labels                     |    547 |        29 |            213218 |
| training_matrix_hydrocore.csv                    | main_training_matrix                |    547 |        23 |            171079 |
| training_matrix_basin_context.csv                | basin_context_training_matrix       |    547 |        52 |            452045 |
| training_matrix_optical_matched_0d.csv           | optical_sensitivity                 |    188 |        32 |             69702 |
| training_matrix_optical_matched_1d.csv           | optical_sensitivity                 |    297 |        32 |            110597 |
| training_matrix_optical_matched_3d.csv           | optical_sensitivity_main            |    374 |        32 |            139717 |
| training_matrix_optical_matched_3d_hls.csv       | optical_sensitivity_sensor_specific |    144 |        32 |             43101 |
| training_matrix_optical_matched_3d_landsat.csv   | optical_sensitivity_sensor_specific |    184 |        32 |             76922 |
| training_matrix_optical_matched_3d_sentinel2.csv | optical_sensitivity_sensor_specific |     46 |        32 |             20490 |
| training_matrix_optical_matched_7d.csv           | optical_sensitivity_wide_window     |    420 |        32 |            156783 |
| prediction_grid_daily_hydrocore.csv              | future_prediction_grid_x_only       |  52594 |        16 |          11444171 |
| prediction_grid_daily_with_basin_context.csv     | future_prediction_grid_x_only       |  52594 |        45 |          38415317 |
| basin_attributes_curated.csv                     | basin_attribute_long                |    240 |        17 |             83075 |
| basin_attributes_curated_wide.csv                | basin_attribute_wide                |      6 |        30 |              3631 |
| basin_context_gold.csv                           | basin_context                       |      6 |        12 |             16405 |
| daily_discharge_gold.csv                         | daily_discharge                     | 154370 |        13 |          29115877 |
| daily_hydroclimate_gold.csv                      | daily_hydroclimate                  |  52726 |        19 |          23071138 |
| optical_timeseries_gold.csv                      | optical_time_series                 | 142058 |        27 |          58661525 |
| lab_optical_proxy_gold.csv                       | lab_optical_mechanism_only          |    882 |        23 |            323529 |
| roi_catalog_gold.csv                             | roi_metadata                        |     39 |        14 |             62404 |

## Verification Snapshot

| table_name                                       | status   | row_count_ok   | sha256_ok   |
|:-------------------------------------------------|:---------|:---------------|:------------|
| doc_labels_gold.csv                              | ok       | True           | True        |
| training_matrix_hydrocore.csv                    | ok       | True           | True        |
| training_matrix_basin_context.csv                | ok       | True           | True        |
| training_matrix_optical_matched_0d.csv           | ok       | True           | True        |
| training_matrix_optical_matched_1d.csv           | ok       | True           | True        |
| training_matrix_optical_matched_3d.csv           | ok       | True           | True        |
| training_matrix_optical_matched_3d_hls.csv       | ok       | True           | True        |
| training_matrix_optical_matched_3d_landsat.csv   | ok       | True           | True        |
| training_matrix_optical_matched_3d_sentinel2.csv | ok       | True           | True        |
| training_matrix_optical_matched_7d.csv           | ok       | True           | True        |
| prediction_grid_daily_hydrocore.csv              | ok       | True           | True        |
| prediction_grid_daily_with_basin_context.csv     | ok       | True           | True        |
| basin_attributes_curated.csv                     | ok       | True           | True        |
| basin_attributes_curated_wide.csv                | ok       | True           | True        |
| basin_context_gold.csv                           | ok       | True           | True        |
| daily_discharge_gold.csv                         | ok       | True           | True        |
| daily_hydroclimate_gold.csv                      | ok       | True           | True        |
| optical_timeseries_gold.csv                      | ok       | True           | True        |
| lab_optical_proxy_gold.csv                       | ok       | True           | True        |
| roi_catalog_gold.csv                             | ok       | True           | True        |
