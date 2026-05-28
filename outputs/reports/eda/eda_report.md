# EDA Report

## 1. Data contract status

- freeze_id: `data_freeze_gold_20260526_v1`
- contract tables ok: `20/20`
- hash mismatches: `0`
- row count mismatches: `0`
- schema/leakage issues: `0`

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

## 2. Gold matrix inventory

| table_name                                       |   rows |   columns |   n_rivers |   n_years | min_date   | max_date   |
|:-------------------------------------------------|-------:|----------:|-----------:|----------:|:-----------|:-----------|
| training_matrix_hydrocore.csv                    |    547 |        23 |          6 |        21 | 2003-06-18 | 2024-08-24 |
| training_matrix_basin_context.csv                |    547 |        52 |          6 |        21 | 2003-06-18 | 2024-08-24 |
| training_matrix_optical_matched_0d.csv           |    188 |        32 |          6 |        18 | 2004-06-14 | 2024-08-24 |
| training_matrix_optical_matched_1d.csv           |    297 |        32 |          6 |        19 | 2004-04-07 | 2024-08-24 |
| training_matrix_optical_matched_3d.csv           |    374 |        32 |          6 |        21 | 2003-08-12 | 2024-08-24 |
| training_matrix_optical_matched_7d.csv           |    420 |        32 |          6 |        21 | 2003-08-12 | 2024-08-24 |
| training_matrix_optical_matched_3d_hls.csv       |    144 |        32 |          6 |         9 | 2016-03-03 | 2024-08-19 |
| training_matrix_optical_matched_3d_landsat.csv   |    184 |        32 |          6 |        17 | 2003-08-12 | 2021-06-15 |
| training_matrix_optical_matched_3d_sentinel2.csv |     46 |        32 |          6 |         7 | 2018-01-23 | 2024-08-24 |
| prediction_grid_daily_hydrocore.csv              |  52594 |        16 |          6 |        26 | 2000-01-01 | 2025-12-31 |
| basin_attributes_curated.csv                     |    240 |        17 |          6 |           |            |            |
| basin_attributes_curated_wide.csv                |      6 |        30 |          6 |           |            |            |
| doc_labels_gold.csv                              |    547 |        29 |          6 |        21 | 2003-06-18 | 2024-08-24 |
| daily_discharge_gold.csv                         | 154370 |        13 |          6 |        91 | 1936-01-01 | 2026-01-31 |
| daily_hydroclimate_gold.csv                      |  52726 |        19 |          6 |        26 | 2000-01-01 | 2025-12-31 |
| optical_timeseries_gold.csv                      | 142058 |        27 |          6 |        23 | 2003-02-22 | 2025-12-09 |

## 3. DOC label coverage

| river     |   row_count | min_date   | max_date   |   n_years |   n_months |   may_july_samples |   non_may_july_samples |
|:----------|------------:|:-----------|:-----------|----------:|-----------:|-------------------:|-----------------------:|
| Kolyma    |          87 | 2003-08-26 | 2021-11-08 |        17 |         12 |                 31 |                     56 |
| Lena      |          87 | 2003-08-12 | 2021-11-25 |        17 |         12 |                 28 |                     59 |
| Mackenzie |         103 | 2003-06-24 | 2024-08-19 |        21 |         12 |                 34 |                     69 |
| Ob        |          87 | 2003-07-16 | 2021-11-27 |        17 |         12 |                 32 |                     55 |
| Yenisey   |          87 | 2004-03-19 | 2021-11-17 |        16 |         12 |                 31 |                     56 |
| Yukon     |          96 | 2003-06-18 | 2024-08-24 |        20 |         12 |                 37 |                     59 |

## 4. River/year/month/season coverage

Season windows are provisional and descriptive only. The spring freshet window is May-July; early season is May-June; summer is July-August; late season is September-October.

| season_window              | river     |   row_count |
|:---------------------------|:----------|------------:|
| early_season               | ALL       |         146 |
| late_season                | ALL       |          87 |
| spring_freshet_provisional | ALL       |         193 |
| summer                     | ALL       |         114 |
| early_season               | Kolyma    |          23 |
| early_season               | Lena      |          24 |
| early_season               | Mackenzie |          22 |
| early_season               | Ob        |          24 |
| early_season               | Yenisey   |          25 |
| early_season               | Yukon     |          28 |
| late_season                | Kolyma    |          15 |
| late_season                | Lena      |          15 |
| late_season                | Mackenzie |          14 |
| late_season                | Ob        |          12 |
| late_season                | Yenisey   |          14 |
| late_season                | Yukon     |          17 |
| spring_freshet_provisional | Kolyma    |          31 |
| spring_freshet_provisional | Lena      |          28 |
| spring_freshet_provisional | Mackenzie |          34 |
| spring_freshet_provisional | Ob        |          32 |
| spring_freshet_provisional | Yenisey   |          31 |
| spring_freshet_provisional | Yukon     |          37 |
| summer                     | Kolyma    |          19 |
| summer                     | Lena      |          15 |
| summer                     | Mackenzie |          25 |
| summer                     | Ob        |          18 |
| summer                     | Yenisey   |          16 |
| summer                     | Yukon     |          21 |

## 5. Hydrocore predictor completeness

| column                   |   nonmissing_count |   missing_count |   missing_rate | flag_missing_rate_gt_0_25   |
|:-------------------------|-------------------:|----------------:|---------------:|:----------------------------|
| Q_m3s                    |                545 |               2 |     0.00365631 | False                       |
| temperature_2m_C         |                513 |              34 |     0.0621572  | False                       |
| positive_degree_day_Cday |                513 |              34 |     0.0621572  | False                       |
| snow_cover_fraction      |                390 |             157 |     0.28702    | True                        |
| snow_depletion_rate_7d   |                329 |             218 |     0.398537   | True                        |
| surface_runoff_m         |                513 |              34 |     0.0621572  | False                       |
| sin_doy                  |                547 |               0 |     0          | False                       |
| cos_doy                  |                547 |               0 |     0          | False                       |

## 6. DOC distribution and outliers

| variable   | group_type    | group_value                |   n |   min |    p05 |    p25 |   median |    mean |      p75 |     p95 |   max |     std |
|:-----------|:--------------|:---------------------------|----:|------:|-------:|-------:|---------:|--------:|---------:|--------:|------:|--------:|
| DOC_mgC_L  | overall       | overall                    | 547 | 2.1   | 2.7    | 4      |   5.6    | 6.6514  |  8.5805  | 13.3    |  23.5 | 3.53786 |
| DOC_mgC_L  | river         | Kolyma                     |  87 | 2.349 | 2.5979 | 3.2    |   4.3    | 5.33453 |  6.2     | 11.4    |  18.4 | 3.02775 |
| DOC_mgC_L  | river         | Lena                       |  87 | 3.2   | 4.8    | 6.6    |   7.7    | 9.22337 | 10.655   | 18.02   |  23.5 | 4.36482 |
| DOC_mgC_L  | river         | Mackenzie                  | 103 | 2.3   | 3.2946 | 4      |   4.6    | 4.76082 |  5.4     |  6.6962 |   8.1 | 1.1063  |
| DOC_mgC_L  | river         | Ob                         |  87 | 4     | 4.7    | 7.55   |   9.6    | 9.3904  | 11       | 13.27   |  16.6 | 2.69074 |
| DOC_mgC_L  | river         | Yenisey                    |  87 | 2.2   | 2.8    | 3.35   |   5.2    | 5.9123  |  8.3805  | 11.22   |  13   | 2.92887 |
| DOC_mgC_L  | river         | Yukon                      |  96 | 2.1   | 2.4    | 3      |   5.1    | 5.73    |  7.225   | 13.225  |  15.9 | 3.31771 |
| DOC_mgC_L  | month         | 1                          |  29 | 2.3   | 2.6782 | 3.1    |   3.524  | 4.73959 |  6.6     |  8.76   |   9.9 | 2.29018 |
| DOC_mgC_L  | month         | 2                          |  30 | 2.4   | 2.545  | 2.903  |   3.95   | 4.6744  |  5.85625 |  8.65   |  10.6 | 2.09677 |
| DOC_mgC_L  | month         | 3                          |  39 | 2.1   | 2.2    | 2.8945 |   3.5    | 4.22021 |  4.892   |  7.5    |  12.7 | 2.09306 |
| DOC_mgC_L  | month         | 4                          |  33 | 2.2   | 2.4    | 3      |   3.7    | 4.12527 |  5.137   |  6.74   |   7.3 | 1.44037 |
| DOC_mgC_L  | month         | 5                          |  33 | 2.371 | 2.5    | 3.1    |   7.374  | 8.98755 | 13.9     | 17.86   |  23.4 | 5.94417 |
| DOC_mgC_L  | month         | 6                          | 113 | 4.039 | 4.76   | 6.9    |   9.4    | 9.62462 | 11.4     | 17.02   |  23.5 | 3.63091 |
| DOC_mgC_L  | month         | 7                          |  47 | 2.349 | 3.03   | 4.3885 |   5.8    | 6.76653 |  8.15    | 12.431  |  16.6 | 3.18285 |
| DOC_mgC_L  | month         | 8                          |  67 | 2.6   | 2.83   | 4.1535 |   5.4    | 6.07524 |  6.95    | 12.15   |  16   | 2.85678 |
| DOC_mgC_L  | month         | 9                          |  50 | 2.7   | 3.335  | 4.45   |   5.6665 | 6.17762 |  7.4285  | 10.21   |  14.8 | 2.40237 |
| DOC_mgC_L  | month         | 10                         |  37 | 3     | 3.36   | 5.2    |   6.6    | 6.26357 |  7.2     |  8.66   |  11.8 | 1.83477 |
| DOC_mgC_L  | month         | 11                         |  36 | 2.6   | 3.15   | 4.42   |   5.7775 | 6.23222 |  7.85    | 10.625  |  11.6 | 2.38143 |
| DOC_mgC_L  | month         | 12                         |  33 | 2.554 | 2.96   | 4.1    |   4.6    | 5.62673 |  7.3     | 10.386  |  11.1 | 2.42649 |
| DOC_mgC_L  | season_window | early_season               | 146 | 2.371 | 3.025  | 6.525  |   9.3255 | 9.48062 | 12.025   | 17.5    |  23.5 | 4.2488  |
| DOC_mgC_L  | season_window | late_season                |  87 | 2.7   | 3.26   | 4.6695 |   6      | 6.21417 |  7.35    |  9.9632 |  14.8 | 2.1678  |
| DOC_mgC_L  | season_window | spring_freshet_provisional | 193 | 2.349 | 3      | 5.4    |   8.561  | 8.81968 | 11.03    | 16.64   |  23.5 | 4.17426 |
| DOC_mgC_L  | season_window | summer                     | 114 | 2.349 | 2.865  | 4.318  |   5.4145 | 6.36025 |  8.075   | 12.3805 |  16.6 | 3.00124 |

## 7. Q/discharge distribution

| variable   | group_type   | group_value   |   n |        min |        p05 |        p25 |      median |        mean |         p75 |          p95 |         max |          std |
|:-----------|:-------------|:--------------|----:|-----------:|-----------:|-----------:|------------:|------------:|------------:|-------------:|------------:|-------------:|
| Q_m3s      | river        | Kolyma        |  85 |  148       |  158.8     |  366       |  4360       |  5632.41    |  7510       |  20020       |  24300      |  6191.31     |
| Q_m3s      | river        | Lena          |  87 | 1000       | 2131.2     | 3480       | 22300       | 31542.7     | 43700       | 106800       | 163000      | 36675.4      |
| Q_m3s      | river        | Mackenzie     | 103 | 2470       | 3030.1     | 3865       | 10600       | 10651.6     | 14776.5     |  24258.5     |  28800      |  7059.15     |
| Q_m3s      | river        | Ob            |  87 | 3350       | 4364       | 5760       | 10800       | 17248.8     | 31100       |  35040       |  36400      | 12113.7      |
| Q_m3s      | river        | Yenisey       |  87 | 5850       | 7380       | 9205       | 14700       | 24597.9     | 25900       |  80930       |  98200      | 23190.6      |
| Q_m3s      | river        | Yukon         |  96 | 1175       | 1412.25    | 2236       | 10321.5     |  9682.94    | 13684       |  19892.5     |  33414      |  6717.25     |
| log_Q      | river        | Kolyma        |  85 |    4.99721 |    5.06739 |    5.90263 |     8.38023 |     7.73692 |     8.92399 |      9.90432 |     10.0982 |     1.61551  |
| log_Q      | river        | Lena          |  87 |    6.90776 |    7.66367 |    8.15409 |    10.0123  |     9.52976 |    10.6851  |     11.5786  |     12.0015 |     1.41987  |
| log_Q      | river        | Mackenzie     | 103 |    7.81197 |    8.01635 |    8.25968 |     9.26861 |     9.02647 |     9.60079 |     10.0965  |     10.2681 |     0.734927 |
| log_Q      | river        | Ob            |  87 |    8.11672 |    8.37991 |    8.65866 |     9.2873  |     9.46692 |    10.345   |     10.4642  |     10.5023 |     0.793178 |
| log_Q      | river        | Yenisey       |  87 |    8.6742  |    8.90651 |    9.1275  |     9.5956  |     9.77995 |    10.162   |     11.3011  |     11.4948 |     0.770792 |
| log_Q      | river        | Yukon         |  96 |    7.06902 |    7.2527  |    7.71219 |     9.24182 |     8.82632 |     9.52398 |      9.89804 |     10.4167 |     0.954846 |

## 8. Hydroclimate predictor distribution

| variable                 | group_type   | group_value   |   n |            min |            p05 |            p25 |          median |            mean |             p75 |              p95 |             max |             std |
|:-------------------------|:-------------|:--------------|----:|---------------:|---------------:|---------------:|----------------:|----------------:|----------------:|-----------------:|----------------:|----------------:|
| Q_m3s                    | river        | Kolyma        |  85 |  148           |  158.8         |  366           |  4360           |  5632.41        |  7510           |  20020           |  24300          |  6191.31        |
| Q_m3s                    | river        | Lena          |  87 | 1000           | 2131.2         | 3480           | 22300           | 31542.7         | 43700           | 106800           | 163000          | 36675.4         |
| Q_m3s                    | river        | Mackenzie     | 103 | 2470           | 3030.1         | 3865           | 10600           | 10651.6         | 14776.5         |  24258.5         |  28800          |  7059.15        |
| Q_m3s                    | river        | Ob            |  87 | 3350           | 4364           | 5760           | 10800           | 17248.8         | 31100           |  35040           |  36400          | 12113.7         |
| Q_m3s                    | river        | Yenisey       |  87 | 5850           | 7380           | 9205           | 14700           | 24597.9         | 25900           |  80930           |  98200          | 23190.6         |
| Q_m3s                    | river        | Yukon         |  96 | 1175           | 1412.25        | 2236           | 10321.5         |  9682.94        | 13684           |  19892.5         |  33414          |  6717.25        |
| temperature_2m_C         | river        | Kolyma        |  87 |  -43.2613      |  -36.2082      |  -14.2959      |     3.30073     |    -3.05458     |    11.089       |     16.3796      |     21.4347     |    17.7108      |
| temperature_2m_C         | river        | Lena          |  87 |  -47.7546      |  -40.2322      |  -21.5608      |    -0.422039    |    -4.85279     |    12.2436      |     20.2785      |     26.7544     |    20.2622      |
| temperature_2m_C         | river        | Mackenzie     | 103 |  -32.3605      |  -28.4536      |  -18.2696      |     5.17756     |    -1.15225     |    12.4523      |     18.6435      |     26.1337     |    16.7733      |
| temperature_2m_C         | river        | Ob            |  87 |  -37.9342      |  -25.3184      |   -8.84082     |     2.43969     |    -0.135093    |    11.0046      |     16.4804      |     24.3039     |    14.2077      |
| temperature_2m_C         | river        | Yenisey       |  53 |   -7.79529     |   -3.14305     |    3.77764     |     8.40819     |     7.33208     |    12.2525      |     15.5305      |     24.1291     |     6.38603     |
| temperature_2m_C         | river        | Yukon         |  96 |  -31.4009      |  -20.205       |   -1.13882     |     6.60587     |     3.19299     |    11.1536      |     16.3241      |     19.1243     |    11.4491      |
| positive_degree_day_Cday | river        | Kolyma        |  87 |    0           |    0           |    0           |     3.30073     |     5.58139     |    11.089       |     16.3796      |     21.4347     |     6.34896     |
| positive_degree_day_Cday | river        | Lena          |  87 |    0           |    0           |    0           |     0           |     6.1387      |    12.2436      |     20.2785      |     26.7544     |     7.70608     |
| positive_degree_day_Cday | river        | Mackenzie     | 103 |    0           |    0           |    0           |     5.17756     |     6.74076     |    12.4523      |     18.6435      |     26.1337     |     7.25881     |
| positive_degree_day_Cday | river        | Ob            |  87 |    0           |    0           |    0           |     2.43969     |     5.81431     |    11.0046      |     16.4804      |     24.3039     |     6.65301     |
| positive_degree_day_Cday | river        | Yenisey       |  53 |    0           |    0           |    3.77764     |     8.40819     |     7.78438     |    12.2525      |     15.5305      |     24.1291     |     5.61695     |
| positive_degree_day_Cday | river        | Yukon         |  96 |    0           |    0           |    0           |     6.60587     |     6.53732     |    11.1536      |     16.3241      |     19.1243     |     5.85193     |
| snow_cover_fraction      | river        | Kolyma        |  64 |    0           |    0           |    0.00677223  |     0.101329    |     0.261159    |     0.512516    |      0.859663    |      0.988545   |     0.316851    |
| snow_cover_fraction      | river        | Lena          |  59 |    0           |    0           |    0.00210188  |     0.103087    |     0.260778    |     0.599296    |      0.803355    |      1          |     0.315334    |
| snow_cover_fraction      | river        | Mackenzie     |  74 |    0           |    0           |    0.000116536 |     0.0812385   |     0.245863    |     0.492531    |      0.850861    |      0.999477   |     0.318774    |
| snow_cover_fraction      | river        | Ob            |  54 |    0           |    0           |    0.000533939 |     0.0799469   |     0.302105    |     0.645158    |      0.985311    |      1          |     0.363578    |
| snow_cover_fraction      | river        | Yenisey       |  62 |    0           |    1.7145e-06  |    0.0270965   |     0.305916    |     0.381256    |     0.690734    |      0.95254     |      1          |     0.349464    |
| snow_cover_fraction      | river        | Yukon         |  77 |    0           |    0           |    0.012908    |     0.127203    |     0.260342    |     0.553502    |      0.726897    |      0.980648   |     0.29121     |
| snow_depletion_rate_7d   | river        | Kolyma        |  55 |   -0.999657    |   -0.879023    |   -0.182133    |    -0.00416262  |    -0.0829621   |     0.0569103   |      0.327405    |      0.972113   |     0.38097     |
| snow_depletion_rate_7d   | river        | Lena          |  52 |   -1           |   -0.974202    |   -0.0934998   |    -0.00194266  |    -0.0893787   |     0.0489647   |      0.42869     |      0.986921   |     0.411562    |
| snow_depletion_rate_7d   | river        | Mackenzie     |  66 |   -1           |   -0.948815    |   -0.0559508   |    -0.000573661 |    -0.0666969   |     0.0392774   |      0.754146    |      1          |     0.421646    |
| snow_depletion_rate_7d   | river        | Ob            |  47 |   -0.984568    |   -0.800831    |   -0.146477    |    -0.0016511   |    -0.0376592   |     0.0539123   |      0.9308      |      0.995084   |     0.455535    |
| snow_depletion_rate_7d   | river        | Yenisey       |  50 |   -0.974085    |   -0.951429    |   -0.162431    |    -0.00144856  |    -0.0414949   |     0.129217    |      0.751308    |      0.926232   |     0.456111    |
| snow_depletion_rate_7d   | river        | Yukon         |  59 |   -0.939987    |   -0.818109    |   -0.107636    |    -0.0149326   |    -0.0864374   |     0.0239534   |      0.315856    |      0.968351   |     0.341517    |
| surface_runoff_m         | river        | Kolyma        |  87 |    0           |    0           |    0           |     1.03452e-07 |     2.98544e-05 |     7.71007e-06 |      6.61475e-05 |      0.00087327 |     0.000121751 |
| surface_runoff_m         | river        | Lena          |  87 |    0           |    0           |    0           |     2.72108e-08 |     4.26102e-05 |     4.74829e-06 |      9.64529e-05 |      0.00175157 |     0.000224553 |
| surface_runoff_m         | river        | Mackenzie     | 103 |    0           |    0           |    0           |     5.72205e-08 |     3.2495e-05  |     1.89006e-06 |      3.83916e-05 |      0.00229352 |     0.000232379 |
| surface_runoff_m         | river        | Ob            |  87 |    0           |    0           |    0           |     6.37262e-08 |     0.000156296 |     1.25713e-05 |      0.000338423 |      0.00796673 |     0.000889692 |
| surface_runoff_m         | river        | Yenisey       |  53 |    1.49012e-08 |    1.17009e-06 |    1.08213e-05 |     0.000155057 |     0.0240791   |     0.00104238  |      0.175396    |      0.376133   |     0.076677    |

## 9. DOC-Q-season descriptive relationships

The correlations below are descriptive Spearman summaries only. They are not model results.

| river     | variable            |   n |   spearman_r |     p_value | interpretation_flag   |
|:----------|:--------------------|----:|-------------:|------------:|:----------------------|
| Kolyma    | Q_m3s               |  85 |    0.784699  | 1.00946e-21 | strong                |
| Kolyma    | log_Q               |  85 |    0.784699  | 1.00946e-21 | strong                |
| Kolyma    | temperature_2m_C    |  87 |    0.596965  | 2.80023e-10 | moderate              |
| Kolyma    | snow_cover_fraction |  64 |   -0.505882  | 1.35077e-05 | moderate              |
| Kolyma    | surface_runoff_m    |  87 |    0.253623  | 0.0174847   | weak                  |
| Lena      | Q_m3s               |  87 |    0.640736  | 3.3975e-12  | strong                |
| Lena      | log_Q               |  87 |    0.640736  | 3.3975e-12  | strong                |
| Lena      | temperature_2m_C    |  87 |    0.313388  | 0.0029564   | moderate              |
| Lena      | snow_cover_fraction |  59 |   -0.404491  | 0.00132553  | moderate              |
| Lena      | surface_runoff_m    |  87 |    0.280162  | 0.00833331  | weak                  |
| Mackenzie | Q_m3s               | 103 |    0.54258   | 1.21657e-09 | moderate              |
| Mackenzie | log_Q               | 103 |    0.54258   | 1.21657e-09 | moderate              |
| Mackenzie | temperature_2m_C    | 103 |    0.313425  | 0.00118117  | moderate              |
| Mackenzie | snow_cover_fraction |  74 |   -0.446041  | 5.29033e-05 | moderate              |
| Mackenzie | surface_runoff_m    | 103 |    0.0481405 | 0.629964    | weak;sign_uncertain   |
| Ob        | Q_m3s               |  87 |    0.578967  | 1.3841e-09  | moderate              |
| Ob        | log_Q               |  87 |    0.578967  | 1.3841e-09  | moderate              |
| Ob        | temperature_2m_C    |  87 |    0.622689  | 2.31035e-11 | strong                |
| Ob        | snow_cover_fraction |  54 |   -0.335379  | 0.0127213   | moderate              |
| Ob        | surface_runoff_m    |  87 |    0.338974  | 0.00121779  | moderate              |
| Yenisey   | Q_m3s               |  87 |    0.82425   | 7.97283e-27 | strong                |
| Yenisey   | log_Q               |  87 |    0.82425   | 7.97283e-27 | strong                |
| Yenisey   | temperature_2m_C    |  53 |    0.265347  | 0.0545704   | weak                  |
| Yenisey   | snow_cover_fraction |  62 |   -0.234127  | 0.0668993   | weak                  |
| Yenisey   | surface_runoff_m    |  53 |    0.441627  | 0.000798105 | moderate              |
| Yukon     | Q_m3s               |  96 |    0.833556  | 5.89847e-31 | strong                |
| Yukon     | log_Q               |  96 |    0.833556  | 5.89847e-31 | strong                |
| Yukon     | temperature_2m_C    |  96 |    0.45941   | 1.67881e-06 | moderate              |
| Yukon     | snow_cover_fraction |  77 |   -0.500596  | 2.22158e-06 | moderate              |
| Yukon     | surface_runoff_m    |  96 |    0.227319  | 0.0256705   | weak                  |

## 10. Optical matched subset audit

| table_name                                       | window       |   row_count |   unique_labels |   rivers_represented |   years_represented |   sensors_represented |   median_abs_days_offset |   p90_abs_days_offset |   median_valid_water_pixels |
|:-------------------------------------------------|:-------------|------------:|----------------:|---------------------:|--------------------:|----------------------:|-------------------------:|----------------------:|----------------------------:|
| training_matrix_optical_matched_0d.csv           | 0d           |         188 |             188 |                    6 |                  18 |                     3 |                        0 |                     0 |                         0   |
| training_matrix_optical_matched_1d.csv           | 1d           |         297 |             297 |                    6 |                  19 |                     3 |                        0 |                     1 |                         0   |
| training_matrix_optical_matched_3d.csv           | 3d           |         374 |             374 |                    6 |                  21 |                     3 |                        0 |                     2 |                         0   |
| training_matrix_optical_matched_7d.csv           | 7d           |         420 |             420 |                    6 |                  21 |                     3 |                        1 |                     4 |                         0   |
| training_matrix_optical_matched_3d_hls.csv       | 3d_hls       |         144 |             144 |                    6 |                   9 |                     1 |                        0 |                     1 |                         0   |
| training_matrix_optical_matched_3d_landsat.csv   | 3d_landsat   |         184 |             184 |                    6 |                  17 |                     1 |                        1 |                     3 |                        81.5 |
| training_matrix_optical_matched_3d_sentinel2.csv | 3d_sentinel2 |          46 |              46 |                    6 |                   7 |                     1 |                        0 |                     1 |                       539   |

| table_name                                       |   row_count |   unique_labels |   doc_median_subset |   doc_median_full_hydrocore |   doc_median_difference |   q_median_subset |   q_median_full_hydrocore |   q_median_difference |   month_distribution_difference |   river_composition_difference |
|:-------------------------------------------------|------------:|----------------:|--------------------:|----------------------------:|------------------------:|------------------:|--------------------------:|----------------------:|--------------------------------:|-------------------------------:|
| training_matrix_optical_matched_0d.csv           |         188 |             188 |              5.4    |                         5.6 |                 -0.2    |           11192.5 |                     10052 |                1140.5 |                        0.162482 |                      0.0723677 |
| training_matrix_optical_matched_1d.csv           |         297 |             297 |              5.5    |                         5.6 |                 -0.1    |           11400   |                     10052 |                1348   |                        0.12064  |                      0.0596397 |
| training_matrix_optical_matched_3d.csv           |         374 |             374 |              5.572  |                         5.6 |                 -0.028  |           11666.5 |                     10052 |                1614.5 |                        0.129628 |                      0.0613311 |
| training_matrix_optical_matched_7d.csv           |         420 |             420 |              5.6    |                         5.6 |                  0      |           11751.5 |                     10052 |                1699.5 |                        0.118312 |                      0.0509576 |
| training_matrix_optical_matched_3d_hls.csv       |         144 |             144 |              5      |                         5.6 |                 -0.6    |            7875   |                     10052 |               -2177   |                        0.150759 |                      0.0403209 |
| training_matrix_optical_matched_3d_landsat.csv   |         184 |             184 |              6.2    |                         5.6 |                  0.6    |           14300   |                     10052 |                4248   |                        0.302838 |                      0.0967133 |
| training_matrix_optical_matched_3d_sentinel2.csv |          46 |              46 |              5.7505 |                         5.6 |                  0.1505 |            9115   |                     10052 |                -937   |                        0.206939 |                      0.151061  |

## 11. Sensor-specific optical subset audit

| table_name                                       | window       | sensor     |   row_count |   unique_labels |   median_abs_days_offset |
|:-------------------------------------------------|:-------------|:-----------|------------:|----------------:|-------------------------:|
| training_matrix_optical_matched_0d.csv           | 0d           | HLS        |          98 |              98 |                        0 |
| training_matrix_optical_matched_0d.csv           | 0d           | Landsat    |          54 |              54 |                        0 |
| training_matrix_optical_matched_0d.csv           | 0d           | Sentinel-2 |          36 |              36 |                        0 |
| training_matrix_optical_matched_1d.csv           | 1d           | HLS        |         137 |             137 |                        0 |
| training_matrix_optical_matched_1d.csv           | 1d           | Landsat    |         115 |             115 |                        1 |
| training_matrix_optical_matched_1d.csv           | 1d           | Sentinel-2 |          45 |              45 |                        0 |
| training_matrix_optical_matched_3d.csv           | 3d           | HLS        |         144 |             144 |                        0 |
| training_matrix_optical_matched_3d.csv           | 3d           | Landsat    |         184 |             184 |                        1 |
| training_matrix_optical_matched_3d.csv           | 3d           | Sentinel-2 |          46 |              46 |                        0 |
| training_matrix_optical_matched_7d.csv           | 7d           | HLS        |         147 |             147 |                        0 |
| training_matrix_optical_matched_7d.csv           | 7d           | Landsat    |         226 |             226 |                        1 |
| training_matrix_optical_matched_7d.csv           | 7d           | Sentinel-2 |          47 |              47 |                        0 |
| training_matrix_optical_matched_3d_hls.csv       | 3d_hls       | HLS        |         144 |             144 |                        0 |
| training_matrix_optical_matched_3d_landsat.csv   | 3d_landsat   | Landsat    |         184 |             184 |                        1 |
| training_matrix_optical_matched_3d_sentinel2.csv | 3d_sentinel2 | Sentinel-2 |          46 |              46 |                        0 |

## 12. Basin attribute audit

| metric                         | category     |   value |
|:-------------------------------|:-------------|--------:|
| curated_attributes             | all          |     240 |
| model_use_true                 | all          |     174 |
| mechanism_use_true             | all          |     174 |
| needs_area_weighted_refinement | all          |     240 |
| wide_candidate_columns         | all          |      29 |
| attributes_by_category         | hydrology    |      96 |
| attributes_by_category         | other        |      66 |
| attributes_by_category         | physiography |      60 |
| attributes_by_category         | water_lake   |      18 |
| excluded_from_modeling         | all          |      66 |

| attribute_name   | source_field    | attribute_category   | model_use   | mechanism_use   | needs_area_weighted_refinement   | candidate_reason                                                                      | warning                                          |
|:-----------------|:----------------|:---------------------|:------------|:----------------|:---------------------------------|:--------------------------------------------------------------------------------------|:-------------------------------------------------|
| lka_pc_sse       | lka_pc_sse_mean | water_lake           | True        | True            | True                             | Interpretable basin hydrology/physiography candidate; only 6 river-level units exist. | Candidate list only; no model fitting performed. |
| lka_pc_sse       | lka_pc_sse_mean | water_lake           | True        | True            | True                             | Interpretable basin hydrology/physiography candidate; only 6 river-level units exist. | Candidate list only; no model fitting performed. |
| lka_pc_sse       | lka_pc_sse_mean | water_lake           | True        | True            | True                             | Interpretable basin hydrology/physiography candidate; only 6 river-level units exist. | Candidate list only; no model fitting performed. |
| lka_pc_sse       | lka_pc_sse_mean | water_lake           | True        | True            | True                             | Interpretable basin hydrology/physiography candidate; only 6 river-level units exist. | Candidate list only; no model fitting performed. |
| lka_pc_sse       | lka_pc_sse_mean | water_lake           | True        | True            | True                             | Interpretable basin hydrology/physiography candidate; only 6 river-level units exist. | Candidate list only; no model fitting performed. |
| lka_pc_sse       | lka_pc_sse_mean | water_lake           | True        | True            | True                             | Interpretable basin hydrology/physiography candidate; only 6 river-level units exist. | Candidate list only; no model fitting performed. |
| lka_pc_use       | lka_pc_use_mean | water_lake           | True        | True            | True                             | Interpretable basin hydrology/physiography candidate; only 6 river-level units exist. | Candidate list only; no model fitting performed. |
| lka_pc_use       | lka_pc_use_mean | water_lake           | True        | True            | True                             | Interpretable basin hydrology/physiography candidate; only 6 river-level units exist. | Candidate list only; no model fitting performed. |
| lka_pc_use       | lka_pc_use_mean | water_lake           | True        | True            | True                             | Interpretable basin hydrology/physiography candidate; only 6 river-level units exist. | Candidate list only; no model fitting performed. |
| lka_pc_use       | lka_pc_use_mean | water_lake           | True        | True            | True                             | Interpretable basin hydrology/physiography candidate; only 6 river-level units exist. | Candidate list only; no model fitting performed. |

Only six river-level basin units exist, so basin attributes should be treated as sensitivity or mechanism context unless a later modeling design explicitly handles that limitation.

## 13. Prediction grid coverage

The prediction grid is x-only. This EDA does not generate DOC predictions.

| river     |   row_count | min_date   | max_date   |   date_span_days |   distinct_days |   missing_dates | date_continuity_ok   |   missing_q_rate |   missing_hydroclimate_rate |   available_may_july_days |
|:----------|------------:|:-----------|:-----------|-----------------:|----------------:|----------------:|:---------------------|-----------------:|----------------------------:|--------------------------:|
| Kolyma    |        9497 | 2000-01-01 | 2025-12-31 |             9497 |            9497 |               0 | True                 |       0.0206381  |                    0.501211 |                      2392 |
| Lena      |        9497 | 2000-01-01 | 2025-12-31 |             9497 |            9497 |               0 | True                 |       0.0153733  |                    0.49542  |                      2392 |
| Mackenzie |        9472 | 2000-01-01 | 2025-12-06 |             9472 |            9472 |               0 | True                 |       0.00348395 |                    0.509291 |                      2392 |
| Ob        |        9476 | 2000-01-01 | 2025-12-10 |             9476 |            9476 |               0 | True                 |       0.00633179 |                    0.558991 |                      2392 |
| Yenisey   |        5241 | 2000-03-03 | 2025-11-03 |             9377 |            5241 |            4136 | False                |       0          |                    0.346499 |                      2313 |
| Yukon     |        9411 | 2000-01-01 | 2025-10-06 |             9411 |            9411 |               0 | True                 |       0.0484539  |                    0.541813 |                      2392 |

## 14. Candidate modeling scopes

| scope                          | required_data                                    |   available_rows |   n_rivers |   n_years | minimum_threshold_met   | recommended_for_phase_3   | caveat                                                     |
|:-------------------------------|:-------------------------------------------------|-----------------:|-----------:|----------:|:------------------------|:--------------------------|:-----------------------------------------------------------|
| season_only_baseline           | DOC + season terms                               |              547 |          6 |        21 | True                    | True                      | Descriptive seasonal baseline only after EDA.              |
| q_season_baseline              | DOC + Q + season terms                           |              545 |          6 |        21 | True                    | True                      | No model fitted in EDA phase.                              |
| hydroclimate_complete_case     | DOC + Q + hydroclimate complete cases            |              325 |          6 |        21 | False                   | True                      | Complete-case analysis may change sample composition.      |
| hydroclimate_missingness_aware | DOC + Q + hydroclimate with missingness strategy |              547 |          6 |        21 | True                    | True                      | Requires explicit missingness policy before fitting.       |
| river_effects_model            | DOC + predictors + river grouping                |              547 |          6 |        21 | True                    | True                      | Only six rivers; partial pooling may be useful later.      |
| leave_one_year_out_cv          | multiple years per river                         |              547 |          6 |        21 | True                    | True                      | Temporal folds may be uneven by river.                     |
| leave_one_river_out_cv         | six river groups                                 |              547 |          6 |        21 | True                    | False                     | High-risk extrapolation with only six river units.         |
| optical_3d_any_sensor          | 3-day optical matched subset                     |              374 |          6 |        21 | True                    | True                      | Sensitivity scope, not primary baseline.                   |
| optical_3d_hls_only            | 3-day HLS optical subset                         |              144 |          6 |        21 | True                    | True                      | Sensor-specific sensitivity only.                          |
| optical_3d_landsat_only        | 3-day Landsat optical subset                     |              184 |          6 |        21 | True                    | True                      | Sensor-specific sensitivity only.                          |
| optical_3d_sentinel2_only      | 3-day Sentinel-2 optical subset                  |               46 |          6 |        21 | False                   | False                     | Small subset; likely underpowered.                         |
| basin_context_sensitivity      | basin context matrix                             |              547 |          6 |        21 | True                    | False                     | Only six river-level basin units; use sensitivity framing. |
| daily_prediction_grid_ready    | daily x-only prediction grid                     |            52594 |          6 |        21 | True                    | True                      | Grid is x-only; no DOC predictions generated now.          |

## 15. Recommended baseline modeling sequence

1. Season-only baseline for calibration sanity checks.
2. Q + season baseline.
3. Hydroclimate missingness-aware baseline after documenting imputation or complete-case policy.
4. River-aware model structure if phase 3 explicitly addresses the six-river grouping limit.
5. Optical and basin-context sensitivity analyses after the primary hydrocore baseline is stable.

## 16. Risks and caveats

- Season windows are provisional and should not be interpreted as final hydrologic freshet definitions.
- Optical matched subsets may have month, river, DOC, or Q composition differences from the full hydrocore set.
- Sentinel-2 3-day matched sample size is small compared with any-sensor and Landsat subsets.
- Basin context has only six river-level units, limiting standalone basin-attribute inference.
- Correlations are descriptive and can be confounded by season and river structure.

## 17. Explicit statement: no model trained, no prediction, no flux

No model was trained. No DOC prediction was generated. No flux was generated. Only frozen gold data were read.

## EDA-specific checks

| check_name                    | passed   | message                                      |
|:------------------------------|:---------|:---------------------------------------------|
| gold_contract_ok              | True     | Gold contract verified before EDA.           |
| schema_checks_ok              | True     | Schema and leakage checks passed before EDA. |
| no_model_output_dirs          | True     | []                                           |
| no_model_binaries             | True     | model_binary_count=0                         |
| figures_generated_or_optional | True     | figures_generated=10                         |

## Generated artifacts

- EDA tables generated: `26`
- EDA figures generated: `10`
