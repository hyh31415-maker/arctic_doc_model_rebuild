# Baseline DOC Concentration Model Report

## 1. Scope and guardrails

This phase trains simple DOC concentration baseline models for cross-validation diagnostics only. It does not generate production daily DOC predictions and does not compute flux.

- model input table: `training_matrix_hydrocore.csv`
- excluded in this phase: prediction grid, optical matrices, basin context matrices, lab optical/CDOM, flux products
- model artifact policy: no final production model artifact is saved

## 2. Data contract status

- freeze_id: `data_freeze_gold_20260526_v1`
- source_tag: `data_freeze_gold_20260526_v1`
- gold_data_dir: `D:\Hao\Desktop\冰冻圈水文\北极大河\arctic_doc_data_audit\data\processed\gold`
- contract_tables_ok: `20/20`
- hash_mismatches: `0`
- row_count_mismatches: `0`

## 3. Input matrix summary

| feature_set                         | required_columns                                                                                                            |   rows_total |   rows_available |   rows_dropped |   drop_rate | sensitivity_only   | caveat                                                                                         |
|:------------------------------------|:----------------------------------------------------------------------------------------------------------------------------|-------------:|-----------------:|---------------:|------------:|:-------------------|:-----------------------------------------------------------------------------------------------|
| F0_intercept_only                   | nan                                                                                                                         |          547 |              547 |              0 |  0          | False              | nan                                                                                            |
| F1_season_only                      | sin_doy;cos_doy                                                                                                             |          547 |              547 |              0 |  0          | False              | nan                                                                                            |
| F2_q_season                         | log_Q;sin_doy;cos_doy                                                                                                       |          547 |              545 |              2 |  0.00365631 | False              | nan                                                                                            |
| F3_q_season_river_fixed             | log_Q;sin_doy;cos_doy;river                                                                                                 |          547 |              545 |              2 |  0.00365631 | False              | Leave-one-river-out is a structural stress test for unseen river categories.                   |
| F4_reduced_hydroclimate             | log_Q;sin_doy;cos_doy;temperature_2m_C;positive_degree_day_Cday;surface_runoff_m                                            |          547 |              511 |             36 |  0.0658135  | False              | nan                                                                                            |
| F5_snow_hydroclimate_complete_case  | log_Q;sin_doy;cos_doy;temperature_2m_C;positive_degree_day_Cday;surface_runoff_m;snow_cover_fraction;snow_depletion_rate_7d |          547 |              325 |            222 |  0.40585    | True               | Snow variables have high missingness; do not use as main model without a missingness strategy. |
| F6_reduced_hydroclimate_river_fixed | log_Q;sin_doy;cos_doy;temperature_2m_C;positive_degree_day_Cday;surface_runoff_m;river                                      |          547 |              511 |             36 |  0.0658135  | False              | River fixed effects are not reliable for held-out unseen rivers.                               |

## 4. Feature sets

| feature_set                         | description                                                                      | numeric_features                                                                                                            | categorical_features   | all_features                                                                                                                | sensitivity_only   | caveat                                                                                         |
|:------------------------------------|:---------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------|:-----------------------|:----------------------------------------------------------------------------------------------------------------------------|:-------------------|:-----------------------------------------------------------------------------------------------|
| F0_intercept_only                   | Training-fold mean DOC baseline.                                                 | nan                                                                                                                         | nan                    | nan                                                                                                                         | False              | nan                                                                                            |
| F1_season_only                      | Seasonal harmonic baseline.                                                      | sin_doy;cos_doy                                                                                                             | nan                    | sin_doy;cos_doy                                                                                                             | False              | nan                                                                                            |
| F2_q_season                         | Discharge plus seasonal harmonics.                                               | log_Q;sin_doy;cos_doy                                                                                                       | nan                    | log_Q;sin_doy;cos_doy                                                                                                       | False              | nan                                                                                            |
| F3_q_season_river_fixed             | Discharge, seasonal harmonics, and river one-hot fixed effects.                  | log_Q;sin_doy;cos_doy                                                                                                       | river                  | log_Q;sin_doy;cos_doy;river                                                                                                 | False              | Leave-one-river-out is a structural stress test for unseen river categories.                   |
| F4_reduced_hydroclimate             | Q, season, temperature, positive degree days, and surface runoff complete cases. | log_Q;sin_doy;cos_doy;temperature_2m_C;positive_degree_day_Cday;surface_runoff_m                                            | nan                    | log_Q;sin_doy;cos_doy;temperature_2m_C;positive_degree_day_Cday;surface_runoff_m                                            | False              | nan                                                                                            |
| F5_snow_hydroclimate_complete_case  | Complete-case snow hydroclimate sensitivity.                                     | log_Q;sin_doy;cos_doy;temperature_2m_C;positive_degree_day_Cday;surface_runoff_m;snow_cover_fraction;snow_depletion_rate_7d | nan                    | log_Q;sin_doy;cos_doy;temperature_2m_C;positive_degree_day_Cday;surface_runoff_m;snow_cover_fraction;snow_depletion_rate_7d | True               | Snow variables have high missingness; do not use as main model without a missingness strategy. |
| F6_reduced_hydroclimate_river_fixed | Reduced hydroclimate model with river one-hot fixed effects.                     | log_Q;sin_doy;cos_doy;temperature_2m_C;positive_degree_day_Cday;surface_runoff_m                                            | river                  | log_Q;sin_doy;cos_doy;temperature_2m_C;positive_degree_day_Cday;surface_runoff_m;river                                      | False              | River fixed effects are not reliable for held-out unseen rivers.                               |

## 5. Validation schemes

| validation_scheme     | validation_role           | group_column   | description                                                             | stress_test   | primary_for_model_selection   |
|:----------------------|:--------------------------|:---------------|:------------------------------------------------------------------------|:--------------|:------------------------------|
| leave_one_year_out    | primary                   | year           | Leave one sampled year out at a time.                                   | False         | True                          |
| river_year_groupkfold | secondary_structure_check | river_year     | Five-fold GroupKFold using river-year groups.                           | False         | False                         |
| leave_one_river_out   | stress_test               | river          | Held-out river stress test; not primary selection with only six rivers. | True          | False                         |

## 6. Overall metrics

| model_id          | feature_set                         | validation_scheme   |   n_test_total |   n_folds |   n_train_total |     rmse |     mae |   median_absolute_error |         r2 |   bias_mean |   bias_median |   spearman_r |   pearson_r |
|:------------------|:------------------------------------|:--------------------|---------------:|----------:|----------------:|---------:|--------:|------------------------:|-----------:|------------:|--------------:|-------------:|------------:|
| ridge_alpha_10    | F3_q_season_river_fixed             | leave_one_river_out |            545 |         6 |            2725 |  3.21482 | 2.58054 |                 2.1609  |  0.173736  |  0.0337747  |     -0.610648 |     0.492592 |   0.496948  |
| ridge_alpha_1     | F3_q_season_river_fixed             | leave_one_river_out |            545 |         6 |            2725 |  3.24625 | 2.60623 |                 2.16244 |  0.157498  |  0.0129035  |     -0.750323 |     0.49441  |   0.503605  |
| ridge_alpha_0.1   | F3_q_season_river_fixed             | leave_one_river_out |            545 |         6 |            2725 |  3.25036 | 2.60945 |                 2.1513  |  0.155366  |  0.0098076  |     -0.778702 |     0.494656 |   0.504444  |
| linear_regression | F3_q_season_river_fixed             | leave_one_river_out |            545 |         6 |            2725 |  3.25083 | 2.60981 |                 2.15205 |  0.155122  |  0.00944799 |     -0.781955 |     0.494778 |   0.504539  |
| ridge_alpha_10    | F2_q_season                         | leave_one_river_out |            545 |         6 |            2725 |  3.30881 | 2.6423  |                 2.19429 |  0.124716  |  0.184245   |     -0.415853 |     0.470717 |   0.462754  |
| ridge_alpha_1     | F2_q_season                         | leave_one_river_out |            545 |         6 |            2725 |  3.34638 | 2.67146 |                 2.19161 |  0.104725  |  0.200405   |     -0.381573 |     0.468214 |   0.460947  |
| ridge_alpha_0.1   | F2_q_season                         | leave_one_river_out |            545 |         6 |            2725 |  3.35062 | 2.67471 |                 2.19688 |  0.102453  |  0.202176   |     -0.38101  |     0.468103 |   0.460737  |
| linear_regression | F2_q_season                         | leave_one_river_out |            545 |         6 |            2725 |  3.3511  | 2.67508 |                 2.19746 |  0.102197  |  0.202375   |     -0.381145 |     0.468138 |   0.460713  |
| ridge_alpha_10    | F1_season_only                      | leave_one_river_out |            547 |         6 |            2735 |  3.56533 | 2.8218  |                 2.48392 | -0.0174489 | -0.0246045  |     -0.740466 |     0.168133 |   0.171304  |
| ridge_alpha_1     | F1_season_only                      | leave_one_river_out |            547 |         6 |            2735 |  3.56661 | 2.82183 |                 2.49169 | -0.0181814 | -0.0246938  |     -0.721704 |     0.171258 |   0.174295  |
| ridge_alpha_0.1   | F1_season_only                      | leave_one_river_out |            547 |         6 |            2735 |  3.56675 | 2.82185 |                 2.49458 | -0.0182624 | -0.0247029  |     -0.721371 |     0.171593 |   0.174594  |
| linear_regression | F1_season_only                      | leave_one_river_out |            547 |         6 |            2735 |  3.56677 | 2.82185 |                 2.49435 | -0.0182714 | -0.0247039  |     -0.721334 |     0.171629 |   0.174627  |
| mean_baseline     | F0_intercept_only                   | leave_one_river_out |            547 |         6 |            2735 |  3.74059 | 2.98112 |                 2.60881 | -0.11994   | -0.0190923  |     -1.10153  |    -0.507149 |  -0.522718  |
| linear_regression | F6_reduced_hydroclimate_river_fixed | leave_one_river_out |            511 |         6 |            2555 |  6.50806 | 2.99078 |                 2.17259 | -2.35581   |  0.700334   |     -0.47518  |     0.538609 |   0.154987  |
| ridge_alpha_0.1   | F6_reduced_hydroclimate_river_fixed | leave_one_river_out |            511 |         6 |            2555 |  6.5107  | 2.99089 |                 2.17135 | -2.35854   |  0.700686   |     -0.472258 |     0.538376 |   0.154588  |
| ridge_alpha_1     | F6_reduced_hydroclimate_river_fixed | leave_one_river_out |            511 |         6 |            2555 |  6.5336  | 2.99173 |                 2.16045 | -2.3822    |  0.703682   |     -0.446863 |     0.536741 |   0.15111   |
| linear_regression | F4_reduced_hydroclimate             | leave_one_river_out |            511 |         6 |            2555 |  6.65693 | 3.18407 |                 2.28052 | -2.5111    |  0.878088   |     -0.270759 |     0.491769 |   0.138614  |
| ridge_alpha_0.1   | F4_reduced_hydroclimate             | leave_one_river_out |            511 |         6 |            2555 |  6.65747 | 3.18354 |                 2.2757  | -2.51167   |  0.877741   |     -0.2702   |     0.491754 |   0.138435  |
| ridge_alpha_1     | F4_reduced_hydroclimate             | leave_one_river_out |            511 |         6 |            2555 |  6.66214 | 3.17889 |                 2.27228 | -2.5166    |  0.874666   |     -0.265159 |     0.491305 |   0.136855  |
| ridge_alpha_10    | F4_reduced_hydroclimate             | leave_one_river_out |            511 |         6 |            2555 |  6.69273 | 3.14006 |                 2.25313 | -2.54896   |  0.84737    |     -0.244516 |     0.488897 |   0.123159  |
| ridge_alpha_10    | F6_reduced_hydroclimate_river_fixed | leave_one_river_out |            511 |         6 |            2555 |  6.695   | 2.9971  |                 2.16451 | -2.55138   |  0.721008   |     -0.373483 |     0.52271  |   0.124427  |
| ridge_alpha_10    | F5_snow_hydroclimate_complete_case  | leave_one_river_out |            325 |         6 |            1625 |  9.89791 | 3.59575 |                 2.21278 | -5.76244   |  1.23509    |     -0.3337   |     0.524955 |   0.0564864 |
| ridge_alpha_1     | F5_snow_hydroclimate_complete_case  | leave_one_river_out |            325 |         6 |            1625 | 10.7657  | 3.74296 |                 2.31563 | -7.00026   |  1.36282    |     -0.376271 |     0.524445 |   0.0507308 |
| ridge_alpha_0.1   | F5_snow_hydroclimate_complete_case  | leave_one_river_out |            325 |         6 |            1625 | 10.8798  | 3.76234 |                 2.32665 | -7.17064   |  1.37894    |     -0.379437 |     0.523671 |   0.0498829 |
| linear_regression | F5_snow_hydroclimate_complete_case  | leave_one_river_out |            325 |         6 |            1625 | 10.8929  | 3.76456 |                 2.32795 | -7.19039   |  1.38079    |     -0.381918 |     0.523497 |   0.0497838 |
| ridge_alpha_1     | F6_reduced_hydroclimate_river_fixed | leave_one_year_out  |            511 |        21 |           10220 |  2.24078 | 1.66492 |                 1.22834 |  0.602175  |  0.00998951 |     -0.219564 |     0.803006 |   0.776022  |
| ridge_alpha_0.1   | F6_reduced_hydroclimate_river_fixed | leave_one_year_out  |            511 |        21 |           10220 |  2.24085 | 1.66672 |                 1.22685 |  0.602149  |  0.00983827 |     -0.21782  |     0.802892 |   0.776051  |
| linear_regression | F6_reduced_hydroclimate_river_fixed | leave_one_year_out  |            511 |        21 |           10220 |  2.24087 | 1.66693 |                 1.22376 |  0.60214   |  0.00982082 |     -0.216126 |     0.802786 |   0.776052  |
| ridge_alpha_10    | F6_reduced_hydroclimate_river_fixed | leave_one_year_out  |            511 |        21 |           10220 |  2.25038 | 1.66063 |                 1.24358 |  0.598759  |  0.0110511  |     -0.239618 |     0.805209 |   0.774489  |
| ridge_alpha_1     | F3_q_season_river_fixed             | leave_one_year_out  |            545 |        21 |           10900 |  2.25785 | 1.71611 |                 1.34859 |  0.592437  |  0.00277652 |     -0.199575 |     0.787391 |   0.769721  |

## 7. Metrics by river

| model_id          | feature_set                        | validation_scheme   | river     |   n_test_total |   n_folds |   n_train_total |    rmse |      mae |   median_absolute_error |          r2 |    bias_mean |   bias_median |   spearman_r |   pearson_r |
|:------------------|:-----------------------------------|:--------------------|:----------|---------------:|----------:|----------------:|--------:|---------:|------------------------:|------------:|-------------:|--------------:|-------------:|------------:|
| linear_regression | F1_season_only                     | leave_one_year_out  | Kolyma    |             87 |        17 |             nan | 3.0236  | 2.51178  |                2.46509  | -0.00885683 | -1.3571      |    -1.92448   |    0.505355  |    0.441526 |
| linear_regression | F1_season_only                     | leave_one_year_out  | Lena      |             87 |        17 |             nan | 4.79071 | 3.29206  |                1.71253  | -0.218675   |  2.71196     |     1.49398   |    0.327187  |    0.439911 |
| linear_regression | F1_season_only                     | leave_one_year_out  | Mackenzie |            103 |        21 |             nan | 2.39247 | 2.02867  |                2.04108  | -3.7226     | -1.90699     |    -2.04108   |    0.298489  |    0.303047 |
| linear_regression | F1_season_only                     | leave_one_year_out  | Ob        |             87 |        17 |             nan | 3.68532 | 3.16185  |                2.91931  | -0.897703   |  2.78614     |     2.8168    |    0.475016  |    0.436976 |
| linear_regression | F1_season_only                     | leave_one_year_out  | Yenisey   |             87 |        16 |             nan | 2.39471 | 1.99967  |                1.75373  |  0.323719   | -0.706273    |    -0.854244  |    0.663884  |    0.648917 |
| linear_regression | F1_season_only                     | leave_one_year_out  | Yukon     |             96 |        20 |             nan | 3.17738 | 2.64042  |                2.28209  |  0.0731517  | -1.04694     |    -1.65641   |    0.430037  |    0.420591 |
| linear_regression | F2_q_season                        | leave_one_year_out  | Kolyma    |             85 |        17 |             nan | 2.506   | 1.95338  |                1.79552  |  0.316102   |  1.24283     |     1.50039   |    0.779525  |    0.719618 |
| linear_regression | F2_q_season                        | leave_one_year_out  | Lena      |             87 |        17 |             nan | 3.62574 | 2.65693  |                1.99433  |  0.301956   |  1.70444     |     1.3973    |    0.611754  |    0.687593 |
| linear_regression | F2_q_season                        | leave_one_year_out  | Mackenzie |            103 |        21 |             nan | 2.15479 | 1.92122  |                1.92825  | -2.83088    | -1.87504     |    -1.92825   |    0.536622  |    0.550104 |
| linear_regression | F2_q_season                        | leave_one_year_out  | Ob        |             87 |        17 |             nan | 3.11455 | 2.52816  |                2.08257  | -0.355405   |  1.98262     |     1.71768   |    0.406789  |    0.44565  |
| linear_regression | F2_q_season                        | leave_one_year_out  | Yenisey   |             87 |        16 |             nan | 2.88423 | 2.40234  |                2.08422  |  0.018975   | -2.06252     |    -1.88651   |    0.568664  |    0.783935 |
| linear_regression | F2_q_season                        | leave_one_year_out  | Yukon     |             96 |        20 |             nan | 2.37334 | 1.90163  |                1.55922  |  0.482883   | -0.524217    |    -1.29533   |    0.823442  |    0.791407 |
| linear_regression | F3_q_season_river_fixed            | leave_one_year_out  | Kolyma    |             85 |        17 |             nan | 2.27549 | 1.90879  |                1.76231  |  0.436129   |  0.0150566   |     0.0931854 |    0.781373  |    0.726651 |
| linear_regression | F3_q_season_river_fixed            | leave_one_year_out  | Lena      |             87 |        17 |             nan | 3.12378 | 2.36754  |                1.85762  |  0.481856   |  0.0019171   |    -0.173274  |    0.617205  |    0.696767 |
| linear_regression | F3_q_season_river_fixed            | leave_one_year_out  | Mackenzie |            103 |        21 |             nan | 1.07496 | 0.831322 |                0.619908 |  0.0466114  | -0.000223631 |    -0.0838382 |    0.548993  |    0.564514 |
| linear_regression | F3_q_season_river_fixed            | leave_one_year_out  | Ob        |             87 |        17 |             nan | 2.49347 | 2.04478  |                1.72122  |  0.131266   |  0.00141914  |    -0.13519   |    0.354669  |    0.396492 |
| linear_regression | F3_q_season_river_fixed            | leave_one_year_out  | Yenisey   |             87 |        16 |             nan | 2.04666 | 1.70681  |                1.58231  |  0.506016   | -0.00346665  |     0.144159  |    0.459062  |    0.742734 |
| linear_regression | F3_q_season_river_fixed            | leave_one_year_out  | Yukon     |             96 |        20 |             nan | 2.20084 | 1.62377  |                1.12828  |  0.555321   |  0.00210725  |    -0.76855   |    0.848127  |    0.813858 |
| linear_regression | F4_reduced_hydroclimate            | leave_one_year_out  | Kolyma    |             85 |        17 |             nan | 2.46148 | 1.8561   |                1.39861  |  0.340189   |  1.35576     |     1.19835   |    0.787765  |    0.75062  |
| linear_regression | F4_reduced_hydroclimate            | leave_one_year_out  | Lena      |             87 |        17 |             nan | 3.30225 | 2.35051  |                1.73134  |  0.420959   |  1.09511     |     0.65498   |    0.635552  |    0.715616 |
| linear_regression | F4_reduced_hydroclimate            | leave_one_year_out  | Mackenzie |            103 |        21 |             nan | 2.42305 | 2.2121   |                2.15065  | -3.84409    | -2.20581     |    -2.15065   |    0.516319  |    0.564081 |
| linear_regression | F4_reduced_hydroclimate            | leave_one_year_out  | Ob        |             87 |        17 |             nan | 3.01735 | 2.42128  |                2.10132  | -0.272126   |  1.64801     |     1.39705   |    0.333823  |    0.361652 |
| linear_regression | F4_reduced_hydroclimate            | leave_one_year_out  | Yenisey   |             53 |        16 |             nan | 2.50816 | 1.93737  |                1.53531  |  0.189505   | -1.13821     |    -1.18112   |    0.60327   |    0.598143 |
| linear_regression | F4_reduced_hydroclimate            | leave_one_year_out  | Yukon     |             96 |        20 |             nan | 2.35548 | 1.90065  |                1.65337  |  0.490633   | -0.55365     |    -1.24967   |    0.823706  |    0.793702 |
| linear_regression | F5_snow_hydroclimate_complete_case | leave_one_year_out  | Kolyma    |             55 |        15 |             nan | 2.56869 | 1.90494  |                1.42772  |  0.403911   |  1.44886     |     1.16289   |    0.809773  |    0.780354 |
| linear_regression | F5_snow_hydroclimate_complete_case | leave_one_year_out  | Lena      |             52 |        17 |             nan | 3.49085 | 2.50445  |                1.73995  |  0.501834   |  0.993745    |     0.340022  |    0.725941  |    0.753647 |
| linear_regression | F5_snow_hydroclimate_complete_case | leave_one_year_out  | Mackenzie |             66 |        20 |             nan | 2.45881 | 2.20007  |                2.14208  | -3.28045    | -2.19854     |    -2.14208   |    0.567434  |    0.621302 |
| linear_regression | F5_snow_hydroclimate_complete_case | leave_one_year_out  | Ob        |             47 |        16 |             nan | 3.65853 | 3.00663  |                2.64728  | -0.968942   |  2.13462     |     2.12083   |    0.0297316 |    0.08846  |
| linear_regression | F5_snow_hydroclimate_complete_case | leave_one_year_out  | Yenisey   |             46 |        16 |             nan | 2.48951 | 1.85677  |                1.50342  |  0.172506   | -1.15074     |    -1.15483   |    0.586381  |    0.599813 |
| linear_regression | F5_snow_hydroclimate_complete_case | leave_one_year_out  | Yukon     |             59 |        19 |             nan | 2.36576 | 1.87912  |                1.43635  |  0.537381   | -0.434751    |    -1.11374   |    0.801292  |    0.810426 |

## 8. Metrics by year

| model_id          | feature_set    | validation_scheme   |   year |   n_test_total |   n_folds |   n_train_total |      rmse |       mae |   median_absolute_error |           r2 |   bias_mean |   bias_median |   spearman_r |   pearson_r |
|:------------------|:---------------|:--------------------|-------:|---------------:|----------:|----------------:|----------:|----------:|------------------------:|-------------:|------------:|--------------:|-------------:|------------:|
| linear_regression | F1_season_only | leave_one_year_out  |   2003 |              5 |         1 |             nan | 2.94927   | 2.77065   |               2.60061   |   0.00841283 |  -1.06696   |    -2.50771   |    0.7       |    0.775925 |
| linear_regression | F1_season_only | leave_one_year_out  |   2004 |             42 |         1 |             nan | 3.24984   | 2.80138   |               2.63933   |   0.0820417  |  -0.413093  |    -0.853887  |    0.228452  |    0.323197 |
| linear_regression | F1_season_only | leave_one_year_out  |   2005 |             42 |         1 |             nan | 2.90552   | 2.41771   |               2.16694   |   0.0829345  |   0.113571  |     0.315544  |    0.195752  |    0.291871 |
| linear_regression | F1_season_only | leave_one_year_out  |   2006 |             11 |         1 |             nan | 3.55362   | 3.00316   |               2.98171   |  -0.0655186  |   1.67071   |     1.99441   |    0.363636  |    0.412419 |
| linear_regression | F1_season_only | leave_one_year_out  |   2007 |              1 |         1 |             nan | 0.447913  | 0.447913  |               0.447913  | nan          |  -0.447913  |    -0.447913  |  nan         |  nan        |
| linear_regression | F1_season_only | leave_one_year_out  |   2009 |             29 |         1 |             nan | 3.80239   | 2.98441   |               2.46509   |   0.0406853  |   0.83424   |     0.377866  |    0.215184  |    0.297268 |
| linear_regression | F1_season_only | leave_one_year_out  |   2010 |             31 |         1 |             nan | 4.32385   | 3.37822   |               2.60331   |  -0.201997   |   2.07163   |     1.96937   |    0.0213882 |    0.273083 |
| linear_regression | F1_season_only | leave_one_year_out  |   2011 |             30 |         1 |             nan | 5.03183   | 3.50883   |               2.07666   |  -0.173047   |   2.57897   |     1.35369   |    0.283519  |    0.402243 |
| linear_regression | F1_season_only | leave_one_year_out  |   2012 |             23 |         1 |             nan | 3.21813   | 2.43271   |               1.91865   |   0.173576   |  -0.267749  |    -1.01659   |    0.463897  |    0.424408 |
| linear_regression | F1_season_only | leave_one_year_out  |   2013 |             33 |         1 |             nan | 3.22504   | 2.64079   |               2.55696   |   0.0260456  |  -0.258691  |    -0.634809  |    0.173928  |    0.241145 |
| linear_regression | F1_season_only | leave_one_year_out  |   2014 |             35 |         1 |             nan | 4.1238    | 2.83235   |               1.8545    |   0.13795    |   0.39018   |    -0.860184  |    0.386039  |    0.399023 |
| linear_regression | F1_season_only | leave_one_year_out  |   2015 |             36 |         1 |             nan | 3.37348   | 2.83636   |               2.81095   |  -0.0998167  |  -0.88499   |    -1.57009   |    0.204379  |    0.172103 |
| linear_regression | F1_season_only | leave_one_year_out  |   2016 |             37 |         1 |             nan | 2.27446   | 1.90336   |               1.49347   |  -0.0168114  |  -1.02256   |    -0.985665  |    0.429724  |    0.451286 |
| linear_regression | F1_season_only | leave_one_year_out  |   2017 |             35 |         1 |             nan | 2.61787   | 2.18856   |               2.31821   |  -0.00539886 |  -0.288088  |    -0.76616   |    0.249492  |    0.260335 |
| linear_regression | F1_season_only | leave_one_year_out  |   2018 |             35 |         1 |             nan | 2.87436   | 2.41137   |               2.20885   |  -0.187737   |  -0.538236  |    -1.36056   |    0.0770794 |    0.108754 |
| linear_regression | F1_season_only | leave_one_year_out  |   2019 |             26 |         1 |             nan | 3.44319   | 2.68074   |               2.29459   |   0.145624   |  -0.384276  |    -1.29912   |    0.472018  |    0.404916 |
| linear_regression | F1_season_only | leave_one_year_out  |   2020 |             31 |         1 |             nan | 2.68111   | 2.12362   |               1.9338    |   0.17246    |  -0.0060444 |    -0.375016  |    0.412826  |    0.420192 |
| linear_regression | F1_season_only | leave_one_year_out  |   2021 |             37 |         1 |             nan | 2.46193   | 2.03063   |               1.69414   |  -0.171932   |  -0.574864  |    -0.577572  |    0.170362  |    0.211738 |
| linear_regression | F1_season_only | leave_one_year_out  |   2022 |             10 |         1 |             nan | 2.99423   | 2.33922   |               1.70754   |   0.0648069  |  -0.687761  |    -1.19095   |    0.236364  |    0.342485 |
| linear_regression | F1_season_only | leave_one_year_out  |   2023 |             12 |         1 |             nan | 2.42019   | 1.99365   |               1.76366   |   0.111933   |  -0.526674  |    -0.527159  |    0.293706  |    0.409342 |
| linear_regression | F1_season_only | leave_one_year_out  |   2024 |              6 |         1 |             nan | 3.41892   | 3.26708   |               3.62518   | -14.2466     |  -3.26708   |    -3.62518   |    0.314286  |    0.363188 |
| linear_regression | F2_q_season    | leave_one_year_out  |   2003 |              5 |         1 |             nan | 2.11119   | 1.97751   |               2.22681   |   0.491893   |  -0.456054  |    -0.736359  |    0.7       |    0.72554  |
| linear_regression | F2_q_season    | leave_one_year_out  |   2004 |             42 |         1 |             nan | 2.45742   | 2.11584   |               1.9604    |   0.475122   |  -0.628188  |    -0.914761  |    0.740182  |    0.741922 |
| linear_regression | F2_q_season    | leave_one_year_out  |   2005 |             42 |         1 |             nan | 2.28775   | 1.85637   |               1.52273   |   0.431447   |   0.0186026 |    -0.569039  |    0.574856  |    0.656875 |
| linear_regression | F2_q_season    | leave_one_year_out  |   2006 |             11 |         1 |             nan | 3.00154   | 2.58851   |               2.17203   |   0.23984    |   1.342     |     2.05558   |    0.5       |    0.642117 |
| linear_regression | F2_q_season    | leave_one_year_out  |   2007 |              1 |         1 |             nan | 0.0726259 | 0.0726259 |               0.0726259 | nan          |   0.0726259 |     0.0726259 |  nan         |  nan        |
| linear_regression | F2_q_season    | leave_one_year_out  |   2009 |             28 |         1 |             nan | 2.99397   | 2.55509   |               1.96801   |   0.380382   |   0.10393   |    -1.13211   |    0.586094  |    0.642859 |
| linear_regression | F2_q_season    | leave_one_year_out  |   2010 |             30 |         1 |             nan | 3.53858   | 2.65757   |               1.56125   |   0.203224   |   1.53028   |     0.845411  |    0.603228  |    0.620696 |
| linear_regression | F2_q_season    | leave_one_year_out  |   2011 |             30 |         1 |             nan | 4.10901   | 2.94192   |               1.93471   |   0.217765   |   2.10116   |     0.933581  |    0.773016  |    0.75156  |
| linear_regression | F2_q_season    | leave_one_year_out  |   2012 |             23 |         1 |             nan | 2.6213    | 2.13183   |               1.68247   |   0.451683   |  -0.188459  |    -0.864042  |    0.528787  |    0.679349 |

## 9. Metrics by season window

Season windows are provisional descriptive windows, not final hydrologic freshet definitions.

| model_id          | feature_set                         | validation_scheme   | season_window              |   n_test_total |   n_folds |   n_train_total |    rmse |     mae |   median_absolute_error |         r2 |   bias_mean |   bias_median |   spearman_r |   pearson_r |
|:------------------|:------------------------------------|:--------------------|:---------------------------|---------------:|----------:|----------------:|--------:|--------:|------------------------:|-----------:|------------:|--------------:|-------------:|------------:|
| linear_regression | F1_season_only                      | leave_one_year_out  | early_season               |            146 |        20 |             nan | 4.54865 | 3.50408 |                2.64991  | -0.154033  |  1.63355    |     1.38136   |  -0.106475   | -0.0184621  |
| linear_regression | F1_season_only                      | leave_one_year_out  | late_season                |             87 |        18 |             nan | 2.2492  | 1.80168 |                1.49347  | -0.0890328 | -0.153044   |    -0.217369  |  -0.177824   | -0.0675681  |
| linear_regression | F1_season_only                      | leave_one_year_out  | spring_freshet_provisional |            193 |        20 |             nan | 4.29981 | 3.37349 |                2.81422  | -0.0665848 |  0.920721   |     0.541587  |  -0.220363   | -0.133174   |
| linear_regression | F1_season_only                      | leave_one_year_out  | summer                     |            114 |        20 |             nan | 3.28758 | 2.89501 |                2.90534  | -0.210532  | -1.39552    |    -2.22104   |   0.0595102  |  0.0878755  |
| linear_regression | F2_q_season                         | leave_one_year_out  | early_season               |            146 |        20 |             nan | 3.49246 | 2.59671 |                1.81109  |  0.319679  |  0.895728   |     0.698792  |   0.654888   |  0.625141   |
| linear_regression | F2_q_season                         | leave_one_year_out  | late_season                |             87 |        18 |             nan | 1.93308 | 1.51164 |                1.32663  |  0.195575  | -0.197196   |    -0.502238  |   0.438341   |  0.457234   |
| linear_regression | F2_q_season                         | leave_one_year_out  | spring_freshet_provisional |            193 |        20 |             nan | 3.3174  | 2.5333  |                1.99433  |  0.36512   |  0.495896   |    -0.175167  |   0.695364   |  0.64682    |
| linear_regression | F2_q_season                         | leave_one_year_out  | summer                     |            114 |        20 |             nan | 2.68241 | 2.2998  |                2.14585  |  0.194111  | -0.772268   |    -1.59453   |   0.616779   |  0.524418   |
| linear_regression | F3_q_season_river_fixed             | leave_one_year_out  | early_season               |            146 |        20 |             nan | 2.97709 | 2.23929 |                1.63823  |  0.505649  |  0.614987   |     0.280813  |   0.744271   |  0.72876    |
| linear_regression | F3_q_season_river_fixed             | leave_one_year_out  | late_season                |             87 |        18 |             nan | 1.67668 | 1.31751 |                1.0077   |  0.394822  | -0.176439   |    -0.119029  |   0.693163   |  0.65958    |
| linear_regression | F3_q_season_river_fixed             | leave_one_year_out  | spring_freshet_provisional |            193 |        20 |             nan | 2.81707 | 2.12625 |                1.57906  |  0.542182  |  0.340476   |     0.115216  |   0.772395   |  0.746284   |
| linear_regression | F3_q_season_river_fixed             | leave_one_year_out  | summer                     |            114 |        20 |             nan | 2.22591 | 1.74915 |                1.524    |  0.445067  | -0.577846   |    -0.691102  |   0.657616   |  0.695166   |
| linear_regression | F4_reduced_hydroclimate             | leave_one_year_out  | early_season               |            146 |        20 |             nan | 3.45063 | 2.60583 |                1.97062  |  0.335878  |  0.724815   |     0.382975  |   0.616368   |  0.609965   |
| linear_regression | F4_reduced_hydroclimate             | leave_one_year_out  | late_season                |             85 |        18 |             nan | 1.88218 | 1.45053 |                1.14278  |  0.233355  | -0.00792445 |    -0.37057   |   0.521297   |  0.498183   |
| linear_regression | F4_reduced_hydroclimate             | leave_one_year_out  | spring_freshet_provisional |            193 |        20 |             nan | 3.27888 | 2.53128 |                2.02424  |  0.379777  |  0.365659   |    -0.10849   |   0.665269   |  0.633868   |
| linear_regression | F4_reduced_hydroclimate             | leave_one_year_out  | summer                     |            114 |        20 |             nan | 2.65436 | 2.25199 |                2.21914  |  0.210879  | -0.701193   |    -1.52593   |   0.608082   |  0.516989   |
| linear_regression | F5_snow_hydroclimate_complete_case  | leave_one_year_out  | early_season               |            126 |        20 |             nan | 3.37472 | 2.57981 |                2.05066  |  0.362707  |  0.252917   |    -0.177571  |   0.611833   |  0.609102   |
| linear_regression | F5_snow_hydroclimate_complete_case  | leave_one_year_out  | late_season                |             64 |        18 |             nan | 1.9641  | 1.48814 |                1.13479  |  0.119571  |  0.118337   |    -0.377878  |   0.514249   |  0.459955   |
| linear_regression | F5_snow_hydroclimate_complete_case  | leave_one_year_out  | spring_freshet_provisional |            163 |        20 |             nan | 3.23683 | 2.51751 |                2.08035  |  0.395471  |  0.102106   |    -0.38608   |   0.641741   |  0.63567    |
| linear_regression | F5_snow_hydroclimate_complete_case  | leave_one_year_out  | summer                     |             93 |        20 |             nan | 2.7197  | 2.22324 |                1.98686  |  0.248771  | -0.229721   |    -1.16832   |   0.605277   |  0.504214   |
| linear_regression | F6_reduced_hydroclimate_river_fixed | leave_one_year_out  | early_season               |            146 |        20 |             nan | 2.95695 | 2.17118 |                1.50592  |  0.512313  |  0.464519   |     0.181875  |   0.73748    |  0.726437   |
| linear_regression | F6_reduced_hydroclimate_river_fixed | leave_one_year_out  | late_season                |             85 |        18 |             nan | 1.59429 | 1.23617 |                0.889317 |  0.449945  |  0.0168934  |     0.0169573 |   0.705217   |  0.683014   |
| linear_regression | F6_reduced_hydroclimate_river_fixed | leave_one_year_out  | spring_freshet_provisional |            193 |        20 |             nan | 2.78156 | 2.05551 |                1.51012  |  0.553651  |  0.22492    |    -0.154772  |   0.775047   |  0.751231   |
| linear_regression | F6_reduced_hydroclimate_river_fixed | leave_one_year_out  | summer                     |            114 |        20 |             nan | 2.1538  | 1.67901 |                1.43009  |  0.480439  | -0.529525   |    -0.715259  |   0.708409   |  0.715616   |
| mean_baseline     | F0_intercept_only                   | leave_one_year_out  | early_season               |            146 |        20 |             nan | 5.1377  | 3.95789 |                3.22926  | -0.472276  |  2.86709    |     2.71816   |  -0.350649   | -0.338539   |
| mean_baseline     | F0_intercept_only                   | leave_one_year_out  | late_season                |             87 |        18 |             nan | 2.20192 | 1.71052 |                1.49687  | -0.0437264 | -0.446023   |    -0.721485  |  -0.00278276 |  0.00226255 |
| mean_baseline     | F0_intercept_only                   | leave_one_year_out  | spring_freshet_provisional |            193 |        20 |             nan | 4.73104 | 3.59936 |                2.83839  | -0.291253  |  2.1927     |     1.92554   |  -0.354762   | -0.341196   |
| mean_baseline     | F0_intercept_only                   | leave_one_year_out  | summer                     |            114 |        20 |             nan | 3.00873 | 2.43368 |                2.17788  | -0.0138892 | -0.290475   |    -1.2161    |  -0.111603   | -0.0900866  |
| ridge_alpha_0.1   | F1_season_only                      | leave_one_year_out  | early_season               |            146 |        20 |             nan | 4.54874 | 3.50414 |                2.64989  | -0.154078  |  1.63379    |     1.38158   |  -0.106475   | -0.0184886  |
| ridge_alpha_0.1   | F1_season_only                      | leave_one_year_out  | late_season                |             87 |        18 |             nan | 2.24918 | 1.80165 |                1.49322  | -0.089009  | -0.153092   |    -0.217392  |  -0.177824   | -0.067569   |

## 10. Residual diagnostics

| model_id      | feature_set       | validation_scheme   | river     |   month |   residual_mgC_L |   abs_residual_mgC_L |
|:--------------|:------------------|:--------------------|:----------|--------:|-----------------:|---------------------:|
| mean_baseline | F0_intercept_only | leave_one_year_out  | Kolyma    |       8 |       -3.75003   |            3.75003   |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Lena      |       8 |       -1.45003   |            1.45003   |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Mackenzie |       6 |       -1.15003   |            1.15003   |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Ob        |       7 |        4.34997   |            4.34997   |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Yukon     |       6 |        2.74997   |            2.74997   |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Kolyma    |       6 |        3.67343   |            3.67343   |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Kolyma    |       6 |        2.47343   |            2.47343   |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Kolyma    |       6 |       -0.226566  |            0.226566  |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Kolyma    |       7 |       -0.726566  |            0.726566  |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Kolyma    |       8 |       -2.72657   |            2.72657   |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Kolyma    |       8 |       -2.22657   |            2.22657   |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Kolyma    |       9 |       -2.72657   |            2.72657   |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Lena      |       4 |        0.0734337 |            0.0734337 |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Lena      |       6 |        8.17343   |            8.17343   |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Lena      |       6 |        5.77343   |            5.77343   |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Lena      |       8 |        0.0734337 |            0.0734337 |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Lena      |       8 |       -0.126566  |            0.126566  |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Lena      |      10 |       -0.0265663 |            0.0265663 |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Lena      |      10 |        0.773434  |            0.773434  |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Mackenzie |       3 |       -3.02657   |            3.02657   |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Mackenzie |       6 |       -1.52657   |            1.52657   |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Mackenzie |       6 |       -2.02657   |            2.02657   |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Mackenzie |       7 |       -2.92657   |            2.92657   |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Mackenzie |       8 |       -3.32657   |            3.32657   |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Mackenzie |       8 |       -3.32657   |            3.32657   |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Mackenzie |       9 |       -3.52657   |            3.52657   |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Ob        |       4 |       -1.12657   |            1.12657   |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Ob        |       6 |        2.07343   |            2.07343   |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Ob        |       6 |        2.07343   |            2.07343   |
| mean_baseline | F0_intercept_only | leave_one_year_out  | Ob        |       7 |        5.37343   |            5.37343   |

## 11. Best baseline candidates

Primary ranking uses leave-one-year-out RMSE and MAE with penalties for low sample count and snow complete-case sample loss.

| model_id          | feature_set                         | validation_scheme   |   n_test_total |   n_folds |   n_train_total |    rmse |     mae |   median_absolute_error |       r2 |   bias_mean |   bias_median |   spearman_r |   pearson_r |   rows_available_for_feature_set |   sample_penalty |   snow_complete_case_penalty |   ranking_score |   primary_rank | ranking_basis                                   | selection_caveat                |
|:------------------|:------------------------------------|:--------------------|---------------:|----------:|----------------:|--------:|--------:|------------------------:|---------:|------------:|--------------:|-------------:|------------:|---------------------------------:|-----------------:|-----------------------------:|----------------:|---------------:|:------------------------------------------------|:--------------------------------|
| ridge_alpha_1     | F6_reduced_hydroclimate_river_fixed | leave_one_year_out  |            511 |        21 |           10220 | 2.24078 | 1.66492 |                 1.22834 | 0.602175 |  0.00998951 |     -0.219564 |     0.803006 |    0.776022 |                              511 |                0 |                            0 |         2.24078 |              1 | leave_one_year_out_rmse_mae_with_sample_penalty | Primary LOYO ranking candidate. |
| ridge_alpha_0.1   | F6_reduced_hydroclimate_river_fixed | leave_one_year_out  |            511 |        21 |           10220 | 2.24085 | 1.66672 |                 1.22685 | 0.602149 |  0.00983827 |     -0.21782  |     0.802892 |    0.776051 |                              511 |                0 |                            0 |         2.24085 |              2 | leave_one_year_out_rmse_mae_with_sample_penalty | Primary LOYO ranking candidate. |
| linear_regression | F6_reduced_hydroclimate_river_fixed | leave_one_year_out  |            511 |        21 |           10220 | 2.24087 | 1.66693 |                 1.22376 | 0.60214  |  0.00982082 |     -0.216126 |     0.802786 |    0.776052 |                              511 |                0 |                            0 |         2.24087 |              3 | leave_one_year_out_rmse_mae_with_sample_penalty | Primary LOYO ranking candidate. |
| ridge_alpha_10    | F6_reduced_hydroclimate_river_fixed | leave_one_year_out  |            511 |        21 |           10220 | 2.25038 | 1.66063 |                 1.24358 | 0.598759 |  0.0110511  |     -0.239618 |     0.805209 |    0.774489 |                              511 |                0 |                            0 |         2.25038 |              4 | leave_one_year_out_rmse_mae_with_sample_penalty | Primary LOYO ranking candidate. |
| ridge_alpha_1     | F3_q_season_river_fixed             | leave_one_year_out  |            545 |        21 |           10900 | 2.25785 | 1.71611 |                 1.34859 | 0.592437 |  0.00277652 |     -0.199575 |     0.787391 |    0.769721 |                              545 |                0 |                            0 |         2.25785 |              5 | leave_one_year_out_rmse_mae_with_sample_penalty | Primary LOYO ranking candidate. |
| ridge_alpha_0.1   | F3_q_season_river_fixed             | leave_one_year_out  |            545 |        21 |           10900 | 2.25786 | 1.71749 |                 1.35636 | 0.592431 |  0.00266876 |     -0.176015 |     0.786819 |    0.769767 |                              545 |                0 |                            0 |         2.25786 |              6 | leave_one_year_out_rmse_mae_with_sample_penalty | Primary LOYO ranking candidate. |
| linear_regression | F3_q_season_river_fixed             | leave_one_year_out  |            545 |        21 |           10900 | 2.25788 | 1.71765 |                 1.35557 | 0.592424 |  0.00265638 |     -0.176146 |     0.78678  |    0.76977  |                              545 |                0 |                            0 |         2.25788 |              7 | leave_one_year_out_rmse_mae_with_sample_penalty | Primary LOYO ranking candidate. |
| ridge_alpha_10    | F3_q_season_river_fixed             | leave_one_year_out  |            545 |        21 |           10900 | 2.26819 | 1.71567 |                 1.31852 | 0.588694 |  0.00358828 |     -0.264282 |     0.791755 |    0.767959 |                              545 |                0 |                            0 |         2.26819 |              8 | leave_one_year_out_rmse_mae_with_sample_penalty | Primary LOYO ranking candidate. |

Do not use leave-one-river-out as the primary winner selection because it is a high-risk extrapolation stress test with only six rivers.

## 12. Why optical/basin/flux are excluded in this phase

- Optical matrices are reserved for a later optical sensitivity phase.
- Basin context matrices are reserved for a later basin sensitivity phase because only six river-level units exist.
- Prediction grid outputs are x-only and are not used to generate daily DOC predictions in this phase.
- Flux requires production DOC predictions and discharge integration, so it is explicitly out of scope.

## 13. Recommended next phase

Model refinement or optical sensitivity, after deciding whether the reduced hydroclimate LOYO candidate is stable enough for a baseline reference.

## 14. Explicit statement

- DOC concentration models were trained for validation only.
- No production daily DOC prediction was generated.
- No DOC flux was generated.
- Gold data were not modified.
