# arctic_doc_model_rebuild

Clean model rebuild repository for Arctic river DOC modeling from the frozen gold data freeze `data_freeze_gold_20260526_v1`.

This repository does not own or modify source data. It reads frozen gold CSVs from `hyh31415-maker/arctic_doc_data_audit` and starts with data contract verification only. Model training begins only after the gold contract, schema checks, and leakage checks pass.

No model is trained in the initial setup. No DOC prediction is generated. No flux is generated.

## Data Source

- source repo: `hyh31415-maker/arctic_doc_data_audit`
- source tag: `data_freeze_gold_20260526_v1`
- local gold data: `D:/Hao/Desktop/冰冻圈水文/北极大河/arctic_doc_data_audit/data/processed/gold/`
- override: `ARCTIC_DOC_GOLD_DIR`

Path priority:

1. `ARCTIC_DOC_GOLD_DIR`
2. `configs/local_paths.yaml`
3. built-in default local path

The repository must read only `data/processed/gold/*` from the sealed data repository.

## Quick Start

```powershell
python -m pip install -e .[test]
copy configs/local_paths.example.yaml configs/local_paths.yaml
# edit gold_data_dir if needed
python -m arctic_doc_model_rebuild.cli verify-gold-data
python -m arctic_doc_model_rebuild.cli summarize-gold-data
python -m arctic_doc_model_rebuild.cli data-contract-report
python -m pytest
```

Or use the console entrypoint:

```powershell
arctic-doc-model verify-gold-data
arctic-doc-model summarize-gold-data
arctic-doc-model data-contract-report
```

If the gold directory is not found:

```powershell
$env:ARCTIC_DOC_GOLD_DIR="D:/Hao/Desktop/冰冻圈水文/北极大河/arctic_doc_data_audit/data/processed/gold"
```

## Rules

- Do not read raw/interim/canonical tables.
- Do not modify gold data.
- Do not promote candidate labels.
- Do not use lab optical/CDOM as daily production predictors.
- Do not treat satellite reflectance as DOC observation.
- Do not generate model binaries, DOC prediction outputs, or flux products during data contract verification.

## Initial Outputs

- `outputs/reports/data_contract_report.md`
- `outputs/reports/gold_data_summary_report.md`
- `outputs/tables/gold_table_verification.csv`
- `outputs/tables/model_input_schema_check.csv`
- `outputs/tables/gold_input_inventory.csv`
- `outputs/tables/gold_matrix_missingness.csv`
- `outputs/tables/gold_matrix_by_river.csv`
- `outputs/tables/gold_matrix_by_year.csv`

## EDA Phase

```powershell
python -m arctic_doc_model_rebuild.cli verify-gold-data
python -m arctic_doc_model_rebuild.cli run-eda
python -m pytest
```

EDA writes descriptive reports, tables, and lightweight figures under `outputs/reports/eda/`, `outputs/tables/eda/`, and `outputs/figures/eda/`.

EDA does not train models, does not generate DOC predictions, and does not generate flux.

## Baseline Model Phase 1

```powershell
python -m arctic_doc_model_rebuild.cli verify-gold-data
python -m arctic_doc_model_rebuild.cli run-baseline-models
python -m arctic_doc_model_rebuild.cli baseline-report
python -m pytest
```

Baseline phase 1 trains simple DOC concentration models for cross-validation diagnostics only. It reads only `training_matrix_hydrocore.csv` as model input. It does not read the daily prediction grid, optical matrices, or basin context matrices; it does not generate production daily DOC prediction or flux.

## Baseline Refinement Phase

```powershell
python -m arctic_doc_model_rebuild.cli verify-gold-data
python -m arctic_doc_model_rebuild.cli run-baseline-refinement
python -m arctic_doc_model_rebuild.cli baseline-refinement-report
python -m pytest
```

Baseline refinement compares F3 and F6 on the same hydrocore complete-case subset, checks raw versus log target sensitivity, and writes validation-only diagnostics. It still reads only `training_matrix_hydrocore.csv` as model input and does not generate production daily DOC prediction or flux.

## Baseline Finalization Phase

```powershell
python -m arctic_doc_model_rebuild.cli verify-gold-data
python -m arctic_doc_model_rebuild.cli finalize-baseline
python -m arctic_doc_model_rebuild.cli baseline-final-report
python -m pytest
```

Baseline finalization freezes the selected validation comparator as `F3_q_season_river_fixed + ridge_alpha_1`, writes model decision tables and YAML specs, and hands off to optical sensitivity. It does not train new model families, does not use optical or basin matrices, and does not generate production daily DOC prediction or flux.

## Optical Sensitivity Phase

```powershell
python -m arctic_doc_model_rebuild.cli verify-gold-data
python -m arctic_doc_model_rebuild.cli run-optical-sensitivity
python -m arctic_doc_model_rebuild.cli optical-sensitivity-report
python -m pytest
```

Optical sensitivity tests whether satellite optical proxy variables improve the finalized `F3_q_season_river_fixed + ridge_alpha_1` baseline on identical optical-matched subsets. It trains validation-only DOC concentration diagnostics, does not read the daily prediction grid or basin context matrices, and does not generate production daily DOC prediction or flux.

## ROI Final QC Phase

```powershell
python -m arctic_doc_model_rebuild.cli verify-gold-data
python -m arctic_doc_model_rebuild.cli roi-final-qc
python -m arctic_doc_model_rebuild.cli roi-final-qc-report
python -m pytest
```

ROI final QC audits frozen ROI metadata, optical valid-water support, and 3-day optical match integrity after the optical sensitivity result. It does not recalculate ROI, does not re-extract GEE, does not train DOC models, and does not generate production daily DOC prediction or flux.

## Concentration Uncertainty Phase

```powershell
python -m arctic_doc_model_rebuild.cli verify-gold-data
python -m arctic_doc_model_rebuild.cli run-concentration-uncertainty
python -m arctic_doc_model_rebuild.cli concentration-uncertainty-report
python -m pytest
```

Concentration uncertainty evaluates validation-only residual intervals, fold stability, river bias, high-DOC behavior, calibration, bootstrap coefficient stability, and production-readiness for the finalized F3 concentration baseline. It does not load the prediction grid, optical matrices, basin context matrices, or lab optical/CDOM data, and it does not generate production daily DOC prediction or flux.

## Bias-aware Model Refinement Phase

```powershell
python -m arctic_doc_model_rebuild.cli verify-gold-data
python -m arctic_doc_model_rebuild.cli run-bias-aware-refinement
python -m arctic_doc_model_rebuild.cli bias-aware-refinement-report
python -m pytest
```

Bias-aware refinement tests interpretable nonlinear, river-interaction, robust-regression, and log-target sensitivity variants against the finalized F3 baseline. It remains validation-only, reads only `training_matrix_hydrocore.csv` as model input, does not load prediction grids or optical/basin matrices, and does not generate production daily DOC prediction or flux.

## Guarded Daily DOC Prediction Phase

```powershell
python -m arctic_doc_model_rebuild.cli verify-gold-data
python -m arctic_doc_model_rebuild.cli freeze-production-candidate
python -m arctic_doc_model_rebuild.cli run-daily-doc-prediction
python -m arctic_doc_model_rebuild.cli daily-doc-prediction-report
python -m pytest
```

This phase freezes the refined `R4_river_specific_Q_and_season + linear_regression` production candidate, fits it on eligible hydrocore training rows, and generates guarded daily DOC concentration predictions from `prediction_grid_daily_hydrocore.csv`. It attaches empirical validation-residual intervals and writes flux-readiness diagnostics, but it does not calculate DOC flux or create daily/annual/snowmelt flux products.

## Guarded DOC Flux Calculation Phase

```powershell
python -m arctic_doc_model_rebuild.cli verify-gold-data
python -m arctic_doc_model_rebuild.cli run-doc-flux
python -m arctic_doc_model_rebuild.cli doc-flux-report
python -m pytest
```

This phase calculates guarded daily and annual DOC flux from the existing guarded daily DOC concentration prediction table. It does not retrain models, does not regenerate DOC predictions, does not modify gold data, and does not read optical/basin/lab features. Flux intervals propagate DOC concentration empirical residual intervals only; discharge uncertainty is not propagated. May-July flux is provisional and is not a final snowmelt contribution estimate.
