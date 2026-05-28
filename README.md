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
