# Production Candidate Handoff

Generated: 2026-05-28T13:36:22Z

- freeze_id: `data_freeze_gold_20260526_v1`
- model_spec_id: `production_candidate_r4_river_specific_q_and_season_linear`
- model: `R4_river_specific_Q_and_season + LinearRegression`
- production daily DOC prediction: allowed in guarded mode
- flux: not allowed

## Inputs

- `data/processed/gold/training_matrix_hydrocore.csv`
- `data/processed/gold/prediction_grid_daily_hydrocore.csv`

## Caveats

- within six ArcticGRO rivers only
- no cross-river extrapolation
- high-DOC caveat carried forward
- fold stability caveat carried forward
- empirical intervals are derived from validation residuals
- optical, basin, lab optical/CDOM, and flux features are excluded
