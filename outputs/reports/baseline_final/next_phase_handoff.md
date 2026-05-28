# Next Phase Handoff

Recommended next phase: Optical Sensitivity Phase.

Use:

- `training_matrix_optical_matched_3d.csv`
- `training_matrix_optical_matched_3d_hls.csv`
- `training_matrix_optical_matched_3d_landsat.csv`
- `training_matrix_optical_matched_3d_sentinel2.csv`

Baseline comparator:

- `primary_baseline_f3_q_season_river_fixed_ridge_alpha_1`

Question:

Does optical proxy improve validation metrics over F3 on the same optical-matched subset?

Guardrails:

- Optical is proxy, not DOC observation.
- Compare on the same sample.
- Sensor-specific sensitivity is required.
- Do not generate daily DOC prediction.
- Do not compute flux.
