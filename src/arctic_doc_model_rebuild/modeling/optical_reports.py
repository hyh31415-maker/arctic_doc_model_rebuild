from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..gold_contract import load_contract
from ..reports import _md_table, utc_now


def optical_incremental_value_status(ranking: pd.DataFrame) -> str:
    if ranking.empty:
        return "no"
    primary = ranking[
        ranking["dataset_id"].eq("any_sensor_3d")
        & ranking["validation_scheme"].eq("leave_one_year_out")
        & ranking.get("is_optical_proxy_feature_set", pd.Series(True, index=ranking.index)).astype(bool)
    ].copy()
    if not primary.empty and primary["classification"].eq("optical_improves_baseline").any():
        return "yes"
    if not primary.empty and primary["classification"].eq("optical_marginal").any():
        return "marginal"
    sensor_specific = ranking[
        ranking["dataset_id"].isin(["hls_3d", "landsat_3d", "sentinel2_3d"])
        & ranking["classification"].eq("optical_improves_baseline")
        & ranking.get("is_optical_proxy_feature_set", pd.Series(True, index=ranking.index)).astype(bool)
    ]
    if not sensor_specific.empty:
        return "sensor-specific only"
    return "no"


def write_optical_sensitivity_report(table_dir: Path, report_dir: Path, report_path: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    contract = load_contract()

    dataset_registry = pd.read_csv(table_dir / "optical_dataset_registry.csv")
    feature_registry = pd.read_csv(table_dir / "optical_feature_set_registry.csv")
    validation_registry = pd.read_csv(table_dir / "optical_validation_registry.csv")
    overall = pd.read_csv(table_dir / "optical_metrics_overall.csv")
    deltas = pd.read_csv(table_dir / "optical_same_sample_deltas.csv")
    ranking = pd.read_csv(table_dir / "optical_model_ranking.csv")
    bias = pd.read_csv(table_dir / "optical_bias_audit.csv")
    folds = pd.read_csv(table_dir / "optical_fold_summary.csv")

    primary_deltas = deltas[
        deltas["dataset_id"].eq("any_sensor_3d")
        & deltas["validation_scheme"].eq("leave_one_year_out")
    ].sort_values(["classification_rank", "rmse_reduction"], ascending=[True, False])
    sensor_deltas = deltas[
        deltas["dataset_id"].isin(["hls_3d", "landsat_3d", "sentinel2_3d"])
        & deltas["validation_scheme"].eq("leave_one_year_out")
    ].sort_values(["dataset_id", "classification_rank", "rmse_reduction"], ascending=[True, True, False])
    status = optical_incremental_value_status(ranking)

    lines = [
        "# Optical Sensitivity Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "## 1. Scope and guardrails",
        "",
        "This phase trains validation-only DOC concentration models to test whether satellite optical proxy variables add incremental value over the finalized F3 baseline on identical optical-matched subsets.",
        "",
        "No production daily DOC prediction is generated. No DOC flux is generated. The prediction grid, basin context matrices, and lab optical/CDOM table are not used.",
        "",
        "## 2. Baseline comparator",
        "",
        "- primary baseline comparator: `F3_q_season_river_fixed + ridge_alpha_1`",
        "- comparator feature set in this phase: `B0_F3_same_subset`",
        "- target: raw `DOC_mgC_L`",
        f"- freeze_id: `{contract['freeze_id']}`",
        "",
        "## 3. Input optical datasets",
        "",
        _md_table(dataset_registry, max_rows=20),
        "",
        "## 4. Feature sets",
        "",
        _md_table(feature_registry, max_rows=20),
        "",
        "## 5. Validation schemes",
        "",
        _md_table(validation_registry, max_rows=10),
        "",
        "## 6. Same-sample comparison logic",
        "",
        "For each optical dataset and optical feature set, `B0_F3_same_subset` is evaluated on the exact same rows as the optical candidate. Positive RMSE/MAE reductions mean the candidate improves over the F3 comparator on that subset. `O1_quality_only` is a match-quality bias check, not evidence of optical reflectance proxy skill.",
        "",
        "## 7. Any-sensor window results",
        "",
        _md_table(primary_deltas.head(40), max_rows=40),
        "",
        "## 8. Sensor-specific 3d results",
        "",
        _md_table(sensor_deltas.head(60), max_rows=60),
        "",
        "## 9. Bias and residual diagnostics",
        "",
        _md_table(bias.head(60), max_rows=60),
        "",
        "## 10. Fold stability",
        "",
        _md_table(folds[folds["dataset_id"].eq("any_sensor_3d")].head(60), max_rows=60),
        "",
        "## 11. Does optical improve F3 baseline?",
        "",
        f"Answer: `{status}`.",
        "",
        _md_table(ranking.head(30), max_rows=30),
        "",
        "## 12. Recommended next step",
        "",
        "If the answer is `yes`, carry the best optical feature set into a guarded model refinement phase on the same optical-matched samples. If the answer is `marginal`, `no`, or `sensor-specific only`, keep optical as sensitivity evidence rather than a production candidate.",
        "",
        "## 13. Explicit statements",
        "",
        "- Validation-only DOC concentration models were trained.",
        "- No production daily DOC prediction was generated.",
        "- No DOC flux was generated.",
        "- Gold data were not modified.",
        "- Optical reflectance is a proxy, not DOC observation.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
