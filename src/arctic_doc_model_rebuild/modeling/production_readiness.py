from __future__ import annotations

import pandas as pd


def production_readiness_decision(
    *,
    river_bias: pd.DataFrame,
    fold_stability: pd.DataFrame,
    intervals: pd.DataFrame,
    roi_qc: pd.DataFrame,
    optical_ranking: pd.DataFrame,
    no_production_generated: bool,
) -> pd.DataFrame:
    severe_river_bias = False
    if not river_bias.empty:
        severe_river_bias = bool(
            river_bias["flag_abs_bias_gt_1"].astype(bool).any()
            or river_bias["flag_rmse_gt_1_25x_overall"].astype(bool).any()
        )
    unstable_fold = False
    if not fold_stability.empty:
        primary = fold_stability[fold_stability["model_role"].eq("primary_baseline")]
        unstable_fold = bool(primary["unstable_fold"].astype(bool).any()) if not primary.empty else False
    intervals_available = not intervals.empty
    roi_reopen = bool(roi_qc["reopen_freeze_recommendation"].astype(str).str.lower().eq("yes").any()) if not roi_qc.empty else True
    roi_caveated = bool(roi_qc["roi_decision"].astype(str).str.contains("caveat|review", case=False, regex=True).any()) if not roi_qc.empty else True
    optical_improves = False
    if not optical_ranking.empty and "is_optical_proxy_feature_set" in optical_ranking.columns:
        primary = optical_ranking[
            optical_ranking["dataset_id"].eq("any_sensor_3d")
            & optical_ranking["is_optical_proxy_feature_set"].astype(bool)
        ]
        optical_improves = bool(primary["classification"].eq("optical_improves_baseline").any()) if not primary.empty else False

    rows = [
        {
            "decision_item": "primary_model_selected",
            "status": "true",
            "evidence": "Baseline finalization selected F3_q_season_river_fixed + ridge_alpha_1.",
            "recommendation": "Use as primary concentration baseline comparator.",
            "blocking_for_prediction": False,
        },
        {
            "decision_item": "optical_excluded",
            "status": "true" if not optical_improves else "false",
            "evidence": "Optical sensitivity found no robust incremental value over F3." if not optical_improves else "Optical model improved primary ranking unexpectedly.",
            "recommendation": "Keep optical excluded from primary production path.",
            "blocking_for_prediction": bool(optical_improves),
        },
        {
            "decision_item": "hydroclimate_extension_optional",
            "status": "true",
            "evidence": "Hydroclimate extension remained extension_not_primary in baseline finalization.",
            "recommendation": "Do not require F6 for initial production readiness.",
            "blocking_for_prediction": False,
        },
        {
            "decision_item": "roi_qc_completed",
            "status": "true_with_caveats" if roi_caveated and not roi_reopen else "false" if roi_reopen else "true",
            "evidence": "ROI QC completed; no freeze reopen recommended." if not roi_reopen else "ROI QC recommends reopening the data freeze.",
            "recommendation": "Carry ROI caveats forward." if roi_caveated and not roi_reopen else "Proceed." if not roi_reopen else "Return to data repo and create a new freeze.",
            "blocking_for_prediction": bool(roi_reopen),
        },
        {
            "decision_item": "residual_interval_available",
            "status": "true" if intervals_available else "false",
            "evidence": f"Empirical residual interval rows: {len(intervals)}.",
            "recommendation": "Use intervals as validation residual intervals, not final production prediction intervals.",
            "blocking_for_prediction": not intervals_available,
        },
        {
            "decision_item": "river_bias_acceptable",
            "status": "false" if severe_river_bias else "true",
            "evidence": "Severe river bias flag present." if severe_river_bias else "No river bias exceeded configured severe thresholds.",
            "recommendation": "Refine model before production prediction." if severe_river_bias else "Proceed with river caveats.",
            "blocking_for_prediction": bool(severe_river_bias),
        },
        {
            "decision_item": "fold_stability_acceptable",
            "status": "true_with_caveats" if unstable_fold else "true",
            "evidence": "One or more LOYO folds had high RMSE or high bias." if unstable_fold else "No primary LOYO fold exceeded instability thresholds.",
            "recommendation": "Carry fold stability caveats into production planning." if unstable_fold else "Proceed.",
            "blocking_for_prediction": False,
        },
        {
            "decision_item": "log_target_status",
            "status": "sensitivity_only",
            "evidence": "Log target remains sensitivity_candidate_only and is not promoted in this phase.",
            "recommendation": "Do not promote without a separate residual-bias review.",
            "blocking_for_prediction": False,
        },
    ]

    blockers = [row for row in rows if row["blocking_for_prediction"]]
    if blockers:
        ready_status = "false"
        ready_recommendation = "Resolve blocking items before production daily DOC prediction."
    elif roi_caveated or unstable_fold:
        ready_status = "true_with_caveats"
        ready_recommendation = "Proceed only with documented caveats and validation-residual intervals."
    else:
        ready_status = "true"
        ready_recommendation = "Proceed to a guarded production daily DOC prediction phase."
    rows.append(
        {
            "decision_item": "ready_for_production_daily_prediction",
            "status": ready_status if no_production_generated else "false",
            "evidence": "No production prediction was generated in this phase." if no_production_generated else "Production prediction artifact exists or was generated.",
            "recommendation": ready_recommendation if no_production_generated else "Remove production artifacts and rerun diagnostics.",
            "blocking_for_prediction": ready_status == "false" or not no_production_generated,
        }
    )
    return pd.DataFrame(rows)
