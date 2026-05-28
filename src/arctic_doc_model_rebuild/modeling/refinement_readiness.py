from __future__ import annotations

import pandas as pd


VALID_READINESS_STATUS = {"true", "true_with_caveats", "false"}


def refined_readiness_decision(
    *,
    recommendation: pd.DataFrame,
    deltas: pd.DataFrame,
    ranking: pd.DataFrame,
    diagnostic_audit: pd.DataFrame,
    no_production_generated: bool,
) -> pd.DataFrame:
    selected = recommendation[recommendation["decision_item"].eq("recommended_primary_model_after_refinement")]
    selected_model = selected["status"].iloc[0] if not selected.empty else "F3_q_season_river_fixed + ridge_alpha_1"
    selected_new_model = bool(selected["recommended_action"].astype(str).str.contains("replace", case=False, regex=False).any()) if not selected.empty else False
    selected_key = ""
    if not selected.empty and {"feature_set", "model_id", "target_scale"}.issubset(selected.columns):
        first = selected.iloc[0]
        selected_key = f"{first['feature_set']}:{first['model_id']}:{first['target_scale']}:leave_one_year_out"

    primary_delta = deltas[deltas["candidate_key"].eq(selected_key)].head(1) if "candidate_key" in deltas.columns else pd.DataFrame()
    if primary_delta.empty:
        f3 = ranking[ranking["candidate_key"].eq("B0_F3_finalized:ridge_alpha_1:raw:leave_one_year_out")]
        severe_river = bool(f3["lena_rmse"].iloc[0] > f3["f3_lena_rmse"].iloc[0] * 1.0) if not f3.empty and "f3_lena_rmse" in f3.columns else True
        high_doc_ok = False
        instability_ok = False
    else:
        row = primary_delta.iloc[0]
        severe_river = bool(row["delta_lena_rmse_vs_f3"] < 0)
        high_doc_ok = bool(row["delta_extreme_high_bias_abs_vs_f3"] >= 0 and row["delta_high_doc_bias_abs_vs_f3"] >= 0)
        instability_ok = bool(row["delta_unstable_fold_count_vs_f3"] >= 0)

    audit_bug = bool(
        diagnostic_audit["requires_code_fix"].astype(str).str.lower().isin({"true", "1"}).any()
    ) if not diagnostic_audit.empty and "requires_code_fix" in diagnostic_audit.columns else False

    rows = [
        {
            "decision_item": "primary_model_after_refinement",
            "status": selected_model,
            "evidence": "Refinement selection table reviewed candidate improvements against finalized F3.",
            "recommendation": "Use the selected validation-only concentration model as the next production candidate.",
            "blocking_for_prediction": False,
        },
        {
            "decision_item": "river_bias_acceptable",
            "status": "false" if severe_river else "true_with_caveats",
            "evidence": "Lena RMSE did not improve enough to remove caveat." if severe_river else "Lena RMSE did not worsen and improved relative to F3.",
            "recommendation": "Further model refinement before production." if severe_river and not selected_new_model else "Proceed with river-specific caveats.",
            "blocking_for_prediction": bool(severe_river and not selected_new_model),
        },
        {
            "decision_item": "high_doc_bias_acceptable",
            "status": "true_with_caveats" if high_doc_ok else "false",
            "evidence": "High-DOC bias improved or did not worsen." if high_doc_ok else "High-DOC or extreme-high bias remains a concern.",
            "recommendation": "Carry high-DOC residual caveat forward." if high_doc_ok else "Further residual refinement needed.",
            "blocking_for_prediction": not high_doc_ok,
        },
        {
            "decision_item": "fold_stability_acceptable",
            "status": "true_with_caveats" if instability_ok else "false",
            "evidence": "Unstable fold count did not increase." if instability_ok else "Fold instability count increased or remained blocking.",
            "recommendation": "Carry LOYO fold caveats forward." if instability_ok else "Further temporal stability review needed.",
            "blocking_for_prediction": not instability_ok,
        },
        {
            "decision_item": "empirical_interval_strategy",
            "status": "true",
            "evidence": "Concentration uncertainty phase produced empirical residual intervals.",
            "recommendation": "Use validation residual intervals with clear caveats; do not treat as final production intervals until production run is authorized.",
            "blocking_for_prediction": False,
        },
    ]
    blockers = [row for row in rows if row["blocking_for_prediction"]]
    if audit_bug:
        blockers.append({"decision_item": "diagnostic_audit_code_fix"})
    if blockers:
        ready_status = "false"
        ready_recommendation = "Resolve remaining bias/stability blockers before production daily DOC prediction."
    elif selected_new_model:
        ready_status = "true_with_caveats"
        ready_recommendation = "Proceed only after freezing the refined model spec and carrying empirical interval caveats."
    else:
        ready_status = "true_with_caveats"
        ready_recommendation = "Proceed only with finalized F3 and explicit empirical residual interval caveats."
    rows.append(
        {
            "decision_item": "ready_for_production_daily_prediction",
            "status": ready_status if no_production_generated else "false",
            "evidence": "No production prediction was generated in this phase." if no_production_generated else "Production prediction artifact exists or was generated.",
            "recommendation": ready_recommendation if no_production_generated else "Remove production artifacts and rerun guardrails.",
            "blocking_for_prediction": bool((ready_status == "false") or not no_production_generated),
        }
    )
    return pd.DataFrame(rows)
