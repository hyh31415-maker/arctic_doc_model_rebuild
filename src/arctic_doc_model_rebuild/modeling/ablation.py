from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .feature_sets import FeatureSet


COMMON_SUBSET_COLUMNS = [
    "DOC_mgC_L",
    "Q_m3s",
    "sin_doy",
    "cos_doy",
    "temperature_2m_C",
    "positive_degree_day_Cday",
    "surface_runoff_m",
]


@dataclass(frozen=True)
class AblationFeatureSet:
    ablation_id: str
    feature_set: FeatureSet


ABLATION_FEATURE_SETS = [
    AblationFeatureSet("A0", FeatureSet("F0_intercept_only", "Same-sample intercept-only baseline.", ())),
    AblationFeatureSet("A1", FeatureSet("F1_season_only", "Same-sample season-only baseline.", ("sin_doy", "cos_doy"))),
    AblationFeatureSet("A2", FeatureSet("F2_q_season", "Same-sample Q plus season.", ("log_Q", "sin_doy", "cos_doy"))),
    AblationFeatureSet(
        "A3",
        FeatureSet("F3_q_season_river_fixed", "Same-sample Q, season, river fixed effects.", ("log_Q", "sin_doy", "cos_doy"), ("river",)),
    ),
    AblationFeatureSet(
        "A4",
        FeatureSet(
            "F4_reduced_hydroclimate",
            "Same-sample reduced hydroclimate without river fixed effects.",
            ("log_Q", "sin_doy", "cos_doy", "temperature_2m_C", "positive_degree_day_Cday", "surface_runoff_m"),
        ),
    ),
    AblationFeatureSet(
        "A5",
        FeatureSet(
            "F6_reduced_hydroclimate_river_fixed",
            "Same-sample reduced hydroclimate with river fixed effects.",
            ("log_Q", "sin_doy", "cos_doy", "temperature_2m_C", "positive_degree_day_Cday", "surface_runoff_m"),
            ("river",),
        ),
    ),
]

DELTA_COMPARISONS = [
    ("F2_minus_F1_value_of_Q", "F1_season_only", "F2_q_season"),
    ("F3_minus_F2_value_of_river_fixed_effects", "F2_q_season", "F3_q_season_river_fixed"),
    ("F4_minus_F2_value_of_hydroclimate_without_river", "F2_q_season", "F4_reduced_hydroclimate"),
    ("F6_minus_F3_incremental_hydroclimate_after_river", "F3_q_season_river_fixed", "F6_reduced_hydroclimate_river_fixed"),
    ("F6_minus_F4_value_of_river_after_hydroclimate", "F4_reduced_hydroclimate", "F6_reduced_hydroclimate_river_fixed"),
]


def common_subset(frame: pd.DataFrame) -> pd.DataFrame:
    subset = frame.dropna(subset=COMMON_SUBSET_COLUMNS).copy()
    subset = subset[subset["Q_m3s"] > 0].copy()
    return subset.reset_index(drop=True)


def ablation_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    primary = metrics[metrics["validation_scheme"].eq("leave_one_year_out")].copy()
    for comparison, baseline_feature, candidate_feature in DELTA_COMPARISONS:
        for (model_id, validation_scheme), group in primary.groupby(["model_id", "validation_scheme"]):
            baseline = group[group["feature_set"].eq(baseline_feature)]
            candidate = group[group["feature_set"].eq(candidate_feature)]
            if baseline.empty or candidate.empty:
                continue
            baseline_row = baseline.iloc[0]
            candidate_row = candidate.iloc[0]
            rows.append(
                {
                    "comparison": comparison,
                    "model_id": model_id,
                    "validation_scheme": validation_scheme,
                    "baseline_feature_set": baseline_feature,
                    "candidate_feature_set": candidate_feature,
                    "baseline_rmse": baseline_row["rmse"],
                    "candidate_rmse": candidate_row["rmse"],
                    "rmse_reduction": baseline_row["rmse"] - candidate_row["rmse"],
                    "baseline_mae": baseline_row["mae"],
                    "candidate_mae": candidate_row["mae"],
                    "mae_reduction": baseline_row["mae"] - candidate_row["mae"],
                    "baseline_r2": baseline_row["r2"],
                    "candidate_r2": candidate_row["r2"],
                    "r2_gain": candidate_row["r2"] - baseline_row["r2"],
                    "baseline_bias_mean": baseline_row["bias_mean"],
                    "candidate_bias_mean": candidate_row["bias_mean"],
                    "bias_mean_change_abs": abs(candidate_row["bias_mean"]) - abs(baseline_row["bias_mean"]),
                }
            )
    return pd.DataFrame(rows)
