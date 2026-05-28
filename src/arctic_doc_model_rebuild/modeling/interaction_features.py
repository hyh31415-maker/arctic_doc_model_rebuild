from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BiasFeatureSet:
    feature_set: str
    description: str
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...] = ()
    target_scale: str = "raw"
    caveat: str = ""

    @property
    def required_features(self) -> tuple[str, ...]:
        return (*self.numeric_features, *self.categorical_features)


def river_interaction_columns(rivers: list[str], variable: str) -> list[str]:
    return [f"{river}__x__{variable}" for river in rivers]


def add_river_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    rivers = sorted(out["river"].dropna().astype(str).unique())
    for river in rivers:
        mask = out["river"].astype(str).eq(river)
        for variable in ["log_Q", "sin_doy", "cos_doy"]:
            out[f"{river}__x__{variable}"] = out[variable].where(mask, 0.0)
    return out


def bias_feature_sets(rivers: list[str]) -> list[BiasFeatureSet]:
    river_log_q = tuple(river_interaction_columns(rivers, "log_Q"))
    river_sin = tuple(river_interaction_columns(rivers, "sin_doy"))
    river_cos = tuple(river_interaction_columns(rivers, "cos_doy"))
    base = ("log_Q", "sin_doy", "cos_doy")
    return [
        BiasFeatureSet("B0_F3_finalized", "Finalized F3 baseline comparator.", base, ("river",)),
        BiasFeatureSet("R1_nonlinear_Q", "F3 plus quadratic log discharge.", (*base, "log_Q_squared"), ("river",)),
        BiasFeatureSet("R2_river_specific_Q_slope", "F3 plus river-by-log_Q interactions.", (*base, *river_log_q), ("river",)),
        BiasFeatureSet("R3_river_specific_seasonality", "F3 plus river-specific seasonal harmonic interactions.", (*base, *river_sin, *river_cos), ("river",)),
        BiasFeatureSet("R4_river_specific_Q_and_season", "F3 plus river-specific Q and seasonal interactions.", (*base, *river_log_q, *river_sin, *river_cos), ("river",)),
        BiasFeatureSet("R5_season_window", "F3 plus provisional season-window category.", base, ("river", "season_window")),
        BiasFeatureSet("R6_high_flow_indicator", "F3 plus within-river high-flow flag and log_Q interaction.", (*base, "high_flow_flag", "log_Q_high_flow"), ("river",)),
        BiasFeatureSet("R7_robust_huber_F3", "F3 features with Huber robust regression.", base, ("river",), caveat="HuberRegressor sensitivity only."),
        BiasFeatureSet("R8_log_target_F3", "F3 features with log target and exp inverse transform.", base, ("river",), target_scale="log", caveat="Log target cannot be promoted in this phase."),
        BiasFeatureSet("R9_log_target_R2", "River-specific Q slope with log target.", (*base, *river_log_q), ("river",), target_scale="log", caveat="Log target cannot be promoted in this phase."),
    ]


def bias_feature_set_registry(feature_sets: list[BiasFeatureSet]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_set": item.feature_set,
                "description": item.description,
                "numeric_features": ";".join(item.numeric_features),
                "categorical_features": ";".join(item.categorical_features),
                "target_scale": item.target_scale,
                "caveat": item.caveat,
                "uses_optical": False,
                "uses_basin_context": False,
                "uses_prediction_grid": False,
            }
            for item in feature_sets
        ]
    )
