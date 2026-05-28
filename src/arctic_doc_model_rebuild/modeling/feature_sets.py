from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


TARGET_COLUMN = "DOC_mgC_L"
REQUIRED_HYDROCORE_COLUMNS = [
    "label_id",
    "river",
    "date",
    "year",
    "doy",
    "DOC_mgC_L",
    "Q_m3s",
    "sin_doy",
    "cos_doy",
    "temperature_2m_C",
    "positive_degree_day_Cday",
    "snow_cover_fraction",
    "snow_depletion_rate_7d",
    "surface_runoff_m",
]
FORBIDDEN_FEATURE_COLUMNS = {
    "A254",
    "A350",
    "A355",
    "A375",
    "A412",
    "A420",
    "A440",
    "SUVA254",
    "spectral_slope_275_295",
    "spectral_slope_350_400",
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
    "ndwi",
    "mndwi",
    "red_green_ratio",
    "green_blue_ratio",
}


@dataclass(frozen=True)
class FeatureSet:
    feature_set: str
    description: str
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...] = ()
    sensitivity_only: bool = False
    caveat: str = ""

    @property
    def required_features(self) -> tuple[str, ...]:
        return (*self.numeric_features, *self.categorical_features)


FEATURE_SETS = [
    FeatureSet("F0_intercept_only", "Training-fold mean DOC baseline.", ()),
    FeatureSet("F1_season_only", "Seasonal harmonic baseline.", ("sin_doy", "cos_doy")),
    FeatureSet("F2_q_season", "Discharge plus seasonal harmonics.", ("log_Q", "sin_doy", "cos_doy")),
    FeatureSet(
        "F3_q_season_river_fixed",
        "Discharge, seasonal harmonics, and river one-hot fixed effects.",
        ("log_Q", "sin_doy", "cos_doy"),
        ("river",),
        caveat="Leave-one-river-out is a structural stress test for unseen river categories.",
    ),
    FeatureSet(
        "F4_reduced_hydroclimate",
        "Q, season, temperature, positive degree days, and surface runoff complete cases.",
        ("log_Q", "sin_doy", "cos_doy", "temperature_2m_C", "positive_degree_day_Cday", "surface_runoff_m"),
    ),
    FeatureSet(
        "F5_snow_hydroclimate_complete_case",
        "Complete-case snow hydroclimate sensitivity.",
        (
            "log_Q",
            "sin_doy",
            "cos_doy",
            "temperature_2m_C",
            "positive_degree_day_Cday",
            "surface_runoff_m",
            "snow_cover_fraction",
            "snow_depletion_rate_7d",
        ),
        sensitivity_only=True,
        caveat="Snow variables have high missingness; do not use as main model without a missingness strategy.",
    ),
    FeatureSet(
        "F6_reduced_hydroclimate_river_fixed",
        "Reduced hydroclimate model with river one-hot fixed effects.",
        ("log_Q", "sin_doy", "cos_doy", "temperature_2m_C", "positive_degree_day_Cday", "surface_runoff_m"),
        ("river",),
        caveat="River fixed effects are not reliable for held-out unseen rivers.",
    ),
]


def feature_set_registry() -> pd.DataFrame:
    rows = []
    for feature_set in FEATURE_SETS:
        rows.append(
            {
                "feature_set": feature_set.feature_set,
                "description": feature_set.description,
                "numeric_features": ";".join(feature_set.numeric_features),
                "categorical_features": ";".join(feature_set.categorical_features),
                "all_features": ";".join(feature_set.required_features),
                "sensitivity_only": feature_set.sensitivity_only,
                "caveat": feature_set.caveat,
            }
        )
    return pd.DataFrame(rows)


def get_feature_set(name: str) -> FeatureSet:
    for feature_set in FEATURE_SETS:
        if feature_set.feature_set == name:
            return feature_set
    raise KeyError(f"Unknown feature set: {name}")
