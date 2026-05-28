from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class OpticalDataset:
    dataset_id: str
    table_name: str
    window: str
    sensor_scope: str
    primary_window: bool = False
    primary_sensor_check: bool = False
    underpowered_threshold: int = 60


@dataclass(frozen=True)
class OpticalFeatureSet:
    feature_set: str
    description: str
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...] = ()
    any_sensor_only: bool = False
    baseline_comparator: bool = False

    @property
    def required_features(self) -> tuple[str, ...]:
        return (*self.numeric_features, *self.categorical_features)


@dataclass(frozen=True)
class OpticalModelSpec:
    model_id: str
    model_family: str
    model_class: str
    alpha: float | None = None


OPTICAL_DATASETS = [
    OpticalDataset("any_sensor_0d", "training_matrix_optical_matched_0d.csv", "0d", "any_sensor"),
    OpticalDataset("any_sensor_1d", "training_matrix_optical_matched_1d.csv", "1d", "any_sensor"),
    OpticalDataset("any_sensor_3d", "training_matrix_optical_matched_3d.csv", "3d", "any_sensor", primary_window=True),
    OpticalDataset("any_sensor_7d", "training_matrix_optical_matched_7d.csv", "7d", "any_sensor"),
    OpticalDataset("hls_3d", "training_matrix_optical_matched_3d_hls.csv", "3d", "hls", primary_sensor_check=True),
    OpticalDataset("landsat_3d", "training_matrix_optical_matched_3d_landsat.csv", "3d", "landsat", primary_sensor_check=True),
    OpticalDataset("sentinel2_3d", "training_matrix_optical_matched_3d_sentinel2.csv", "3d", "sentinel2", primary_sensor_check=True),
]

OPTICAL_FEATURE_SETS = [
    OpticalFeatureSet(
        "B0_F3_same_subset",
        "Primary F3 baseline comparator evaluated on the same rows as each optical candidate.",
        ("log_Q", "sin_doy", "cos_doy"),
        ("river",),
        baseline_comparator=True,
    ),
    OpticalFeatureSet(
        "O1_quality_only",
        "F3 plus optical matching quality variables to audit days-offset and water-pixel bias.",
        ("log_Q", "sin_doy", "cos_doy", "abs_days_offset", "pct_valid_water_pixels"),
        ("river",),
    ),
    OpticalFeatureSet(
        "O2_indices",
        "F3 plus NDWI/MNDWI and simple reflectance ratios.",
        ("log_Q", "sin_doy", "cos_doy", "ndwi", "mndwi", "red_green_ratio", "green_blue_ratio"),
        ("river",),
    ),
    OpticalFeatureSet(
        "O3_bands",
        "F3 plus optical reflectance bands.",
        ("log_Q", "sin_doy", "cos_doy", "blue", "green", "red", "nir", "swir1", "swir2"),
        ("river",),
    ),
    OpticalFeatureSet(
        "O4_bands_indices",
        "F3 plus reflectance bands, NDWI/MNDWI, and ratios.",
        (
            "log_Q",
            "sin_doy",
            "cos_doy",
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
        ),
        ("river",),
    ),
    OpticalFeatureSet(
        "O5_sensor_aware_any_sensor",
        "Any-sensor model with sensor one-hot, bands, indices, water-pixel fraction, and match offset.",
        (
            "log_Q",
            "sin_doy",
            "cos_doy",
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
            "pct_valid_water_pixels",
            "abs_days_offset",
        ),
        ("river", "sensor"),
        any_sensor_only=True,
    ),
]

OPTICAL_MODEL_SPECS = [
    OpticalModelSpec("linear_regression", "linear", "LinearRegression"),
    OpticalModelSpec("ridge_alpha_0.1", "ridge", "Ridge", alpha=0.1),
    OpticalModelSpec("ridge_alpha_1", "ridge", "Ridge", alpha=1.0),
    OpticalModelSpec("ridge_alpha_10", "ridge", "Ridge", alpha=10.0),
    OpticalModelSpec("ridge_alpha_100", "ridge", "Ridge", alpha=100.0),
]

BASELINE_COMPARATOR = OPTICAL_FEATURE_SETS[0]
BASELINE_MODEL_ID = "ridge_alpha_1"

FORBIDDEN_OPTICAL_PHASE_FEATURES = {
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
    "basin_SUB_AREA",
    "HYBAS_ID_mean",
    "NEXT_DOWN_mean",
    "PFAF_ID_mean",
    "DOC_mgC_L",
    "predicted_DOC",
    "pred_DOC",
    "prediction",
    "flux",
    "TgC",
    "Mg_day",
}


def optical_dataset_registry() -> pd.DataFrame:
    return pd.DataFrame([dataset.__dict__ for dataset in OPTICAL_DATASETS])


def optical_feature_set_registry() -> pd.DataFrame:
    rows = []
    for feature_set in OPTICAL_FEATURE_SETS:
        rows.append(
            {
                "feature_set": feature_set.feature_set,
                "description": feature_set.description,
                "numeric_features": ";".join(feature_set.numeric_features),
                "categorical_features": ";".join(feature_set.categorical_features),
                "all_features": ";".join(feature_set.required_features),
                "any_sensor_only": feature_set.any_sensor_only,
                "baseline_comparator": feature_set.baseline_comparator,
                "validation_only": True,
            }
        )
    return pd.DataFrame(rows)


def optical_model_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_id": spec.model_id,
                "model_family": spec.model_family,
                "model_class": spec.model_class,
                "alpha": spec.alpha if spec.alpha is not None else "",
                "simple_model_only": True,
                "validation_only": True,
            }
            for spec in OPTICAL_MODEL_SPECS
        ]
    )


def optical_feature_sets_for_dataset(dataset: OpticalDataset) -> list[OpticalFeatureSet]:
    if dataset.sensor_scope == "any_sensor":
        return [feature_set for feature_set in OPTICAL_FEATURE_SETS if not feature_set.baseline_comparator]
    return [
        feature_set
        for feature_set in OPTICAL_FEATURE_SETS
        if not feature_set.baseline_comparator and not feature_set.any_sensor_only
    ]


def validate_optical_feature_sets() -> None:
    used = set()
    for feature_set in OPTICAL_FEATURE_SETS:
        used.update(feature_set.required_features)
    forbidden = used.intersection(FORBIDDEN_OPTICAL_PHASE_FEATURES)
    if forbidden:
        raise RuntimeError(f"Forbidden optical-phase features configured: {sorted(forbidden)}")
