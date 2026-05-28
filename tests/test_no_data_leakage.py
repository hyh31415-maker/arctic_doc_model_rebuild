from __future__ import annotations

import pandas as pd

from arctic_doc_model_rebuild.schema_checks import (
    check_basin_attributes,
    check_hydrocore_matrix,
    check_optical_matched_matrix,
    check_prediction_grid,
)


def _check_map(rows):
    return {row["check_name"]: row for row in rows}


def test_hydrocore_matrix_does_not_contain_lab_optical_columns() -> None:
    frame = pd.DataFrame(
        {
            "label_id": range(500),
            "river": ["Yukon"] * 500,
            "date": ["2020-01-01"] * 500,
            "DOC_mgC_L": [5.0] * 500,
            "Q_m3s": [100.0] * 500,
            "doy": [1] * 500,
            "sin_doy": [0.0] * 500,
            "cos_doy": [1.0] * 500,
            "temperature_2m_C": [-10.0] * 500,
            "positive_degree_day_Cday": [0.0] * 500,
            "snow_cover_fraction": [1.0] * 500,
            "snow_depletion_rate_7d": [0.0] * 500,
            "surface_runoff_m": [0.0] * 500,
        }
    )
    checks = _check_map(check_hydrocore_matrix(frame))
    assert checks["hydrocore_no_lab_optical"]["passed"]
    assert checks["hydrocore_no_prediction_flux"]["passed"]


def test_prediction_grid_does_not_contain_doc_prediction_or_flux_columns() -> None:
    frame = pd.DataFrame(
        {
            "river": ["Yukon"] * 2,
            "date": ["2020-01-01", "2020-01-02"],
            "Q_m3s": [100.0, 101.0],
            "doy": [1, 2],
            "sin_doy": [0.0, 0.1],
            "cos_doy": [1.0, 0.9],
        }
    )
    checks = _check_map(check_prediction_grid(frame))
    assert checks["prediction_grid_no_doc"]["passed"]
    assert checks["prediction_grid_no_prediction_flux"]["passed"]


def test_optical_matrix_contains_optical_columns_and_sensor() -> None:
    frame = pd.DataFrame(
        {
            "label_id": ["a"],
            "river": ["Yukon"],
            "date": ["2020-01-01"],
            "DOC_mgC_L": [5.0],
            "sensor": ["HLS"],
            "optical_date": ["2020-01-01"],
            "days_offset": [0],
            "blue": [0.1],
            "green": [0.2],
            "red": [0.3],
            "nir": [0.4],
            "ndwi": [0.5],
            "mndwi": [0.6],
        }
    )
    checks = _check_map(check_optical_matched_matrix(frame))
    assert checks["optical_has_bands"]["passed"]
    assert checks["optical_has_sensor"]["passed"]
    assert checks["optical_no_lab_optical"]["passed"]


def test_basin_attributes_do_not_treat_id_means_as_predictors() -> None:
    rivers = ["Ob", "Yenisey", "Lena", "Mackenzie", "Yukon", "Kolyma"]
    rows = []
    for river in rivers:
        rows.append(
            {
                "river": river,
                "upstream_area_km2": 1000,
                "source_field": "HYBAS_ID_mean",
                "model_use": False,
                "attribute_name": "HYBAS_ID",
            }
        )
        rows.append(
            {
                "river": river,
                "upstream_area_km2": 1000,
                "source_field": "SUB_AREA_mean",
                "model_use": True,
                "attribute_name": "SUB_AREA",
            }
        )
    long_frame = pd.DataFrame(rows)
    wide_frame = pd.DataFrame({"river": rivers, "basin_SUB_AREA": [1000] * 6})
    checks = _check_map(check_basin_attributes(long_frame, wide_frame))
    assert checks["basin_six_rivers"]["passed"]
    assert checks["basin_id_means_not_predictors"]["passed"]
    assert checks["basin_wide_no_id_means"]["passed"]
    assert checks["basin_attributes_exist"]["passed"]
