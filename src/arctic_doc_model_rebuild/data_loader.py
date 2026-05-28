from __future__ import annotations

from pathlib import Path

import pandas as pd

from .gold_contract import require_gold_data_dir, table_path


def _load_gold_csv(table_name: str) -> pd.DataFrame:
    gold_dir = require_gold_data_dir()
    destination = table_path(table_name, gold_dir=gold_dir)
    if not destination.exists():
        raise FileNotFoundError(f"Missing expected gold table: {destination}")
    return pd.read_csv(destination, low_memory=False)


def load_hydrocore() -> pd.DataFrame:
    return _load_gold_csv("training_matrix_hydrocore.csv")


def load_basin_context_matrix() -> pd.DataFrame:
    return _load_gold_csv("training_matrix_basin_context.csv")


def load_optical_matched(window: str = "3d", sensor: str | None = None) -> pd.DataFrame:
    normalized_window = window.lower().strip()
    if sensor is None:
        table_name = f"training_matrix_optical_matched_{normalized_window}.csv"
    else:
        normalized_sensor = sensor.lower().replace("-", "").replace("_", "")
        sensor_map = {"hls": "hls", "landsat": "landsat", "sentinel2": "sentinel2"}
        if normalized_sensor not in sensor_map:
            raise ValueError("sensor must be one of: hls, landsat, sentinel2")
        table_name = f"training_matrix_optical_matched_{normalized_window}_{sensor_map[normalized_sensor]}.csv"
    return _load_gold_csv(table_name)


def load_prediction_grid(with_basin: bool = False) -> pd.DataFrame:
    if with_basin:
        return _load_gold_csv("prediction_grid_daily_with_basin_context.csv")
    return _load_gold_csv("prediction_grid_daily_hydrocore.csv")


def load_basin_attributes(long: bool = True) -> pd.DataFrame:
    if long:
        return _load_gold_csv("basin_attributes_curated.csv")
    return _load_gold_csv("basin_attributes_curated_wide.csv")


def gold_table_file(table_name: str) -> Path:
    return table_path(table_name, gold_dir=require_gold_data_dir())
