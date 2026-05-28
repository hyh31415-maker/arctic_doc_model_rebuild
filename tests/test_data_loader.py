from __future__ import annotations

import pandas as pd
import pytest

from arctic_doc_model_rebuild.data_loader import load_hydrocore
from arctic_doc_model_rebuild.schema_checks import check_hydrocore_matrix


HYDRO_COLUMNS = [
    "label_id",
    "river",
    "date",
    "DOC_mgC_L",
    "Q_m3s",
    "doy",
    "sin_doy",
    "cos_doy",
    "temperature_2m_C",
    "positive_degree_day_Cday",
    "snow_cover_fraction",
    "snow_depletion_rate_7d",
    "surface_runoff_m",
]


def _write_hydrocore(directory, value: str = "gold") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    row = {column: 1 for column in HYDRO_COLUMNS}
    row.update({"label_id": value, "river": "Yukon", "date": "2020-01-01"})
    pd.DataFrame([row]).to_csv(directory / "training_matrix_hydrocore.csv", index=False)


def test_loader_refuses_missing_gold_dir(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ARCTIC_DOC_GOLD_DIR", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="Gold data directory was not found"):
        load_hydrocore()


def test_loader_reads_only_resolved_gold_dir(monkeypatch, tmp_path) -> None:
    gold_dir = tmp_path / "gold"
    _write_hydrocore(gold_dir, value="gold-row")
    monkeypatch.setenv("ARCTIC_DOC_GOLD_DIR", str(gold_dir))
    frame = load_hydrocore()
    assert frame.loc[0, "label_id"] == "gold-row"


def test_loader_does_not_read_raw_interim_or_canonical_paths(monkeypatch, tmp_path) -> None:
    gold_dir = tmp_path / "gold"
    _write_hydrocore(gold_dir, value="gold-row")
    for directory in [tmp_path / "raw", tmp_path / "interim", tmp_path / "canonical"]:
        _write_hydrocore(directory, value="poison")
    monkeypatch.setenv("ARCTIC_DOC_GOLD_DIR", str(gold_dir))
    frame = load_hydrocore()
    assert set(frame["label_id"]) == {"gold-row"}


def test_hydrocore_loader_fixture_has_expected_columns(monkeypatch, tmp_path) -> None:
    gold_dir = tmp_path / "gold"
    _write_hydrocore(gold_dir)
    monkeypatch.setenv("ARCTIC_DOC_GOLD_DIR", str(gold_dir))
    frame = load_hydrocore()
    checks = check_hydrocore_matrix(frame)
    required = next(item for item in checks if item["check_name"] == "hydrocore_required_columns")
    assert required["passed"]
