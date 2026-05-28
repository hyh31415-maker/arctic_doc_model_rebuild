from __future__ import annotations

import hashlib

from arctic_doc_model_rebuild.gold_contract import load_contract, sha256_file, verify_table_hash


def test_contract_loads() -> None:
    contract = load_contract()
    assert contract["freeze_id"] == "data_freeze_gold_20260526_v1"
    assert contract["source_repo"] == "hyh31415-maker/arctic_doc_data_audit"


def test_expected_tables_are_listed() -> None:
    tables = load_contract()["expected_tables"]
    assert "doc_labels_gold.csv" in tables
    assert "training_matrix_hydrocore.csv" in tables
    assert "prediction_grid_daily_hydrocore.csv" in tables


def test_forbidden_columns_are_listed() -> None:
    contract = load_contract()
    forbidden = set(contract["forbidden_predictor_columns"])
    assert {"A254", "A375", "A440", "SUVA254"}.issubset(forbidden)
    assert "DOC_mgC_L" in contract["forbidden_prediction_grid_columns"]


def test_hash_verification_function_works(tmp_path) -> None:
    target = tmp_path / "tiny.csv"
    target.write_text("a,b\n1,2\n", encoding="utf-8")
    expected = hashlib.sha256(target.read_bytes()).hexdigest()
    assert sha256_file(target) == expected
    assert verify_table_hash(target, expected)
