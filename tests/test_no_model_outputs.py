from __future__ import annotations

import pandas as pd

from arctic_doc_model_rebuild import cli
from arctic_doc_model_rebuild.paths import project_root


def test_verify_command_does_not_create_model_prediction_or_flux_dirs(monkeypatch) -> None:
    verification = pd.DataFrame(
        [
            {
                "table_name": "training_matrix_hydrocore.csv",
                "role": "main_training_matrix",
                "expected_row_count": 1,
                "actual_row_count": 1,
                "row_count_ok": True,
                "expected_sha256": "0" * 64,
                "actual_sha256": "0" * 64,
                "sha256_ok": True,
                "exists": True,
                "status": "ok",
                "message": "ok",
            }
        ]
    )
    schema = pd.DataFrame(
        [
            {
                "check_name": "no_leakage",
                "table_name": "training_matrix_hydrocore.csv",
                "passed": True,
                "status": "ok",
                "severity": "error",
                "message": "ok",
            }
        ]
    )
    monkeypatch.setattr(cli, "write_verification_outputs", lambda: (verification, schema))
    assert cli.main(["verify-gold-data"]) == 0
    root = project_root()
    assert not (root / "outputs" / "models").exists()
    assert not (root / "outputs" / "predictions").exists()
    assert not (root / "outputs" / "flux").exists()
    forbidden = [
        item
        for item in root.rglob("*")
        if item.is_file() and item.suffix.lower() in {".joblib", ".pkl", ".pickle"}
    ]
    assert forbidden == []
