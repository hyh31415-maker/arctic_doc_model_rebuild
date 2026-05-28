from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .paths import CONFIG_DIR, path


DEFAULT_GOLD_DATA_DIR = Path("D:/Hao/Desktop/冰冻圈水文/北极大河/arctic_doc_data_audit/data/processed/gold")
CONTRACT_PATH = CONFIG_DIR / "gold_data_contract.yaml"
LOCAL_PATHS_PATH = CONFIG_DIR / "local_paths.yaml"


@dataclass(frozen=True)
class GoldDataLocation:
    data_dir: Path
    source: str


def load_contract() -> dict[str, Any]:
    with CONTRACT_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_local_paths() -> dict[str, Any]:
    if not LOCAL_PATHS_PATH.exists():
        return {}
    with LOCAL_PATHS_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_gold_data_location() -> GoldDataLocation:
    env_value = os.environ.get("ARCTIC_DOC_GOLD_DIR", "").strip()
    if env_value:
        return GoldDataLocation(Path(env_value).expanduser(), "ARCTIC_DOC_GOLD_DIR")

    local_paths = _load_local_paths()
    configured = str(local_paths.get("gold_data_dir", "")).strip()
    if configured:
        return GoldDataLocation(Path(configured).expanduser(), "configs/local_paths.yaml")

    return GoldDataLocation(DEFAULT_GOLD_DATA_DIR, "built-in default fallback")


def resolve_gold_data_dir() -> Path:
    return resolve_gold_data_location().data_dir


def require_gold_data_dir() -> Path:
    gold_dir = resolve_gold_data_dir()
    if not gold_dir.exists() or not gold_dir.is_dir():
        raise FileNotFoundError(
            "Gold data directory was not found. Set it with: "
            '$env:ARCTIC_DOC_GOLD_DIR="D:/Hao/Desktop/冰冻圈水文/北极大河/arctic_doc_data_audit/data/processed/gold"'
        )
    return gold_dir


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_table_hash(file_path: Path, expected_sha256: str) -> bool:
    return sha256_file(file_path).lower() == expected_sha256.lower()


def _row_count(file_path: Path) -> int:
    return len(pd.read_csv(file_path, usecols=[0], low_memory=False))


def verify_table_row_count(file_path: Path, expected_row_count: int) -> bool:
    return _row_count(file_path) == int(expected_row_count)


def expected_table_names(contract: dict[str, Any] | None = None) -> list[str]:
    contract = contract or load_contract()
    return list(contract.get("expected_tables", {}).keys())


def _is_inside(directory: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def table_path(table_name: str, gold_dir: Path | None = None) -> Path:
    contract = load_contract()
    if table_name not in contract.get("expected_tables", {}):
        raise KeyError(f"Table is not part of the gold contract: {table_name}")
    root = gold_dir or require_gold_data_dir()
    candidate = root / table_name
    if not _is_inside(root, candidate):
        raise ValueError(f"Refusing to read outside the resolved gold data dir: {candidate}")
    return candidate


def verify_all_gold_tables() -> pd.DataFrame:
    contract = load_contract()
    gold_dir = require_gold_data_dir()
    rows: list[dict[str, Any]] = []
    for table_name, spec in contract.get("expected_tables", {}).items():
        file_path = gold_dir / table_name
        role = spec.get("role", "")
        expected_row_count = int(spec.get("row_count", 0))
        expected_sha256 = str(spec.get("sha256", "")).lower()
        row = {
            "table_name": table_name,
            "role": role,
            "expected_row_count": expected_row_count,
            "actual_row_count": "",
            "row_count_ok": False,
            "expected_sha256": expected_sha256,
            "actual_sha256": "",
            "sha256_ok": False,
            "exists": file_path.exists(),
            "status": "missing",
            "message": "",
        }
        if not file_path.exists():
            row["message"] = f"Missing expected gold table: {file_path}"
            rows.append(row)
            continue
        try:
            actual_row_count = _row_count(file_path)
            actual_sha256 = sha256_file(file_path)
            row.update(
                {
                    "actual_row_count": actual_row_count,
                    "row_count_ok": actual_row_count == expected_row_count,
                    "actual_sha256": actual_sha256,
                    "sha256_ok": actual_sha256 == expected_sha256,
                }
            )
            if not row["row_count_ok"]:
                row["status"] = "row_count_mismatch"
                row["message"] = f"Expected {expected_row_count} rows, found {actual_row_count}."
            elif not row["sha256_ok"]:
                row["status"] = "sha256_mismatch"
                row["message"] = "SHA256 does not match the frozen contract."
            else:
                row["status"] = "ok"
                row["message"] = "Verified against frozen gold contract."
        except Exception as exc:
            row["status"] = "read_error"
            row["message"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def verification_problem_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {"missing": 0, "row_count_mismatch": 0, "sha256_mismatch": 0, "read_error": 0}
    return {
        "missing": int(frame["status"].eq("missing").sum()),
        "row_count_mismatch": int(frame["status"].eq("row_count_mismatch").sum()),
        "sha256_mismatch": int(frame["status"].eq("sha256_mismatch").sum()),
        "read_error": int(frame["status"].eq("read_error").sum()),
    }
