from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .data_loader import gold_table_file
from .gold_contract import load_contract, resolve_gold_data_location, sha256_file, verify_all_gold_tables, verification_problem_counts
from .paths import REPORT_DIR, TABLE_DIR, ensure_output_dirs
from .schema_checks import issue_count, run_all_schema_checks
from .modeling.diagnostics import ALLOWED_DOC_MODEL_ARTIFACTS


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_csv(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8")
    return destination


def _md_table(frame: pd.DataFrame, max_rows: int = 50) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.head(max_rows).to_markdown(index=False)


def model_output_status() -> dict[str, Any]:
    outputs = Path("outputs")
    forbidden_dirs = [outputs / "predictions", outputs / "flux"]
    forbidden_files = []
    if outputs.exists():
        forbidden_files = [
            item
            for item in outputs.rglob("*")
            if item.is_file() and item.suffix.lower() in {".joblib", ".pkl", ".pickle"} and item.name not in ALLOWED_DOC_MODEL_ARTIFACTS
        ]
    return {
        "outputs_models_exists": (outputs / "models").exists(),
        "outputs_predictions_exists": (outputs / "predictions").exists(),
        "outputs_flux_exists": (outputs / "flux").exists(),
        "model_binary_count": len(forbidden_files),
        "forbidden_dirs_present": [str(item) for item in forbidden_dirs if item.exists()],
    }


def write_verification_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_output_dirs()
    verification = verify_all_gold_tables()
    schema = run_all_schema_checks()
    _write_csv(verification, TABLE_DIR / "gold_table_verification.csv")
    _write_csv(schema, TABLE_DIR / "model_input_schema_check.csv")
    write_data_contract_report(verification, schema)
    return verification, schema


def write_data_contract_report(verification: pd.DataFrame | None = None, schema: pd.DataFrame | None = None) -> Path:
    ensure_output_dirs()
    contract = load_contract()
    location = resolve_gold_data_location()
    verification = verification if verification is not None else verify_all_gold_tables()
    schema = schema if schema is not None else run_all_schema_checks()
    counts = verification_problem_counts(verification)
    output_status = model_output_status()
    schema_issues = issue_count(schema)
    hash_mismatches = counts.get("sha256_mismatch", 0)
    lines = [
        "# Gold Data Contract Report",
        "",
        f"Generated: {utc_now()}",
        "",
        f"- freeze_id: `{contract['freeze_id']}`",
        f"- source_repo: `{contract['source_repo']}`",
        f"- source_tag: `{contract['source_tag']}`",
        f"- data_dir: `{location.data_dir}`",
        f"- data_dir_source: `{location.source}`",
        f"- expected_tables: `{len(contract.get('expected_tables', {}))}`",
        f"- verified_tables_ok: `{int(verification['status'].eq('ok').sum())}`",
        f"- hash_mismatches: `{hash_mismatches}`",
        f"- row_count_mismatches: `{counts.get('row_count_mismatch', 0)}`",
        f"- missing_tables: `{counts.get('missing', 0)}`",
        f"- read_errors: `{counts.get('read_error', 0)}`",
        f"- schema_issues: `{schema_issues}`",
        f"- forbidden_output_dirs_present: `{len(output_status['forbidden_dirs_present'])}`",
        f"- model_binary_count: `{output_status['model_binary_count']}`",
        "",
        "No model was trained.",
        "No DOC prediction was generated.",
        "No flux was generated.",
        "Only gold freeze data were read.",
        "",
        "## Table Statuses",
        "",
        _md_table(verification[["table_name", "role", "expected_row_count", "actual_row_count", "row_count_ok", "sha256_ok", "status"]]),
        "",
        "## Schema And Leakage Checks",
        "",
        _md_table(schema[["check_name", "table_name", "passed", "status", "message"]]),
        "",
        "## Next Recommended Step",
        "",
        "EDA phase, after the data contract remains fully passing.",
    ]
    destination = REPORT_DIR / "data_contract_report.md"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def summarize_gold_data() -> tuple[dict[str, Path], pd.DataFrame]:
    ensure_output_dirs()
    contract = load_contract()
    verification = verify_all_gold_tables()
    inventory_rows: list[dict[str, Any]] = []
    missingness_rows: list[dict[str, Any]] = []
    by_river_rows: list[dict[str, Any]] = []
    by_year_rows: list[dict[str, Any]] = []

    for table_name, spec in contract.get("expected_tables", {}).items():
        file_path = gold_table_file(table_name)
        if not file_path.exists():
            continue
        frame = pd.read_csv(file_path, low_memory=False)
        inventory_rows.append(
            {
                "table_name": table_name,
                "role": spec.get("role", ""),
                "rows": len(frame),
                "columns": len(frame.columns),
                "file_size_bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
        for column in frame.columns:
            missing = int(frame[column].isna().sum())
            missingness_rows.append(
                {
                    "table_name": table_name,
                    "column": column,
                    "missing_count": missing,
                    "missing_rate": missing / len(frame) if len(frame) else 0.0,
                }
            )
        if "river" in frame.columns:
            for river, count in frame.groupby("river", dropna=False).size().items():
                by_river_rows.append({"table_name": table_name, "river": river, "row_count": int(count)})
        if "year" in frame.columns:
            years = frame["year"]
        elif "date" in frame.columns:
            years = pd.to_datetime(frame["date"], errors="coerce").dt.year
        else:
            years = pd.Series(dtype="float64")
        if not years.empty:
            for year, count in years.dropna().astype(int).groupby(years.dropna().astype(int)).size().items():
                by_year_rows.append({"table_name": table_name, "year": int(year), "row_count": int(count)})

    inventory = pd.DataFrame(inventory_rows)
    missingness = pd.DataFrame(missingness_rows)
    by_river = pd.DataFrame(by_river_rows)
    by_year = pd.DataFrame(by_year_rows)
    paths = {
        "inventory": _write_csv(inventory, TABLE_DIR / "gold_input_inventory.csv"),
        "missingness": _write_csv(missingness, TABLE_DIR / "gold_matrix_missingness.csv"),
        "by_river": _write_csv(by_river, TABLE_DIR / "gold_matrix_by_river.csv"),
        "by_year": _write_csv(by_year, TABLE_DIR / "gold_matrix_by_year.csv"),
    }
    report_lines = [
        "# Gold Data Summary Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "No model was trained. No DOC prediction was generated. No flux was generated.",
        "",
        "## Inventory",
        "",
        _md_table(inventory[["table_name", "role", "rows", "columns", "file_size_bytes"]]),
        "",
        "## Verification Snapshot",
        "",
        _md_table(verification[["table_name", "status", "row_count_ok", "sha256_ok"]]),
    ]
    paths["report"] = REPORT_DIR / "gold_data_summary_report.md"
    paths["report"].write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return paths, inventory
