from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def path(*parts: str | Path) -> Path:
    return project_root().joinpath(*map(Path, parts))


CONFIG_DIR = path("configs")
OUTPUT_DIR = path("outputs")
REPORT_DIR = path("outputs", "reports")
TABLE_DIR = path("outputs", "tables")
LOG_DIR = path("outputs", "logs")


def ensure_output_dirs() -> None:
    for directory in [REPORT_DIR, TABLE_DIR, LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
