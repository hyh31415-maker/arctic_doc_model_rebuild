from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..gold_contract import require_gold_data_dir, sha256_file, table_path
from ..paths import REPORT_DIR, TABLE_DIR, path
from .flux_qc import as_bool
from .snowmelt_window_reports import SNOWMELT_REPORT_DIR, SNOWMELT_TABLE_DIR


SNOWMELT_FIGURE_DIR = path("outputs", "figures", "snowmelt_windows")
DOC_FLUX_TABLE_DIR = TABLE_DIR / "doc_flux"
FLUX_INTERPRETATION_TABLE_DIR = TABLE_DIR / "flux_interpretation"
MAY_JULY_TABLE_DIR = TABLE_DIR / "may_july_flux"
ANNUAL_TREND_TABLE_DIR = TABLE_DIR / "annual_flux_trends"

WINDOW_IDS = [
    "fixed_may_july_reference",
    "discharge_centered_freshet",
    "q75_peak_contiguous",
    "snow_depletion_assisted",
    "common_overlap_w1_w2",
]


def required_snowmelt_input_paths() -> dict[str, Path]:
    gold_dir = require_gold_data_dir()
    return {
        "daily_doc_flux": DOC_FLUX_TABLE_DIR / "daily_doc_flux.csv",
        "annual_doc_flux_summary": DOC_FLUX_TABLE_DIR / "annual_doc_flux_summary.csv",
        "provisional_may_july_flux_summary": DOC_FLUX_TABLE_DIR / "provisional_may_july_flux_summary.csv",
        "annual_flux_analysis_cohorts": FLUX_INTERPRETATION_TABLE_DIR / "annual_flux_analysis_cohorts.csv",
        "provisional_may_july_cohort_summary": FLUX_INTERPRETATION_TABLE_DIR / "provisional_may_july_cohort_summary.csv",
        "may_july_flux_interpretation": MAY_JULY_TABLE_DIR / "may_july_flux_interpretation_by_river_year.csv",
        "annual_flux_trends_by_river": ANNUAL_TREND_TABLE_DIR / "annual_flux_trends_by_river.csv",
        "may_july_flux_report": REPORT_DIR / "may_july_flux" / "may_july_flux_interpretation_report.md",
        "annual_flux_trend_report": REPORT_DIR / "annual_flux_trends" / "annual_flux_trend_report.md",
        "prediction_grid_daily_hydrocore": table_path("prediction_grid_daily_hydrocore.csv", gold_dir=gold_dir),
        "daily_hydroclimate_gold": table_path("daily_hydroclimate_gold.csv", gold_dir=gold_dir),
        "daily_discharge_gold": table_path("daily_discharge_gold.csv", gold_dir=gold_dir),
    }


def _ensure_dirs() -> None:
    for directory in [SNOWMELT_TABLE_DIR, SNOWMELT_REPORT_DIR, SNOWMELT_FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def _read_csv(destination: Path) -> pd.DataFrame:
    if not destination.exists():
        raise FileNotFoundError(f"Required snowmelt window input is missing: {destination}")
    return pd.read_csv(destination, low_memory=False)


def _read_text(destination: Path) -> str:
    if not destination.exists():
        raise FileNotFoundError(f"Required snowmelt window input is missing: {destination}")
    return destination.read_text(encoding="utf-8")


def _write_csv(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8")
    return destination


def _input_hashes() -> dict[Path, str]:
    return {destination: sha256_file(destination) for destination in required_snowmelt_input_paths().values()}


def verify_snowmelt_inputs_unchanged(before: dict[Path, str]) -> None:
    after = _input_hashes()
    changed = [str(destination) for destination, before_hash in before.items() if after.get(destination) != before_hash]
    if changed:
        raise RuntimeError(f"Snowmelt window inputs changed during analysis: {changed}")


def _load_definition_inputs() -> dict[str, Any]:
    paths = required_snowmelt_input_paths()
    return {
        "daily_flux": _read_csv(paths["daily_doc_flux"]),
        "annual": _read_csv(paths["annual_doc_flux_summary"]),
        "annual_cohorts": _read_csv(paths["annual_flux_analysis_cohorts"]),
        "prediction_grid": _read_csv(paths["prediction_grid_daily_hydrocore"]),
        "hydroclimate": _read_csv(paths["daily_hydroclimate_gold"]),
        "discharge": _read_csv(paths["daily_discharge_gold"]),
        "may_july_report_text": _read_text(paths["may_july_flux_report"]),
        "annual_trend_report_text": _read_text(paths["annual_flux_trend_report"]),
    }


def _prepare_daily_flux(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    for column in ["Q_m3s", "daily_flux_TgC_day"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _prepare_hydrology(inputs: dict[str, Any]) -> pd.DataFrame:
    grid = inputs["prediction_grid"].copy()
    grid["date"] = pd.to_datetime(grid["date"], errors="coerce")
    grid["year"] = pd.to_numeric(grid["year"], errors="coerce").astype("Int64")
    for column in [
        "doy",
        "Q_m3s",
        "positive_degree_day_Cday",
        "snow_cover_fraction",
        "snow_depletion_rate_7d",
        "surface_runoff_m",
    ]:
        if column in grid.columns:
            grid[column] = pd.to_numeric(grid[column], errors="coerce")

    hydro = inputs["hydroclimate"].copy()
    hydro["date"] = pd.to_datetime(hydro["date"], errors="coerce")
    hydro_cols = ["river", "date", "positive_degree_day_Cday", "snow_cover_fraction", "snow_depletion_rate_7d", "surface_runoff_m"]
    hydro = hydro[[column for column in hydro_cols if column in hydro.columns]].drop_duplicates(["river", "date"])
    hydro = hydro.rename(columns={column: f"{column}_hydro_gold" for column in hydro.columns if column not in {"river", "date"}})
    grid = grid.merge(hydro, on=["river", "date"], how="left")
    for column in ["positive_degree_day_Cday", "snow_cover_fraction", "snow_depletion_rate_7d", "surface_runoff_m"]:
        fallback = f"{column}_hydro_gold"
        if fallback in grid.columns:
            grid[column] = grid[column].where(grid[column].notna(), grid[fallback])

    discharge = inputs["discharge"].copy()
    discharge["date"] = pd.to_datetime(discharge["date"], errors="coerce")
    if "is_primary_discharge" in discharge.columns:
        discharge = discharge[as_bool(discharge["is_primary_discharge"])]
    discharge = discharge[["river", "date", "Q_m3s"]].drop_duplicates(["river", "date"]).rename(columns={"Q_m3s": "Q_m3s_discharge_gold"})
    grid = grid.merge(discharge, on=["river", "date"], how="left")
    grid["Q_m3s"] = grid["Q_m3s"].where(grid["Q_m3s"].notna(), grid["Q_m3s_discharge_gold"])
    grid["month"] = grid["date"].dt.month
    grid["doy"] = grid["date"].dt.dayofyear
    return grid.sort_values(["river", "date"]).reset_index(drop=True)


def _candidate_season(group: pd.DataFrame) -> pd.DataFrame:
    return group[group["month"].between(3, 8, inclusive="both")].copy().sort_values("date")


def _inclusive_days(start: pd.Timestamp | None, end: pd.Timestamp | None) -> int:
    if pd.isna(start) or pd.isna(end) or start is None or end is None or end < start:
        return 0
    return int((end.normalize() - start.normalize()).days) + 1


def _flux_count(daily_flux: pd.DataFrame, river: str, start: pd.Timestamp | None, end: pd.Timestamp | None) -> int:
    if _inclusive_days(start, end) == 0:
        return 0
    subset = daily_flux[
        daily_flux["river"].astype(str).eq(str(river))
        & daily_flux["date"].between(start, end, inclusive="both")
        & daily_flux["flux_status"].astype(str).eq("calculated")
    ]
    return int(len(subset))


def _peak_info(candidate: pd.DataFrame) -> tuple[pd.Timestamp | None, float]:
    q = pd.to_numeric(candidate.get("Q_m3s"), errors="coerce")
    work = candidate[q.notna()].copy()
    if work.empty:
        return None, float("nan")
    idx = work["Q_m3s"].astype(float).idxmax()
    return pd.Timestamp(work.loc[idx, "date"]), float(work.loc[idx, "Q_m3s"])


def _w1_dates(candidate: pd.DataFrame) -> dict[str, Any]:
    peak_date, peak_q = _peak_info(candidate)
    if peak_date is None:
        return {"start": pd.NaT, "end": pd.NaT, "peak": pd.NaT, "peak_q": np.nan, "status": "missing_q", "fallback": False}
    q = pd.to_numeric(candidate["Q_m3s"], errors="coerce")
    q25 = float(q.quantile(0.25))
    q50 = float(q.quantile(0.50))
    before = candidate[candidate["date"].le(peak_date)].copy()
    before["above25"] = pd.to_numeric(before["Q_m3s"], errors="coerce").ge(q25)
    above_counts = []
    values = before["above25"].to_numpy()
    for i in range(len(values)):
        above_counts.append(int(values[i : min(i + 7, len(values))].sum()))
    before["next7_above25"] = above_counts
    start_candidates = before[before["next7_above25"].ge(5)]
    fallback = start_candidates.empty
    start = pd.Timestamp(start_candidates.iloc[0]["date"]) if not start_candidates.empty else peak_date - pd.Timedelta(days=30)
    after = candidate[candidate["date"].ge(peak_date)].copy()
    below = after[pd.to_numeric(after["Q_m3s"], errors="coerce").lt(q50)]
    recession = pd.Timestamp(below.iloc[0]["date"]) if not below.empty else peak_date + pd.Timedelta(days=30)
    end = max(recession, peak_date + pd.Timedelta(days=30))
    season_start = pd.Timestamp(year=int(peak_date.year), month=3, day=1)
    season_end = pd.Timestamp(year=int(peak_date.year), month=8, day=31)
    start = max(start, season_start)
    end = min(end, season_end)
    return {"start": start, "end": end, "peak": peak_date, "peak_q": peak_q, "status": "ok", "fallback": fallback}


def _w2_dates(candidate: pd.DataFrame) -> dict[str, Any]:
    peak_date, peak_q = _peak_info(candidate)
    if peak_date is None:
        return {"start": pd.NaT, "end": pd.NaT, "peak": pd.NaT, "peak_q": np.nan, "status": "missing_q", "fallback": False}
    work = candidate.copy().sort_values("date").reset_index(drop=True)
    q75 = float(pd.to_numeric(work["Q_m3s"], errors="coerce").quantile(0.75))
    work["in_q75"] = pd.to_numeric(work["Q_m3s"], errors="coerce").ge(q75)
    work["period_id"] = (work["in_q75"] != work["in_q75"].shift()).cumsum()
    peak_row = work[work["date"].eq(peak_date)]
    if peak_row.empty or not bool(peak_row.iloc[0]["in_q75"]):
        return {"start": peak_date, "end": peak_date, "peak": peak_date, "peak_q": peak_q, "status": "q75_peak_missing", "fallback": True}
    period = peak_row.iloc[0]["period_id"]
    chosen = work[work["period_id"].eq(period) & work["in_q75"]]
    return {
        "start": pd.Timestamp(chosen["date"].min()),
        "end": pd.Timestamp(chosen["date"].max()),
        "peak": peak_date,
        "peak_q": peak_q,
        "status": "ok",
        "fallback": False,
    }


def _w3_dates(candidate: pd.DataFrame, w1: dict[str, Any]) -> dict[str, Any]:
    peak_date, peak_q = _peak_info(candidate)
    if peak_date is None:
        return {"start": pd.NaT, "end": pd.NaT, "peak": pd.NaT, "peak_q": np.nan, "status": "missing_q", "fallback": False, "snow_used": False}
    snow_cols = ["positive_degree_day_Cday", "snow_cover_fraction", "snow_depletion_rate_7d", "surface_runoff_m"]
    snow_available = candidate[snow_cols].notna().sum().sum() >= 20
    if not snow_available:
        return {
            "start": w1["start"],
            "end": w1["end"],
            "peak": peak_date,
            "peak_q": peak_q,
            "status": "snow_data_missing_fallback_to_w1",
            "fallback": True,
            "snow_used": False,
        }
    work = candidate.copy().sort_values("date").reset_index(drop=True)
    pdd_positive = pd.to_numeric(work["positive_degree_day_Cday"], errors="coerce").gt(0)
    pdd_run = pdd_positive.rolling(3, min_periods=3).sum().ge(3)
    snow_change = pd.to_numeric(work["snow_depletion_rate_7d"], errors="coerce").abs().gt(0.02)
    runoff = pd.to_numeric(work["surface_runoff_m"], errors="coerce").gt(0.00001)
    start_candidates = work[pdd_run & (snow_change | runoff)]
    fallback = start_candidates.empty
    start = pd.Timestamp(start_candidates.iloc[0]["date"]) if not start_candidates.empty else w1["start"]
    if pd.notna(start) and start > peak_date:
        start = w1["start"]
        fallback = True
    after_start = work[work["date"].ge(start)]
    snow_end = after_start[pd.to_numeric(after_start["snow_cover_fraction"], errors="coerce").le(0.20)]
    end = pd.Timestamp(snow_end.iloc[0]["date"]) if not snow_end.empty else w1["end"]
    if pd.isna(end) or end < peak_date:
        end = w1["end"]
        fallback = True
    return {
        "start": start,
        "end": end,
        "peak": peak_date,
        "peak_q": peak_q,
        "status": "ok" if not fallback else "partial_snow_rule_fallback",
        "fallback": fallback,
        "snow_used": True,
    }


def _w4_dates(w1: dict[str, Any], w2: dict[str, Any]) -> dict[str, Any]:
    if pd.isna(w1["start"]) or pd.isna(w1["end"]) or pd.isna(w2["start"]) or pd.isna(w2["end"]):
        return {"start": pd.NaT, "end": pd.NaT, "peak": w1.get("peak", pd.NaT), "peak_q": w1.get("peak_q", np.nan), "status": "missing_component_window", "fallback": False}
    start = max(pd.Timestamp(w1["start"]), pd.Timestamp(w2["start"]))
    end = min(pd.Timestamp(w1["end"]), pd.Timestamp(w2["end"]))
    status = "ok" if _inclusive_days(start, end) >= 14 else "insufficient_overlap"
    return {"start": start, "end": end, "peak": w1["peak"], "peak_q": w1["peak_q"], "status": status, "fallback": False}


def _definition_confidence(status: str, fallback: bool, coverage: float, length: int, peak_inside: bool, n_days_with_flux: int, window_id: str) -> str:
    plausible = 14 <= int(length) <= 120 and peak_inside and n_days_with_flux > 0
    if window_id == "fixed_may_july_reference":
        return "high" if coverage >= 0.95 and n_days_with_flux > 0 else "low"
    if status in {"missing_q", "missing_component_window", "insufficient_overlap"} or not plausible or coverage < 0.90:
        return "low"
    if fallback or status.startswith("snow_data_missing") or coverage < 0.95:
        return "medium"
    return "high"


def _definition_row(
    river: str,
    year: int,
    window_id: str,
    values: dict[str, Any],
    daily_flux: pd.DataFrame,
    status_override: str | None = None,
) -> dict[str, Any]:
    start = pd.Timestamp(values["start"]) if pd.notna(values.get("start")) else pd.NaT
    end = pd.Timestamp(values["end"]) if pd.notna(values.get("end")) else pd.NaT
    peak = pd.Timestamp(values["peak"]) if pd.notna(values.get("peak")) else pd.NaT
    expected = _inclusive_days(start, end)
    n_flux = _flux_count(daily_flux, river, start, end)
    coverage = float(n_flux / expected) if expected else np.nan
    peak_inside = bool(pd.notna(peak) and pd.notna(start) and pd.notna(end) and start <= peak <= end)
    fallback = bool(values.get("fallback", False))
    status = status_override or str(values.get("status", "ok"))
    confidence = _definition_confidence(status, fallback, coverage, expected, peak_inside, n_flux, window_id)
    caveats: list[str] = []
    if fallback:
        caveats.append("fallback_used")
    if not peak_inside and window_id != "fixed_may_july_reference":
        caveats.append("peak_not_inside_window")
    if expected and not (14 <= expected <= 120):
        caveats.append("implausible_window_length")
    if pd.notna(coverage) and coverage < 0.95:
        caveats.append("coverage_lt_0_95")
    if status != "ok":
        caveats.append(status)
    if window_id == "fixed_may_july_reference":
        caveats.append("reference_only")
    return {
        "river": river,
        "year": int(year),
        "window_id": window_id,
        "start_date": start.date().isoformat() if pd.notna(start) else "",
        "end_date": end.date().isoformat() if pd.notna(end) else "",
        "peak_q_date": peak.date().isoformat() if pd.notna(peak) else "",
        "peak_Q_m3s": values.get("peak_q", np.nan),
        "window_length_days": expected,
        "definition_status": status,
        "fallback_used": fallback,
        "snow_data_used": bool(values.get("snow_used", False)),
        "q_data_used": window_id != "fixed_may_july_reference" and pd.notna(values.get("peak")),
        "n_days_with_flux": n_flux,
        "n_days_expected": expected,
        "coverage_rate": coverage,
        "window_confidence_tier": confidence,
        "caveat_reason": ";".join(dict.fromkeys(caveats)) if caveats else "ok",
    }


def _define_for_group(river: str, year: int, group: pd.DataFrame, daily_flux: pd.DataFrame) -> list[dict[str, Any]]:
    candidate = _candidate_season(group)
    w1 = _w1_dates(candidate)
    w2 = _w2_dates(candidate)
    w3 = _w3_dates(candidate, w1)
    w4 = _w4_dates(w1, w2)
    fixed_start = pd.Timestamp(year=int(year), month=5, day=1)
    fixed_end = pd.Timestamp(year=int(year), month=7, day=31)
    may_july_peak, may_july_peak_q = _peak_info(group[group["date"].between(fixed_start, fixed_end, inclusive="both")])
    fixed = {"start": fixed_start, "end": fixed_end, "peak": may_july_peak, "peak_q": may_july_peak_q, "status": "reference_only", "fallback": False}
    return [
        _definition_row(river, year, "fixed_may_july_reference", fixed, daily_flux, status_override="reference_only"),
        _definition_row(river, year, "discharge_centered_freshet", w1, daily_flux),
        _definition_row(river, year, "q75_peak_contiguous", w2, daily_flux),
        _definition_row(river, year, "snow_depletion_assisted", w3, daily_flux),
        _definition_row(river, year, "common_overlap_w1_w2", w4, daily_flux),
    ]


def define_snowmelt_windows() -> dict[str, Any]:
    _ensure_dirs()
    before_hashes = _input_hashes()
    inputs = _load_definition_inputs()
    daily_flux = _prepare_daily_flux(inputs["daily_flux"])
    hydrology = _prepare_hydrology(inputs)
    annual = inputs["annual"].copy()
    annual["year"] = pd.to_numeric(annual["year"], errors="coerce").astype("Int64")
    rows: list[dict[str, Any]] = []
    keys = annual[["river", "year"]].dropna().drop_duplicates().sort_values(["river", "year"])
    for key in keys.itertuples(index=False):
        group = hydrology[hydrology["river"].astype(str).eq(str(key.river)) & hydrology["year"].eq(int(key.year))]
        if group.empty:
            for window_id in WINDOW_IDS:
                rows.append(
                    {
                        "river": key.river,
                        "year": int(key.year),
                        "window_id": window_id,
                        "start_date": "",
                        "end_date": "",
                        "peak_q_date": "",
                        "peak_Q_m3s": np.nan,
                        "window_length_days": 0,
                        "definition_status": "missing_hydrology",
                        "fallback_used": False,
                        "snow_data_used": False,
                        "q_data_used": False,
                        "n_days_with_flux": 0,
                        "n_days_expected": 0,
                        "coverage_rate": np.nan,
                        "window_confidence_tier": "low",
                        "caveat_reason": "missing_hydrology",
                    }
                )
            continue
        rows.extend(_define_for_group(str(key.river), int(key.year), group, daily_flux))
    definitions = pd.DataFrame(rows)
    definitions_path = _write_csv(definitions, SNOWMELT_TABLE_DIR / "snowmelt_window_definitions_by_river_year.csv")
    verify_snowmelt_inputs_unchanged(before_hashes)
    return {"tables": [definitions_path], "definitions": definitions}


__all__ = [
    "SNOWMELT_TABLE_DIR",
    "SNOWMELT_REPORT_DIR",
    "SNOWMELT_FIGURE_DIR",
    "WINDOW_IDS",
    "required_snowmelt_input_paths",
    "verify_snowmelt_inputs_unchanged",
    "define_snowmelt_windows",
]
