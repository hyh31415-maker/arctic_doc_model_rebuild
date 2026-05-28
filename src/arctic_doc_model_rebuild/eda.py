from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .gold_contract import load_contract, require_gold_data_dir, sha256_file, table_path, verification_problem_counts, verify_all_gold_tables
from .paths import REPORT_DIR, TABLE_DIR, path
from .plotting import generate_eda_figures
from .reports import _md_table, model_output_status
from .schema_checks import issue_count, run_all_schema_checks


EDA_TABLE_DIR = TABLE_DIR / "eda"
EDA_REPORT_DIR = REPORT_DIR / "eda"
EDA_FIGURE_DIR = path("outputs", "figures", "eda")

EDA_READ_TABLES = [
    "training_matrix_hydrocore.csv",
    "training_matrix_basin_context.csv",
    "training_matrix_optical_matched_0d.csv",
    "training_matrix_optical_matched_1d.csv",
    "training_matrix_optical_matched_3d.csv",
    "training_matrix_optical_matched_7d.csv",
    "training_matrix_optical_matched_3d_hls.csv",
    "training_matrix_optical_matched_3d_landsat.csv",
    "training_matrix_optical_matched_3d_sentinel2.csv",
    "prediction_grid_daily_hydrocore.csv",
    "basin_attributes_curated.csv",
    "basin_attributes_curated_wide.csv",
    "doc_labels_gold.csv",
    "daily_discharge_gold.csv",
    "daily_hydroclimate_gold.csv",
    "optical_timeseries_gold.csv",
]

HYDROCORE_PREDICTORS = [
    "Q_m3s",
    "temperature_2m_C",
    "positive_degree_day_Cday",
    "snow_cover_fraction",
    "snow_depletion_rate_7d",
    "surface_runoff_m",
    "sin_doy",
    "cos_doy",
]
HYDROCLIMATE_PREDICTORS = [
    "temperature_2m_C",
    "positive_degree_day_Cday",
    "snow_cover_fraction",
    "snow_depletion_rate_7d",
    "surface_runoff_m",
]
OPTICAL_TABLES = [
    "training_matrix_optical_matched_0d.csv",
    "training_matrix_optical_matched_1d.csv",
    "training_matrix_optical_matched_3d.csv",
    "training_matrix_optical_matched_7d.csv",
    "training_matrix_optical_matched_3d_hls.csv",
    "training_matrix_optical_matched_3d_landsat.csv",
    "training_matrix_optical_matched_3d_sentinel2.csv",
]
OPTICAL_PREDICTORS = ["blue", "green", "red", "nir", "swir1", "swir2", "ndwi", "mndwi", "red_green_ratio", "green_blue_ratio"]
ID_TOPOLOGY_FIELDS = {"HYBAS_ID", "HYBAS_ID_mean", "NEXT_DOWN", "NEXT_DOWN_mean", "PFAF_ID", "PFAF_ID_mean", "HYRIV_ID", "BAS_ID"}
MODEL_SCOPES = [
    "season_only_baseline",
    "q_season_baseline",
    "hydroclimate_complete_case",
    "hydroclimate_missingness_aware",
    "river_effects_model",
    "leave_one_year_out_cv",
    "leave_one_river_out_cv",
    "optical_3d_any_sensor",
    "optical_3d_hls_only",
    "optical_3d_landsat_only",
    "optical_3d_sentinel2_only",
    "basin_context_sensitivity",
    "daily_prediction_grid_ready",
]


@dataclass(frozen=True)
class EDAResult:
    report_path: Path
    table_paths: list[Path]
    figure_paths: list[Path]
    verification: pd.DataFrame
    schema_checks: pd.DataFrame


def _ensure_eda_dirs() -> None:
    for directory in [EDA_TABLE_DIR, EDA_REPORT_DIR, EDA_FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def _write_csv(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8")
    return destination


def _read_gold_table(table_name: str) -> pd.DataFrame:
    if table_name not in EDA_READ_TABLES:
        raise KeyError(f"EDA is not allowed to read this table: {table_name}")
    gold_dir = require_gold_data_dir()
    destination = table_path(table_name, gold_dir=gold_dir)
    if not destination.exists():
        raise FileNotFoundError(f"Missing expected gold table: {destination}")
    return pd.read_csv(destination, low_memory=False)


def _load_eda_tables() -> dict[str, pd.DataFrame]:
    return {table_name: _read_gold_table(table_name) for table_name in EDA_READ_TABLES}


def _with_dates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["year"] = out["date"].dt.year if "year" not in out.columns else pd.to_numeric(out["year"], errors="coerce")
        out["month"] = out["date"].dt.month
    return out


def _season_memberships(frame: pd.DataFrame) -> pd.DataFrame:
    dated = _with_dates(frame)
    rows: list[pd.DataFrame] = []
    windows = {
        "spring_freshet_provisional": [5, 6, 7],
        "early_season": [5, 6],
        "summer": [7, 8],
        "late_season": [9, 10],
    }
    for name, months in windows.items():
        subset = dated[dated["month"].isin(months)].copy()
        subset["season_window"] = name
        rows.append(subset)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _safe_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _summary_stats(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"n": 0, "min": np.nan, "p05": np.nan, "p25": np.nan, "median": np.nan, "mean": np.nan, "p75": np.nan, "p95": np.nan, "max": np.nan, "std": np.nan}
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "p05": float(values.quantile(0.05)),
        "p25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "p75": float(values.quantile(0.75)),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
    }


def _distribution_rows(frame: pd.DataFrame, variable: str, group_type: str, group_column: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if variable not in frame.columns:
        return rows
    if group_column is None:
        row = {"variable": variable, "group_type": group_type, "group_value": "overall"}
        row.update(_summary_stats(frame[variable]))
        rows.append(row)
        return rows
    if group_column not in frame.columns:
        return rows
    for value, subset in frame.groupby(group_column, dropna=False):
        row = {"variable": variable, "group_type": group_type, "group_value": value}
        row.update(_summary_stats(subset[variable]))
        rows.append(row)
    return rows


def _total_variation(subset: pd.Series, full: pd.Series) -> float:
    subset_dist = subset.astype(str).value_counts(normalize=True)
    full_dist = full.astype(str).value_counts(normalize=True)
    keys = sorted(set(subset_dist.index).union(full_dist.index))
    return float(sum(abs(subset_dist.get(key, 0.0) - full_dist.get(key, 0.0)) for key in keys) / 2.0)


def _spearman_p_value(r: float, n: int) -> float:
    if n < 4 or not np.isfinite(r) or abs(r) >= 1:
        return float("nan")
    z = math.atanh(float(r)) * math.sqrt(max(n - 3, 1))
    return float(math.erfc(abs(z) / math.sqrt(2)))


def _spearman_pair(frame: pd.DataFrame, x: str, y: str = "DOC_mgC_L") -> tuple[int, float, float]:
    subset = frame[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(subset) < 2 or subset[x].nunique() < 2 or subset[y].nunique() < 2:
        return len(subset), float("nan"), float("nan")
    rho = float(subset[x].rank().corr(subset[y].rank()))
    return len(subset), rho, _spearman_p_value(rho, len(subset))


def _interpret_spearman(n: int, rho: float, p_value: float) -> str:
    flags: list[str] = []
    if n < 10:
        flags.append("n_lt_10")
    if np.isfinite(rho):
        strength = abs(rho)
        if strength < 0.3:
            flags.append("weak")
        elif strength < 0.6:
            flags.append("moderate")
        else:
            flags.append("strong")
    if not np.isfinite(p_value) or p_value >= 0.1:
        flags.append("sign_uncertain")
    return ";".join(flags) if flags else "weak"


def _gold_matrix_inventory(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    contract = load_contract()
    rows = []
    for table_name, frame in tables.items():
        dated = _with_dates(frame)
        file_path = table_path(table_name, gold_dir=require_gold_data_dir())
        rows.append(
            {
                "table_name": table_name,
                "role": contract.get("expected_tables", {}).get(table_name, {}).get("role", "eda_support"),
                "rows": len(frame),
                "columns": len(frame.columns),
                "n_rivers": int(frame["river"].nunique()) if "river" in frame.columns else "",
                "n_years": int(dated["year"].dropna().nunique()) if "year" in dated.columns else "",
                "min_date": dated["date"].min().date().isoformat() if "date" in dated.columns and dated["date"].notna().any() else "",
                "max_date": dated["date"].max().date().isoformat() if "date" in dated.columns and dated["date"].notna().any() else "",
                "file_size_bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
    return pd.DataFrame(rows)


def _coverage_tables(hydrocore: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frame = _with_dates(hydrocore)
    counts_by_river = (
        frame.groupby("river")
        .agg(
            row_count=("label_id", "size"),
            min_date=("date", "min"),
            max_date=("date", "max"),
            n_years=("year", "nunique"),
            n_months=("month", "nunique"),
            may_july_samples=("month", lambda value: int(value.isin([5, 6, 7]).sum())),
            non_may_july_samples=("month", lambda value: int((~value.isin([5, 6, 7])).sum())),
        )
        .reset_index()
    )
    counts_by_river["min_date"] = counts_by_river["min_date"].dt.date.astype(str)
    counts_by_river["max_date"] = counts_by_river["max_date"].dt.date.astype(str)
    by_year = frame.groupby("year", dropna=False).size().reset_index(name="row_count")
    by_month = frame.groupby("month", dropna=False).size().reset_index(name="row_count")
    by_river_year = frame.groupby(["river", "year"], dropna=False).size().reset_index(name="row_count")
    by_river_month = frame.groupby(["river", "month"], dropna=False).size().reset_index(name="row_count")
    season = _season_memberships(frame)
    season_counts = season.groupby(["season_window", "river"], dropna=False).size().reset_index(name="row_count")
    overall = season.groupby("season_window", dropna=False).size().reset_index(name="row_count")
    overall["river"] = "ALL"
    season_counts = pd.concat([overall[["season_window", "river", "row_count"]], season_counts], ignore_index=True)
    return {
        "doc_label_counts_by_river": counts_by_river,
        "doc_label_counts_by_year": by_year,
        "doc_label_counts_by_month": by_month,
        "doc_label_counts_by_river_year": by_river_year,
        "doc_label_counts_by_river_month": by_river_month,
        "season_window_counts": season_counts,
    }


def _missingness_tables(hydrocore: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frame = _with_dates(hydrocore)
    rows = []
    for column in HYDROCORE_PREDICTORS:
        values = frame[column] if column in frame.columns else pd.Series([pd.NA] * len(frame))
        missing_count = int(values.isna().sum())
        missing_rate = missing_count / len(frame) if len(frame) else 0.0
        rows.append(
            {
                "column": column,
                "nonmissing_count": int(values.notna().sum()),
                "missing_count": missing_count,
                "missing_rate": missing_rate,
                "flag_missing_rate_gt_0_25": missing_rate > 0.25,
            }
        )
    by_column = pd.DataFrame(rows)

    def grouped_missing(group_columns: list[str], threshold: float) -> pd.DataFrame:
        out_rows = []
        for keys, subset in frame.groupby(group_columns, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            base = dict(zip(group_columns, keys))
            for column in HYDROCORE_PREDICTORS:
                values = subset[column] if column in subset.columns else pd.Series([pd.NA] * len(subset))
                missing_rate = float(values.isna().mean()) if len(subset) else 0.0
                row = base.copy()
                row.update(
                    {
                        "column": column,
                        "rows": len(subset),
                        "missing_count": int(values.isna().sum()),
                        "missing_rate": missing_rate,
                        "flag_missing_rate": missing_rate > threshold,
                    }
                )
                out_rows.append(row)
        return pd.DataFrame(out_rows)

    return {
        "hydrocore_missingness_by_column": by_column,
        "hydrocore_missingness_by_river": grouped_missing(["river"], 0.4),
        "hydrocore_missingness_by_year": grouped_missing(["year"], 0.5),
        "hydrocore_missingness_by_river_year": grouped_missing(["river", "year"], 0.5),
    }


def _distribution_tables(hydrocore: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frame = _safe_numeric(_with_dates(hydrocore), ["DOC_mgC_L", *HYDROCORE_PREDICTORS])
    frame["log_DOC"] = np.where(frame["DOC_mgC_L"] > 0, np.log(frame["DOC_mgC_L"]), np.nan)
    frame["log_Q"] = np.where(frame["Q_m3s"] > 0, np.log(frame["Q_m3s"]), np.nan)
    season = _season_memberships(frame)

    doc_rows = []
    for group_type, column in [("overall", None), ("river", "river"), ("month", "month")]:
        doc_rows.extend(_distribution_rows(frame, "DOC_mgC_L", group_type, column))
    doc_rows.extend(_distribution_rows(season, "DOC_mgC_L", "season_window", "season_window"))

    predictor_rows = []
    for variable in ["Q_m3s", *HYDROCLIMATE_PREDICTORS, "log_Q"]:
        predictor_rows.extend(_distribution_rows(frame, variable, "river", "river"))
    hydro_summary = pd.DataFrame(predictor_rows)
    q_distribution = hydro_summary[hydro_summary["variable"].isin(["Q_m3s", "log_Q"])].copy()

    season_summary = []
    for (river, season_window), subset in season.groupby(["river", "season_window"], dropna=False):
        row = {"river": river, "season_window": season_window}
        row.update(_summary_stats(subset["DOC_mgC_L"]))
        season_summary.append(row)

    return {
        "doc_distribution_by_river": pd.DataFrame(doc_rows),
        "q_distribution_by_river": q_distribution,
        "hydrocore_predictor_summary": hydro_summary,
        "doc_season_summary_by_river": pd.DataFrame(season_summary),
        "_hydrocore_with_transforms": frame,
    }


def _spearman_table(hydrocore_with_transforms: pd.DataFrame) -> pd.DataFrame:
    variables = ["Q_m3s", "log_Q", "temperature_2m_C", "snow_cover_fraction", "surface_runoff_m"]
    rows = []
    for river, subset in hydrocore_with_transforms.groupby("river", dropna=False):
        for variable in variables:
            if variable not in subset.columns:
                continue
            n, rho, p_value = _spearman_pair(subset, variable)
            rows.append(
                {
                    "river": river,
                    "variable": variable,
                    "n": n,
                    "spearman_r": rho,
                    "p_value": p_value,
                    "interpretation_flag": _interpret_spearman(n, rho, p_value),
                }
            )
    return pd.DataFrame(rows)


def _optical_tables(tables: dict[str, pd.DataFrame], hydrocore: pd.DataFrame) -> dict[str, pd.DataFrame]:
    hydro = _safe_numeric(_with_dates(hydrocore), ["DOC_mgC_L", "Q_m3s"])
    window_rows = []
    sensor_rows = []
    bias_rows = []
    predictor_rows = []
    full_doc_median = pd.to_numeric(hydro["DOC_mgC_L"], errors="coerce").median()
    full_q_median = pd.to_numeric(hydro["Q_m3s"], errors="coerce").median()
    for table_name in OPTICAL_TABLES:
        frame = _safe_numeric(_with_dates(tables[table_name]), ["DOC_mgC_L", "Q_m3s", "days_offset", *OPTICAL_PREDICTORS, "n_valid_water_pixels", "pct_valid_water_pixels"])
        window = table_name.replace("training_matrix_optical_matched_", "").replace(".csv", "")
        abs_offset = frame["days_offset"].abs() if "days_offset" in frame.columns else pd.Series(dtype=float)
        valid_pixels = pd.to_numeric(frame.get("n_valid_water_pixels", pd.Series(dtype=float)), errors="coerce")
        window_rows.append(
            {
                "table_name": table_name,
                "window": window,
                "row_count": len(frame),
                "unique_labels": int(frame["label_id"].nunique()) if "label_id" in frame.columns else "",
                "rivers_represented": int(frame["river"].nunique()) if "river" in frame.columns else "",
                "years_represented": int(frame["year"].nunique()) if "year" in frame.columns else "",
                "sensors_represented": int(frame["sensor"].nunique()) if "sensor" in frame.columns else "",
                "median_abs_days_offset": float(abs_offset.median()) if not abs_offset.empty else np.nan,
                "p90_abs_days_offset": float(abs_offset.quantile(0.9)) if not abs_offset.empty else np.nan,
                "median_valid_water_pixels": float(valid_pixels.median()) if not valid_pixels.empty else np.nan,
            }
        )
        if "sensor" in frame.columns:
            for sensor, subset in frame.groupby("sensor", dropna=False):
                sensor_rows.append(
                    {
                        "table_name": table_name,
                        "window": window,
                        "sensor": sensor,
                        "row_count": len(subset),
                        "unique_labels": int(subset["label_id"].nunique()) if "label_id" in subset.columns else "",
                        "median_abs_days_offset": float(subset["days_offset"].abs().median()) if "days_offset" in subset.columns else np.nan,
                    }
                )
                for variable in OPTICAL_PREDICTORS:
                    if variable in subset.columns:
                        row = {"table_name": table_name, "sensor": sensor, "variable": variable}
                        row.update(_summary_stats(subset[variable]))
                        predictor_rows.append(row)
        subset_doc_median = pd.to_numeric(frame["DOC_mgC_L"], errors="coerce").median() if "DOC_mgC_L" in frame.columns else np.nan
        subset_q_median = pd.to_numeric(frame["Q_m3s"], errors="coerce").median() if "Q_m3s" in frame.columns else np.nan
        bias_rows.append(
            {
                "table_name": table_name,
                "row_count": len(frame),
                "unique_labels": int(frame["label_id"].nunique()) if "label_id" in frame.columns else "",
                "doc_median_subset": subset_doc_median,
                "doc_median_full_hydrocore": full_doc_median,
                "doc_median_difference": subset_doc_median - full_doc_median if pd.notna(subset_doc_median) else np.nan,
                "q_median_subset": subset_q_median,
                "q_median_full_hydrocore": full_q_median,
                "q_median_difference": subset_q_median - full_q_median if pd.notna(subset_q_median) else np.nan,
                "month_distribution_difference": _total_variation(frame["month"], hydro["month"]) if "month" in frame.columns else np.nan,
                "river_composition_difference": _total_variation(frame["river"], hydro["river"]) if "river" in frame.columns else np.nan,
            }
        )
    return {
        "optical_match_counts_by_window": pd.DataFrame(window_rows),
        "optical_match_counts_by_sensor": pd.DataFrame(sensor_rows),
        "optical_match_bias_audit": pd.DataFrame(bias_rows),
        "optical_predictor_summary_by_sensor": pd.DataFrame(predictor_rows),
    }


def _basin_attribute_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    long = tables["basin_attributes_curated.csv"].copy()
    wide = tables["basin_attributes_curated_wide.csv"].copy()
    model_use = long.get("model_use", pd.Series(dtype=str)).astype(str).str.lower().isin({"true", "1", "yes"})
    mechanism_use = long.get("mechanism_use", pd.Series(dtype=str)).astype(str).str.lower().isin({"true", "1", "yes"})
    needs_refinement = long.get("needs_area_weighted_refinement", pd.Series(dtype=str)).astype(str).str.lower().isin({"true", "1", "yes"})
    summary_rows = [
        {"metric": "curated_attributes", "category": "all", "value": int(len(long))},
        {"metric": "model_use_true", "category": "all", "value": int(model_use.sum())},
        {"metric": "mechanism_use_true", "category": "all", "value": int(mechanism_use.sum())},
        {"metric": "needs_area_weighted_refinement", "category": "all", "value": int(needs_refinement.sum())},
        {"metric": "wide_candidate_columns", "category": "all", "value": int(max(len(wide.columns) - 1, 0))},
    ]
    if "attribute_category" in long.columns:
        for category, count in long.groupby("attribute_category").size().items():
            summary_rows.append({"metric": "attributes_by_category", "category": category, "value": int(count)})
    excluded = long[~model_use]
    summary_rows.append({"metric": "excluded_from_modeling", "category": "all", "value": int(len(excluded))})

    preferred_terms = ["run", "dis", "lake", "water", "area", "slope", "elevation", "snow", "wet", "temperature", "precip"]
    candidates = long[model_use].copy()
    if "source_field" in candidates.columns:
        candidates = candidates[~candidates["source_field"].astype(str).isin(ID_TOPOLOGY_FIELDS)]
    candidates = candidates[~candidates.get("attribute_name", pd.Series(dtype=str)).astype(str).isin(ID_TOPOLOGY_FIELDS)]
    candidates["preference_score"] = candidates.apply(
        lambda row: sum(term in " ".join([str(row.get("attribute_name", "")), str(row.get("source_field", "")), str(row.get("attribute_category", ""))]).lower() for term in preferred_terms),
        axis=1,
    )
    candidate_rows = []
    for _, row in candidates.sort_values(["preference_score", "attribute_name"], ascending=[False, True]).head(10).iterrows():
        candidate_rows.append(
            {
                "attribute_name": row.get("attribute_name", ""),
                "source_field": row.get("source_field", ""),
                "attribute_category": row.get("attribute_category", ""),
                "model_use": row.get("model_use", ""),
                "mechanism_use": row.get("mechanism_use", ""),
                "needs_area_weighted_refinement": row.get("needs_area_weighted_refinement", ""),
                "candidate_reason": "Interpretable basin hydrology/physiography candidate; only 6 river-level units exist.",
                "warning": "Candidate list only; no model fitting performed.",
            }
        )
    return {
        "basin_attribute_summary": pd.DataFrame(summary_rows),
        "basin_attribute_model_candidate_list": pd.DataFrame(candidate_rows),
    }


def _prediction_grid_tables(prediction_grid: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frame = _with_dates(_safe_numeric(prediction_grid, ["Q_m3s", *HYDROCLIMATE_PREDICTORS]))
    hydro_cols = [column for column in HYDROCLIMATE_PREDICTORS if column in frame.columns]
    rows = []
    for river, subset in frame.groupby("river", dropna=False):
        dates = pd.to_datetime(subset["date"], errors="coerce").dropna()
        span_days = int((dates.max() - dates.min()).days + 1) if not dates.empty else 0
        distinct_days = int(dates.dt.date.nunique()) if not dates.empty else 0
        rows.append(
            {
                "river": river,
                "row_count": len(subset),
                "min_date": dates.min().date().isoformat() if not dates.empty else "",
                "max_date": dates.max().date().isoformat() if not dates.empty else "",
                "date_span_days": span_days,
                "distinct_days": distinct_days,
                "missing_dates": max(span_days - distinct_days, 0),
                "date_continuity_ok": span_days == distinct_days,
                "missing_q_rate": float(subset["Q_m3s"].isna().mean()) if "Q_m3s" in subset.columns else 1.0,
                "missing_hydroclimate_rate": float(subset[hydro_cols].isna().any(axis=1).mean()) if hydro_cols else 1.0,
                "available_may_july_days": int(subset["month"].isin([5, 6, 7]).sum()),
            }
        )
    year_rows = []
    for (river, year), subset in frame.groupby(["river", "year"], dropna=False):
        year_rows.append(
            {
                "river": river,
                "year": int(year) if pd.notna(year) else "",
                "row_count": len(subset),
                "available_may_july_days": int(subset["month"].isin([5, 6, 7]).sum()),
                "missing_q_rate": float(subset["Q_m3s"].isna().mean()) if "Q_m3s" in subset.columns else 1.0,
                "missing_hydroclimate_rate": float(subset[hydro_cols].isna().any(axis=1).mean()) if hydro_cols else 1.0,
            }
        )
    return {
        "prediction_grid_coverage_by_river": pd.DataFrame(rows),
        "prediction_grid_coverage_by_year": pd.DataFrame(year_rows),
    }


def _model_scope_feasibility(tables: dict[str, pd.DataFrame], hydrocore_with_transforms: pd.DataFrame) -> pd.DataFrame:
    hydro = hydrocore_with_transforms
    n_rivers = int(hydro["river"].nunique())
    n_years = int(hydro["year"].nunique())
    complete_hydro = int(hydro[["DOC_mgC_L", "Q_m3s", "sin_doy", "cos_doy", *HYDROCLIMATE_PREDICTORS]].dropna().shape[0])
    q_complete = int(hydro[["DOC_mgC_L", "Q_m3s", "sin_doy", "cos_doy"]].dropna().shape[0])
    optical_3d = tables["training_matrix_optical_matched_3d.csv"]
    hls = tables["training_matrix_optical_matched_3d_hls.csv"]
    landsat = tables["training_matrix_optical_matched_3d_landsat.csv"]
    sentinel2 = tables["training_matrix_optical_matched_3d_sentinel2.csv"]
    pred_grid = tables["prediction_grid_daily_hydrocore.csv"]
    basin = tables["training_matrix_basin_context.csv"]

    def row(scope: str, required_data: str, available_rows: int, threshold: bool, recommended: bool, caveat: str) -> dict[str, Any]:
        return {
            "scope": scope,
            "required_data": required_data,
            "available_rows": int(available_rows),
            "n_rivers": n_rivers,
            "n_years": n_years,
            "minimum_threshold_met": bool(threshold),
            "recommended_for_phase_3": bool(recommended),
            "caveat": caveat,
        }

    rows = [
        row("season_only_baseline", "DOC + season terms", len(hydro), len(hydro) >= 500, True, "Descriptive seasonal baseline only after EDA."),
        row("q_season_baseline", "DOC + Q + season terms", q_complete, q_complete >= 500, True, "No model fitted in EDA phase."),
        row("hydroclimate_complete_case", "DOC + Q + hydroclimate complete cases", complete_hydro, complete_hydro >= 400, True, "Complete-case analysis may change sample composition."),
        row("hydroclimate_missingness_aware", "DOC + Q + hydroclimate with missingness strategy", len(hydro), len(hydro) >= 500, True, "Requires explicit missingness policy before fitting."),
        row("river_effects_model", "DOC + predictors + river grouping", len(hydro), n_rivers >= 5, True, "Only six rivers; partial pooling may be useful later."),
        row("leave_one_year_out_cv", "multiple years per river", len(hydro), n_years >= 5, True, "Temporal folds may be uneven by river."),
        row("leave_one_river_out_cv", "six river groups", len(hydro), n_rivers >= 5, False, "High-risk extrapolation with only six river units."),
        row("optical_3d_any_sensor", "3-day optical matched subset", len(optical_3d), len(optical_3d) >= 100, True, "Sensitivity scope, not primary baseline."),
        row("optical_3d_hls_only", "3-day HLS optical subset", len(hls), len(hls) >= 100, True, "Sensor-specific sensitivity only."),
        row("optical_3d_landsat_only", "3-day Landsat optical subset", len(landsat), len(landsat) >= 100, True, "Sensor-specific sensitivity only."),
        row("optical_3d_sentinel2_only", "3-day Sentinel-2 optical subset", len(sentinel2), len(sentinel2) >= 100, False, "Small subset; likely underpowered."),
        row("basin_context_sensitivity", "basin context matrix", len(basin), len(basin) >= 500 and n_rivers == 6, False, "Only six river-level basin units; use sensitivity framing."),
        row("daily_prediction_grid_ready", "daily x-only prediction grid", len(pred_grid), len(pred_grid) >= 50000, True, "Grid is x-only; no DOC predictions generated now."),
    ]
    return pd.DataFrame(rows)


def _eda_checks(verification: pd.DataFrame, schema: pd.DataFrame, figure_paths: list[Path]) -> pd.DataFrame:
    output_status = model_output_status()
    rows = [
        {"check_name": "gold_contract_ok", "passed": bool(verification["status"].eq("ok").all()), "message": "Gold contract verified before EDA."},
        {"check_name": "schema_checks_ok", "passed": issue_count(schema) == 0, "message": "Schema and leakage checks passed before EDA."},
        {"check_name": "no_model_output_dirs", "passed": len(output_status["forbidden_dirs_present"]) == 0, "message": str(output_status["forbidden_dirs_present"])},
        {"check_name": "no_model_binaries", "passed": output_status["model_binary_count"] == 0, "message": f"model_binary_count={output_status['model_binary_count']}"},
        {"check_name": "figures_generated_or_optional", "passed": True, "message": f"figures_generated={len(figure_paths)}"},
    ]
    return pd.DataFrame(rows)


def _write_report(
    *,
    verification: pd.DataFrame,
    schema: pd.DataFrame,
    table_frames: dict[str, pd.DataFrame],
    table_paths: list[Path],
    figure_paths: list[Path],
) -> Path:
    counts = verification_problem_counts(verification)
    scope = table_frames["model_scope_feasibility"]
    inventory = table_frames["gold_matrix_inventory"]
    river_counts = table_frames["doc_label_counts_by_river"]
    predictor_missing = table_frames["hydrocore_missingness_by_column"]
    doc_distribution = table_frames["doc_distribution_by_river"]
    q_distribution = table_frames["q_distribution_by_river"]
    correlations = table_frames["doc_q_spearman_by_river"]
    optical_bias = table_frames["optical_match_bias_audit"]
    basin_summary = table_frames["basin_attribute_summary"]
    prediction_grid = table_frames["prediction_grid_coverage_by_river"]
    eda_checks = table_frames["eda_check_results"]

    sections = [
        "# EDA Report",
        "",
        "## 1. Data contract status",
        "",
        f"- freeze_id: `data_freeze_gold_20260526_v1`",
        f"- contract tables ok: `{int(verification['status'].eq('ok').sum())}/{len(verification)}`",
        f"- hash mismatches: `{counts['sha256_mismatch']}`",
        f"- row count mismatches: `{counts['row_count_mismatch']}`",
        f"- schema/leakage issues: `{issue_count(schema)}`",
        "",
        _md_table(verification[["table_name", "status", "row_count_ok", "sha256_ok"]], max_rows=25),
        "",
        "## 2. Gold matrix inventory",
        "",
        _md_table(inventory[["table_name", "rows", "columns", "n_rivers", "n_years", "min_date", "max_date"]], max_rows=30),
        "",
        "## 3. DOC label coverage",
        "",
        _md_table(river_counts, max_rows=20),
        "",
        "## 4. River/year/month/season coverage",
        "",
        "Season windows are provisional and descriptive only. The spring freshet window is May-July; early season is May-June; summer is July-August; late season is September-October.",
        "",
        _md_table(table_frames["season_window_counts"], max_rows=40),
        "",
        "## 5. Hydrocore predictor completeness",
        "",
        _md_table(predictor_missing, max_rows=20),
        "",
        "## 6. DOC distribution and outliers",
        "",
        _md_table(doc_distribution, max_rows=30),
        "",
        "## 7. Q/discharge distribution",
        "",
        _md_table(q_distribution, max_rows=30),
        "",
        "## 8. Hydroclimate predictor distribution",
        "",
        _md_table(table_frames["hydrocore_predictor_summary"], max_rows=35),
        "",
        "## 9. DOC-Q-season descriptive relationships",
        "",
        "The correlations below are descriptive Spearman summaries only. They are not model results.",
        "",
        _md_table(correlations, max_rows=35),
        "",
        "## 10. Optical matched subset audit",
        "",
        _md_table(table_frames["optical_match_counts_by_window"], max_rows=20),
        "",
        _md_table(optical_bias, max_rows=20),
        "",
        "## 11. Sensor-specific optical subset audit",
        "",
        _md_table(table_frames["optical_match_counts_by_sensor"], max_rows=25),
        "",
        "## 12. Basin attribute audit",
        "",
        _md_table(basin_summary, max_rows=30),
        "",
        _md_table(table_frames["basin_attribute_model_candidate_list"], max_rows=10),
        "",
        "Only six river-level basin units exist, so basin attributes should be treated as sensitivity or mechanism context unless a later modeling design explicitly handles that limitation.",
        "",
        "## 13. Prediction grid coverage",
        "",
        "The prediction grid is x-only. This EDA does not generate DOC predictions.",
        "",
        _md_table(prediction_grid, max_rows=20),
        "",
        "## 14. Candidate modeling scopes",
        "",
        _md_table(scope, max_rows=20),
        "",
        "## 15. Recommended baseline modeling sequence",
        "",
        "1. Season-only baseline for calibration sanity checks.",
        "2. Q + season baseline.",
        "3. Hydroclimate missingness-aware baseline after documenting imputation or complete-case policy.",
        "4. River-aware model structure if phase 3 explicitly addresses the six-river grouping limit.",
        "5. Optical and basin-context sensitivity analyses after the primary hydrocore baseline is stable.",
        "",
        "## 16. Risks and caveats",
        "",
        "- Season windows are provisional and should not be interpreted as final hydrologic freshet definitions.",
        "- Optical matched subsets may have month, river, DOC, or Q composition differences from the full hydrocore set.",
        "- Sentinel-2 3-day matched sample size is small compared with any-sensor and Landsat subsets.",
        "- Basin context has only six river-level units, limiting standalone basin-attribute inference.",
        "- Correlations are descriptive and can be confounded by season and river structure.",
        "",
        "## 17. Explicit statement: no model trained, no prediction, no flux",
        "",
        "No model was trained. No DOC prediction was generated. No flux was generated. Only frozen gold data were read.",
        "",
        "## EDA-specific checks",
        "",
        _md_table(eda_checks, max_rows=10),
        "",
        "## Generated artifacts",
        "",
        f"- EDA tables generated: `{len(table_paths)}`",
        f"- EDA figures generated: `{len(figure_paths)}`",
    ]
    destination = EDA_REPORT_DIR / "eda_report.md"
    destination.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return destination


def run_eda() -> EDAResult:
    _ensure_eda_dirs()
    verification = verify_all_gold_tables()
    schema = run_all_schema_checks()
    counts = verification_problem_counts(verification)
    if any(counts.values()) or issue_count(schema) > 0:
        raise RuntimeError("Gold data contract or schema checks failed; refusing to run EDA.")

    tables = _load_eda_tables()
    hydrocore = tables["training_matrix_hydrocore.csv"]

    table_frames: dict[str, pd.DataFrame] = {}
    table_frames["gold_matrix_inventory"] = _gold_matrix_inventory(tables)
    table_frames.update(_coverage_tables(hydrocore))
    table_frames.update(_missingness_tables(hydrocore))
    distribution = _distribution_tables(hydrocore)
    hydrocore_with_transforms = distribution.pop("_hydrocore_with_transforms")
    table_frames.update(distribution)
    table_frames["doc_q_spearman_by_river"] = _spearman_table(hydrocore_with_transforms)
    table_frames.update(_optical_tables(tables, hydrocore))
    table_frames.update(_basin_attribute_tables(tables))
    table_frames.update(_prediction_grid_tables(tables["prediction_grid_daily_hydrocore.csv"]))
    table_frames["model_scope_feasibility"] = _model_scope_feasibility(tables, hydrocore_with_transforms)

    figure_paths = generate_eda_figures(
        hydrocore=hydrocore_with_transforms,
        missing_by_river=table_frames["hydrocore_missingness_by_river"],
        optical_window_counts=table_frames["optical_match_counts_by_window"],
        optical_sensor_counts=table_frames["optical_match_counts_by_sensor"],
        prediction_grid_by_river=table_frames["prediction_grid_coverage_by_river"],
        basin_attribute_summary=table_frames["basin_attribute_summary"],
        figure_dir=EDA_FIGURE_DIR,
    )
    table_frames["eda_check_results"] = _eda_checks(verification, schema, figure_paths)

    table_paths: list[Path] = []
    for name, frame in table_frames.items():
        if name.startswith("_"):
            continue
        table_paths.append(_write_csv(frame, EDA_TABLE_DIR / f"{name}.csv"))

    report_path = _write_report(
        verification=verification,
        schema=schema,
        table_frames=table_frames,
        table_paths=table_paths,
        figure_paths=figure_paths,
    )
    return EDAResult(report_path=report_path, table_paths=table_paths, figure_paths=figure_paths, verification=verification, schema_checks=schema)
