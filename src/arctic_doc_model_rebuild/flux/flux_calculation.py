from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..gold_contract import sha256_file
from ..paths import CONFIG_DIR, REPORT_DIR, TABLE_DIR, path
from .flux_qc import annual_confidence_tier, as_bool, daily_confidence_tier, analysis_readiness_status
from .flux_reports import DOC_FLUX_REPORT_PATH, write_doc_flux_report
from .flux_uncertainty import attach_flux_intervals
from .flux_units import doc_q_to_flux_kg_day, kg_day_to_mg_day, kg_day_to_tg_day


FREEZE_ID = "data_freeze_gold_20260526_v1"
DOC_FLUX_TABLE_DIR = TABLE_DIR / "doc_flux"
DOC_FLUX_REPORT_DIR = REPORT_DIR / "doc_flux"
DOC_FLUX_FIGURE_DIR = path("outputs", "figures", "doc_flux")
DAILY_DOC_TABLE_DIR = TABLE_DIR / "daily_doc_prediction"
DAILY_DOC_REPORT_DIR = REPORT_DIR / "daily_doc_prediction"

DAILY_DOC_PREDICTION_PATH = DAILY_DOC_TABLE_DIR / "daily_doc_prediction.csv"
DAILY_DOC_QC_PATH = DAILY_DOC_TABLE_DIR / "daily_doc_prediction_qc_summary.csv"
DAILY_DOC_RANGE_FLAGS_PATH = DAILY_DOC_TABLE_DIR / "daily_doc_prediction_range_flags.csv"
DAILY_DOC_INTERVAL_SUMMARY_PATH = DAILY_DOC_TABLE_DIR / "daily_doc_prediction_interval_summary.csv"
FLUX_READINESS_PATH = DAILY_DOC_TABLE_DIR / "flux_readiness_decision.csv"
DAILY_DOC_REPORT_PATH = DAILY_DOC_REPORT_DIR / "daily_doc_prediction_report.md"
PRODUCTION_SPEC_PATH = CONFIG_DIR / "model_specs" / "production_candidate_r4_river_specific_q_and_season_linear.yaml"
BIAS_REPORT_PATH = REPORT_DIR / "bias_refinement" / "bias_refinement_report.md"

DAILY_DOC_FLUX_PATH = DOC_FLUX_TABLE_DIR / "daily_doc_flux.csv"
ANNUAL_DOC_FLUX_PATH = DOC_FLUX_TABLE_DIR / "annual_doc_flux_summary.csv"
MAY_JULY_DOC_FLUX_PATH = DOC_FLUX_TABLE_DIR / "provisional_may_july_flux_summary.csv"
PERIOD_DOC_FLUX_PATH = DOC_FLUX_TABLE_DIR / "river_period_flux_summary.csv"


def _ensure_dirs() -> None:
    for directory in [DOC_FLUX_TABLE_DIR, DOC_FLUX_REPORT_DIR, DOC_FLUX_FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def _write_csv(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8")
    return destination


def _read_required_csv(destination: Path) -> pd.DataFrame:
    if not destination.exists():
        raise FileNotFoundError(f"Required DOC flux input is missing: {destination}")
    return pd.read_csv(destination, low_memory=False)


def _read_required_text(destination: Path) -> str:
    if not destination.exists():
        raise FileNotFoundError(f"Required DOC flux metadata is missing: {destination}")
    return destination.read_text(encoding="utf-8")


def _verify_contract_snapshot() -> None:
    verification_path = TABLE_DIR / "gold_table_verification.csv"
    if not verification_path.exists():
        raise FileNotFoundError("Run `python -m arctic_doc_model_rebuild.cli verify-gold-data` before DOC flux calculation.")
    verification = pd.read_csv(verification_path)
    required = {"training_matrix_hydrocore.csv", "prediction_grid_daily_hydrocore.csv"}
    status = verification[verification["table_name"].isin(required)]
    if set(status["table_name"]) != required or not status["status"].eq("ok").all():
        raise RuntimeError("Required gold tables were not verified before DOC flux calculation.")


def _load_inputs() -> dict[str, Any]:
    for destination in [
        DAILY_DOC_PREDICTION_PATH,
        DAILY_DOC_QC_PATH,
        DAILY_DOC_RANGE_FLAGS_PATH,
        DAILY_DOC_INTERVAL_SUMMARY_PATH,
        FLUX_READINESS_PATH,
        DAILY_DOC_REPORT_PATH,
        PRODUCTION_SPEC_PATH,
        BIAS_REPORT_PATH,
    ]:
        if not destination.exists():
            raise FileNotFoundError(f"Required DOC flux input is missing: {destination}")
    readiness = _read_required_csv(FLUX_READINESS_PATH)
    ready = readiness[readiness["decision_item"].eq("ready_for_flux_calculation")]
    if ready.empty or ready.iloc[0]["status"] not in {"true", "true_with_caveats"}:
        raise RuntimeError("Daily DOC prediction phase did not authorize flux calculation.")
    spec = yaml.safe_load(PRODUCTION_SPEC_PATH.read_text(encoding="utf-8"))
    if spec.get("flux_allowed") is not False:
        raise RuntimeError("Production candidate spec must explicitly mark flux_allowed=false; flux is a separate derived product.")
    return {
        "daily_prediction": _read_required_csv(DAILY_DOC_PREDICTION_PATH),
        "daily_qc": _read_required_csv(DAILY_DOC_QC_PATH),
        "daily_range_flags": _read_required_csv(DAILY_DOC_RANGE_FLAGS_PATH),
        "interval_summary": _read_required_csv(DAILY_DOC_INTERVAL_SUMMARY_PATH),
        "flux_readiness": readiness,
        "daily_report_text": _read_required_text(DAILY_DOC_REPORT_PATH),
        "bias_report_text": _read_required_text(BIAS_REPORT_PATH),
        "spec": spec,
    }


def _prepare_prediction_input(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for column in ["year", "month", "doy", "Q_m3s", "DOC_predicted_mgC_L", "DOC_predicted_raw_mgC_L"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in [
        "outside_training_logQ_range",
        "outside_training_doy_range",
        "outside_training_year_range",
        "point_prediction_clipped_at_zero",
        "lower_interval_clipped_at_zero",
    ]:
        out[column] = as_bool(out[column]) if column in out.columns else False
    out["interval_lower_clipped_at_zero"] = out["lower_interval_clipped_at_zero"]
    interval_cols = [
        "DOC_prediction_interval_80_lower",
        "DOC_prediction_interval_80_upper",
        "DOC_prediction_interval_90_lower",
        "DOC_prediction_interval_90_upper",
        "DOC_prediction_interval_95_lower",
        "DOC_prediction_interval_95_upper",
    ]
    for column in interval_cols:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _valid_prediction_mask(frame: pd.DataFrame) -> pd.Series:
    interval_cols = [
        "DOC_prediction_interval_80_lower",
        "DOC_prediction_interval_80_upper",
        "DOC_prediction_interval_90_lower",
        "DOC_prediction_interval_90_upper",
        "DOC_prediction_interval_95_lower",
        "DOC_prediction_interval_95_upper",
    ]
    return (
        frame["prediction_status"].astype(str).eq("predicted")
        & frame["Q_m3s"].notna()
        & (frame["Q_m3s"] > 0)
        & frame["DOC_predicted_mgC_L"].notna()
        & frame[interval_cols].notna().all(axis=1)
    )


def _missing_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if str(row.get("prediction_status")) != "predicted":
        reasons.append(str(row.get("missing_predictor_reason") or "not_predicted"))
    if pd.isna(row.get("Q_m3s")) or float(row.get("Q_m3s", np.nan)) <= 0:
        reasons.append("missing_or_nonpositive_Q_m3s")
    if pd.isna(row.get("DOC_predicted_mgC_L")):
        reasons.append("missing_DOC_prediction")
    for level in ["80", "90", "95"]:
        if pd.isna(row.get(f"DOC_prediction_interval_{level}_lower")) or pd.isna(row.get(f"DOC_prediction_interval_{level}_upper")):
            reasons.append(f"missing_DOC_interval_{level}")
    return ";".join(sorted({reason for reason in reasons if reason and reason != "nan"}))


def _daily_flux(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = _prepare_prediction_input(predictions)
    valid = _valid_prediction_mask(prepared)
    missing = prepared[~valid].copy()
    missing["flux_status"] = "not_calculated"
    missing["missing_flux_reason"] = missing.apply(_missing_reason, axis=1)

    flux = prepared[valid].copy()
    flux["flux_id"] = "flux_" + flux["prediction_id"].astype(str)
    flux["daily_flux_kgC_day"] = doc_q_to_flux_kg_day(flux["DOC_predicted_mgC_L"], flux["Q_m3s"])
    flux["daily_flux_MgC_day"] = kg_day_to_mg_day(flux["daily_flux_kgC_day"])
    flux["daily_flux_TgC_day"] = kg_day_to_tg_day(flux["daily_flux_kgC_day"])
    flux = attach_flux_intervals(flux)
    flux["flux_status"] = "calculated"
    flux["daily_confidence_tier"] = flux.apply(daily_confidence_tier, axis=1)
    flux["is_doc_prediction"] = True
    flux["is_flux"] = True
    flux["notes"] = "guarded_doc_flux_from_daily_doc_prediction;DOC_uncertainty_only;no_discharge_uncertainty"
    keep = [
        "flux_id",
        "river",
        "date",
        "year",
        "month",
        "doy",
        "Q_m3s",
        "DOC_predicted_mgC_L",
        "DOC_predicted_raw_mgC_L",
        "DOC_prediction_interval_80_lower",
        "DOC_prediction_interval_80_upper",
        "DOC_prediction_interval_90_lower",
        "DOC_prediction_interval_90_upper",
        "DOC_prediction_interval_95_lower",
        "DOC_prediction_interval_95_upper",
        "daily_flux_kgC_day",
        "daily_flux_MgC_day",
        "daily_flux_TgC_day",
        "daily_flux_80_lower_kgC_day",
        "daily_flux_80_upper_kgC_day",
        "daily_flux_90_lower_kgC_day",
        "daily_flux_90_upper_kgC_day",
        "daily_flux_95_lower_kgC_day",
        "daily_flux_95_upper_kgC_day",
        "daily_flux_80_lower_TgC_day",
        "daily_flux_80_upper_TgC_day",
        "daily_flux_90_lower_TgC_day",
        "daily_flux_90_upper_TgC_day",
        "daily_flux_95_lower_TgC_day",
        "daily_flux_95_upper_TgC_day",
        "prediction_status",
        "flux_status",
        "daily_confidence_tier",
        "outside_training_logQ_range",
        "outside_training_doy_range",
        "outside_training_year_range",
        "point_prediction_clipped_at_zero",
        "interval_lower_clipped_at_zero",
        "model_spec_id",
        "model_fit_id",
        "freeze_id",
        "is_doc_prediction",
        "is_flux",
        "notes",
        "interval_source_scope",
        "interval_source_n",
    ]
    flux["year"] = flux["year"].astype(int)
    flux["month"] = flux["month"].astype(int)
    flux["doy"] = flux["doy"].astype(int)
    flux["date"] = flux["date"].dt.date.astype(str)
    return flux[keep].reset_index(drop=True), missing.reset_index(drop=True)


def _expected_days(predictions: pd.DataFrame, months: set[int] | None = None) -> pd.DataFrame:
    prepared = _prepare_prediction_input(predictions)
    if months is not None:
        prepared = prepared[prepared["month"].isin(months)].copy()
    grouped = (
        prepared.groupby(["river", "year"], dropna=False)
        .agg(
            n_days_expected=("prediction_id", "count"),
            date_min=("date", "min"),
            date_max=("date", "max"),
        )
        .reset_index()
    )
    grouped["year"] = grouped["year"].astype(int)
    grouped["date_min"] = grouped["date_min"].dt.date.astype(str)
    grouped["date_max"] = grouped["date_max"].dt.date.astype(str)
    return grouped


def _summarize_flux_by_year(daily_flux: pd.DataFrame, predictions: pd.DataFrame, months: set[int] | None = None) -> pd.DataFrame:
    flux = daily_flux.copy()
    if months is not None:
        flux = flux[flux["month"].isin(months)].copy()
    expected = _expected_days(predictions, months)
    if flux.empty:
        return expected.assign(n_days_with_flux=0, coverage_rate=0.0)
    tier_counts = flux.pivot_table(index=["river", "year"], columns="daily_confidence_tier", values="flux_id", aggfunc="count", fill_value=0)
    for tier in ["high", "medium", "low"]:
        if tier not in tier_counts.columns:
            tier_counts[tier] = 0
    tier_counts = tier_counts.reset_index().rename(
        columns={"high": "n_high_confidence_days", "medium": "n_medium_confidence_days", "low": "n_low_confidence_days"}
    )
    by_tier_flux = flux.pivot_table(index=["river", "year"], columns="daily_confidence_tier", values="daily_flux_TgC_day", aggfunc="sum", fill_value=0)
    for tier in ["high", "medium", "low"]:
        if tier not in by_tier_flux.columns:
            by_tier_flux[tier] = 0.0
    by_tier_flux = by_tier_flux.reset_index().rename(
        columns={
            "high": "flux_from_high_confidence_days_TgC",
            "medium": "flux_from_medium_confidence_days_TgC",
            "low": "flux_from_low_confidence_days_TgC",
        }
    )
    grouped = (
        flux.groupby(["river", "year"], dropna=False)
        .agg(
            n_days_with_flux=("flux_id", "count"),
            annual_flux_kgC=("daily_flux_kgC_day", "sum"),
            annual_flux_MgC=("daily_flux_MgC_day", "sum"),
            annual_flux_TgC=("daily_flux_TgC_day", "sum"),
            annual_flux_80_lower_TgC=("daily_flux_80_lower_TgC_day", "sum"),
            annual_flux_80_upper_TgC=("daily_flux_80_upper_TgC_day", "sum"),
            annual_flux_90_lower_TgC=("daily_flux_90_lower_TgC_day", "sum"),
            annual_flux_90_upper_TgC=("daily_flux_90_upper_TgC_day", "sum"),
            annual_flux_95_lower_TgC=("daily_flux_95_lower_TgC_day", "sum"),
            annual_flux_95_upper_TgC=("daily_flux_95_upper_TgC_day", "sum"),
            n_outside_training_logQ_days=("outside_training_logQ_range", "sum"),
            n_outside_training_doy_days=("outside_training_doy_range", "sum"),
            n_outside_training_year_days=("outside_training_year_range", "sum"),
            n_point_prediction_clipped_days=("point_prediction_clipped_at_zero", "sum"),
            n_interval_lower_clipped_days=("interval_lower_clipped_at_zero", "sum"),
        )
        .reset_index()
    )
    out = expected.merge(grouped, on=["river", "year"], how="left")
    out = out.merge(tier_counts, on=["river", "year"], how="left").merge(by_tier_flux, on=["river", "year"], how="left")
    numeric_fill_zero = [
        column
        for column in out.columns
        if column.startswith(("annual_flux", "n_", "flux_from_")) and column not in {"n_days_expected"}
    ]
    out[numeric_fill_zero] = out[numeric_fill_zero].fillna(0)
    out["coverage_rate"] = np.where(out["n_days_expected"] > 0, out["n_days_with_flux"] / out["n_days_expected"], np.nan)
    out["fraction_flux_from_low_confidence_days"] = np.where(
        out["annual_flux_TgC"] > 0, out["flux_from_low_confidence_days_TgC"] / out["annual_flux_TgC"], np.nan
    )
    out["annual_confidence_tier"] = out.apply(
        lambda row: annual_confidence_tier(row["coverage_rate"], row["fraction_flux_from_low_confidence_days"]), axis=1
    )
    out["notes"] = "DOC_flux_from_guarded_daily_concentration_predictions;DOC_uncertainty_only"
    return out


def _provisional_may_july(annual: pd.DataFrame, may_july: pd.DataFrame) -> pd.DataFrame:
    out = may_july.copy()
    annual_lookup = annual[["river", "year", "annual_flux_TgC", "coverage_rate"]].rename(
        columns={"annual_flux_TgC": "annual_total_flux_TgC", "coverage_rate": "annual_coverage_rate"}
    )
    out = out.merge(annual_lookup, on=["river", "year"], how="left")
    out["may_july_flux_fraction_of_annual"] = np.where(
        (out["annual_coverage_rate"] >= 0.95) & (out["annual_total_flux_TgC"] > 0),
        out["annual_flux_TgC"] / out["annual_total_flux_TgC"],
        np.nan,
    )
    out = out.rename(
        columns={
            "annual_flux_kgC": "may_july_flux_kgC",
            "annual_flux_MgC": "may_july_flux_MgC",
            "annual_flux_TgC": "may_july_flux_TgC",
            "annual_flux_80_lower_TgC": "may_july_flux_80_lower_TgC",
            "annual_flux_80_upper_TgC": "may_july_flux_80_upper_TgC",
            "annual_flux_90_lower_TgC": "may_july_flux_90_lower_TgC",
            "annual_flux_90_upper_TgC": "may_july_flux_90_upper_TgC",
            "annual_flux_95_lower_TgC": "may_july_flux_95_lower_TgC",
            "annual_flux_95_upper_TgC": "may_july_flux_95_upper_TgC",
            "annual_confidence_tier": "may_july_confidence_tier",
        }
    )
    out["window_label"] = "provisional_may_july_freshet_window_not_final_snowmelt"
    return out


def _period_summary(annual: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("full_prediction_period", lambda frame: frame),
        ("training_overlap_period", lambda frame: frame[(frame["year"] >= 2003) & (frame["year"] <= 2024)]),
        ("early_hindcast_period", lambda frame: frame[(frame["year"] >= 2000) & (frame["year"] <= 2002)]),
        ("recent_extension_period", lambda frame: frame[frame["year"].eq(2025)]),
    ]
    rows: list[dict[str, Any]] = []
    for period, selector in specs:
        subset = selector(annual)
        for river, group in subset.groupby("river", dropna=False):
            if group.empty:
                continue
            counts = group["annual_confidence_tier"].value_counts().to_dict()
            rows.append(
                {
                    "river": river,
                    "period": period,
                    "mean_annual_flux_TgC": group["annual_flux_TgC"].mean(),
                    "median_annual_flux_TgC": group["annual_flux_TgC"].median(),
                    "min_annual_flux_TgC": group["annual_flux_TgC"].min(),
                    "max_annual_flux_TgC": group["annual_flux_TgC"].max(),
                    "mean_annual_flux_90_lower_TgC": group["annual_flux_90_lower_TgC"].mean(),
                    "mean_annual_flux_90_upper_TgC": group["annual_flux_90_upper_TgC"].mean(),
                    "mean_coverage_rate": group["coverage_rate"].mean(),
                    "mean_fraction_flux_from_low_confidence_days": group["fraction_flux_from_low_confidence_days"].mean(),
                    "n_years": int(group["year"].nunique()),
                    "confidence_summary": ";".join(f"{key}:{value}" for key, value in sorted(counts.items())),
                }
            )
    return pd.DataFrame(rows)


def _range_flags(daily_flux: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    columns = ["scope", "flag_type", "river", "year", "date", "flux_id", "value", "threshold", "notes"]
    rows: list[dict[str, Any]] = []

    def add_daily(flag_type: str, subset: pd.DataFrame, value_column: str, threshold: float | str, notes: str) -> None:
        for row in subset.itertuples(index=False):
            rows.append(
                {
                    "scope": "daily",
                    "flag_type": flag_type,
                    "river": row.river,
                    "year": int(row.year),
                    "date": row.date,
                    "flux_id": row.flux_id,
                    "value": getattr(row, value_column),
                    "threshold": threshold,
                    "notes": notes,
                }
            )

    add_daily("daily_flux_negative", daily_flux[daily_flux["daily_flux_kgC_day"] < 0], "daily_flux_kgC_day", 0.0, "Flag only; rows are retained.")
    add_daily("daily_flux_nan", daily_flux[daily_flux["daily_flux_kgC_day"].isna()], "daily_flux_kgC_day", "not_nan", "Flag only; rows are retained.")
    for river, group in daily_flux.groupby("river", dropna=False):
        median = float(group["daily_flux_kgC_day"].median())
        threshold = median * 5.0
        if threshold > 0:
            subset = group[group["daily_flux_kgC_day"] > threshold]
            add_daily(
                "daily_flux_extremely_high_relative_to_river_distribution",
                subset,
                "daily_flux_kgC_day",
                threshold,
                "Threshold is five times the river median daily flux.",
            )

    for river, group in annual.groupby("river", dropna=False):
        median = float(group["annual_flux_TgC"].median())
        high = median * 3.0
        low = median / 3.0 if median > 0 else np.nan
        high_subset = group[group["annual_flux_TgC"] > high]
        low_subset = group[group["annual_flux_TgC"] < low] if pd.notna(low) else group.iloc[0:0]
        for flag_type, subset, threshold in [
            ("annual_flux_gt_river_median_3x", high_subset, high),
            ("annual_flux_lt_river_median_div_3", low_subset, low),
        ]:
            for row in subset.itertuples(index=False):
                rows.append(
                    {
                        "scope": "annual",
                        "flag_type": flag_type,
                        "river": row.river,
                        "year": int(row.year),
                        "date": "",
                        "flux_id": "",
                        "value": row.annual_flux_TgC,
                        "threshold": threshold,
                        "notes": "Annual relative range flag; row retained.",
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def _confidence_summary(daily_flux: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(scope: str, group_value: str, frame: pd.DataFrame) -> None:
        total = len(frame)
        flux_total = frame["daily_flux_TgC_day"].sum() if not frame.empty else 0.0
        for tier in ["high", "medium", "low"]:
            subset = frame[frame["daily_confidence_tier"].eq(tier)]
            rows.append(
                {
                    "scope": scope,
                    "group_value": group_value,
                    "daily_confidence_tier": tier,
                    "n_days": int(len(subset)),
                    "fraction_days": float(len(subset) / total) if total else np.nan,
                    "flux_TgC": float(subset["daily_flux_TgC_day"].sum()) if not subset.empty else 0.0,
                    "fraction_flux": float(subset["daily_flux_TgC_day"].sum() / flux_total) if flux_total > 0 else np.nan,
                }
            )

    add("overall", "overall", daily_flux)
    for river, group in daily_flux.groupby("river", dropna=False):
        add("river", str(river), group)
    for row in annual.itertuples(index=False):
        rows.append(
            {
                "scope": "annual",
                "group_value": f"{row.river}_{int(row.year)}",
                "daily_confidence_tier": row.annual_confidence_tier,
                "n_days": int(row.n_days_with_flux),
                "fraction_days": float(row.coverage_rate),
                "flux_TgC": float(row.annual_flux_TgC),
                "fraction_flux": 1.0,
            }
        )
    return pd.DataFrame(rows)


def _qc_summary(
    predictions: pd.DataFrame,
    daily_flux: pd.DataFrame,
    annual: pd.DataFrame,
    range_flags: pd.DataFrame,
    missing: pd.DataFrame,
    flux_readiness: pd.DataFrame,
) -> pd.DataFrame:
    ready = flux_readiness[flux_readiness["decision_item"].eq("ready_for_flux_calculation")]
    ready_status = ready["status"].iloc[0] if not ready.empty else "unknown"
    readiness = analysis_readiness_status(annual, range_flags)
    min_coverage = float(annual["coverage_rate"].min()) if not annual.empty else 0.0
    max_low_fraction = float(annual["fraction_flux_from_low_confidence_days"].fillna(0).max()) if not annual.empty else 1.0
    rows = [
        {
            "qc_item": "daily_prediction_input_rows",
            "status": "true" if not predictions.empty else "false",
            "value": len(predictions),
            "notes": "Existing guarded daily DOC predictions were used; prediction grid was not reread.",
        },
        {
            "qc_item": "daily_flux_rows",
            "status": "true" if not daily_flux.empty else "false",
            "value": len(daily_flux),
            "notes": "Rows with valid Q and DOC prediction.",
        },
        {
            "qc_item": "input_flux_readiness_status",
            "status": ready_status,
            "value": ready_status,
            "notes": "Carried from daily DOC prediction phase.",
        },
        {
            "qc_item": "missing_flux_rows",
            "status": "true_with_caveats" if len(missing) else "true",
            "value": len(missing),
            "notes": "Rows retained in missing-row audit; no imputation performed.",
        },
        {
            "qc_item": "minimum_annual_coverage_rate",
            "status": "true" if min_coverage >= 0.98 else "true_with_caveats" if min_coverage >= 0.95 else "false",
            "value": min_coverage,
            "notes": "Annual expected days are based on the daily DOC prediction table span.",
        },
        {
            "qc_item": "maximum_fraction_flux_from_low_confidence_days",
            "status": "true" if max_low_fraction < 0.10 else "true_with_caveats" if max_low_fraction < 0.25 else "false",
            "value": max_low_fraction,
            "notes": "Low confidence is driven by logQ/doy extrapolation or point clipping.",
        },
        {
            "qc_item": "range_flag_rows",
            "status": "true" if range_flags.empty else "true_with_caveats",
            "value": len(range_flags),
            "notes": "Flags only; no rows are dropped.",
        },
        {
            "qc_item": "discharge_uncertainty_propagated",
            "status": "false",
            "value": "false",
            "notes": "Flux intervals propagate DOC concentration empirical residual intervals only.",
        },
        {
            "qc_item": "new_doc_model_trained",
            "status": "false",
            "value": "false",
            "notes": "No model fitting or refitting is performed in this phase.",
        },
        {
            "qc_item": "ready_for_trend_or_snowmelt_analysis",
            "status": readiness,
            "value": readiness,
            "notes": "Proceed only with caveats if status is true_with_caveats; May-July is provisional.",
        },
    ]
    return pd.DataFrame(rows)


def _missing_flux_rows(missing: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "prediction_id",
        "river",
        "date",
        "year",
        "month",
        "doy",
        "Q_m3s",
        "DOC_predicted_mgC_L",
        "prediction_status",
        "missing_predictor_reason",
        "flux_status",
        "missing_flux_reason",
        "model_spec_id",
        "model_fit_id",
        "freeze_id",
    ]
    out = missing.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date.astype(str)
    return out[[column for column in keep if column in out.columns]].reset_index(drop=True)


def _make_figures(daily_flux: pd.DataFrame, annual: pd.DataFrame, may_july: pd.DataFrame, confidence: pd.DataFrame) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    DOC_FLUX_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def save(fig, name: str) -> None:
        fig.tight_layout()
        destination = DOC_FLUX_FIGURE_DIR / name
        fig.savefig(destination, dpi=120)
        plt.close(fig)
        paths.append(destination)

    if not annual.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        means = annual.groupby("river")["annual_flux_TgC"].mean().sort_values()
        ax.barh(means.index.astype(str), means.values)
        ax.set_xlabel("Mean annual DOC flux (Tg C)")
        ax.set_title("Annual DOC flux by river")
        save(fig, "annual_flux_by_river.png")

        fig, ax = plt.subplots(figsize=(9, 5))
        for river, group in annual.groupby("river"):
            ax.plot(group["year"], group["annual_flux_TgC"], marker="o", linewidth=1.0, label=str(river))
        ax.set_xlabel("Year")
        ax.set_ylabel("Annual DOC flux (Tg C)")
        ax.set_title("Annual DOC flux time series")
        ax.legend(fontsize="x-small")
        save(fig, "annual_flux_timeseries_by_river.png")

        fig, ax = plt.subplots(figsize=(9, 5))
        for river, group in annual.groupby("river"):
            ax.plot(group["year"], group["annual_flux_TgC"], linewidth=1.0, label=str(river))
            ax.fill_between(
                group["year"].to_numpy(dtype=float),
                group["annual_flux_90_lower_TgC"].to_numpy(dtype=float),
                group["annual_flux_90_upper_TgC"].to_numpy(dtype=float),
                alpha=0.15,
            )
        ax.set_xlabel("Year")
        ax.set_ylabel("Annual DOC flux (Tg C)")
        ax.set_title("Annual DOC flux with 90% DOC interval")
        ax.legend(fontsize="x-small")
        save(fig, "annual_flux_with_90pct_interval_by_river.png")

    if not may_july.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        data = [group["may_july_flux_fraction_of_annual"].dropna().to_numpy() for _, group in may_july.groupby("river")]
        labels = [str(river) for river, _ in may_july.groupby("river")]
        if data:
            try:
                ax.boxplot(data, tick_labels=labels, showfliers=False)
            except TypeError:
                ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_ylabel("May-July fraction of annual flux")
        ax.set_title("Provisional May-July flux fraction by river")
        save(fig, "provisional_may_july_fraction_by_river.png")

    if not annual.empty:
        tier = annual.pivot_table(index="year", columns="annual_confidence_tier", values="n_days_with_flux", aggfunc="sum", fill_value=0)
        for column in ["high", "medium", "low"]:
            if column not in tier.columns:
                tier[column] = 0
        fig, ax = plt.subplots(figsize=(9, 4.8))
        bottom = np.zeros(len(tier))
        for column in ["high", "medium", "low"]:
            ax.bar(tier.index.astype(str), tier[column].to_numpy(), bottom=bottom, label=column)
            bottom += tier[column].to_numpy()
        ax.set_xlabel("Year")
        ax.set_ylabel("Daily flux rows")
        ax.set_title("Flux confidence tier by year")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)
        save(fig, "flux_confidence_tier_by_year.png")

    if not daily_flux.empty:
        sample_rivers = sorted(daily_flux["river"].dropna().unique())[:6]
        fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=False)
        for ax, river in zip(axes.ravel(), sample_rivers):
            group = daily_flux[daily_flux["river"].eq(river)].copy()
            group["date"] = pd.to_datetime(group["date"], errors="coerce")
            ax.plot(group["date"], group["daily_flux_TgC_day"], linewidth=0.6)
            ax.set_title(str(river))
            ax.set_ylabel("Tg C/day")
        save(fig, "daily_flux_timeseries_examples.png")

    return paths


def run_doc_flux() -> dict[str, Any]:
    _ensure_dirs()
    _verify_contract_snapshot()
    inputs = _load_inputs()
    prediction_hash_before = sha256_file(DAILY_DOC_PREDICTION_PATH)
    predictions = inputs["daily_prediction"]
    daily_flux, missing = _daily_flux(predictions)
    annual = _summarize_flux_by_year(daily_flux, predictions)
    may_july_raw = _summarize_flux_by_year(daily_flux, predictions, months={5, 6, 7})
    may_july = _provisional_may_july(annual, may_july_raw)
    period = _period_summary(annual)
    range_flags = _range_flags(daily_flux, annual)
    missing_rows = _missing_flux_rows(missing)
    confidence = _confidence_summary(daily_flux, annual)
    qc = _qc_summary(predictions, daily_flux, annual, range_flags, missing_rows, inputs["flux_readiness"])

    table_paths = [
        _write_csv(daily_flux, DAILY_DOC_FLUX_PATH),
        _write_csv(annual, ANNUAL_DOC_FLUX_PATH),
        _write_csv(may_july, MAY_JULY_DOC_FLUX_PATH),
        _write_csv(period, PERIOD_DOC_FLUX_PATH),
        _write_csv(qc, DOC_FLUX_TABLE_DIR / "doc_flux_qc_summary.csv"),
        _write_csv(range_flags, DOC_FLUX_TABLE_DIR / "doc_flux_range_flags.csv"),
        _write_csv(missing_rows, DOC_FLUX_TABLE_DIR / "doc_flux_missing_rows.csv"),
        _write_csv(confidence, DOC_FLUX_TABLE_DIR / "doc_flux_confidence_tier_summary.csv"),
    ]
    figure_paths = _make_figures(daily_flux, annual, may_july, confidence)
    report_path = write_doc_flux_report()
    if sha256_file(DAILY_DOC_PREDICTION_PATH) != prediction_hash_before:
        raise RuntimeError("Daily DOC prediction input changed during DOC flux calculation.")
    return {
        "tables": table_paths,
        "figures": figure_paths,
        "report": report_path,
        "daily_flux": daily_flux,
        "annual_flux": annual,
        "may_july_flux": may_july,
        "qc": qc,
        "ready_for_trend_or_snowmelt_analysis": analysis_readiness_status(annual, range_flags),
    }


__all__ = [
    "DOC_FLUX_TABLE_DIR",
    "DOC_FLUX_REPORT_DIR",
    "DOC_FLUX_FIGURE_DIR",
    "DOC_FLUX_REPORT_PATH",
    "run_doc_flux",
    "write_doc_flux_report",
]
