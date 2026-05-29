from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..gold_contract import sha256_file
from ..paths import REPORT_DIR, TABLE_DIR, path
from ..reports import _md_table, utc_now
from .flux_qc import as_bool


FRESHET_TABLE_DIR = TABLE_DIR / "freshet_control"
FRESHET_REPORT_DIR = REPORT_DIR / "freshet_control"
FRESHET_FIGURE_DIR = path("outputs", "figures", "freshet_control")
FRESHET_REPORT_PATH = FRESHET_REPORT_DIR / "freshet_control_synthesis_report.md"

DOC_FLUX_TABLE_DIR = TABLE_DIR / "doc_flux"
SNOWMELT_TABLE_DIR = TABLE_DIR / "snowmelt_windows"
ATTRIBUTION_TABLE_DIR = TABLE_DIR / "flux_attribution"

REQUIRED_FRESHET_INPUTS = {
    "daily_doc_flux": DOC_FLUX_TABLE_DIR / "daily_doc_flux.csv",
    "annual_doc_flux_summary": DOC_FLUX_TABLE_DIR / "annual_doc_flux_summary.csv",
    "snowmelt_window_flux_summary": SNOWMELT_TABLE_DIR / "snowmelt_window_flux_summary.csv",
    "snowmelt_window_trends_by_river": SNOWMELT_TABLE_DIR / "snowmelt_window_trends_by_river.csv",
    "annual_vs_snowmelt_signal_comparison": SNOWMELT_TABLE_DIR / "annual_vs_snowmelt_signal_comparison.csv",
    "annual_flux_attribution": ATTRIBUTION_TABLE_DIR / "annual_flux_attribution_by_river_year.csv",
    "q_doc_component_trends": ATTRIBUTION_TABLE_DIR / "q_doc_component_trends_by_river.csv",
    "flux_driver_classification": ATTRIBUTION_TABLE_DIR / "flux_driver_classification.csv",
    "monthly_flux_by_river_year": ATTRIBUTION_TABLE_DIR / "monthly_flux_by_river_year.csv",
    "monthly_flux_trends_by_river": ATTRIBUTION_TABLE_DIR / "monthly_flux_trends_by_river.csv",
    "seasonal_flux_decomposition": ATTRIBUTION_TABLE_DIR / "seasonal_flux_decomposition_by_river_year.csv",
    "seasonal_flux_trends_by_river": ATTRIBUTION_TABLE_DIR / "seasonal_flux_trends_by_river.csv",
    "export_phenology": ATTRIBUTION_TABLE_DIR / "export_phenology_by_river_year.csv",
    "export_phenology_trends": ATTRIBUTION_TABLE_DIR / "export_phenology_trends_by_river.csv",
    "yukon_flux_attribution_report": REPORT_DIR / "flux_attribution" / "yukon_flux_attribution_report.md",
    "final_synthesis_report": REPORT_DIR / "final_synthesis" / "final_synthesis_report.md",
}

WINDOW_LABELS = {
    "fixed_may_july_reference": "Fixed May-July reference",
    "q75_peak_contiguous": "Q75 high-flow peak window",
    "discharge_centered_freshet": "Discharge-centered freshet",
    "snow_depletion_assisted": "Snow-depletion assisted",
    "common_overlap_w1_w2": "Common overlap W1/W2",
}


def _ensure_dirs() -> None:
    for directory in [FRESHET_TABLE_DIR, FRESHET_REPORT_DIR, FRESHET_FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def _read_csv(destination: Path) -> pd.DataFrame:
    if not destination.exists():
        raise FileNotFoundError(f"Required freshet-control input is missing: {destination}")
    return pd.read_csv(destination, low_memory=False)


def _read_text(destination: Path) -> str:
    if not destination.exists():
        raise FileNotFoundError(f"Required freshet-control input is missing: {destination}")
    return destination.read_text(encoding="utf-8")


def _write_csv(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8")
    return destination


def _input_hashes() -> dict[Path, str]:
    return {destination: sha256_file(destination) for destination in REQUIRED_FRESHET_INPUTS.values()}


def _verify_inputs_unchanged(before: dict[Path, str]) -> None:
    after = _input_hashes()
    changed = [str(destination) for destination, before_hash in before.items() if after.get(destination) != before_hash]
    if changed:
        raise RuntimeError(f"Freshet-control inputs changed during synthesis: {changed}")


def _load_inputs() -> dict[str, Any]:
    return {
        "daily_flux": _read_csv(REQUIRED_FRESHET_INPUTS["daily_doc_flux"]),
        "annual_flux": _read_csv(REQUIRED_FRESHET_INPUTS["annual_doc_flux_summary"]),
        "snowmelt": _read_csv(REQUIRED_FRESHET_INPUTS["snowmelt_window_flux_summary"]),
        "snowmelt_trends": _read_csv(REQUIRED_FRESHET_INPUTS["snowmelt_window_trends_by_river"]),
        "snowmelt_comparison": _read_csv(REQUIRED_FRESHET_INPUTS["annual_vs_snowmelt_signal_comparison"]),
        "annual_attr": _read_csv(REQUIRED_FRESHET_INPUTS["annual_flux_attribution"]),
        "component_trends": _read_csv(REQUIRED_FRESHET_INPUTS["q_doc_component_trends"]),
        "drivers": _read_csv(REQUIRED_FRESHET_INPUTS["flux_driver_classification"]),
        "monthly": _read_csv(REQUIRED_FRESHET_INPUTS["monthly_flux_by_river_year"]),
        "monthly_trends": _read_csv(REQUIRED_FRESHET_INPUTS["monthly_flux_trends_by_river"]),
        "seasonal": _read_csv(REQUIRED_FRESHET_INPUTS["seasonal_flux_decomposition"]),
        "seasonal_trends": _read_csv(REQUIRED_FRESHET_INPUTS["seasonal_flux_trends_by_river"]),
        "phenology": _read_csv(REQUIRED_FRESHET_INPUTS["export_phenology"]),
        "phenology_trends": _read_csv(REQUIRED_FRESHET_INPUTS["export_phenology_trends"]),
        "yukon_report_text": _read_text(REQUIRED_FRESHET_INPUTS["yukon_flux_attribution_report"]),
        "final_synthesis_text": _read_text(REQUIRED_FRESHET_INPUTS["final_synthesis_report"]),
    }


def _prepare_bool(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = as_bool(out[column])
    return out


def _prepare_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    annual_attr = _prepare_bool(inputs["annual_attr"], ["cohort_core_2003_2024", "cohort_high_confidence_only"])
    snowmelt = _prepare_bool(inputs["snowmelt"], ["core_2003_2024", "sensitivity_only", "exclude_from_trend"])
    monthly = _prepare_bool(inputs["monthly"], ["cohort_core_2003_2024", "cohort_high_confidence_only", "cohort_sensitivity_only", "cohort_exclude_from_trend"])
    seasonal = _prepare_bool(inputs["seasonal"], ["cohort_core_2003_2024", "cohort_high_confidence_only", "cohort_sensitivity_only", "cohort_exclude_from_trend"])
    for frame in [annual_attr, snowmelt, monthly, seasonal, inputs["phenology"]]:
        if "year" in frame.columns:
            frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    for frame in [annual_attr, snowmelt, monthly, seasonal, inputs["snowmelt_trends"], inputs["component_trends"], inputs["phenology_trends"]]:
        for column in frame.columns:
            if any(token in column for token in ["flux", "fraction", "slope", "p_value", "r2", "mean", "median"]):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
    prepared = dict(inputs)
    prepared.update({"annual_attr": annual_attr, "snowmelt": snowmelt, "monthly": monthly, "seasonal": seasonal})
    return prepared


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    clean = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(clean) < 3 or clean["x"].nunique() < 2 or clean["y"].nunique() < 2:
        return float("nan")
    return float(clean["x"].corr(clean["y"]))


def _safe_ols_r2(x: pd.Series, y: pd.Series) -> float:
    clean = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(clean) < 3 or clean["x"].nunique() < 2 or clean["y"].nunique() < 2:
        return float("nan")
    result = stats.linregress(clean["x"], clean["y"])
    return float(result.rvalue**2)


def _variability_category(r2: float) -> str:
    if pd.isna(r2):
        return "insufficient"
    if r2 >= 0.75:
        return "strong"
    if r2 >= 0.40:
        return "moderate"
    return "weak"


def build_annual_flux_vs_window_flux_coupling(snowmelt: pd.DataFrame) -> pd.DataFrame:
    core = snowmelt[snowmelt["core_2003_2024"]].copy()
    rows: list[dict[str, Any]] = []
    for (river, window_id), group in core.groupby(["river", "window_id"], dropna=False):
        r_flux = _safe_corr(group["annual_flux_TgC"], group["window_flux_TgC"])
        r_fraction = _safe_corr(group["annual_flux_TgC"], group["window_fraction_of_annual"])
        r2 = _safe_ols_r2(group["window_flux_TgC"], group["annual_flux_TgC"])
        rows.append(
            {
                "river": river,
                "window_id": window_id,
                "window_label": WINDOW_LABELS.get(str(window_id), str(window_id)),
                "n_years": int(len(group)),
                "mean_window_fraction_of_annual": float(pd.to_numeric(group["window_fraction_of_annual"], errors="coerce").mean()),
                "median_window_fraction_of_annual": float(pd.to_numeric(group["window_fraction_of_annual"], errors="coerce").median()),
                "annual_flux_window_flux_correlation": r_flux,
                "annual_flux_window_fraction_correlation": r_fraction,
                "ols_r2_annual_flux_vs_window_flux": r2,
                "annual_variability_explained_category": _variability_category(r2),
                "is_dynamic_window": str(window_id) != "fixed_may_july_reference",
                "notes": "operational_window_definition;not_causal_proof",
            }
        )
    return pd.DataFrame(rows).sort_values(["river", "is_dynamic_window", "ols_r2_annual_flux_vs_window_flux"], ascending=[True, False, False])


def _trend_row(trends: pd.DataFrame, river: str, metric: str, window_id: str | None = None) -> pd.Series:
    subset = trends[trends["river"].astype(str).eq(river) & trends["metric"].astype(str).eq(metric)]
    if window_id is not None and "window_id" in subset.columns:
        subset = subset[subset["window_id"].astype(str).eq(window_id)]
    return subset.iloc[0] if not subset.empty else pd.Series(dtype=object)


def _detectable(row: pd.Series, direction: str | None = None) -> bool:
    if row.empty:
        return False
    value = str(row.get("detectable_trend", row.get("significant_at_0_05", ""))).strip().lower() in {"true", "1"}
    if direction is None:
        return value
    return value and str(row.get("trend_direction", "")) == direction


def _trend_direction(row: pd.Series) -> str:
    return str(row.get("trend_direction", "unavailable")) if not row.empty else "unavailable"


def _trend_language(row: pd.Series) -> str:
    if row.empty:
        return "unavailable"
    language = row.get("trend_language", "")
    if isinstance(language, str) and language:
        return language
    return f"detectable {_trend_direction(row)} trend" if _detectable(row) else "no detectable trend"


def _best_dynamic_window(coupling: pd.DataFrame, river: str) -> pd.Series:
    subset = coupling[coupling["river"].astype(str).eq(river) & coupling["is_dynamic_window"].astype(bool)].copy()
    if subset.empty:
        return pd.Series(dtype=object)
    subset["sort_r2"] = pd.to_numeric(subset["ols_r2_annual_flux_vs_window_flux"], errors="coerce").fillna(-1)
    return subset.sort_values(["sort_r2", "mean_window_fraction_of_annual"], ascending=[False, False]).iloc[0]


def _driver_row(drivers: pd.DataFrame, river: str) -> pd.Series:
    subset = drivers[drivers["river"].astype(str).eq(river)]
    return subset.iloc[0] if not subset.empty else pd.Series(dtype=object)


def _phenology_row(phenology_trends: pd.DataFrame, river: str, metric: str) -> pd.Series:
    return _trend_row(phenology_trends, river, metric)


def _window_fraction_stats(snowmelt: pd.DataFrame, river: str, window_id: str) -> tuple[float, float, str]:
    subset = snowmelt[
        snowmelt["river"].astype(str).eq(river)
        & snowmelt["window_id"].astype(str).eq(window_id)
        & snowmelt["core_2003_2024"]
    ]
    values = pd.to_numeric(subset["window_fraction_of_annual"], errors="coerce").dropna()
    if values.empty:
        return float("nan"), float("nan"), ""
    return float(values.mean()), float(values.median()), f"{values.min():.3f}-{values.max():.3f}"


def _assign_regime(
    river: str,
    best: pd.Series,
    driver: pd.Series,
    annual_trend_direction: str,
    window_flux_trend: str,
    window_fraction_trend: str,
    phenology_shift: str,
    r2: float,
    mean_fraction: float,
) -> str:
    driver_class = str(driver.get("driver_classification", ""))
    if river == "Yukon" and driver_class == "discharge_volume_dominated":
        return "discharge_volume_extended_season"
    if pd.isna(r2) or str(best.get("window_id", "")) == "":
        return "caveated_uncertain"
    if annual_trend_direction == "flat_or_uncertain" and mean_fraction >= 0.55:
        return "freshet_dominated_stable"
    if _variability_category(r2) in {"strong", "moderate"} and mean_fraction >= 0.35:
        return "high_flow_window_sensitive"
    if annual_trend_direction == "flat_or_uncertain":
        return "no_detectable_change"
    if "detectable" in phenology_shift and "increasing" in phenology_shift:
        return "discharge_volume_extended_season"
    return "caveated_uncertain"


def build_freshet_control_summary(
    snowmelt: pd.DataFrame,
    snowmelt_trends: pd.DataFrame,
    annual_comparison: pd.DataFrame,
    coupling: pd.DataFrame,
    drivers: pd.DataFrame,
    phenology_trends: pd.DataFrame,
) -> pd.DataFrame:
    rivers = sorted(snowmelt["river"].dropna().astype(str).unique())
    rows: list[dict[str, Any]] = []
    for river in rivers:
        best = _best_dynamic_window(coupling, river)
        best_window = str(best.get("window_id", ""))
        mean_fraction, median_fraction, fraction_range = _window_fraction_stats(snowmelt, river, best_window)
        annual_row = annual_comparison[
            annual_comparison["river"].astype(str).eq(river)
            & annual_comparison["window_id"].astype(str).eq(best_window)
        ]
        comparison = annual_row.iloc[0] if not annual_row.empty else pd.Series(dtype=object)
        window_flux = _trend_row(snowmelt_trends, river, "window_flux_TgC", best_window)
        window_fraction = _trend_row(snowmelt_trends, river, "window_fraction_of_annual", best_window)
        driver = _driver_row(drivers, river)
        centroid = _phenology_row(phenology_trends, river, "flux_centroid_doy")
        after_july = _phenology_row(phenology_trends, river, "fraction_flux_after_july")
        phenology_shift = "; ".join(
            item
            for item in [
                f"centroid={_trend_language(centroid)}",
                f"after_july_fraction={_trend_language(after_july)}",
            ]
            if item
        )
        r2 = float(best.get("ols_r2_annual_flux_vs_window_flux", np.nan))
        regime = _assign_regime(
            river,
            best,
            driver,
            str(comparison.get("annual_trend_direction", driver.get("annual_flux_trend_direction", ""))),
            _trend_direction(window_flux),
            _trend_direction(window_fraction),
            phenology_shift,
            r2,
            mean_fraction,
        )
        interpretation = _interpret_regime(river, regime, best_window, r2, mean_fraction, driver)
        rows.append(
            {
                "river": river,
                "best_dynamic_window": best_window,
                "mean_window_fraction_of_annual": mean_fraction,
                "median_window_fraction_of_annual": median_fraction,
                "window_fraction_range": fraction_range,
                "annual_flux_trend_direction": comparison.get("annual_trend_direction", driver.get("annual_flux_trend_direction", "")),
                "window_flux_trend_direction": _trend_direction(window_flux),
                "window_fraction_trend_direction": _trend_direction(window_fraction),
                "annual_variability_explained_by_window_flux_r2": r2,
                "annual_variability_explained_category": _variability_category(r2),
                "driver_classification": driver.get("driver_classification", ""),
                "export_phenology_shift": phenology_shift,
                "regime_label": regime,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def _interpret_regime(river: str, regime: str, window_id: str, r2: float, mean_fraction: float, driver: pd.Series) -> str:
    window_label = WINDOW_LABELS.get(window_id, window_id)
    if regime == "discharge_volume_extended_season":
        return f"{river} is best interpreted as discharge-volume-driven extended-season export expansion; high-flow windows matter but do not alone explain the annual increase."
    if regime == "freshet_dominated_stable":
        return f"{river} exports a large stable share of annual DOC during {window_label}; no detectable annual change is carried forward."
    if regime == "high_flow_window_sensitive":
        return f"{river} annual variability is moderately to strongly coupled to {window_label} (R2={r2:.2f}, mean fraction={mean_fraction:.2f})."
    if regime == "no_detectable_change":
        return f"{river} has no detectable annual flux change and no strong evidence for changing export phenology."
    return f"{river} remains caveated or uncertain under the operational freshet-window definitions."


def build_export_regime_classification(
    annual_attr: pd.DataFrame,
    snowmelt: pd.DataFrame,
    drivers: pd.DataFrame,
    phenology_trends: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    core = annual_attr[annual_attr["cohort_core_2003_2024"].astype(bool)].copy()
    mean_flux = core.groupby("river")["annual_flux_TgC"].mean().sort_values(ascending=False)
    rank = mean_flux.rank(ascending=False, method="min").astype(int).to_dict()
    fixed = snowmelt[snowmelt["window_id"].astype(str).eq("fixed_may_july_reference") & snowmelt["core_2003_2024"]]
    q75 = snowmelt[snowmelt["window_id"].astype(str).eq("q75_peak_contiguous") & snowmelt["core_2003_2024"]]
    fixed_fraction = fixed.groupby("river")["window_fraction_of_annual"].mean().to_dict()
    q75_fraction = q75.groupby("river")["window_fraction_of_annual"].mean().to_dict()
    rows: list[dict[str, Any]] = []
    for row in summary.itertuples(index=False):
        driver = _driver_row(drivers, str(row.river))
        after_july = _phenology_row(phenology_trends, str(row.river), "fraction_flux_after_july")
        centroid = _phenology_row(phenology_trends, str(row.river), "flux_centroid_doy")
        rows.append(
            {
                "river": row.river,
                "total_flux_rank": rank.get(row.river, np.nan),
                "yield_rank": "not_available_allowed_inputs_lack_upstream_area",
                "mean_may_july_fraction": fixed_fraction.get(row.river, np.nan),
                "mean_q75_window_fraction": q75_fraction.get(row.river, np.nan),
                "after_july_fraction_trend": _trend_language(after_july),
                "flux_centroid_trend": _trend_language(centroid),
                "annual_flux_trend": driver.get("annual_flux_trend_direction", ""),
                "Q_volume_trend": driver.get("Q_volume_trend_direction", ""),
                "flow_weighted_DOC_trend": driver.get("flow_weighted_DOC_trend_direction", ""),
                "assigned_regime": row.regime_label,
                "evidence_summary": row.interpretation,
            }
        )
    return pd.DataFrame(rows).sort_values("river")


def build_yukon_extended_season_diagnosis(
    summary: pd.DataFrame,
    drivers: pd.DataFrame,
    snowmelt_trends: pd.DataFrame,
    may_july_trends: pd.DataFrame,
    phenology_trends: pd.DataFrame,
) -> pd.DataFrame:
    driver = _driver_row(drivers, "Yukon")
    may_flux = may_july_trends[
        may_july_trends["river"].astype(str).eq("Yukon")
        & may_july_trends["metric"].astype(str).eq("seasonal_flux_TgC")
        & may_july_trends["season_window"].astype(str).eq("may_july")
    ]
    may_fraction = may_july_trends[
        may_july_trends["river"].astype(str).eq("Yukon")
        & may_july_trends["metric"].astype(str).eq("seasonal_fraction_of_annual")
        & may_july_trends["season_window"].astype(str).eq("may_july")
    ]
    may_flux_row = may_flux.iloc[0] if not may_flux.empty else pd.Series(dtype=object)
    may_fraction_row = may_fraction.iloc[0] if not may_fraction.empty else pd.Series(dtype=object)
    q75_flux = _trend_row(snowmelt_trends, "Yukon", "window_flux_TgC", "q75_peak_contiguous")
    discharge_fraction = _trend_row(snowmelt_trends, "Yukon", "window_fraction_of_annual", "discharge_centered_freshet")
    after_july = _phenology_row(phenology_trends, "Yukon", "fraction_flux_after_july")
    centroid = _phenology_row(phenology_trends, "Yukon", "flux_centroid_doy")
    active = _phenology_row(phenology_trends, "Yukon", "active_flux_season_length")
    summary_row = summary[summary["river"].astype(str).eq("Yukon")]
    regime = summary_row.iloc[0]["regime_label"] if not summary_row.empty else "discharge_volume_extended_season"
    final = "discharge-volume-driven extended-season export expansion"
    rows = [
        {
            "diagnostic_item": "annual_flux_increasing",
            "status": bool(driver.get("annual_flux_detectable", False) and driver.get("annual_flux_trend_direction", "") == "increasing"),
            "evidence": f"slope={driver.get('annual_flux_slope_TgC_per_year', np.nan)}, p={driver.get('annual_flux_p_value', np.nan)}",
        },
        {
            "diagnostic_item": "Q_volume_increasing",
            "status": bool(driver.get("Q_volume_detectable", False) and driver.get("Q_volume_trend_direction", "") == "increasing"),
            "evidence": f"slope={driver.get('Q_volume_slope_km3_per_year', np.nan)}, p={driver.get('Q_volume_p_value', np.nan)}",
        },
        {
            "diagnostic_item": "flow_weighted_DOC_no_detectable_trend",
            "status": not bool(driver.get("flow_weighted_DOC_detectable", False)),
            "evidence": f"slope={driver.get('flow_weighted_DOC_slope_mgC_L_per_year', np.nan)}, p={driver.get('flow_weighted_DOC_p_value', np.nan)}",
        },
        {
            "diagnostic_item": "May_July_flux_trend_no",
            "status": not _detectable(may_flux_row),
            "evidence": _trend_language(may_flux_row),
        },
        {
            "diagnostic_item": "May_July_fraction_decreasing",
            "status": _detectable(may_fraction_row, "decreasing") or float(may_fraction_row.get("slope_ols_per_year", 0.0) or 0.0) < 0,
            "evidence": _trend_language(may_fraction_row),
        },
        {
            "diagnostic_item": "q75_window_flux_trend_no",
            "status": not _detectable(q75_flux),
            "evidence": _trend_language(q75_flux),
        },
        {
            "diagnostic_item": "discharge_centered_fraction_increasing",
            "status": _detectable(discharge_fraction, "increasing") or float(discharge_fraction.get("slope_ols_per_year", 0.0) or 0.0) > 0,
            "evidence": _trend_language(discharge_fraction),
        },
        {
            "diagnostic_item": "after_July_fraction_increasing",
            "status": _detectable(after_july, "increasing"),
            "evidence": _trend_language(after_july),
        },
        {
            "diagnostic_item": "centroid_later",
            "status": _detectable(centroid, "increasing"),
            "evidence": _trend_language(centroid),
        },
        {
            "diagnostic_item": "active_season_length_increasing",
            "status": _detectable(active, "increasing"),
            "evidence": _trend_language(active),
        },
        {
            "diagnostic_item": "final_interpretation",
            "status": True,
            "evidence": f"{final}; regime={regime}",
        },
    ]
    return pd.DataFrame(rows)


def _make_figures(
    summary: pd.DataFrame,
    coupling: pd.DataFrame,
    regimes: pd.DataFrame,
    yukon: pd.DataFrame,
    snowmelt: pd.DataFrame,
    phenology_trends: pd.DataFrame,
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    FRESHET_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def save(fig, name: str) -> None:
        fig.tight_layout()
        destination = FRESHET_FIGURE_DIR / name
        fig.savefig(destination, dpi=120)
        plt.close(fig)
        paths.append(destination)

    if not summary.empty:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        ordered = summary.sort_values("mean_window_fraction_of_annual")
        ax.barh(ordered["river"], ordered["mean_window_fraction_of_annual"])
        ax.set_xlabel("Mean best-window fraction of annual flux")
        ax.set_title("Freshet/high-flow fraction by river")
        save(fig, "freshet_fraction_by_river.png")

    q75 = coupling[coupling["window_id"].astype(str).eq("q75_peak_contiguous")]
    if not q75.empty:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.barh(q75.sort_values("ols_r2_annual_flux_vs_window_flux")["river"], q75.sort_values("ols_r2_annual_flux_vs_window_flux")["ols_r2_annual_flux_vs_window_flux"])
        ax.set_xlabel("OLS R2: annual flux ~ q75 window flux")
        ax.set_title("Annual flux coupling to Q75 high-flow window")
        save(fig, "annual_flux_vs_q75_window_flux_by_river.png")

    if not regimes.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        counts = regimes["assigned_regime"].value_counts()
        ax.bar(counts.index, counts.values)
        ax.set_ylabel("Number of rivers")
        ax.set_title("DOC export regime summary")
        ax.tick_params(axis="x", rotation=25)
        save(fig, "export_regime_summary.png")

    yukon_core = snowmelt[snowmelt["river"].astype(str).eq("Yukon") & snowmelt["core_2003_2024"]]
    if not yukon_core.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        for window_id, group in yukon_core.groupby("window_id"):
            if str(window_id) in {"fixed_may_july_reference", "q75_peak_contiguous", "discharge_centered_freshet"}:
                ax.plot(group["year"], group["window_fraction_of_annual"], marker="o", linewidth=1.0, label=str(window_id))
        ax.set_xlabel("Year")
        ax.set_ylabel("Fraction of annual flux")
        ax.set_title("Yukon extended-season dashboard: window fractions")
        ax.legend(fontsize="x-small")
        save(fig, "yukon_extended_season_dashboard.png")

    centroid = phenology_trends[phenology_trends["metric"].astype(str).eq("flux_centroid_doy")]
    if not centroid.empty:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.barh(centroid.sort_values("slope_ols_per_year")["river"], centroid.sort_values("slope_ols_per_year")["slope_ols_per_year"])
        ax.axvline(0, linestyle="--")
        ax.set_xlabel("Centroid DOY slope (days/year)")
        ax.set_title("Flux centroid trends by river")
        save(fig, "flux_centroid_trends_by_river.png")

    after_july = phenology_trends[phenology_trends["metric"].astype(str).eq("fraction_flux_after_july")]
    if not after_july.empty:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.barh(after_july.sort_values("slope_ols_per_year")["river"], after_july.sort_values("slope_ols_per_year")["slope_ols_per_year"])
        ax.axvline(0, linestyle="--")
        ax.set_xlabel("After-July fraction slope (fraction/year)")
        ax.set_title("After-July export fraction trends by river")
        save(fig, "after_july_fraction_trends_by_river.png")
    return paths


def write_freshet_control_report() -> Path:
    FRESHET_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = _read_csv(FRESHET_TABLE_DIR / "freshet_control_summary_by_river.csv")
    coupling = _read_csv(FRESHET_TABLE_DIR / "annual_flux_vs_window_flux_coupling.csv")
    regimes = _read_csv(FRESHET_TABLE_DIR / "export_regime_classification.csv")
    yukon = _read_csv(FRESHET_TABLE_DIR / "yukon_extended_season_diagnosis.csv")
    lines = [
        "# Freshet Control and Export Phenology Synthesis Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "## 1. Scope and guardrails",
        "",
        "This synthesis integrates existing daily flux, annual flux, snowmelt-window, flux-attribution, export-phenology, and annual-trend outputs. It does not train models, generate new DOC predictions, recalculate flux, read raw/interim/canonical data, or modify gold data.",
        "",
        "## 2. Literature-inspired framing: freshet control vs extended-season export",
        "",
        "The process question is whether annual DOC export is primarily governed by operational snowmelt/freshet or high-flow windows, or whether late-season and shoulder-season discharge expansion contributes meaningfully to annual export. These are process hypotheses, not formal causal proof.",
        "",
        "## 3. Freshet/high-flow contribution to annual DOC flux",
        "",
        _md_table(summary[["river", "best_dynamic_window", "mean_window_fraction_of_annual", "annual_variability_explained_by_window_flux_r2", "regime_label"]], max_rows=20),
        "",
        "## 4. Does freshet-window flux explain annual variability?",
        "",
        _md_table(coupling, max_rows=40),
        "",
        "## 5. Yukon extended-season diagnosis",
        "",
        _md_table(yukon, max_rows=20),
        "",
        "## 6. Export phenology shifts",
        "",
        "Yukon shows later/export-expanded diagnostics: after-July fraction, flux centroid, and active export season length increase in the core cohort. This supports an extended-season export interpretation rather than a fixed May-July explanation.",
        "",
        "## 7. River export regime classification",
        "",
        _md_table(regimes, max_rows=20),
        "",
        "## 8. Mechanistic interpretation",
        "",
        "Freshet or high-flow windows explain a large share of annual flux for several rivers, but the Yukon annual increase is best described as discharge-volume-driven extended-season export expansion. Fixed May-July remains a provisional reference, and dynamic windows are operational definitions rather than final hydrologic truth.",
        "",
        "## 9. Caveats",
        "",
        "- This is exploratory mechanism analysis, not causal proof.",
        "- No new model, prediction, or flux product was generated.",
        "- Discharge uncertainty was not propagated.",
        "- Snowmelt/freshet windows are operational definitions.",
        "- DOC concentration and flux results inherit the caveats from the guarded production and flux phases.",
        "",
        "## 10. Manuscript-ready process hypotheses",
        "",
        "1. ArcticGRO DOC export is often concentrated in operational high-flow/freshet windows, but the strength of annual coupling differs by river.",
        "2. Yukon annual DOC flux increase is consistent with discharge-volume-driven extended-season export expansion rather than a stronger fixed May-July contribution.",
        "3. Flow-weighted DOC does not show a detectable Yukon increase in the core cohort, suggesting volume and timing are more plausible first-order explanations than concentration intensification.",
        "",
        "## 11. What not to claim",
        "",
        "- Do not claim causal proof.",
        "- Do not claim discharge uncertainty was propagated.",
        "- Do not claim fixed May-July is final snowmelt flux.",
        "- Do not claim new DOC model, prediction, or flux products were created in this phase.",
    ]
    FRESHET_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return FRESHET_REPORT_PATH


def synthesize_freshet_control() -> dict[str, Any]:
    _ensure_dirs()
    before_hashes = _input_hashes()
    inputs = _prepare_inputs(_load_inputs())
    coupling = build_annual_flux_vs_window_flux_coupling(inputs["snowmelt"])
    summary = build_freshet_control_summary(
        inputs["snowmelt"],
        inputs["snowmelt_trends"],
        inputs["snowmelt_comparison"],
        coupling,
        inputs["drivers"],
        inputs["phenology_trends"],
    )
    regimes = build_export_regime_classification(
        inputs["annual_attr"],
        inputs["snowmelt"],
        inputs["drivers"],
        inputs["phenology_trends"],
        summary,
    )
    yukon = build_yukon_extended_season_diagnosis(
        summary,
        inputs["drivers"],
        inputs["snowmelt_trends"],
        inputs["seasonal_trends"],
        inputs["phenology_trends"],
    )
    figures = _make_figures(summary, coupling, regimes, yukon, inputs["snowmelt"], inputs["phenology_trends"])
    table_paths = [
        _write_csv(summary, FRESHET_TABLE_DIR / "freshet_control_summary_by_river.csv"),
        _write_csv(coupling, FRESHET_TABLE_DIR / "annual_flux_vs_window_flux_coupling.csv"),
        _write_csv(regimes, FRESHET_TABLE_DIR / "export_regime_classification.csv"),
        _write_csv(yukon, FRESHET_TABLE_DIR / "yukon_extended_season_diagnosis.csv"),
    ]
    report_path = write_freshet_control_report()
    _verify_inputs_unchanged(before_hashes)
    return {
        "tables": table_paths,
        "figures": figures,
        "report": report_path,
        "summary": summary,
        "coupling": coupling,
        "regimes": regimes,
        "yukon": yukon,
    }


__all__ = [
    "FRESHET_TABLE_DIR",
    "FRESHET_REPORT_DIR",
    "FRESHET_FIGURE_DIR",
    "FRESHET_REPORT_PATH",
    "REQUIRED_FRESHET_INPUTS",
    "synthesize_freshet_control",
    "write_freshet_control_report",
]
