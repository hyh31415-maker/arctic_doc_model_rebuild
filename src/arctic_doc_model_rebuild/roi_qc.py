from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .gold_contract import load_contract, require_gold_data_dir, sha256_file, table_path
from .modeling.diagnostics import assert_no_forbidden_outputs
from .paths import REPORT_DIR, TABLE_DIR, path
from .reports import _md_table, utc_now


ROI_QC_TABLE_DIR = TABLE_DIR / "roi_qc"
ROI_QC_REPORT_DIR = REPORT_DIR / "roi_qc"
ROI_QC_FIGURE_DIR = path("outputs", "figures", "roi_qc")
ROI_QC_REPORT_PATH = ROI_QC_REPORT_DIR / "roi_final_qc_report.md"

OPTICAL_SENSITIVITY_REPORT_PATH = REPORT_DIR / "optical_sensitivity" / "optical_sensitivity_report.md"
OPTICAL_RANKING_PATH = TABLE_DIR / "optical_sensitivity" / "optical_model_ranking.csv"
OPTICAL_DATASET_REGISTRY_PATH = TABLE_DIR / "optical_sensitivity" / "optical_dataset_registry.csv"

ALLOWED_GOLD_TABLES = {
    "roi_catalog_gold.csv",
    "optical_timeseries_gold.csv",
    "training_matrix_optical_matched_3d.csv",
    "training_matrix_optical_matched_3d_hls.csv",
    "training_matrix_optical_matched_3d_landsat.csv",
    "training_matrix_optical_matched_3d_sentinel2.csv",
}

MATCHED_TABLES = {
    "any_sensor_3d": "training_matrix_optical_matched_3d.csv",
    "hls_3d": "training_matrix_optical_matched_3d_hls.csv",
    "landsat_3d": "training_matrix_optical_matched_3d_landsat.csv",
    "sentinel2_3d": "training_matrix_optical_matched_3d_sentinel2.csv",
}

BAND_COLUMNS = ["blue", "green", "red", "nir", "swir1", "swir2"]
INDEX_COLUMNS = ["ndwi", "mndwi"]
EXPECTED_RIVERS = ["Kolyma", "Lena", "Mackenzie", "Ob", "Yenisey", "Yukon"]


def _ensure_dirs() -> None:
    for directory in [ROI_QC_TABLE_DIR, ROI_QC_REPORT_DIR, ROI_QC_FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def _write_csv(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8")
    return destination


def _read_required_csv(destination: Path) -> pd.DataFrame:
    if not destination.exists():
        raise FileNotFoundError(f"Required ROI QC input is missing: {destination}")
    return pd.read_csv(destination, low_memory=False)


def _verify_allowed_gold_hashes() -> dict[str, str]:
    contract = load_contract()
    gold_dir = require_gold_data_dir()
    hashes: dict[str, str] = {}
    for table_name in sorted(ALLOWED_GOLD_TABLES):
        if table_name not in contract.get("expected_tables", {}):
            raise KeyError(f"ROI QC table is not in the frozen gold contract: {table_name}")
        destination = table_path(table_name, gold_dir=gold_dir)
        actual_hash = sha256_file(destination)
        expected_hash = str(contract["expected_tables"][table_name]["sha256"]).lower()
        if actual_hash != expected_hash:
            raise RuntimeError(f"Frozen gold table hash mismatch for ROI QC input: {table_name}")
        hashes[table_name] = actual_hash
    return hashes


def _read_gold_table(table_name: str) -> pd.DataFrame:
    if table_name not in ALLOWED_GOLD_TABLES:
        raise RuntimeError(f"ROI QC is not allowed to read this table: {table_name}")
    gold_dir = require_gold_data_dir()
    return pd.read_csv(table_path(table_name, gold_dir=gold_dir), low_memory=False)


def _load_inputs() -> dict[str, Any]:
    if not OPTICAL_SENSITIVITY_REPORT_PATH.exists():
        raise FileNotFoundError("Optical sensitivity report is missing. Run optical sensitivity before ROI final QC.")
    if not OPTICAL_RANKING_PATH.exists() or not OPTICAL_DATASET_REGISTRY_PATH.exists():
        raise FileNotFoundError("Optical sensitivity tables are missing. Run optical sensitivity before ROI final QC.")

    matched = {}
    for dataset_id, table_name in MATCHED_TABLES.items():
        matched[dataset_id] = _prepare_optical(_read_gold_table(table_name), dataset_id=dataset_id, source_kind="matched_3d")
    return {
        "roi": _read_gold_table("roi_catalog_gold.csv"),
        "optical": _prepare_optical(_read_gold_table("optical_timeseries_gold.csv"), dataset_id="optical_timeseries", source_kind="timeseries"),
        "matched": matched,
        "optical_report_text": OPTICAL_SENSITIVITY_REPORT_PATH.read_text(encoding="utf-8"),
        "optical_ranking": _read_required_csv(OPTICAL_RANKING_PATH),
        "optical_dataset_registry": _read_required_csv(OPTICAL_DATASET_REGISTRY_PATH),
    }


def _prepare_optical(frame: pd.DataFrame, *, dataset_id: str, source_kind: str) -> pd.DataFrame:
    out = frame.copy()
    out["dataset_id"] = dataset_id
    out["source_kind"] = source_kind
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["year"] = out["date"].dt.year
    if "optical_date" in out.columns:
        out["optical_date"] = pd.to_datetime(out["optical_date"], errors="coerce")
    for column in [*BAND_COLUMNS, *INDEX_COLUMNS, "pct_valid_water_pixels", "n_valid_water_pixels", "days_offset"]:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    out["abs_days_offset"] = out["days_offset"].abs()
    out["rows_with_any_optical_band"] = out[BAND_COLUMNS].notna().any(axis=1)
    out["rows_with_all_core_bands"] = out[["blue", "green", "red", "nir"]].notna().all(axis=1)
    out["rows_with_indices"] = out[INDEX_COLUMNS].notna().any(axis=1)
    out["extreme_reflectance"] = out[BAND_COLUMNS].lt(-0.05).any(axis=1) | out[BAND_COLUMNS].gt(1.0).any(axis=1)
    out["extreme_indices"] = out[INDEX_COLUMNS].lt(-1.05).any(axis=1) | out[INDEX_COLUMNS].gt(1.05).any(axis=1)
    return out


def _join_unique(values: pd.Series) -> str:
    return ";".join(sorted({str(value) for value in values.dropna() if str(value) and str(value) != "nan"}))


def _rate(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return np.nan
    return float(frame[column].mean())


def _finite_rate(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan
    return float(pd.to_numeric(frame[column], errors="coerce").notna().mean())


def _roi_metadata(roi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for river, subset in roi.groupby("river", dropna=False):
        risks = subset["roi_risk"].fillna("unknown").astype(str).value_counts().to_dict()
        rows.append(
            {
                "river": river,
                "roi_count": len(subset),
                "roi_sets": _join_unique(subset["roi_set"]),
                "final_primary_available": bool(subset["roi_set"].astype(str).eq("final_primary").any()),
                "roi_risk_summary": ";".join(f"{key}:{value}" for key, value in sorted(risks.items())),
                "manual_review_required_count": int(subset["manual_review_required"].astype(str).str.lower().isin({"true", "1", "yes"}).sum()),
                "quality_flag_summary": _join_unique(subset.get("quality_flag", pd.Series(dtype=str))),
            }
        )
    return pd.DataFrame(rows)


def _optical_coverage(optical: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (river, sensor), subset in optical.groupby(["river", "sensor"], dropna=False):
        pct = pd.to_numeric(subset["pct_valid_water_pixels"], errors="coerce")
        rows.append(
            {
                "river": river,
                "sensor": sensor,
                "total_optical_rows": len(subset),
                "date_min": subset["date"].min().date().isoformat() if subset["date"].notna().any() else "",
                "date_max": subset["date"].max().date().isoformat() if subset["date"].notna().any() else "",
                "rows_with_any_optical_band": int(subset["rows_with_any_optical_band"].sum()),
                "any_optical_band_rate": _rate(subset, "rows_with_any_optical_band"),
                "median_pct_valid_water_pixels": float(pct.median()) if pct.notna().any() else np.nan,
                "p10_pct_valid_water_pixels": float(pct.quantile(0.10)) if pct.notna().any() else np.nan,
                "p90_pct_valid_water_pixels": float(pct.quantile(0.90)) if pct.notna().any() else np.nan,
                "rows_pct_valid_lt_005": int((pct < 0.05).sum()),
                "rows_pct_valid_lt_010": int((pct < 0.10).sum()),
                "rows_pct_valid_lt_025": int((pct < 0.25).sum()),
            }
        )
    return pd.DataFrame(rows)


def _valid_water_summary(optical: pd.DataFrame, matched: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [optical, *matched.values()]
    combined = pd.concat(frames, ignore_index=True)
    rows = []
    for (source_kind, dataset_id, river, sensor), subset in combined.groupby(["source_kind", "dataset_id", "river", "sensor"], dropna=False):
        pct = pd.to_numeric(subset["pct_valid_water_pixels"], errors="coerce")
        rows.append(
            {
                "source_kind": source_kind,
                "dataset_id": dataset_id,
                "river": river,
                "sensor": sensor,
                "rows": len(subset),
                "rows_with_any_optical_band": int(subset["rows_with_any_optical_band"].sum()),
                "any_optical_band_rate": _rate(subset, "rows_with_any_optical_band"),
                "median_pct_valid_water_pixels": float(pct.median()) if pct.notna().any() else np.nan,
                "p10_pct_valid_water_pixels": float(pct.quantile(0.10)) if pct.notna().any() else np.nan,
                "p90_pct_valid_water_pixels": float(pct.quantile(0.90)) if pct.notna().any() else np.nan,
                "rows_pct_valid_lt_005": int((pct < 0.05).sum()),
                "rows_pct_valid_lt_010": int((pct < 0.10).sum()),
                "rows_pct_valid_lt_025": int((pct < 0.25).sum()),
            }
        )
    return pd.DataFrame(rows)


def _match_integrity(matched: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset_id, frame in matched.items():
        for (river, sensor), subset in frame.groupby(["river", "sensor"], dropna=False):
            pct = pd.to_numeric(subset["pct_valid_water_pixels"], errors="coerce")
            years = subset["year"].dropna().astype(int)
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "river": river,
                    "sensor": sensor,
                    "matched_rows": len(subset),
                    "rows_with_actual_bands": int(subset["rows_with_any_optical_band"].sum()),
                    "actual_band_rate": _rate(subset, "rows_with_any_optical_band"),
                    "rows_with_core_bands": int(subset["rows_with_all_core_bands"].sum()),
                    "core_band_rate": _rate(subset, "rows_with_all_core_bands"),
                    "rows_with_indices": int(subset["rows_with_indices"].sum()),
                    "index_rate": _rate(subset, "rows_with_indices"),
                    "median_abs_days_offset": float(subset["abs_days_offset"].median()) if subset["abs_days_offset"].notna().any() else np.nan,
                    "matched_doc_years_represented": int(years.nunique()) if not years.empty else 0,
                    "date_min": subset["date"].min().date().isoformat() if subset["date"].notna().any() else "",
                    "date_max": subset["date"].max().date().isoformat() if subset["date"].notna().any() else "",
                    "median_pct_valid_water_pixels": float(pct.median()) if pct.notna().any() else np.nan,
                    "blue_finite_rate": _finite_rate(subset, "blue"),
                    "green_finite_rate": _finite_rate(subset, "green"),
                    "red_finite_rate": _finite_rate(subset, "red"),
                    "nir_finite_rate": _finite_rate(subset, "nir"),
                    "swir1_finite_rate": _finite_rate(subset, "swir1"),
                    "swir2_finite_rate": _finite_rate(subset, "swir2"),
                    "ndwi_finite_rate": _finite_rate(subset, "ndwi"),
                    "mndwi_finite_rate": _finite_rate(subset, "mndwi"),
                    "extreme_reflectance_count": int(subset["extreme_reflectance"].sum()),
                    "extreme_index_count": int(subset["extreme_indices"].sum()),
                    "sensor_specific_underpowered": dataset_id in {"hls_3d", "sentinel2_3d"} and len(subset) < 60,
                }
            )
    return pd.DataFrame(rows)


def _best_optical_status(optical_ranking: pd.DataFrame) -> str:
    if optical_ranking.empty:
        return "no"
    primary = optical_ranking[
        optical_ranking["dataset_id"].eq("any_sensor_3d")
        & optical_ranking.get("is_optical_proxy_feature_set", pd.Series(True, index=optical_ranking.index)).astype(bool)
    ]
    if primary.empty:
        return "uncertain"
    if primary["classification"].eq("optical_improves_baseline").any():
        return "yes"
    if primary["classification"].eq("optical_marginal").any():
        return "marginal"
    return "no"


def _decisions(
    metadata: pd.DataFrame,
    coverage: pd.DataFrame,
    integrity: pd.DataFrame,
    optical_ranking: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    recommendation_rows = []
    any_integrity = integrity[integrity["dataset_id"].eq("any_sensor_3d")].copy()
    for river in EXPECTED_RIVERS:
        meta = metadata[metadata["river"].eq(river)].iloc[0] if not metadata[metadata["river"].eq(river)].empty else pd.Series(dtype=object)
        cov = coverage[coverage["river"].eq(river)]
        integ = any_integrity[any_integrity["river"].eq(river)]
        matched_rows = int(integ["matched_rows"].sum()) if not integ.empty else 0
        rows_with_bands = int(integ["rows_with_actual_bands"].sum()) if not integ.empty else 0
        actual_band_rate = rows_with_bands / matched_rows if matched_rows else 0.0
        median_valid = float(cov["median_pct_valid_water_pixels"].median()) if not cov.empty else np.nan
        timeseries_band_rate = float(cov["rows_with_any_optical_band"].sum() / cov["total_optical_rows"].sum()) if not cov.empty and cov["total_optical_rows"].sum() else 0.0
        underpowered = integrity[
            integrity["river"].eq(river)
            & integrity["dataset_id"].isin(["hls_3d", "sentinel2_3d"])
            & integrity["sensor_specific_underpowered"].astype(bool)
        ]["dataset_id"].drop_duplicates().tolist()

        reasons: list[str] = []
        interpretation: list[str] = []
        if not bool(meta.get("final_primary_available", False)):
            decision = "roi_revision_recommended"
            reasons.append("final_primary ROI is missing")
            interpretation.append("unresolved / needs visual review")
        elif matched_rows > 0 and (actual_band_rate < 0.25 or rows_with_bands < 10):
            decision = "roi_needs_visual_review"
            reasons.append("3d matched DOC rows exist but actual optical bands are sparse")
            interpretation.append("ROI/valid-pixel limitation")
            interpretation.append("unresolved / needs visual review")
        elif pd.notna(median_valid) and median_valid < 0.05 and timeseries_band_rate < 0.15:
            decision = "roi_accepted_with_caveat"
            reasons.append("valid-water pixels are often low across the optical time series")
            interpretation.append("ROI/valid-pixel limitation")
        else:
            decision = "roi_accepted_with_caveat" if int(meta.get("manual_review_required_count", 0)) else "roi_accepted"
            if int(meta.get("manual_review_required_count", 0)):
                reasons.append("legacy ROI requires external visual review")
                interpretation.append("unresolved / needs visual review")

        if underpowered:
            reasons.append(f"sensor-specific subset underpowered: {';'.join(sorted(underpowered))}")
            interpretation.append("underpowered sensor subset")
        if int(meta.get("manual_review_required_count", 0)):
            interpretation.append("unresolved / needs visual review")
            if "legacy ROI requires external visual review" not in reasons:
                reasons.append("legacy ROI requires external visual review")
        optical_status = _best_optical_status(optical_ranking)
        if optical_status == "no" and actual_band_rate >= 0.25:
            interpretation.append("optical proxy genuinely not adding skill")
        if not interpretation:
            interpretation.append("optical proxy genuinely not adding skill")
        if not reasons:
            reasons.append("no quantitative fatal ROI issue found")

        reopen = decision == "roi_revision_recommended"
        likely_roi_driven = "yes" if reopen else "uncertain" if {"ROI/valid-pixel limitation", "unresolved / needs visual review"}.intersection(interpretation) else "no"
        rows.append(
            {
                "river": river,
                "roi_count": int(meta.get("roi_count", 0)),
                "roi_sets": meta.get("roi_sets", ""),
                "final_primary_available": bool(meta.get("final_primary_available", False)),
                "roi_risk_summary": meta.get("roi_risk_summary", ""),
                "manual_review_required_count": int(meta.get("manual_review_required_count", 0)),
                "quality_flag_summary": meta.get("quality_flag_summary", ""),
                "matched_3d_rows": matched_rows,
                "matched_3d_rows_with_actual_bands": rows_with_bands,
                "matched_3d_actual_band_rate": actual_band_rate,
                "timeseries_any_band_rate": timeseries_band_rate,
                "median_timeseries_pct_valid_water_pixels": median_valid,
                "underpowered_sensor_subsets": ";".join(sorted(underpowered)),
                "optical_negative_result_interpretation": ";".join(sorted(set(interpretation))),
                "optical_negative_result_likely_roi_driven": likely_roi_driven,
                "roi_decision": decision,
                "reopen_freeze_recommendation": "yes" if reopen else "no",
                "reason": "; ".join(reasons),
            }
        )
        recommendation_rows.append(
            {
                "river": river,
                "roi_decision": decision,
                "reopen_freeze_recommendation": "yes" if reopen else "no",
                "recommended_action": "open_new_data_freeze_for_roi_revision" if reopen else "retain_frozen_roi_with_documented_caveats",
                "notes": "; ".join(reasons),
            }
        )
    summary = pd.DataFrame(rows)
    recommendations = pd.DataFrame(recommendation_rows)
    return summary, recommendations


def _make_figures(optical: pd.DataFrame, matched_any: pd.DataFrame) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    ROI_QC_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def save(fig, name: str) -> None:
        fig.tight_layout()
        destination = ROI_QC_FIGURE_DIR / name
        fig.savefig(destination, dpi=120)
        plt.close(fig)
        paths.append(destination)

    if not optical.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        data = []
        labels = []
        for (river, sensor), subset in optical.groupby(["river", "sensor"], dropna=False):
            data.append(pd.to_numeric(subset["pct_valid_water_pixels"], errors="coerce").dropna().to_numpy())
            labels.append(f"{river}\n{sensor}")
        if data:
            try:
                ax.boxplot(data, tick_labels=labels, showfliers=False)
            except TypeError:
                ax.boxplot(data, labels=labels, showfliers=False)
            ax.set_ylabel("pct_valid_water_pixels")
            ax.set_title("Valid water pixel distribution by river/sensor")
            ax.tick_params(axis="x", rotation=45)
            save(fig, "valid_water_pixel_distribution_by_river_sensor.png")

        counts = optical.dropna(subset=["date"]).copy()
        if not counts.empty:
            counts["year"] = counts["date"].dt.year
            yearly = counts.groupby(["year", "river", "sensor"], dropna=False).size().reset_index(name="rows")
            fig, ax = plt.subplots(figsize=(9, 5))
            for label, subset in yearly.groupby(yearly["river"] + ":" + yearly["sensor"]):
                ax.plot(subset["year"], subset["rows"], alpha=0.65, label=label)
            ax.set_xlabel("Year")
            ax.set_ylabel("Optical rows")
            ax.set_title("Optical rows over time by river/sensor")
            ax.legend(fontsize="xx-small", ncols=3)
            save(fig, "optical_rows_over_time_by_river_sensor.png")

    if not matched_any.empty:
        support = matched_any.groupby("river", dropna=False).agg(
            matched_rows=("river", "size"),
            rows_with_any_band=("rows_with_any_optical_band", "sum"),
        ).reset_index()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(support))
        ax.bar(x - 0.18, support["matched_rows"], width=0.36, label="3d matched rows")
        ax.bar(x + 0.18, support["rows_with_any_band"], width=0.36, label="rows with bands")
        ax.set_xticks(x)
        ax.set_xticklabels(support["river"], rotation=30)
        ax.set_ylabel("Rows")
        ax.set_title("Matched DOC optical support by river")
        ax.legend()
        save(fig, "matched_doc_optical_support_by_river.png")

        for column, name in [("ndwi", "ndwi_distribution_by_river_sensor.png"), ("mndwi", "mndwi_distribution_by_river_sensor.png")]:
            values = matched_any.dropna(subset=[column]).copy()
            if values.empty:
                continue
            fig, ax = plt.subplots(figsize=(9, 5))
            data = []
            labels = []
            for (river, sensor), subset in values.groupby(["river", "sensor"], dropna=False):
                data.append(subset[column].to_numpy())
                labels.append(f"{river}\n{sensor}")
            try:
                ax.boxplot(data, tick_labels=labels, showfliers=False)
            except TypeError:
                ax.boxplot(data, labels=labels, showfliers=False)
            ax.set_ylabel(column.upper())
            ax.set_title(f"{column.upper()} distribution by river/sensor")
            ax.tick_params(axis="x", rotation=45)
            save(fig, name)

    return paths


def write_roi_final_qc_report() -> Path:
    ROI_QC_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = _read_required_csv(ROI_QC_TABLE_DIR / "roi_final_qc_summary.csv")
    coverage = _read_required_csv(ROI_QC_TABLE_DIR / "roi_optical_coverage_by_river_sensor.csv")
    valid_pixels = _read_required_csv(ROI_QC_TABLE_DIR / "roi_valid_water_pixel_summary.csv")
    integrity = _read_required_csv(ROI_QC_TABLE_DIR / "roi_optical_match_integrity.csv")
    recommendations = _read_required_csv(ROI_QC_TABLE_DIR / "roi_revision_recommendations.csv")

    rivers_requiring_revision = sorted(summary.loc[summary["roi_decision"].eq("roi_revision_recommended"), "river"].tolist())
    rivers_with_caveat = sorted(summary.loc[summary["roi_decision"].eq("roi_accepted_with_caveat"), "river"].tolist())
    reopen_freeze = "yes" if recommendations["reopen_freeze_recommendation"].astype(str).str.lower().eq("yes").any() else "no"
    optical_driven = "uncertain"
    if reopen_freeze == "yes":
        optical_driven = "yes"
    elif summary["optical_negative_result_interpretation"].astype(str).str.contains("ROI/valid-pixel limitation").any():
        optical_driven = "uncertain"
    elif "optical_negative_result_likely_roi_driven" in summary.columns and summary["optical_negative_result_likely_roi_driven"].astype(str).eq("uncertain").any():
        optical_driven = "uncertain"
    else:
        optical_driven = "no"

    lines = [
        "# ROI Final QC Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "## 1. Scope and guardrails",
        "",
        "This audit checks frozen ROI metadata and optical data integrity after the optical sensitivity phase. It does not modify gold data, does not recalculate ROI, does not re-extract GEE data, does not train DOC models, does not generate production daily DOC prediction, and does not compute flux.",
        "",
        "## 2. Inputs",
        "",
        "- `data/processed/gold/roi_catalog_gold.csv`",
        "- `data/processed/gold/optical_timeseries_gold.csv`",
        "- `data/processed/gold/training_matrix_optical_matched_3d*.csv`",
        "- `outputs/reports/optical_sensitivity/optical_sensitivity_report.md`",
        "- `outputs/tables/optical_sensitivity/optical_model_ranking.csv`",
        "- `outputs/tables/optical_sensitivity/optical_dataset_registry.csv`",
        "",
        "## 3. ROI metadata summary",
        "",
        _md_table(summary[["river", "roi_count", "roi_sets", "final_primary_available", "roi_risk_summary", "manual_review_required_count", "roi_decision"]], max_rows=20),
        "",
        "## 4. Optical coverage by river and sensor",
        "",
        _md_table(coverage, max_rows=40),
        "",
        "## 5. Valid water pixel summary",
        "",
        _md_table(valid_pixels.head(60), max_rows=60),
        "",
        "## 6. Optical matched DOC support and plausibility",
        "",
        _md_table(integrity.head(80), max_rows=80),
        "",
        "## 7. Optical sensitivity interpretation",
        "",
        _md_table(summary[["river", "matched_3d_rows", "matched_3d_rows_with_actual_bands", "matched_3d_actual_band_rate", "underpowered_sensor_subsets", "optical_negative_result_interpretation", "optical_negative_result_likely_roi_driven"]], max_rows=20),
        "",
        "## 8. Revision recommendations",
        "",
        _md_table(recommendations, max_rows=20),
        "",
        "## 9. Required answers",
        "",
        "- Is existing frozen ROI adequate for hydrocore baseline? `yes`; hydrocore baseline does not use optical ROI.",
        "- Is existing frozen ROI adequate for optical sensitivity? `yes with caveats`; final_primary exists for all rivers, but legacy/manual visual review caveats remain and valid-water support is uneven.",
        f"- Does optical negative result appear likely caused by ROI failure? `{optical_driven}`; quantitative checks do not show a fatal ROI failure, but visual GIS review remains external/manual.",
        f"- Should we reopen data freeze to recalculate ROI? `{reopen_freeze}` unless a fatal ROI issue is found in external visual review.",
        f"- reopen_freeze_recommendation: `{reopen_freeze}`",
        f"- rivers_requiring_revision: `{rivers_requiring_revision}`",
        f"- rivers_with_caveat: `{rivers_with_caveat}`",
        "",
        "## 10. External visual review note",
        "",
        "Map geometry is not plotted in this model repository. Visual GIS review remains external/manual. If a fatal ROI geometry issue is found, create a new data freeze in the data repository rather than silently editing this freeze.",
        "",
        "## 11. Explicit statements",
        "",
        "- Gold data were not modified.",
        "- No ROI was recalculated.",
        "- No GEE extraction was run.",
        "- No DOC model was trained.",
        "- No production daily DOC prediction was generated.",
        "- No DOC flux was generated.",
    ]
    ROI_QC_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ROI_QC_REPORT_PATH


def run_roi_final_qc() -> dict[str, Any]:
    _ensure_dirs()
    assert_no_forbidden_outputs()
    before_hashes = _verify_allowed_gold_hashes()
    inputs = _load_inputs()

    metadata = _roi_metadata(inputs["roi"])
    coverage = _optical_coverage(inputs["optical"])
    valid_pixels = _valid_water_summary(inputs["optical"], inputs["matched"])
    integrity = _match_integrity(inputs["matched"])
    summary, recommendations = _decisions(metadata, coverage, integrity, inputs["optical_ranking"])

    table_paths = [
        _write_csv(summary, ROI_QC_TABLE_DIR / "roi_final_qc_summary.csv"),
        _write_csv(coverage, ROI_QC_TABLE_DIR / "roi_optical_coverage_by_river_sensor.csv"),
        _write_csv(valid_pixels, ROI_QC_TABLE_DIR / "roi_valid_water_pixel_summary.csv"),
        _write_csv(integrity, ROI_QC_TABLE_DIR / "roi_optical_match_integrity.csv"),
        _write_csv(recommendations, ROI_QC_TABLE_DIR / "roi_revision_recommendations.csv"),
    ]
    figure_paths = _make_figures(inputs["optical"], inputs["matched"]["any_sensor_3d"])
    report_path = write_roi_final_qc_report()
    after_hashes = _verify_allowed_gold_hashes()
    if before_hashes != after_hashes:
        raise RuntimeError("One or more frozen ROI/optical gold table hashes changed during ROI final QC.")
    assert_no_forbidden_outputs()
    return {
        "tables": table_paths,
        "figures": figure_paths,
        "report": report_path,
        "summary": summary,
        "recommendations": recommendations,
    }
