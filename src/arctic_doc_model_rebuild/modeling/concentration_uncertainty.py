from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression

from ..gold_contract import load_contract, require_gold_data_dir, sha256_file, table_path
from ..paths import CONFIG_DIR, REPORT_DIR, TABLE_DIR, path
from ..reports import _md_table, utc_now
from .baseline_models import MODEL_SPECS, _make_estimator, _prepare_hydrocore, _read_hydrocore
from .diagnostics import assert_no_forbidden_outputs
from .feature_sets import TARGET_COLUMN, FeatureSet, get_feature_set
from .metrics import metric_row
from .production_readiness import production_readiness_decision
from .residual_diagnostics import grouped_residual_summary, high_doc_residual_review
from .target_transforms import inverse_target, target_values
from .uncertainty_intervals import empirical_interval_coverage, empirical_residual_intervals
from .validation import validation_splits


UNCERTAINTY_TABLE_DIR = TABLE_DIR / "concentration_uncertainty"
UNCERTAINTY_REPORT_DIR = REPORT_DIR / "concentration_uncertainty"
UNCERTAINTY_FIGURE_DIR = path("outputs", "figures", "concentration_uncertainty")
UNCERTAINTY_REPORT_PATH = UNCERTAINTY_REPORT_DIR / "concentration_uncertainty_report.md"

BASELINE_DECISION_PATH = TABLE_DIR / "baseline_final" / "baseline_model_decision.csv"
PRIMARY_SPEC_PATH = CONFIG_DIR / "model_specs" / "primary_baseline_f3_q_season_river_fixed_ridge_alpha_1.yaml"
HYDROCLIMATE_SPEC_PATH = CONFIG_DIR / "model_specs" / "hydroclimate_extension_f6_ridge_alpha_1.yaml"
REFINED_RECOMMENDATION_PATH = TABLE_DIR / "baseline_refinement" / "refined_model_recommendation.csv"
SAME_SAMPLE_METRICS_PATH = TABLE_DIR / "baseline_refinement" / "same_sample_ablation_metrics.csv"
LOG_TARGET_METRICS_PATH = TABLE_DIR / "baseline_refinement" / "log_target_sensitivity_metrics.csv"
OPTICAL_RANKING_PATH = TABLE_DIR / "optical_sensitivity" / "optical_model_ranking.csv"
OPTICAL_REPORT_PATH = REPORT_DIR / "optical_sensitivity" / "optical_sensitivity_report.md"
ROI_QC_SUMMARY_PATH = TABLE_DIR / "roi_qc" / "roi_final_qc_summary.csv"
ROI_QC_REPORT_PATH = REPORT_DIR / "roi_qc" / "roi_final_qc_report.md"

ALLOWED_METADATA_PATHS = [
    BASELINE_DECISION_PATH,
    PRIMARY_SPEC_PATH,
    HYDROCLIMATE_SPEC_PATH,
    REFINED_RECOMMENDATION_PATH,
    SAME_SAMPLE_METRICS_PATH,
    LOG_TARGET_METRICS_PATH,
    OPTICAL_RANKING_PATH,
    OPTICAL_REPORT_PATH,
    ROI_QC_SUMMARY_PATH,
    ROI_QC_REPORT_PATH,
]
BOOTSTRAP_N = 200


def _ensure_dirs() -> None:
    for directory in [UNCERTAINTY_TABLE_DIR, UNCERTAINTY_REPORT_DIR, UNCERTAINTY_FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def _write_csv(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8")
    return destination


def _read_required_csv(destination: Path) -> pd.DataFrame:
    if not destination.exists():
        raise FileNotFoundError(f"Required concentration uncertainty input is missing: {destination}")
    return pd.read_csv(destination)


def _read_required_text(destination: Path) -> str:
    if not destination.exists():
        raise FileNotFoundError(f"Required concentration uncertainty input is missing: {destination}")
    return destination.read_text(encoding="utf-8")


def _load_required_metadata() -> dict[str, Any]:
    for destination in ALLOWED_METADATA_PATHS:
        if not destination.exists():
            raise FileNotFoundError(f"Required concentration uncertainty metadata is missing: {destination}")
    baseline_decision = _read_required_csv(BASELINE_DECISION_PATH)
    primary = baseline_decision[baseline_decision["decision_type"].eq("primary_baseline")]
    if primary.empty or primary.iloc[0]["feature_set"] != "F3_q_season_river_fixed" or primary.iloc[0]["model_id"] != "ridge_alpha_1":
        raise RuntimeError("Primary baseline decision must be F3_q_season_river_fixed + ridge_alpha_1.")
    with PRIMARY_SPEC_PATH.open("r", encoding="utf-8") as handle:
        primary_spec = yaml.safe_load(handle)
    with HYDROCLIMATE_SPEC_PATH.open("r", encoding="utf-8") as handle:
        hydro_spec = yaml.safe_load(handle)
    if primary_spec.get("production_prediction_allowed") is not False or hydro_spec.get("production_prediction_allowed") is not False:
        raise RuntimeError("Model specs must disallow production prediction in this phase.")
    return {
        "baseline_decision": baseline_decision,
        "primary_spec": primary_spec,
        "hydro_spec": hydro_spec,
        "refined_recommendation": _read_required_csv(REFINED_RECOMMENDATION_PATH),
        "same_sample_metrics": _read_required_csv(SAME_SAMPLE_METRICS_PATH),
        "log_target_metrics": _read_required_csv(LOG_TARGET_METRICS_PATH),
        "optical_ranking": _read_required_csv(OPTICAL_RANKING_PATH),
        "optical_report": _read_required_text(OPTICAL_REPORT_PATH),
        "roi_qc": _read_required_csv(ROI_QC_SUMMARY_PATH),
        "roi_report": _read_required_text(ROI_QC_REPORT_PATH),
    }


def _verify_contract_snapshot() -> pd.DataFrame:
    verification_path = TABLE_DIR / "gold_table_verification.csv"
    if not verification_path.exists():
        raise FileNotFoundError("Run `python -m arctic_doc_model_rebuild.cli verify-gold-data` before concentration uncertainty.")
    verification = pd.read_csv(verification_path)
    hydro = verification[verification["table_name"].eq("training_matrix_hydrocore.csv")]
    if hydro.empty or hydro.iloc[0]["status"] != "ok":
        raise RuntimeError("training_matrix_hydrocore.csv is not verified in the current gold contract snapshot.")
    return verification


def _read_hydrocore_only() -> tuple[pd.DataFrame, Path]:
    gold_dir = require_gold_data_dir()
    destination = table_path("training_matrix_hydrocore.csv", gold_dir=gold_dir)
    if destination.name != "training_matrix_hydrocore.csv":
        raise RuntimeError(f"Concentration uncertainty cannot read this gold table: {destination.name}")
    contract = load_contract()
    expected_hash = str(contract["expected_tables"]["training_matrix_hydrocore.csv"]["sha256"]).lower()
    if sha256_file(destination) != expected_hash:
        raise RuntimeError("training_matrix_hydrocore.csv hash does not match frozen contract.")
    return _prepare_hydrocore(_read_hydrocore()), destination


def _ridge_alpha_1():
    for spec in MODEL_SPECS:
        if spec.model_id == "ridge_alpha_1":
            return spec
    raise KeyError("ridge_alpha_1 model spec not found")


def _model_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_role": "primary_baseline",
                "feature_set": "F3_q_season_river_fixed",
                "model_id": "ridge_alpha_1",
                "target_scale": "raw",
                "selected_as": "primary_baseline",
                "production_prediction_allowed": False,
                "flux_allowed": False,
            },
            {
                "model_role": "hydroclimate_extension",
                "feature_set": "F6_reduced_hydroclimate_river_fixed",
                "model_id": "ridge_alpha_1",
                "target_scale": "raw",
                "selected_as": "extension_not_primary",
                "production_prediction_allowed": False,
                "flux_allowed": False,
            },
            {
                "model_role": "log_target_sensitivity",
                "feature_set": "F6_reduced_hydroclimate_river_fixed",
                "model_id": "ridge_alpha_1",
                "target_scale": "log",
                "selected_as": "sensitivity_candidate_only",
                "production_prediction_allowed": False,
                "flux_allowed": False,
            },
        ]
    )


def _usable(frame: pd.DataFrame, feature_set: FeatureSet, target_scale: str) -> pd.DataFrame:
    required = [TARGET_COLUMN, "label_id", "river", "date", "year", "month", "Q_m3s", *feature_set.required_features]
    required = list(dict.fromkeys(required))
    usable = frame.dropna(subset=[column for column in required if column in frame.columns]).copy()
    usable = usable[usable["Q_m3s"] > 0].copy()
    if target_scale == "log":
        usable = usable[pd.to_numeric(usable[TARGET_COLUMN], errors="coerce") > 0].copy()
    return usable.reset_index(drop=True)


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, feature_set: FeatureSet, target_scale: str) -> tuple[np.ndarray, np.ndarray]:
    estimator = _make_estimator(_ridge_alpha_1(), feature_set)
    estimator.fit(train[list(feature_set.required_features)], target_values(train, target_scale))
    pred_target = estimator.predict(test[list(feature_set.required_features)])
    pred_mg = inverse_target(pred_target, target_scale)
    return np.asarray(pred_mg, dtype=float), np.asarray(pred_target, dtype=float)


def _run_model_cv(frame: pd.DataFrame, *, model_role: str, feature_set: FeatureSet, target_scale: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = _usable(frame, feature_set, target_scale)
    cv_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for scheme, fold_id, train_idx, test_idx, fold_info in validation_splits(usable):
        train = usable.iloc[train_idx].copy()
        test = usable.iloc[test_idx].copy()
        pred_mg, pred_target = _fit_predict(train, test, feature_set, target_scale)
        fold_metric = metric_row(test[TARGET_COLUMN], pred_mg)
        fold_rows.append(
            {
                "model_role": model_role,
                "feature_set": feature_set.feature_set,
                "model_id": "ridge_alpha_1",
                "target_scale": target_scale,
                "validation_scheme": scheme,
                "fold_id": fold_id,
                "fold_group": fold_info.get("fold_group", ""),
                "fold_year": fold_info.get("fold_year", ""),
                "fold_river": fold_info.get("fold_river", ""),
                "n_train": len(train),
                "n_test": len(test),
                "rivers_in_fold": ";".join(sorted(test["river"].astype(str).unique())),
                "rmse": fold_metric["rmse"],
                "mae": fold_metric["mae"],
                "bias_mean": fold_metric["bias_mean"],
                "bias_median": fold_metric["bias_median"],
                "r2": fold_metric["r2"],
            }
        )
        for row, predicted_mg, predicted_target in zip(test.to_dict("records"), pred_mg, pred_target):
            observed = float(row[TARGET_COLUMN])
            output = {
                "label_id": row["label_id"],
                "river": row["river"],
                "date": pd.Timestamp(row["date"]).date().isoformat(),
                "year": int(row["year"]) if pd.notna(row["year"]) else "",
                "month": int(row["month"]) if pd.notna(row["month"]) else "",
                "Q_m3s": row.get("Q_m3s", np.nan),
                "DOC_observed_mgC_L": observed,
                "DOC_cv_predicted_mgC_L": float(predicted_mg),
                "residual_mgC_L": observed - float(predicted_mg),
                "model_role": model_role,
                "feature_set": feature_set.feature_set,
                "model_id": "ridge_alpha_1",
                "target_scale": target_scale,
                "validation_scheme": scheme,
                "fold_id": fold_id,
                "is_cv_prediction": True,
                "is_production_prediction": False,
            }
            if target_scale == "log":
                output["DOC_cv_predicted_log"] = float(predicted_target)
            cv_rows.append(output)
    return pd.DataFrame(cv_rows), pd.DataFrame(fold_rows)


def _run_all_cv(hydrocore: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = [
        ("primary_baseline", get_feature_set("F3_q_season_river_fixed"), "raw"),
        ("hydroclimate_extension", get_feature_set("F6_reduced_hydroclimate_river_fixed"), "raw"),
        ("log_target_sensitivity", get_feature_set("F6_reduced_hydroclimate_river_fixed"), "log"),
    ]
    pred_frames = []
    fold_frames = []
    for model_role, feature_set, target_scale in specs:
        predictions, folds = _run_model_cv(hydrocore, model_role=model_role, feature_set=feature_set, target_scale=target_scale)
        pred_frames.append(predictions)
        fold_frames.append(folds)
    return pd.concat(pred_frames, ignore_index=True), pd.concat(fold_frames, ignore_index=True)


def _overall_lookup(predictions: pd.DataFrame) -> dict[str, float]:
    rows = {}
    loyo = predictions[predictions["validation_scheme"].eq("leave_one_year_out")]
    for model_role, subset in loyo.groupby("model_role", dropna=False):
        rows[model_role] = metric_row(subset["DOC_observed_mgC_L"], subset["DOC_cv_predicted_mgC_L"])["rmse"]
    return rows


def _fold_stability(folds: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    overall = _overall_lookup(predictions)
    rows = []
    loyo = folds[folds["validation_scheme"].eq("leave_one_year_out") & folds["model_role"].isin(["primary_baseline", "hydroclimate_extension"])].copy()
    for record in loyo.to_dict("records"):
        overall_rmse = overall.get(record["model_role"], np.nan)
        high_rmse = bool(pd.notna(overall_rmse) and record["rmse"] > overall_rmse * 1.5)
        high_bias = bool(abs(record["bias_mean"]) > 1.0)
        small_n = bool(int(record["n_test"]) < 5)
        rows.append(
            {
                **record,
                "overall_rmse": overall_rmse,
                "small_fold_n": small_n,
                "high_rmse": high_rmse,
                "high_bias": high_bias,
                "unstable_fold": high_rmse or high_bias,
            }
        )
    return pd.DataFrame(rows)


def _bias_summary(predictions: pd.DataFrame, group_column: str, overall_rmse: float, roi_qc: pd.DataFrame | None = None) -> pd.DataFrame:
    primary = predictions[predictions["model_role"].eq("primary_baseline") & predictions["validation_scheme"].eq("leave_one_year_out")].copy()
    rows = []
    for group, subset in primary.groupby(group_column, dropna=False):
        metrics = metric_row(subset["DOC_observed_mgC_L"], subset["DOC_cv_predicted_mgC_L"])
        residual = pd.to_numeric(subset["residual_mgC_L"], errors="coerce").dropna()
        row = {
            group_column: group,
            "n": len(subset),
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "bias_mean": metrics["bias_mean"],
            "bias_median": metrics["bias_median"],
            "p05_residual": float(residual.quantile(0.05)) if not residual.empty else np.nan,
            "p95_residual": float(residual.quantile(0.95)) if not residual.empty else np.nan,
            "flag_abs_bias_gt_1": abs(metrics["bias_mean"]) > 1.0,
            "flag_rmse_gt_1_25x_overall": metrics["rmse"] > overall_rmse * 1.25 if pd.notna(overall_rmse) else False,
            "flag_n_lt_20": len(subset) < 20,
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    if group_column == "river" and roi_qc is not None and not roi_qc.empty:
        roi_cols = roi_qc[["river", "roi_decision", "reason", "reopen_freeze_recommendation"]].rename(
            columns={"reason": "roi_caveat", "reopen_freeze_recommendation": "roi_reopen_freeze_recommended"}
        )
        out = out.merge(roi_cols, on="river", how="left")
    return out


def _calibration_diagnostics(primary_loyo: pd.DataFrame) -> pd.DataFrame:
    subset = primary_loyo[["DOC_observed_mgC_L", "DOC_cv_predicted_mgC_L"]].dropna()
    if len(subset) < 2:
        return pd.DataFrame()
    x = subset[["DOC_cv_predicted_mgC_L"]].to_numpy()
    y = subset["DOC_observed_mgC_L"].to_numpy()
    reg = LinearRegression().fit(x, y)
    pred = reg.predict(x)
    return pd.DataFrame(
        [
            {
                "model_role": "primary_baseline",
                "diagnostic": "observed_DOC_equals_a_plus_b_cv_predicted_DOC",
                "calibration_intercept": float(reg.intercept_),
                "calibration_slope": float(reg.coef_[0]),
                "calibration_r2": metric_row(y, pred)["r2"],
                "n": len(subset),
                "mean_prediction": float(subset["DOC_cv_predicted_mgC_L"].mean()),
                "mean_observation": float(subset["DOC_observed_mgC_L"].mean()),
                "prediction_std": float(subset["DOC_cv_predicted_mgC_L"].std(ddof=1)),
                "observation_std": float(subset["DOC_observed_mgC_L"].std(ddof=1)),
                "corrective_model_used": False,
            }
        ]
    )


def _clean_feature_name(name: str) -> str:
    return name.split("__", 1)[-1].replace("river_", "river=")


def _bootstrap_coefficients(hydrocore: pd.DataFrame, *, n_bootstrap: int = BOOTSTRAP_N) -> pd.DataFrame:
    rng = np.random.default_rng(31415)
    feature_set = get_feature_set("F3_q_season_river_fixed")
    usable = _usable(hydrocore, feature_set, "raw")
    coeffs: dict[str, list[float]] = {}
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, len(usable), len(usable))
        sample = usable.iloc[sample_idx].copy()
        estimator = _make_estimator(_ridge_alpha_1(), feature_set)
        estimator.fit(sample[list(feature_set.required_features)], sample[TARGET_COLUMN])
        names = [_clean_feature_name(name) for name in estimator.named_steps["preprocess"].get_feature_names_out()]
        values = np.asarray(estimator.named_steps["model"].coef_, dtype=float)
        for name, value in zip(names, values):
            coeffs.setdefault(name, []).append(float(value))
    rows = []
    for feature, values in coeffs.items():
        series = pd.Series(values, dtype=float)
        positive_rate = float((series > 0).mean())
        negative_rate = float((series < 0).mean())
        if positive_rate >= 0.95:
            sign = "stable_positive"
        elif negative_rate >= 0.95:
            sign = "stable_negative"
        else:
            sign = "mixed"
        rows.append(
            {
                "feature": feature,
                "coef_mean": float(series.mean()),
                "coef_std": float(series.std(ddof=1)),
                "coef_p05": float(series.quantile(0.05)),
                "coef_p50": float(series.quantile(0.50)),
                "coef_p95": float(series.quantile(0.95)),
                "sign_stability": sign,
                "n_bootstrap": n_bootstrap,
            }
        )
    return pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)


def _make_figures(
    *,
    predictions: pd.DataFrame,
    residual_summary: pd.DataFrame,
    fold_stability: pd.DataFrame,
    high_doc: pd.DataFrame,
    calibration: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    UNCERTAINTY_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def save(fig, name: str) -> None:
        fig.tight_layout()
        destination = UNCERTAINTY_FIGURE_DIR / name
        fig.savefig(destination, dpi=120)
        plt.close(fig)
        paths.append(destination)

    primary = predictions[predictions["model_role"].eq("primary_baseline") & predictions["validation_scheme"].eq("leave_one_year_out")].copy()
    if not primary.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(primary["DOC_observed_mgC_L"], primary["DOC_cv_predicted_mgC_L"], s=16, alpha=0.6)
        lo = min(primary["DOC_observed_mgC_L"].min(), primary["DOC_cv_predicted_mgC_L"].min())
        hi = max(primary["DOC_observed_mgC_L"].max(), primary["DOC_cv_predicted_mgC_L"].max())
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlabel("Observed DOC mg C/L")
        ax.set_ylabel("Validation-only CV predicted DOC mg C/L")
        ax.set_title("Primary observed vs CV predicted")
        save(fig, "primary_observed_vs_cv_predicted.png")

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(primary["residual_mgC_L"].dropna(), bins=25)
        ax.axvline(0, linestyle="--")
        ax.set_xlabel("Residual mg C/L")
        ax.set_ylabel("Count")
        ax.set_title("Primary LOYO residual distribution")
        save(fig, "primary_residual_histogram.png")

        for group_column, name in [("river", "residuals_by_river.png"), ("year", "residuals_by_year.png"), ("month", "residuals_by_month.png")]:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            data = [subset["residual_mgC_L"].to_numpy() for _, subset in primary.groupby(group_column)]
            labels = [str(group) for group, _ in primary.groupby(group_column)]
            if data:
                try:
                    ax.boxplot(data, tick_labels=labels, showfliers=False)
                except TypeError:
                    ax.boxplot(data, labels=labels, showfliers=False)
                ax.axhline(0, linestyle="--")
                ax.set_ylabel("Residual mg C/L")
                ax.set_title(f"Primary residuals by {group_column}")
                ax.tick_params(axis="x", rotation=45)
                save(fig, name)

        if not calibration.empty:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(primary["DOC_cv_predicted_mgC_L"], primary["DOC_observed_mgC_L"], s=16, alpha=0.6)
            x = np.linspace(primary["DOC_cv_predicted_mgC_L"].min(), primary["DOC_cv_predicted_mgC_L"].max(), 100)
            intercept = float(calibration["calibration_intercept"].iloc[0])
            slope = float(calibration["calibration_slope"].iloc[0])
            ax.plot(x, intercept + slope * x, label="diagnostic calibration")
            ax.plot(x, x, linestyle="--", label="1:1")
            ax.set_xlabel("CV predicted DOC mg C/L")
            ax.set_ylabel("Observed DOC mg C/L")
            ax.set_title("Calibration diagnostic")
            ax.legend(fontsize="small")
            save(fig, "calibration_plot.png")

    if not high_doc.empty:
        plot = high_doc[high_doc["target_scale"].isin(["raw", "log"])].copy()
        fig, ax = plt.subplots(figsize=(8, 4.8))
        labels = plot["model_role"] + ":" + plot["doc_behavior_group"]
        ax.barh(labels, plot["bias_mean"])
        ax.axvline(0, linestyle="--")
        ax.set_xlabel("Mean residual")
        ax.set_title("High-DOC residual comparison")
        save(fig, "high_doc_residual_comparison.png")

    if not fold_stability.empty:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for role, subset in fold_stability.groupby("model_role"):
            ax.plot(subset["fold_year"].astype(str), subset["rmse"], marker="o", label=role)
        ax.set_xlabel("Held-out year")
        ax.set_ylabel("Fold RMSE")
        ax.set_title("LOYO fold stability")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize="small")
        save(fig, "fold_stability_rmse.png")

    if not bootstrap.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot = bootstrap.sort_values("coef_mean")
        ax.errorbar(plot["coef_mean"], plot["feature"], xerr=[plot["coef_mean"] - plot["coef_p05"], plot["coef_p95"] - plot["coef_mean"]], fmt="o")
        ax.axvline(0, linestyle="--")
        ax.set_xlabel("Bootstrap coefficient")
        ax.set_title("Primary model bootstrap coefficient stability")
        save(fig, "bootstrap_coefficients.png")
    return paths


def write_concentration_uncertainty_report() -> Path:
    UNCERTAINTY_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    contract = load_contract()
    registry = _read_required_csv(UNCERTAINTY_TABLE_DIR / "uncertainty_model_registry.csv")
    residuals = _read_required_csv(UNCERTAINTY_TABLE_DIR / "residual_distribution_summary.csv")
    intervals = _read_required_csv(UNCERTAINTY_TABLE_DIR / "empirical_residual_intervals.csv")
    coverage = _read_required_csv(UNCERTAINTY_TABLE_DIR / "empirical_interval_coverage.csv")
    folds = _read_required_csv(UNCERTAINTY_TABLE_DIR / "fold_stability_summary.csv")
    river_bias = _read_required_csv(UNCERTAINTY_TABLE_DIR / "river_bias_summary.csv")
    high_doc = _read_required_csv(UNCERTAINTY_TABLE_DIR / "high_doc_residual_review.csv")
    calibration = _read_required_csv(UNCERTAINTY_TABLE_DIR / "calibration_diagnostics.csv")
    bootstrap = _read_required_csv(UNCERTAINTY_TABLE_DIR / "bootstrap_coefficient_summary.csv")
    readiness = _read_required_csv(UNCERTAINTY_TABLE_DIR / "production_readiness_decision.csv")
    ready = readiness[readiness["decision_item"].eq("ready_for_production_daily_prediction")]
    ready_status = ready["status"].iloc[0] if not ready.empty else "unknown"
    lines = [
        "# Concentration Uncertainty Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "## 1. Scope and guardrails",
        "",
        "This phase trains validation-only DOC concentration models for uncertainty, residual, fold stability, calibration, and production-readiness diagnostics. It does not generate production daily DOC predictions and does not compute flux.",
        "",
        "## 2. Current model decisions",
        "",
        _md_table(registry, max_rows=10),
        "",
        "## 3. Inputs and exclusions",
        "",
        f"- freeze_id: `{contract['freeze_id']}`",
        "- model input: `data/processed/gold/training_matrix_hydrocore.csv`",
        "- excluded: prediction grid, optical matrices, basin context matrices, lab optical/CDOM.",
        "",
        "## 4. Primary model residual distribution",
        "",
        _md_table(residuals[residuals["model_role"].eq("primary_baseline") & residuals["group_type"].eq("overall")], max_rows=10),
        "",
        "## 5. Empirical residual intervals and coverage",
        "",
        _md_table(intervals, max_rows=30),
        "",
        _md_table(coverage, max_rows=30),
        "",
        "## 6. Fold stability",
        "",
        _md_table(folds.head(40), max_rows=40),
        "",
        "## 7. River-specific bias",
        "",
        _md_table(river_bias, max_rows=20),
        "",
        "## 8. High-DOC residual behavior and log-target status",
        "",
        "Log-target remains sensitivity-only in this phase.",
        "",
        _md_table(high_doc, max_rows=30),
        "",
        "## 9. Calibration diagnostics",
        "",
        "The calibration regression is diagnostic only and is not used as a corrective model.",
        "",
        _md_table(calibration, max_rows=10),
        "",
        "## 10. Bootstrap coefficient stability",
        "",
        _md_table(bootstrap, max_rows=30),
        "",
        "## 11. ROI and optical caveats carried forward",
        "",
        "ROI QC completed with caveats and no freeze reopen recommendation. Optical proxy remains excluded from the primary model because optical sensitivity found no robust incremental value over F3.",
        "",
        "## 12. Production daily prediction readiness",
        "",
        f"ready_for_production_daily_prediction: `{ready_status}`",
        "",
        _md_table(readiness, max_rows=20),
        "",
        "## 13. Recommended next phase",
        "",
        "Proceed to a guarded production daily DOC prediction phase only if `ready_for_production_daily_prediction` is `true` or `true_with_caveats`; otherwise refine the concentration model first.",
        "",
        "## 14. Explicit statements",
        "",
        "- Validation-only DOC concentration models were trained.",
        "- No production daily DOC prediction was generated.",
        "- No DOC flux was generated.",
        "- Gold data were not modified.",
        "- Optical/basin matrices were not used as predictors.",
        "- Prediction grid was not loaded.",
    ]
    UNCERTAINTY_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return UNCERTAINTY_REPORT_PATH


def run_concentration_uncertainty() -> dict[str, Any]:
    _ensure_dirs()
    assert_no_forbidden_outputs()
    metadata = _load_required_metadata()
    verification = _verify_contract_snapshot()
    hydrocore, hydrocore_path = _read_hydrocore_only()
    before_hash = sha256_file(hydrocore_path)

    predictions, fold_summary = _run_all_cv(hydrocore)
    residual_summary = grouped_residual_summary(predictions)
    primary_loyo = predictions[predictions["model_role"].eq("primary_baseline") & predictions["validation_scheme"].eq("leave_one_year_out")].copy()
    intervals = empirical_residual_intervals(primary_loyo)
    coverage = empirical_interval_coverage(primary_loyo, intervals)
    fold_stability = _fold_stability(fold_summary, predictions)
    primary_overall_rmse = metric_row(primary_loyo["DOC_observed_mgC_L"], primary_loyo["DOC_cv_predicted_mgC_L"])["rmse"]
    river_bias = _bias_summary(predictions, "river", primary_overall_rmse, metadata["roi_qc"])
    year_bias = _bias_summary(predictions, "year", primary_overall_rmse)
    month_bias = _bias_summary(predictions, "month", primary_overall_rmse)
    high_doc = high_doc_residual_review(predictions)
    calibration = _calibration_diagnostics(primary_loyo)
    bootstrap = _bootstrap_coefficients(hydrocore)
    readiness = production_readiness_decision(
        river_bias=river_bias,
        fold_stability=fold_stability,
        intervals=intervals,
        roi_qc=metadata["roi_qc"],
        optical_ranking=metadata["optical_ranking"],
        no_production_generated=True,
    )

    table_paths = [
        _write_csv(_model_registry(), UNCERTAINTY_TABLE_DIR / "uncertainty_model_registry.csv"),
        _write_csv(predictions, UNCERTAINTY_TABLE_DIR / "uncertainty_cv_predictions.csv"),
        _write_csv(residual_summary, UNCERTAINTY_TABLE_DIR / "residual_distribution_summary.csv"),
        _write_csv(intervals, UNCERTAINTY_TABLE_DIR / "empirical_residual_intervals.csv"),
        _write_csv(coverage, UNCERTAINTY_TABLE_DIR / "empirical_interval_coverage.csv"),
        _write_csv(fold_stability, UNCERTAINTY_TABLE_DIR / "fold_stability_summary.csv"),
        _write_csv(river_bias, UNCERTAINTY_TABLE_DIR / "river_bias_summary.csv"),
        _write_csv(year_bias, UNCERTAINTY_TABLE_DIR / "year_bias_summary.csv"),
        _write_csv(month_bias, UNCERTAINTY_TABLE_DIR / "month_bias_summary.csv"),
        _write_csv(high_doc, UNCERTAINTY_TABLE_DIR / "high_doc_residual_review.csv"),
        _write_csv(calibration, UNCERTAINTY_TABLE_DIR / "calibration_diagnostics.csv"),
        _write_csv(bootstrap, UNCERTAINTY_TABLE_DIR / "bootstrap_coefficient_summary.csv"),
        _write_csv(readiness, UNCERTAINTY_TABLE_DIR / "production_readiness_decision.csv"),
    ]
    figure_paths = _make_figures(
        predictions=predictions,
        residual_summary=residual_summary,
        fold_stability=fold_stability,
        high_doc=high_doc,
        calibration=calibration,
        bootstrap=bootstrap,
    )
    report_path = write_concentration_uncertainty_report()

    if sha256_file(hydrocore_path) != before_hash:
        raise RuntimeError("training_matrix_hydrocore.csv changed during concentration uncertainty diagnostics.")
    assert_no_forbidden_outputs()
    return {
        "tables": table_paths,
        "figures": figure_paths,
        "report": report_path,
        "predictions": predictions,
        "readiness": readiness,
        "verification": verification,
    }
