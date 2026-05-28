from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..gold_contract import load_contract, require_gold_data_dir, sha256_file, table_path
from ..paths import REPORT_DIR, TABLE_DIR, path
from ..reports import _md_table
from .ablation import ABLATION_FEATURE_SETS, DELTA_COMPARISONS, ablation_deltas, common_subset
from .baseline_models import MODEL_SPECS, MeanBaseline, _make_estimator, _prepare_hydrocore, _read_hydrocore
from .diagnostics import assert_gold_hash_unchanged, assert_no_forbidden_outputs
from .feature_sets import FeatureSet, TARGET_COLUMN
from .metrics import grouped_metrics, metric_row
from .residual_analysis import add_quantiles, add_season_windows, fold_stability, residual_summary
from .target_transforms import inverse_target, target_values
from .validation import validation_scheme_registry, validation_splits


REFINEMENT_TABLE_DIR = TABLE_DIR / "baseline_refinement"
REFINEMENT_REPORT_DIR = REPORT_DIR / "baseline_refinement"
REFINEMENT_FIGURE_DIR = path("outputs", "figures", "baseline_refinement")
REFINEMENT_REPORT_PATH = REFINEMENT_REPORT_DIR / "baseline_refinement_report.md"
REFINEMENT_ALLOWED_INPUT_TABLES = {"training_matrix_hydrocore.csv"}
REFINEMENT_MODELS = [spec for spec in MODEL_SPECS if spec.model_id in {"linear_regression", "ridge_alpha_0.1", "ridge_alpha_1", "ridge_alpha_10"}]
LOG_TARGET_FEATURE_SETS = {
    item.feature_set.feature_set: item.feature_set
    for item in ABLATION_FEATURE_SETS
    if item.feature_set.feature_set
    in {"F2_q_season", "F3_q_season_river_fixed", "F4_reduced_hydroclimate", "F6_reduced_hydroclimate_river_fixed"}
}
LOG_TARGET_MODELS = [spec for spec in MODEL_SPECS if spec.model_id in {"ridge_alpha_1", "ridge_alpha_10"}]


def _ensure_dirs() -> None:
    for directory in [REFINEMENT_TABLE_DIR, REFINEMENT_REPORT_DIR, REFINEMENT_FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def _write_csv(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8")
    return destination


def _read_hydrocore_only() -> tuple[pd.DataFrame, Path]:
    gold_dir = require_gold_data_dir()
    destination = table_path("training_matrix_hydrocore.csv", gold_dir=gold_dir)
    if destination.name not in REFINEMENT_ALLOWED_INPUT_TABLES:
        raise RuntimeError(f"Refinement is not allowed to read this input table: {destination.name}")
    return _prepare_hydrocore(_read_hydrocore()), destination


def _hash_all_gold_tables() -> dict[str, str]:
    contract = load_contract()
    gold_dir = require_gold_data_dir()
    hashes = {}
    for table_name in contract["expected_tables"]:
        destination = table_path(table_name, gold_dir=gold_dir)
        hashes[table_name] = sha256_file(destination)
    return hashes


def _fit_predict(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_set: FeatureSet,
    model_spec,
    target_scale: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not feature_set.required_features:
        estimator = MeanBaseline()
    else:
        estimator = _make_estimator(model_spec, feature_set)
    x_train = train[list(feature_set.required_features)]
    x_test = test[list(feature_set.required_features)]
    y_train = target_values(train, target_scale)
    estimator.fit(x_train, y_train)
    pred_target = estimator.predict(x_test)
    pred_mg = inverse_target(pred_target, target_scale)
    return np.asarray(pred_mg, dtype=float), np.asarray(pred_target, dtype=float)


def _run_cv(
    *,
    frame: pd.DataFrame,
    feature_set: FeatureSet,
    model_spec,
    target_scale: str,
    validation_schemes: set[str],
    ablation_id: str = "",
    sample_scope: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cv_rows = []
    fold_rows = []
    usable = frame.dropna(subset=[TARGET_COLUMN, "label_id", "river", "date", "year", "month", *feature_set.required_features]).copy()
    if target_scale == "log":
        usable = usable[pd.to_numeric(usable[TARGET_COLUMN], errors="coerce") > 0].copy()
    if "log_Q" in feature_set.required_features:
        usable = usable[usable["Q_m3s"] > 0].copy()
    usable = usable.reset_index(drop=True)
    for scheme, fold_id, train_idx, test_idx, fold_info in validation_splits(usable):
        if scheme not in validation_schemes:
            continue
        train = usable.iloc[train_idx].copy()
        test = usable.iloc[test_idx].copy()
        pred_mg, pred_target = _fit_predict(train=train, test=test, feature_set=feature_set, model_spec=model_spec, target_scale=target_scale)
        obs_mg = pd.to_numeric(test[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
        fold_metric = metric_row(obs_mg, pred_mg)
        if target_scale == "log":
            obs_log = target_values(test, "log").to_numpy(dtype=float)
            log_metric = metric_row(obs_log, pred_target)
        else:
            obs_log = np.full(len(test), np.nan)
            log_metric = {key: np.nan for key in ["rmse", "mae", "median_absolute_error", "r2", "bias_mean", "bias_median", "spearman_r", "pearson_r"]}
        fold_rows.append(
            {
                "ablation_id": ablation_id,
                "sample_scope": sample_scope,
                "model_id": model_spec.model_id,
                "feature_set": feature_set.feature_set,
                "target_scale": target_scale,
                "validation_scheme": scheme,
                "fold_id": fold_id,
                "fold_group": fold_info.get("fold_group", ""),
                "fold_year": fold_info.get("fold_year", ""),
                "fold_river": fold_info.get("fold_river", ""),
                "n_train": len(train),
                "n_test": len(test),
                "rivers_in_test": ";".join(sorted(test["river"].astype(str).unique())),
                "rmse": fold_metric["rmse"],
                "mae": fold_metric["mae"],
                "bias_mean": fold_metric["bias_mean"],
                "r2": fold_metric["r2"],
                "rmse_log_space": log_metric["rmse"],
                "mae_log_space": log_metric["mae"],
            }
        )
        for row, observed, predicted_mg, predicted_target, observed_log in zip(test.to_dict("records"), obs_mg, pred_mg, pred_target, obs_log):
            cv_rows.append(
                {
                    "label_id": row["label_id"],
                    "river": row["river"],
                    "date": pd.Timestamp(row["date"]).date().isoformat(),
                    "year": int(row["year"]) if pd.notna(row["year"]) else "",
                    "month": int(row["month"]) if pd.notna(row["month"]) else "",
                    "Q_m3s": row.get("Q_m3s", np.nan),
                    "DOC_observed_mgC_L": float(observed),
                    "DOC_cv_predicted_mgC_L": float(predicted_mg),
                    "residual_mgC_L": float(observed - predicted_mg),
                    "DOC_observed_log": float(observed_log) if np.isfinite(observed_log) else "",
                    "DOC_cv_predicted_log": float(predicted_target) if target_scale == "log" else "",
                    "model_id": model_spec.model_id,
                    "feature_set": feature_set.feature_set,
                    "target_scale": target_scale,
                    "validation_scheme": scheme,
                    "fold_id": fold_id,
                    "ablation_id": ablation_id,
                    "sample_scope": sample_scope,
                    "is_cv_prediction": True,
                    "is_production_prediction": False,
                }
            )
    return pd.DataFrame(cv_rows), pd.DataFrame(fold_rows)


def _overall_metrics(predictions: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows = []
    for keys, subset in predictions.groupby(["model_id", "feature_set", "target_scale", "validation_scheme"], dropna=False):
        model_id, feature_set, target_scale, validation_scheme = keys
        row = {
            "model_id": model_id,
            "feature_set": feature_set,
            "target_scale": target_scale,
            "validation_scheme": validation_scheme,
            "n_test_total": len(subset),
            "n_folds": int(subset["fold_id"].nunique()),
            "ablation_id": subset["ablation_id"].iloc[0] if "ablation_id" in subset.columns else "",
            "sample_scope": subset["sample_scope"].iloc[0] if "sample_scope" in subset.columns else "",
            "common_subset_n": len(subset),
        }
        fold_subset = folds[
            folds["model_id"].eq(model_id)
            & folds["feature_set"].eq(feature_set)
            & folds["target_scale"].eq(target_scale)
            & folds["validation_scheme"].eq(validation_scheme)
        ]
        row["n_train_total"] = int(fold_subset["n_train"].sum()) if not fold_subset.empty else ""
        row.update(metric_row(subset["DOC_observed_mgC_L"], subset["DOC_cv_predicted_mgC_L"]))
        rows.append(row)
    return pd.DataFrame(rows)


def _run_same_sample_ablation(hydrocore: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subset = common_subset(hydrocore)
    pred_frames = []
    fold_frames = []
    for item in ABLATION_FEATURE_SETS:
        for model_spec in REFINEMENT_MODELS:
            cv, folds = _run_cv(
                frame=subset,
                feature_set=item.feature_set,
                model_spec=model_spec,
                target_scale="raw",
                validation_schemes={"leave_one_year_out", "river_year_groupkfold", "leave_one_river_out"},
                ablation_id=item.ablation_id,
                sample_scope="same_sample_common_subset",
            )
            pred_frames.append(cv)
            fold_frames.append(folds)
    predictions = pd.concat(pred_frames, ignore_index=True)
    folds = pd.concat(fold_frames, ignore_index=True)
    metrics = _overall_metrics(predictions, folds)
    metrics["same_sample_n"] = len(subset)
    deltas = ablation_deltas(metrics)
    return metrics, deltas, predictions, folds


def _run_log_target_sensitivity(hydrocore: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_frames = []
    fold_frames = []
    for feature_set in LOG_TARGET_FEATURE_SETS.values():
        for model_spec in LOG_TARGET_MODELS:
            for target_scale in ["raw", "log"]:
                cv, folds = _run_cv(
                    frame=hydrocore,
                    feature_set=feature_set,
                    model_spec=model_spec,
                    target_scale=target_scale,
                    validation_schemes={"leave_one_year_out", "river_year_groupkfold"},
                    sample_scope="feature_specific_complete_case",
                )
                pred_frames.append(cv)
                fold_frames.append(folds)
    predictions = pd.concat(pred_frames, ignore_index=True)
    folds = pd.concat(fold_frames, ignore_index=True)
    metrics = _overall_metrics(predictions, folds)
    log_fold_metrics = folds.groupby(["model_id", "feature_set", "target_scale", "validation_scheme"], dropna=False).agg(
        rmse_log_space_mean=("rmse_log_space", "mean"),
        mae_log_space_mean=("mae_log_space", "mean"),
    ).reset_index()
    metrics = metrics.merge(log_fold_metrics, on=["model_id", "feature_set", "target_scale", "validation_scheme"], how="left")
    return metrics, predictions, folds


def _select_top_predictions(ablation_predictions: pd.DataFrame, log_predictions: pd.DataFrame, log_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = [
        ("F3_q_season_river_fixed", "ridge_alpha_1", "raw", ablation_predictions),
        ("F6_reduced_hydroclimate_river_fixed", "ridge_alpha_1", "raw", ablation_predictions),
        ("F6_reduced_hydroclimate_river_fixed", "ridge_alpha_10", "raw", ablation_predictions),
    ]
    log_loyo = log_metrics[(log_metrics["target_scale"].eq("log")) & (log_metrics["validation_scheme"].eq("leave_one_year_out"))].sort_values(["rmse", "mae"])
    if not log_loyo.empty:
        row = log_loyo.iloc[0]
        selected.append((row["feature_set"], row["model_id"], "log", log_predictions))
    frames = []
    selected_rows = []
    seen = set()
    for feature_set, model_id, target_scale, source in selected:
        key = (feature_set, model_id, target_scale)
        if key in seen:
            continue
        seen.add(key)
        subset = source[
            source["feature_set"].eq(feature_set)
            & source["model_id"].eq(model_id)
            & source["target_scale"].eq(target_scale)
            & source["validation_scheme"].eq("leave_one_year_out")
        ].copy()
        if not subset.empty:
            frames.append(subset)
            selected_rows.append({"feature_set": feature_set, "model_id": model_id, "target_scale": target_scale})
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), pd.DataFrame(selected_rows)


def _residual_tables(top_predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if top_predictions.empty:
        return {
            "residual_summary_by_river": pd.DataFrame(),
            "residual_summary_by_year": pd.DataFrame(),
            "residual_summary_by_month": pd.DataFrame(),
            "residual_summary_by_season_window": pd.DataFrame(),
            "residual_summary_by_doc_quantile": pd.DataFrame(),
            "residual_summary_by_q_quantile": pd.DataFrame(),
        }
    enriched = add_quantiles(top_predictions.copy())
    overall = residual_summary(enriched, [])
    season = add_season_windows(enriched)
    return {
        "residual_summary_by_river": residual_summary(enriched, ["river"], overall),
        "residual_summary_by_year": residual_summary(enriched, ["year"], overall),
        "residual_summary_by_month": residual_summary(enriched, ["month"], overall),
        "residual_summary_by_season_window": residual_summary(season, ["season_window"], overall),
        "residual_summary_by_doc_quantile": residual_summary(enriched, ["doc_quantile"], overall),
        "residual_summary_by_q_quantile": residual_summary(enriched, ["q_quantile"], overall),
    }


def _recommendations(ablation_metrics: pd.DataFrame, deltas: pd.DataFrame, log_metrics: pd.DataFrame) -> pd.DataFrame:
    loyo = ablation_metrics[ablation_metrics["validation_scheme"].eq("leave_one_year_out")].copy()
    f6_vs_f3 = deltas[
        deltas["comparison"].eq("F6_minus_F3_incremental_hydroclimate_after_river")
        & deltas["model_id"].eq("ridge_alpha_1")
        & deltas["validation_scheme"].eq("leave_one_year_out")
    ]
    rmse_delta = float(f6_vs_f3["rmse_reduction"].iloc[0]) if not f6_vs_f3.empty else 0.0
    mae_delta = float(f6_vs_f3["mae_reduction"].iloc[0]) if not f6_vs_f3.empty else 0.0
    f3 = loyo[loyo["feature_set"].eq("F3_q_season_river_fixed") & loyo["model_id"].eq("ridge_alpha_1")]
    f6 = loyo[loyo["feature_set"].eq("F6_reduced_hydroclimate_river_fixed") & loyo["model_id"].eq("ridge_alpha_1")]
    meaningful = rmse_delta >= 0.10 or mae_delta >= 0.05
    if meaningful:
        primary = f6.iloc[0] if not f6.empty else loyo.sort_values("rmse").iloc[0]
        hydro = f3.iloc[0] if not f3.empty else primary
        primary_reason = "F6 improves same-sample LOYO RMSE/MAE meaningfully over F3 without using high-missingness snow variables."
    else:
        primary = f3.iloc[0] if not f3.empty else loyo.sort_values("rmse").iloc[0]
        hydro = f6.iloc[0] if not f6.empty else primary
        primary_reason = "F6 same-sample improvement over F3 is tiny; choose simpler Q+season+river fixed baseline."

    raw_best = log_metrics[(log_metrics["target_scale"].eq("raw")) & (log_metrics["validation_scheme"].eq("leave_one_year_out"))].sort_values(["rmse", "mae"]).head(1)
    log_best = log_metrics[(log_metrics["target_scale"].eq("log")) & (log_metrics["validation_scheme"].eq("leave_one_year_out"))].sort_values(["rmse", "mae"]).head(1)
    log_recommendation = "not_recommended_as_default"
    log_reason = "No log-target result improved both RMSE and MAE over the best raw-target candidate."
    if not raw_best.empty and not log_best.empty:
        if float(log_best["rmse"].iloc[0]) < float(raw_best["rmse"].iloc[0]) and float(log_best["mae"].iloc[0]) < float(raw_best["mae"].iloc[0]):
            log_recommendation = "sensitivity_candidate"
            log_reason = "Best log-target result improves both RMSE and MAE in mg C/L; inspect residual bias before promoting."

    rows = [
        {
            "recommendation_type": "recommended_primary_baseline",
            "feature_set": primary["feature_set"],
            "model_id": primary["model_id"],
            "target_scale": "raw",
            "validation_basis": "same_sample_leave_one_year_out",
            "rmse": primary["rmse"],
            "mae": primary["mae"],
            "recommendation": "primary",
            "reason": primary_reason,
        },
        {
            "recommendation_type": "recommended_hydroclimate_extension",
            "feature_set": hydro["feature_set"],
            "model_id": hydro["model_id"],
            "target_scale": "raw",
            "validation_basis": "same_sample_leave_one_year_out",
            "rmse": hydro["rmse"],
            "mae": hydro["mae"],
            "recommendation": "hydroclimate_extension",
            "reason": f"F6 vs F3 same-sample delta RMSE={rmse_delta:.3f}, MAE={mae_delta:.3f}.",
        },
        {
            "recommendation_type": "recommended_sensitivity_only",
            "feature_set": log_best["feature_set"].iloc[0] if not log_best.empty else "",
            "model_id": log_best["model_id"].iloc[0] if not log_best.empty else "",
            "target_scale": "log",
            "validation_basis": "log_target_leave_one_year_out",
            "rmse": log_best["rmse"].iloc[0] if not log_best.empty else "",
            "mae": log_best["mae"].iloc[0] if not log_best.empty else "",
            "recommendation": log_recommendation,
            "reason": log_reason,
        },
        {
            "recommendation_type": "not_recommended",
            "feature_set": "leave_one_river_out_winner",
            "model_id": "",
            "target_scale": "raw",
            "validation_basis": "leave_one_river_out",
            "rmse": "",
            "mae": "",
            "recommendation": "not_primary",
            "reason": "LORO remains a stress test with six rivers and unseen river fixed-effect categories; do not use as primary ranking.",
        },
    ]
    return pd.DataFrame(rows)


def _make_figures(
    ablation_metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    log_metrics: pd.DataFrame,
    top_predictions: pd.DataFrame,
    stability: pd.DataFrame,
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    REFINEMENT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def save(fig, name: str) -> None:
        fig.tight_layout()
        destination = REFINEMENT_FIGURE_DIR / name
        fig.savefig(destination, dpi=120)
        plt.close(fig)
        paths.append(destination)

    loyo = ablation_metrics[ablation_metrics["validation_scheme"].eq("leave_one_year_out") & ablation_metrics["model_id"].eq("ridge_alpha_1")].copy()
    if not loyo.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.bar(loyo["feature_set"], loyo["rmse"])
        ax.set_ylabel("LOYO RMSE")
        ax.set_title("Same-sample ablation RMSE (ridge alpha 1)")
        ax.tick_params(axis="x", rotation=35)
        save(fig, "same_sample_ablation_rmse.png")

    delta_plot = deltas[deltas["model_id"].eq("ridge_alpha_1") & deltas["validation_scheme"].eq("leave_one_year_out")]
    if not delta_plot.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.bar(delta_plot["comparison"], delta_plot["rmse_reduction"])
        ax.axhline(0, linestyle="--")
        ax.set_ylabel("RMSE reduction")
        ax.set_title("Same-sample incremental RMSE gains")
        ax.tick_params(axis="x", rotation=35)
        save(fig, "same_sample_ablation_delta_rmse.png")

    target_plot = log_metrics[log_metrics["validation_scheme"].eq("leave_one_year_out")].copy()
    if not target_plot.empty:
        labels = target_plot["feature_set"] + ":" + target_plot["model_id"] + ":" + target_plot["target_scale"]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.barh(labels, target_plot["rmse"])
        ax.invert_yaxis()
        ax.set_xlabel("RMSE in mg C/L")
        ax.set_title("Raw vs log target LOYO metrics")
        save(fig, "raw_vs_log_target_metrics.png")

    if not top_predictions.empty:
        top_predictions["model_label"] = top_predictions["feature_set"] + ":" + top_predictions["model_id"] + ":" + top_predictions["target_scale"]
        fig, ax = plt.subplots(figsize=(9, 5))
        data = []
        labels = []
        for label, subset in top_predictions.groupby("model_label"):
            data.append(subset["residual_mgC_L"].to_numpy())
            labels.append(label)
        try:
            ax.boxplot(data, tick_labels=labels, showfliers=False)
        except TypeError:
            ax.boxplot(data, labels=labels, showfliers=False)
        ax.axhline(0, linestyle="--")
        ax.set_ylabel("Residual mg C/L")
        ax.set_title("LOYO residuals by top models")
        ax.tick_params(axis="x", rotation=35)
        save(fig, "residuals_by_river_top_models.png")

        fig, ax = plt.subplots(figsize=(8, 4.8))
        month_bias = top_predictions.groupby(["month", "model_label"])["residual_mgC_L"].mean().reset_index()
        for label, subset in month_bias.groupby("model_label"):
            ax.plot(subset["month"], subset["residual_mgC_L"], marker="o", label=label)
        ax.axhline(0, linestyle="--")
        ax.set_xlabel("Month")
        ax.set_ylabel("Mean residual")
        ax.set_title("LOYO residual bias by month")
        ax.legend(fontsize="x-small")
        save(fig, "residuals_by_month_top_models.png")

    if not stability.empty:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for label, subset in stability.groupby(stability["feature_set"] + ":" + stability["model_id"] + ":" + stability["target_scale"]):
            ax.plot(subset["fold_year"].astype(str), subset["rmse"], marker="o", label=label)
        ax.set_title("LOYO fold stability")
        ax.set_xlabel("Held-out year")
        ax.set_ylabel("RMSE")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize="x-small")
        save(fig, "fold_stability_loyo.png")

    return paths


def write_baseline_refinement_report() -> Path:
    REFINEMENT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    contract = load_contract()
    ablation_metrics = _read_table("same_sample_ablation_metrics.csv")
    deltas = _read_table("same_sample_ablation_deltas.csv")
    log_metrics = _read_table("log_target_sensitivity_metrics.csv")
    recommendation = _read_table("refined_model_recommendation.csv")
    stability = _read_table("fold_stability_leave_one_year_out.csv")
    residual_river = _read_table("residual_summary_by_river.csv")
    loyo = ablation_metrics[ablation_metrics.get("validation_scheme", pd.Series(dtype=str)).eq("leave_one_year_out")].sort_values(["rmse", "mae"]) if not ablation_metrics.empty else pd.DataFrame()
    lines = [
        "# Baseline Refinement Report",
        "",
        "## 1. Scope and guardrails",
        "",
        "This phase trains DOC concentration models for validation diagnostics only. Model input is restricted to `training_matrix_hydrocore.csv`.",
        "",
        "No production daily DOC prediction is generated. No DOC flux is generated. Optical and basin matrices are not used.",
        "",
        "## 2. Why refinement was needed",
        "",
        "Baseline phase 1 showed `F6_reduced_hydroclimate_river_fixed` and `F3_q_season_river_fixed` were close. Refinement compares them on the same complete-case sample and separates the value of Q, river fixed effects, hydroclimate, and target transformation.",
        "",
        "## 3. Same-sample ablation results",
        "",
        f"- freeze_id: `{contract['freeze_id']}`",
        f"- same_sample_n: `{int(ablation_metrics['same_sample_n'].iloc[0]) if 'same_sample_n' in ablation_metrics.columns and not ablation_metrics.empty else ''}`",
        "",
        _md_table(loyo.head(30), max_rows=30),
        "",
        "## 4. Incremental value of Q / river effects / hydroclimate",
        "",
        "Positive reductions indicate that the candidate improves over the baseline.",
        "",
        _md_table(deltas, max_rows=30),
        "",
        "## 5. Raw vs log target sensitivity",
        "",
        "Log-target models are inverse transformed with `exp()` and evaluated in mg C/L. Log-space metrics are also recorded.",
        "",
        _md_table(log_metrics[log_metrics.get("validation_scheme", pd.Series(dtype=str)).eq("leave_one_year_out")].sort_values(["rmse", "mae"]).head(30) if not log_metrics.empty else log_metrics, max_rows=30),
        "",
        "## 6. Residual diagnostics",
        "",
        _md_table(residual_river.head(30) if not residual_river.empty else residual_river, max_rows=30),
        "",
        "## 7. Fold stability",
        "",
        _md_table(stability.head(40) if not stability.empty else stability, max_rows=40),
        "",
        "## 8. Leave-one-river-out stress interpretation",
        "",
        "Leave-one-river-out remains a stress test, not a primary model-selection target. River fixed effects have unseen categories under LORO and are structurally disadvantaged or ambiguous for extrapolation.",
        "",
        "## 9. Recommended primary baseline",
        "",
        _md_table(recommendation, max_rows=10),
        "",
        "## 10. Recommended next phase",
        "",
        "Baseline finalization or optical sensitivity, depending on whether the hydroclimate extension is accepted as primary or retained as an extension.",
        "",
        "## 11. Explicit statements",
        "",
        "- Validation-only DOC concentration models were trained.",
        "- No production daily DOC prediction was generated.",
        "- No DOC flux was generated.",
        "- Gold data were not modified.",
        "- Optical/basin matrices were not used.",
    ]
    REFINEMENT_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return REFINEMENT_REPORT_PATH


def _read_table(name: str) -> pd.DataFrame:
    destination = REFINEMENT_TABLE_DIR / name
    if not destination.exists():
        return pd.DataFrame()
    return pd.read_csv(destination)


def run_baseline_refinement() -> dict[str, Any]:
    _ensure_dirs()
    assert_no_forbidden_outputs()
    before_hashes = _hash_all_gold_tables()
    hydrocore, hydrocore_path = _read_hydrocore_only()
    before_hydrocore_hash = sha256_file(hydrocore_path)

    ablation_metrics, deltas, ablation_predictions, ablation_folds = _run_same_sample_ablation(hydrocore)
    log_metrics, log_predictions, log_folds = _run_log_target_sensitivity(hydrocore)
    top_predictions, selected_top = _select_top_predictions(ablation_predictions, log_predictions, log_metrics)
    residual_tables = _residual_tables(top_predictions)
    top_overall = _overall_metrics(top_predictions, pd.concat([ablation_folds, log_folds], ignore_index=True) if not log_folds.empty else ablation_folds)
    stability = fold_stability(top_predictions, top_overall)
    recommendations = _recommendations(ablation_metrics, deltas, log_metrics)

    table_frames = {
        "same_sample_ablation_metrics": ablation_metrics,
        "same_sample_ablation_deltas": deltas,
        "same_sample_ablation_cv_predictions": ablation_predictions,
        "same_sample_ablation_fold_summary": ablation_folds,
        "log_target_sensitivity_metrics": log_metrics,
        "log_target_cv_predictions": log_predictions,
        "selected_top_model_registry": selected_top,
        "fold_stability_leave_one_year_out": stability,
        "refined_model_recommendation": recommendations,
        **residual_tables,
    }
    table_paths = [_write_csv(frame, REFINEMENT_TABLE_DIR / f"{name}.csv") for name, frame in table_frames.items()]
    figure_paths = _make_figures(ablation_metrics, deltas, log_metrics, top_predictions, stability)
    report_path = write_baseline_refinement_report()

    assert_gold_hash_unchanged(hydrocore_path, before_hydrocore_hash)
    after_hashes = _hash_all_gold_tables()
    if before_hashes != after_hashes:
        raise RuntimeError("One or more frozen gold table hashes changed during baseline refinement.")
    assert_no_forbidden_outputs()
    return {"tables": table_paths, "figures": figure_paths, "report": report_path, "recommendation": recommendations, "same_sample_n": int(ablation_metrics["same_sample_n"].iloc[0])}
