from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..gold_contract import sha256_file
from ..paths import path


FORBIDDEN_OUTPUT_DIRS = [path("outputs", "predictions"), path("outputs", "flux")]
FORBIDDEN_FILE_PATTERNS = ["daily_flux", "annual_flux", "snowmelt_flux"]
ALLOWED_DOC_FLUX_DIR = path("outputs", "tables", "doc_flux")
ALLOWED_DOC_FLUX_OUTPUT_DIRS = [
    path("outputs", "tables", "doc_flux"),
    path("outputs", "reports", "doc_flux"),
    path("outputs", "figures", "doc_flux"),
    path("outputs", "tables", "flux_interpretation"),
    path("outputs", "reports", "flux_interpretation"),
    path("outputs", "figures", "flux_interpretation"),
    path("outputs", "tables", "annual_flux_trends"),
    path("outputs", "reports", "annual_flux_trends"),
    path("outputs", "figures", "annual_flux_trends"),
    path("outputs", "tables", "may_july_flux"),
    path("outputs", "reports", "may_july_flux"),
    path("outputs", "figures", "may_july_flux"),
    path("outputs", "tables", "snowmelt_windows"),
    path("outputs", "reports", "snowmelt_windows"),
    path("outputs", "figures", "snowmelt_windows"),
    path("outputs", "tables", "final_synthesis"),
    path("outputs", "reports", "final_synthesis"),
    path("outputs", "figures", "final_synthesis"),
    path("outputs", "tables", "flux_attribution"),
    path("outputs", "reports", "flux_attribution"),
    path("outputs", "figures", "flux_attribution"),
    path("outputs", "tables", "freshet_control"),
    path("outputs", "reports", "freshet_control"),
    path("outputs", "figures", "freshet_control"),
]
ALLOWED_DOC_MODEL_ARTIFACTS = {
    "production_candidate_r4_daily_doc_model.joblib",
    "production_candidate_r4_daily_doc_model_metadata.json",
}


def assert_gold_hash_unchanged(file_path: Path, before_hash: str) -> None:
    after_hash = sha256_file(file_path)
    if after_hash != before_hash:
        raise RuntimeError(f"Gold input table changed during baseline run: {file_path}")


def assert_no_forbidden_outputs() -> None:
    existing_dirs = [directory for directory in FORBIDDEN_OUTPUT_DIRS if directory.exists()]
    if existing_dirs:
        raise RuntimeError(f"Forbidden production output directories exist: {existing_dirs}")
    forbidden_files = []
    outputs = path("outputs")
    if outputs.exists():
        for item in outputs.rglob("*"):
            if not item.is_file():
                continue
            lower = item.name.lower()
            allowed_doc_flux_output = any(directory in item.parents for directory in ALLOWED_DOC_FLUX_OUTPUT_DIRS)
            if item.suffix.lower() in {".joblib", ".pkl", ".pickle"} and item.name not in ALLOWED_DOC_MODEL_ARTIFACTS:
                forbidden_files.append(item)
            if any(pattern in lower for pattern in FORBIDDEN_FILE_PATTERNS) and not allowed_doc_flux_output:
                forbidden_files.append(item)
            if lower.endswith("_flux.csv") and ALLOWED_DOC_FLUX_DIR not in item.parents:
                forbidden_files.append(item)
            if "outputs\\models" in str(item).lower() or "outputs/models" in str(item).lower():
                if item.name not in ALLOWED_DOC_MODEL_ARTIFACTS:
                    forbidden_files.append(item)
    if forbidden_files:
        raise RuntimeError(f"Forbidden model/prediction/flux artifacts exist: {forbidden_files}")


def _try_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _save(fig, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=120)
    return destination


def make_baseline_figures(
    predictions: pd.DataFrame,
    overall: pd.DataFrame,
    by_river: pd.DataFrame,
    by_month: pd.DataFrame,
    fold_summary: pd.DataFrame,
    ranking: pd.DataFrame,
    figure_dir: Path,
) -> list[Path]:
    plt = _try_matplotlib()
    if plt is None:
        return []
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    best = ranking.head(4)[["model_id", "feature_set"]].drop_duplicates() if not ranking.empty else pd.DataFrame()
    best_keys = {(row.model_id, row.feature_set) for row in best.itertuples(index=False)}
    plot_predictions = predictions[predictions.apply(lambda row: (row["model_id"], row["feature_set"]) in best_keys and row["validation_scheme"] == "leave_one_year_out", axis=1)]
    if not plot_predictions.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        for (model_id, feature_set), subset in plot_predictions.groupby(["model_id", "feature_set"]):
            ax.scatter(subset["DOC_observed_mgC_L"], subset["DOC_cv_predicted_mgC_L"], s=12, alpha=0.55, label=f"{feature_set}:{model_id}")
        lo = min(plot_predictions["DOC_observed_mgC_L"].min(), plot_predictions["DOC_cv_predicted_mgC_L"].min())
        hi = max(plot_predictions["DOC_observed_mgC_L"].max(), plot_predictions["DOC_cv_predicted_mgC_L"].max())
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlabel("Observed DOC mgC/L")
        ax.set_ylabel("CV predicted DOC mgC/L")
        ax.set_title("Observed vs validation-only CV predictions")
        ax.legend(fontsize="x-small")
        paths.append(_save(fig, figure_dir / "observed_vs_cv_predicted_by_model.png"))
        plt.close(fig)

    if not plot_predictions.empty:
        best_first = ranking.iloc[0]
        residual_subset = predictions[
            predictions["model_id"].eq(best_first["model_id"])
            & predictions["feature_set"].eq(best_first["feature_set"])
            & predictions["validation_scheme"].eq("leave_one_year_out")
        ]
        if not residual_subset.empty:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            data = [grp["residual_mgC_L"].to_numpy() for _, grp in residual_subset.groupby("river")]
            labels = [str(name) for name, _ in residual_subset.groupby("river")]
            try:
                ax.boxplot(data, tick_labels=labels, showfliers=False)
            except TypeError:
                ax.boxplot(data, labels=labels, showfliers=False)
            ax.axhline(0, linestyle="--")
            ax.set_title("LOYO residuals by river")
            ax.set_ylabel("Observed - predicted DOC mgC/L")
            ax.tick_params(axis="x", rotation=30)
            paths.append(_save(fig, figure_dir / "residuals_by_river.png"))
            plt.close(fig)

    loyo_overall = overall[overall["validation_scheme"].eq("leave_one_year_out")].sort_values("rmse").head(12)
    if not loyo_overall.empty:
        labels = loyo_overall["feature_set"] + ":" + loyo_overall["model_id"]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(labels, loyo_overall["rmse"])
        ax.invert_yaxis()
        ax.set_xlabel("RMSE")
        ax.set_title("LOYO RMSE by model")
        paths.append(_save(fig, figure_dir / "rmse_by_model.png"))
        plt.close(fig)

    if not by_river.empty and not ranking.empty:
        top_keys = ranking.head(3)[["model_id", "feature_set"]]
        subset_rows = []
        for row in top_keys.itertuples(index=False):
            subset_rows.append(by_river[by_river["model_id"].eq(row.model_id) & by_river["feature_set"].eq(row.feature_set) & by_river["validation_scheme"].eq("leave_one_year_out")])
        bias = pd.concat(subset_rows, ignore_index=True) if subset_rows else pd.DataFrame()
        if not bias.empty:
            pivot = bias.pivot_table(index="river", columns="feature_set", values="bias_mean")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            pivot.plot(kind="bar", ax=ax)
            ax.axhline(0, linestyle="--")
            ax.set_title("LOYO bias by river for top baseline candidates")
            ax.set_ylabel("Mean residual")
            paths.append(_save(fig, figure_dir / "bias_by_river_best_models.png"))
            plt.close(fig)

    if not by_month.empty and not ranking.empty:
        best_first = ranking.iloc[0]
        month_metrics = by_month[
            by_month["model_id"].eq(best_first["model_id"])
            & by_month["feature_set"].eq(best_first["feature_set"])
            & by_month["validation_scheme"].eq("leave_one_year_out")
        ]
        if not month_metrics.empty:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.bar(month_metrics["month"].astype(str), month_metrics["bias_mean"])
            ax.axhline(0, linestyle="--")
            ax.set_xlabel("Month")
            ax.set_ylabel("Mean residual")
            ax.set_title("LOYO residual bias by month")
            paths.append(_save(fig, figure_dir / "residuals_by_month.png"))
            plt.close(fig)

    if not fold_summary.empty:
        loyo_folds = fold_summary[fold_summary["validation_scheme"].eq("leave_one_year_out")].copy()
        if not loyo_folds.empty:
            best_key = ranking.iloc[0] if not ranking.empty else loyo_folds.iloc[0]
            loyo_folds = loyo_folds[
                loyo_folds["model_id"].eq(best_key["model_id"])
                & loyo_folds["feature_set"].eq(best_key["feature_set"])
            ]
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.plot(loyo_folds["fold_year"].astype(str), loyo_folds["rmse"], marker="o")
            ax.set_title("LOYO fold RMSE for top baseline candidate")
            ax.set_xlabel("Held-out year")
            ax.set_ylabel("RMSE")
            ax.tick_params(axis="x", rotation=45)
            paths.append(_save(fig, figure_dir / "cv_fold_performance.png"))
            plt.close(fig)

    return paths
