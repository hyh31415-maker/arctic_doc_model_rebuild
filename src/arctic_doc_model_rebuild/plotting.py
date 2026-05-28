from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


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


def _boxplot_by_group(frame: pd.DataFrame, value: str, group: str, title: str, ylabel: str, destination: Path, log_y: bool = False) -> Path | None:
    plt = _try_matplotlib()
    if plt is None or frame.empty or value not in frame.columns or group not in frame.columns:
        return None
    data = []
    labels = []
    for label, subset in frame.groupby(group):
        values = pd.to_numeric(subset[value], errors="coerce").dropna()
        if not values.empty:
            data.append(values)
            labels.append(str(label))
    if not data:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.8))
    try:
        ax.boxplot(data, tick_labels=labels, showfliers=False)
    except TypeError:
        ax.boxplot(data, labels=labels, showfliers=False)
    if log_y:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(group)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    path = _save(fig, destination)
    plt.close(fig)
    return path


def generate_eda_figures(
    *,
    hydrocore: pd.DataFrame,
    missing_by_river: pd.DataFrame,
    optical_window_counts: pd.DataFrame,
    optical_sensor_counts: pd.DataFrame,
    prediction_grid_by_river: pd.DataFrame,
    basin_attribute_summary: pd.DataFrame,
    figure_dir: Path,
) -> list[Path]:
    plt = _try_matplotlib()
    if plt is None:
        return []
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    doc_river = _boxplot_by_group(
        hydrocore,
        "DOC_mgC_L",
        "river",
        "DOC by river",
        "DOC_mgC_L",
        figure_dir / "doc_by_river_boxplot.png",
    )
    if doc_river:
        paths.append(doc_river)

    if "date" in hydrocore.columns:
        month_frame = hydrocore.copy()
        month_frame["month"] = pd.to_datetime(month_frame["date"], errors="coerce").dt.month
        doc_month = _boxplot_by_group(
            month_frame,
            "DOC_mgC_L",
            "month",
            "DOC by month",
            "DOC_mgC_L",
            figure_dir / "doc_by_month_boxplot.png",
        )
        if doc_month:
            paths.append(doc_month)

    if {"river", "date"}.issubset(hydrocore.columns):
        coverage = hydrocore.copy()
        coverage["date"] = pd.to_datetime(coverage["date"], errors="coerce")
        rivers = sorted(coverage["river"].dropna().astype(str).unique())
        river_index = {river: idx for idx, river in enumerate(rivers)}
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.scatter(coverage["date"], coverage["river"].astype(str).map(river_index), s=12)
        ax.set_yticks(list(river_index.values()), labels=list(river_index.keys()))
        ax.set_title("DOC sample time coverage by river")
        ax.set_xlabel("date")
        ax.set_ylabel("river")
        paths.append(_save(fig, figure_dir / "doc_time_coverage_by_river.png"))
        plt.close(fig)

    q_river = _boxplot_by_group(
        hydrocore,
        "Q_m3s",
        "river",
        "Q by river",
        "Q_m3s",
        figure_dir / "q_by_river_boxplot_logscale.png",
        log_y=True,
    )
    if q_river:
        paths.append(q_river)

    if {"Q_m3s", "DOC_mgC_L", "river"}.issubset(hydrocore.columns):
        scatter = hydrocore.copy()
        scatter["Q_m3s"] = pd.to_numeric(scatter["Q_m3s"], errors="coerce")
        scatter["DOC_mgC_L"] = pd.to_numeric(scatter["DOC_mgC_L"], errors="coerce")
        scatter = scatter[(scatter["Q_m3s"] > 0) & scatter["DOC_mgC_L"].notna()]
        fig, ax = plt.subplots(figsize=(7, 5))
        for river, subset in scatter.groupby("river"):
            ax.scatter(subset["Q_m3s"], subset["DOC_mgC_L"], s=14, label=str(river), alpha=0.75)
        ax.set_xscale("log")
        ax.set_title("DOC vs log(Q) by river")
        ax.set_xlabel("Q_m3s log scale")
        ax.set_ylabel("DOC_mgC_L")
        ax.legend(fontsize="small", ncol=2)
        paths.append(_save(fig, figure_dir / "doc_vs_logq_by_river.png"))
        plt.close(fig)

    if not missing_by_river.empty:
        matrix = missing_by_river.pivot(index="river", columns="column", values="missing_rate").fillna(0)
        fig, ax = plt.subplots(figsize=(9, 4.8))
        image = ax.imshow(matrix.to_numpy(), aspect="auto")
        ax.set_xticks(range(len(matrix.columns)), labels=matrix.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(matrix.index)), labels=matrix.index)
        ax.set_title("Hydrocore missingness by river")
        fig.colorbar(image, ax=ax, label="missing rate")
        paths.append(_save(fig, figure_dir / "hydrocore_missingness_heatmap.png"))
        plt.close(fig)

    if not optical_window_counts.empty:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.bar(optical_window_counts["window"], optical_window_counts["row_count"])
        ax.set_title("Optical matches by window")
        ax.set_xlabel("window")
        ax.set_ylabel("rows")
        paths.append(_save(fig, figure_dir / "optical_matches_by_window.png"))
        plt.close(fig)

    if not optical_sensor_counts.empty:
        sensor_totals = optical_sensor_counts.groupby("sensor", dropna=False)["row_count"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.bar(sensor_totals.index.astype(str), sensor_totals.values)
        ax.set_title("Optical matches by sensor")
        ax.set_xlabel("sensor")
        ax.set_ylabel("rows")
        paths.append(_save(fig, figure_dir / "optical_matches_by_sensor.png"))
        plt.close(fig)

    if not prediction_grid_by_river.empty:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.bar(prediction_grid_by_river["river"].astype(str), prediction_grid_by_river["row_count"])
        ax.set_title("Prediction grid coverage by river")
        ax.set_xlabel("river")
        ax.set_ylabel("rows")
        ax.tick_params(axis="x", rotation=35)
        paths.append(_save(fig, figure_dir / "prediction_grid_coverage_by_river.png"))
        plt.close(fig)

    category_rows = basin_attribute_summary[basin_attribute_summary.get("metric", pd.Series(dtype=str)).eq("attributes_by_category")]
    if not category_rows.empty:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.bar(category_rows["category"].astype(str), pd.to_numeric(category_rows["value"], errors="coerce").fillna(0))
        ax.set_title("Basin attribute categories")
        ax.set_xlabel("category")
        ax.set_ylabel("attributes")
        ax.tick_params(axis="x", rotation=35)
        paths.append(_save(fig, figure_dir / "basin_attribute_categories.png"))
        plt.close(fig)

    return paths
