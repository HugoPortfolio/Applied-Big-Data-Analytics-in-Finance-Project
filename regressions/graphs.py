from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from config import FINAL_DATASET, RESULTS_DIR


# Paper-style palette
GRAY_FILL = "#d0d0d0"
GRAY_FILL_2 = "#bdbdbd"
GRID_GRAY = "#d9d9d9"
BLACK = "#000000"

# Explicitly different colors for the combined time-series figure
COLOR_PREPARED = "#1f77b4"
COLOR_QA = "#d62728"
COLOR_GAP = "#2ca02c"


def load_data() -> pd.DataFrame:
    return pd.read_parquet(FINAL_DATASET)


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "figure.figsize": (8, 5),
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "patch.linewidth": 0.8,
            "text.color": BLACK,
            "axes.labelcolor": BLACK,
            "axes.edgecolor": BLACK,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
        }
    )


def style_ax(ax, add_ygrid: bool = True) -> None:
    if add_ygrid:
        ax.grid(True, axis="y", color=GRID_GRAY, linewidth=0.6)
    else:
        ax.grid(False)

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_color(BLACK)
    ax.spines["bottom"].set_color(BLACK)
    ax.tick_params(axis="both", which="major", length=4, width=0.8, colors=BLACK)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))


def finalize_figure(fig, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def save_prepared_vs_qa_boxplot(df: pd.DataFrame) -> None:
    sample = df[["NegPrepared", "NegQA"]].dropna().copy()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(
        [sample["NegPrepared"], sample["NegQA"]],
        tick_labels=["Prepared remarks", "Q&A"],
        showfliers=False,
        widths=0.5,
        patch_artist=True,
        boxprops=dict(facecolor=GRAY_FILL, edgecolor=BLACK, linewidth=0.9),
        medianprops=dict(color=BLACK, linewidth=1.1),
        whiskerprops=dict(color=BLACK, linewidth=0.9),
        capprops=dict(color=BLACK, linewidth=0.9),
    )
    ax.axhline(0, linestyle="--", linewidth=1.0, color=BLACK)

    ax.set_title("Prepared Remarks and Q&A Negativity", pad=10)
    ax.set_ylabel("Negativity score")

    style_ax(ax, add_ygrid=True)
    finalize_figure(fig, "fig_prepared_vs_qa_boxplot.png")


def save_three_distribution_panels(df: pd.DataFrame) -> None:
    cols = [
        ("NegPrepared", "Prepared-Remarks Negativity"),
        ("NegQA", "Q&A Negativity"),
        ("NegGap", "Relative Q&A Negativity"),
    ]
    sample = df[[c for c, _ in cols]].dropna().copy()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)

    for ax, (col, title) in zip(axes, cols):
        ax.hist(
            sample[col],
            bins=50,
            color=GRAY_FILL,
            edgecolor=BLACK,
            linewidth=0.5,
        )
        ax.axvline(0, linestyle="--", linewidth=1.0, color=BLACK)
        ax.set_title(title, pad=8)
        ax.set_xlabel(col)
        style_ax(ax, add_ygrid=True)

    axes[0].set_ylabel("Frequency")

    finalize_figure(fig, "fig_distributions_prepared_qa_gap.png")


def save_neg_gap_quintile_plot(df: pd.DataFrame) -> None:
    sample = df[["NegGap", "CAR_m1_p1"]].dropna().copy()

    sample["NegGap_quintile"] = pd.qcut(
        sample["NegGap"],
        5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
        duplicates="drop",
    )

    grouped = (
        sample.groupby("NegGap_quintile", observed=True)["CAR_m1_p1"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])
    grouped["ci95"] = 1.96 * grouped["se"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        grouped["NegGap_quintile"],
        grouped["mean"],
        yerr=grouped["ci95"],
        capsize=3,
        color=GRAY_FILL,
        edgecolor=BLACK,
        linewidth=0.8,
        ecolor=BLACK,
    )
    ax.axhline(0, linewidth=1.0, color=BLACK)

    ax.set_title("Mean CAR[-1,+1] by NegGap Quintile", pad=10)
    ax.set_xlabel("NegGap quintile")
    ax.set_ylabel("Mean CAR[-1,+1]")

    style_ax(ax, add_ygrid=True)
    finalize_figure(fig, "fig_neg_gap_quintiles_car.png")


def save_neg_gap_decile_binscatter(df: pd.DataFrame) -> None:
    sample = df[["NegGap", "CAR_m1_p1"]].dropna().copy()

    sample["NegGap_decile"] = pd.qcut(
        sample["NegGap"],
        10,
        labels=False,
        duplicates="drop",
    )

    grouped = (
        sample.groupby("NegGap_decile", observed=True)
        .agg(
            neggap_mean=("NegGap", "mean"),
            car_mean=("CAR_m1_p1", "mean"),
            car_std=("CAR_m1_p1", "std"),
            n=("CAR_m1_p1", "count"),
        )
        .reset_index()
    )
    grouped["se"] = grouped["car_std"] / np.sqrt(grouped["n"])
    grouped["ci95"] = 1.96 * grouped["se"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        grouped["neggap_mean"],
        grouped["car_mean"],
        yerr=grouped["ci95"],
        fmt="o-",
        capsize=3,
        color=BLACK,
        ecolor=BLACK,
        markerfacecolor=GRAY_FILL_2,
        markeredgecolor=BLACK,
        markersize=5,
        linewidth=1.1,
    )
    ax.axhline(0, linewidth=1.0, color=BLACK)
    ax.axvline(0, linestyle="--", linewidth=1.0, color=BLACK)

    ax.set_title("Binned Relation Between NegGap and CAR[-1,+1]", pad=10)
    ax.set_xlabel("Mean NegGap within decile")
    ax.set_ylabel("Mean CAR[-1,+1]")

    style_ax(ax, add_ygrid=True)
    finalize_figure(fig, "fig_neg_gap_binscatter_car.png")


def _year_ticks_from_labels(labels: pd.Series) -> tuple[list[int], list[str]]:
    labels = labels.astype(str).tolist()
    positions: list[int] = []
    tick_labels: list[str] = []
    seen_years: set[str] = set()

    for i, lab in enumerate(labels):
        year = lab[:4]
        if year not in seen_years:
            seen_years.add(year)
            positions.append(i)
            tick_labels.append(year)

    return positions, tick_labels


def save_combined_negativity_time_series(df: pd.DataFrame) -> None:
    if "year_quarter" not in df.columns:
        return

    cols = ["year_quarter", "NegPrepared", "NegQA", "NegGap"]
    sample = df[cols].dropna().copy()

    grouped = (
        sample.groupby("year_quarter", observed=True)[["NegPrepared", "NegQA", "NegGap"]]
        .mean()
        .reset_index()
        .sort_values("year_quarter")
    )

    x = np.arange(len(grouped))
    xticks, xticklabels = _year_ticks_from_labels(grouped["year_quarter"])

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.plot(x, grouped["NegPrepared"], label="Prepared remarks", color=COLOR_PREPARED, linewidth=1.3)
    ax.plot(x, grouped["NegQA"], label="Q&A", color=COLOR_QA, linewidth=1.3)
    ax.plot(x, grouped["NegGap"], label="Relative Q&A negativity", color=COLOR_GAP, linewidth=1.3)
    ax.axhline(0, linestyle="--", linewidth=1.0, color=BLACK)

    ax.set_title("Average Negativity Measures Over Time", pad=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean score")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=35, ha="right")
    ax.margins(x=0.01)

    style_ax(ax, add_ygrid=True)
    ax.legend(frameon=False, loc="best")

    finalize_figure(fig, "fig_time_series_combined_negativity.png")


def save_text_measure_comparison_boxplot(df: pd.DataFrame) -> None:
    cols = ["NegGap", "NegGap_seglenw", "NegGap_segmax"]
    if not set(cols).issubset(df.columns):
        return

    labels = ["Baseline", "Seg.-len. weighted", "Most-negative portion"]
    sample = df[cols].dropna().copy()

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.boxplot(
        [sample[c] for c in cols],
        tick_labels=labels,
        showfliers=False,
        widths=0.5,
        patch_artist=True,
        boxprops=dict(facecolor=GRAY_FILL, edgecolor=BLACK, linewidth=0.9),
        medianprops=dict(color=BLACK, linewidth=1.1),
        whiskerprops=dict(color=BLACK, linewidth=0.9),
        capprops=dict(color=BLACK, linewidth=0.9),
    )
    ax.axhline(0, linestyle="--", linewidth=1.0, color=BLACK)

    ax.set_title("Alternative Transcript-Level Measures of Relative Q&A Negativity", pad=10)
    ax.set_ylabel("NegGap measure")

    style_ax(ax, add_ygrid=True)
    finalize_figure(fig, "fig_compare_text_measures_boxplot.png")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()

    df = load_data()

    save_three_distribution_panels(df)
    save_prepared_vs_qa_boxplot(df)
    save_neg_gap_quintile_plot(df)
    save_neg_gap_decile_binscatter(df)
    save_combined_negativity_time_series(df)
    save_text_measure_comparison_boxplot(df)

    print("Saved graphs.")


if __name__ == "__main__":
    main()


