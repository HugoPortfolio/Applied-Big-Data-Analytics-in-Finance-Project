import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import FINAL_DATASET, RESULTS_DIR


def load_data() -> pd.DataFrame:
    return pd.read_parquet(FINAL_DATASET)


def save_summary_statistics(df: pd.DataFrame):
    cols = [
        "NegPrepared",
        "NegQA",
        "NegGap",
        "CAR_m1_p1",
        "eps_surprise",
        "revenue_surprise",
        "log_marketCap",
        "log_AvgVolume_m20_m1",
        "log_n_tokens_qa",
    ]

    sample = df[cols].dropna().copy()

    summary = sample.describe(percentiles=[0.25, 0.5, 0.75]).T
    summary = summary[["count", "mean", "std", "25%", "50%", "75%"]]
    summary.columns = ["N", "Mean", "Std. Dev.", "P25", "Median", "P75"]

    summary.to_csv(RESULTS_DIR / "summary_statistics.csv")


def save_histogram(df: pd.DataFrame, column: str, title: str, xlabel: str, filename: str):
    sample = df[[column]].dropna().copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sample[column], bins=50)
    ax.axvline(0, linestyle="--", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_neg_gap_histogram(df: pd.DataFrame):
    save_histogram(
        df=df,
        column="NegGap",
        title="Distribution of Negative Gap",
        xlabel="NegGap",
        filename="fig_neg_gap_histogram.png",
    )


def save_neg_prepared_histogram(df: pd.DataFrame):
    save_histogram(
        df=df,
        column="NegPrepared",
        title="Distribution of Prepared Negativity",
        xlabel="NegPrepared",
        filename="fig_neg_prepared_histogram.png",
    )


def save_neg_qa_histogram(df: pd.DataFrame):
    save_histogram(
        df=df,
        column="NegQA",
        title="Distribution of Q&A Negativity",
        xlabel="NegQA",
        filename="fig_neg_qa_histogram.png",
    )


def save_prepared_vs_qa_boxplot(df: pd.DataFrame):
    sample = df[["NegPrepared", "NegQA"]].dropna().copy()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(
        [sample["NegPrepared"], sample["NegQA"]],
        tick_labels=["Prepared", "Q&A"],
        showfliers=False,
    )
    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_title("Prepared Remarks vs Q&A Negativity")
    ax.set_ylabel("Negativity score")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_prepared_vs_qa_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_neg_gap_quintile_plot(df: pd.DataFrame):
    needed = ["NegGap", "CAR_m1_p1"]
    sample = df[needed].dropna().copy()

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
    ax.bar(grouped["NegGap_quintile"], grouped["mean"], yerr=grouped["ci95"], capsize=4)
    ax.axhline(0, linewidth=1)

    ax.set_title("Mean CAR[-1,+1] by NegGap Quintile")
    ax.set_xlabel("NegGap quintile")
    ax.set_ylabel("Mean CAR[-1,+1]")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_neg_gap_quintiles_car.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    grouped.to_csv(RESULTS_DIR / "neg_gap_quintiles_car.csv", index=False)


def save_neg_gap_decile_binscatter(df: pd.DataFrame):
    needed = ["NegGap", "CAR_m1_p1"]
    sample = df[needed].dropna().copy()

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
        capsize=4,
    )
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linestyle="--", linewidth=1)

    ax.set_title("Binned relation between NegGap and CAR[-1,+1]")
    ax.set_xlabel("Mean NegGap within decile")
    ax.set_ylabel("Mean CAR[-1,+1]")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_neg_gap_binscatter_car.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    grouped.to_csv(RESULTS_DIR / "neg_gap_binscatter_car.csv", index=False)


def save_prepared_quintile_car_plot(df: pd.DataFrame):
    needed = ["NegPrepared", "CAR_m1_p1"]
    sample = df[needed].dropna().copy()

    sample["Prepared_quintile"] = pd.qcut(
        sample["NegPrepared"],
        5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
        duplicates="drop",
    )

    grouped = (
        sample.groupby("Prepared_quintile", observed=True)["CAR_m1_p1"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])
    grouped["ci95"] = 1.96 * grouped["se"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(grouped["Prepared_quintile"], grouped["mean"], yerr=grouped["ci95"], capsize=4)
    ax.axhline(0, linewidth=1)

    ax.set_title("Mean CAR[-1,+1] by Prepared Negativity Quintile")
    ax.set_xlabel("Prepared Negativity quintile")
    ax.set_ylabel("Mean CAR[-1,+1]")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_prepared_quintiles_car.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    grouped.to_csv(RESULTS_DIR / "prepared_quintiles_car.csv", index=False)


def save_time_series_plot(df: pd.DataFrame, variable: str, title: str, ylabel: str, filename: str):
    if "year_quarter" not in df.columns:
        return

    sample = df[["year_quarter", variable]].dropna().copy()

    grouped = (
        sample.groupby("year_quarter", observed=True)[variable]
        .mean()
        .reset_index()
        .sort_values("year_quarter")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(grouped["year_quarter"].astype(str), grouped[variable])
    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Year-quarter")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=90)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    grouped.to_csv(RESULTS_DIR / filename.replace(".png", ".csv"), index=False)


def save_neg_prepared_time_series(df: pd.DataFrame):
    save_time_series_plot(
        df=df,
        variable="NegPrepared",
        title="Average Prepared Negativity over Time",
        ylabel="Mean NegPrepared",
        filename="fig_time_series_neg_prepared.png",
    )


def save_neg_qa_time_series(df: pd.DataFrame):
    save_time_series_plot(
        df=df,
        variable="NegQA",
        title="Average Q&A Negativity over Time",
        ylabel="Mean NegQA",
        filename="fig_time_series_neg_qa.png",
    )


def save_neg_gap_time_series(df: pd.DataFrame):
    save_time_series_plot(
        df=df,
        variable="NegGap",
        title="Average Negative Gap over Time",
        ylabel="Mean NegGap",
        filename="fig_time_series_neg_gap.png",
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    save_summary_statistics(df)

    save_neg_gap_histogram(df)
    save_neg_prepared_histogram(df)
    save_neg_qa_histogram(df)

    save_prepared_vs_qa_boxplot(df)

    save_neg_gap_quintile_plot(df)
    save_neg_gap_decile_binscatter(df)
    save_prepared_quintile_car_plot(df)

    save_neg_prepared_time_series(df)
    save_neg_qa_time_series(df)
    save_neg_gap_time_series(df)

    print("Saved summary statistics and graphs.")


if __name__ == "__main__":
    main()