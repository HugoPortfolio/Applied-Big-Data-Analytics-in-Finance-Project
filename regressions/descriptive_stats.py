from __future__ import annotations

import pandas as pd

from config import FINAL_DATASET, CHUNK_DATASET, RESULTS_DIR


SUMMARY_VARS = [
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

CORR_VARS = [
    "NegPrepared",
    "NegQA",
    "NegGap",
    "CAR_m1_p1",
    "eps_surprise",
    "revenue_surprise",
]


def load_final_data() -> pd.DataFrame:
    return pd.read_parquet(FINAL_DATASET)


def load_chunk_data() -> pd.DataFrame:
    df = pd.read_parquet(CHUNK_DATASET)
    print(f"Chunk data loaded from: {CHUNK_DATASET}")
    return df


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


def build_sample_overview(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    firm_col = first_existing(df, ["gvkey", "company_id", "ticker", "permno"])
    call_col = first_existing(df, ["transcript_id", "call_id"])
    date_col = first_existing(df, ["date", "call_date"])
    yq_col = first_existing(df, ["year_quarter"])

    rows: list[tuple[str, object]] = [("Observations", len(df))]

    if firm_col:
        rows.append(("Unique firms", df[firm_col].nunique()))
    if call_col:
        rows.append(("Unique earnings calls", df[call_col].nunique()))
    if yq_col:
        rows.append(("Unique year-quarter", df[yq_col].nunique()))

    if date_col:
        date_series = pd.to_datetime(df[date_col], errors="coerce")
        if date_series.notna().any():
            rows.append(("Start date", str(date_series.min().date())))
            rows.append(("End date", str(date_series.max().date())))

    coverage = None
    if firm_col and yq_col:
        quarters_per_firm = df.groupby(firm_col)[yq_col].nunique()
        rows.append(("Mean quarters per firm", round(quarters_per_firm.mean(), 2)))
        rows.append(("Median quarters per firm", round(quarters_per_firm.median(), 2)))

        coverage = (
            quarters_per_firm.describe(percentiles=[0.25, 0.5, 0.75])
            .to_frame(name="Value")
            .rename(
                index={
                    "count": "Firms",
                    "mean": "Mean quarters",
                    "std": "Std. Dev.",
                    "min": "Min",
                    "25%": "P25",
                    "50%": "Median",
                    "75%": "P75",
                    "max": "Max",
                }
            )
        )

    overview = pd.DataFrame(rows, columns=["Statistic", "Value"])
    return overview, coverage


def build_distribution(df: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
    out = (
        df[col]
        .value_counts(dropna=False)
        .rename_axis(name)
        .reset_index(name="Count")
    )
    out["Share"] = out["Count"] / out["Count"].sum()
    return out


def build_summary_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    out = df[cols].dropna().describe(percentiles=[0.25, 0.5, 0.75]).T
    out = out[["count", "mean", "std", "25%", "50%", "75%", "min", "max"]]
    return out.rename(
        columns={
            "count": "N",
            "mean": "Mean",
            "std": "Std. Dev.",
            "25%": "P25",
            "50%": "Median",
            "75%": "P75",
            "min": "Min",
            "max": "Max",
        }
    )


def build_corr_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    return df[cols].dropna().corr()


def build_measure_comparison(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["NegGap", "NegGap_seglenw", "NegGap_segmax"] if c in df.columns]

    labels = {
        "NegGap": "Baseline NegGap",
        "NegGap_seglenw": "Segment-length-weighted NegGap",
        "NegGap_segmax": "Most-negative-portion NegGap",
    }

    out = (
        df[cols]
        .agg(["mean", "std", "min", "median", "max"])
        .T
        .rename(
            columns={
                "mean": "Mean",
                "std": "Std. Dev.",
                "min": "Min",
                "median": "Median",
                "max": "Max",
            }
        )
    )

    if "NegGap" in cols:
        out["Corr. with baseline"] = df[cols].corr()["NegGap"]

    out.index = [labels[c] for c in out.index]
    return out.reset_index().rename(columns={"index": "Measure"})


def print_title(title: str) -> None:
    print(f"\n{title}\n" + "=" * len(title))


def print_df(df: pd.DataFrame, decimals: int = 4, index: bool = True) -> None:
    out = df.copy()
    num_cols = out.select_dtypes(include="number").columns
    out[num_cols] = out[num_cols].round(decimals)
    print(out.to_string(index=index))


def run_basic_checks(
    final_df: pd.DataFrame,
    section_dist: pd.DataFrame,
    speaker_dist: pd.DataFrame,
) -> None:
    checks = [
        ("Final dataset observations > 0", len(final_df) > 0, len(final_df)),
    ]

    if "transcript_id" in final_df.columns:
        unique_calls = final_df["transcript_id"].nunique()
        checks.append(
            ("One row per call in final dataset", unique_calls == len(final_df), f"{unique_calls} unique calls vs {len(final_df)} rows")
        )

    if "year_quarter" in final_df.columns:
        checks.append(
            ("Year-quarter count > 0", final_df["year_quarter"].nunique() > 0, final_df["year_quarter"].nunique())
        )

    checks.append(
        ("Section shares sum to 1", abs(section_dist["Share"].sum() - 1.0) < 1e-8, section_dist["Share"].sum())
    )
    checks.append(
        ("Speaker shares sum to 1", abs(speaker_dist["Share"].sum() - 1.0) < 1e-8, speaker_dist["Share"].sum())
    )

    sec = dict(zip(section_dist["Section"], section_dist["Count"]))
    spk = dict(zip(speaker_dist["Speaker role"], speaker_dist["Count"]))

    exec_expected = sec.get("Prepared", 0) + sec.get("A", 0)
    analyst_expected = sec.get("Q", 0)

    checks.append(
        ("Executive chunks = Prepared + A", spk.get("Executive") == exec_expected, f"{spk.get('Executive')} vs {exec_expected}")
    )
    checks.append(
        ("Analyst chunks = Q", spk.get("Analyst") == analyst_expected, f"{spk.get('Analyst')} vs {analyst_expected}")
    )

    print_title("Basic checks")
    print_df(pd.DataFrame(checks, columns=["Check", "Pass", "Value"]), decimals=6, index=False)


def latex_block(title: str, df: pd.DataFrame, *, index: bool = True, float_format: str = "%.4f") -> str:
    return f"% {title}\n" + df.to_latex(index=index, float_format=float_format)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    final_df = load_final_data()
    chunk_df = load_chunk_data()

    sample_overview, firm_coverage = build_sample_overview(final_df)
    section_dist = build_distribution(chunk_df, "section", "Section")
    speaker_dist = build_distribution(chunk_df, "speaker_role", "Speaker role")
    summary_table = build_summary_table(final_df, SUMMARY_VARS)
    corr_table = build_corr_table(final_df, CORR_VARS)
    measure_comparison = build_measure_comparison(final_df)

    tables = [
        ("Table 0 - Sample overview", sample_overview, False, 2),
        ("Table 0D - Firm-quarter coverage", firm_coverage, True, 2),
        ("Table 0B - Section distribution", section_dist, False, 4),
        ("Table 0C - Speaker role distribution", speaker_dist, False, 4),
        ("Table 1 - Summary statistics", summary_table, True, 4),
        ("Table 2 - Correlation matrix", corr_table, True, 4),
        ("Appendix Table - Comparison of text measures", measure_comparison, False, 4),
    ]

    for title, df, use_index, dec in tables:
        print_title(title)
        print_df(df, decimals=dec, index=use_index)

    run_basic_checks(final_df, section_dist, speaker_dist)

    latex_blocks = [
        latex_block("TABLE 0: Sample overview", sample_overview, index=False, float_format="%.2f"),
        latex_block("TABLE 0D: Firm-quarter coverage", firm_coverage, index=True, float_format="%.2f"),
        latex_block("TABLE 0B: Section distribution", section_dist, index=False, float_format="%.4f"),
        latex_block("TABLE 0C: Speaker role distribution", speaker_dist, index=False, float_format="%.4f"),
        latex_block("TABLE 1: Summary statistics", summary_table, index=True, float_format="%.4f"),
        latex_block("TABLE 2: Correlation matrix", corr_table, index=True, float_format="%.4f"),
        latex_block("APPENDIX TABLE: Comparison of text measures", measure_comparison, index=False, float_format="%.4f"),
    ]

    output_path = RESULTS_DIR / "descriptive_statistics.txt"
    output_path.write_text("\n\n".join(latex_blocks), encoding="utf-8")

    print(f"\nSaved descriptive statistics to: {output_path}")


if __name__ == "__main__":
    main()