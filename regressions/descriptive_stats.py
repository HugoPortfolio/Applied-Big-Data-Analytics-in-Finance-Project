from __future__ import annotations

from pathlib import Path
import pandas as pd

from config import FINAL_DATASET, RESULTS_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORED_DIR = PROJECT_ROOT / "data" / "scored"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CHUNK_CANDIDATES = [
    SCORED_DIR / "koyfin_chunks_scored_finetuned_sp500.parquet",
    SCORED_DIR / "koyfin_chunks_scored.parquet",
    PROCESSED_DIR / "koyfin_chunks_sp500.parquet",
    PROCESSED_DIR / "koyfin_chunks.parquet",
]

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


def load_chunk_data() -> pd.DataFrame | None:
    for path in CHUNK_CANDIDATES:
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
            print(f"Chunk data loaded from: {path}")
            return df
        except Exception as e:
            print(f"Skipping invalid chunk file: {path}")
            print(f"Reason: {e}")
    print("No valid chunk parquet found. Section tables will be skipped.")
    return None


def find_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def build_sample_overview(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    firm_col = find_first_existing_col(df, ["gvkey", "company_id", "ticker", "permno"])
    call_col = find_first_existing_col(df, ["transcript_id", "call_id"])
    date_col = find_first_existing_col(df, ["date", "call_date"])
    yq_col = find_first_existing_col(df, ["year_quarter"])

    rows: list[tuple[str, object]] = []

    rows.append(("Observations", len(df)))
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

    firm_quarter_distribution = None

    if firm_col and yq_col:
        quarters_per_firm = df.groupby(firm_col)[yq_col].nunique()

        rows.append(("Mean quarters per firm", round(quarters_per_firm.mean(), 2)))
        rows.append(("Median quarters per firm", round(quarters_per_firm.median(), 2)))

        firm_quarter_distribution = (
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

    elif firm_col and call_col:
        calls_per_firm = df.groupby(firm_col)[call_col].nunique()
        rows.append(("Mean calls per firm", round(calls_per_firm.mean(), 2)))
        rows.append(("Median calls per firm", round(calls_per_firm.median(), 2)))

    sample_overview = pd.DataFrame(rows, columns=["Statistic", "Value"])
    return sample_overview, firm_quarter_distribution


def build_section_distribution(chunks: pd.DataFrame | None) -> pd.DataFrame | None:
    if chunks is None or "section" not in chunks.columns:
        return None

    counts = (
        chunks["section"]
        .value_counts(dropna=False)
        .rename_axis("Section")
        .reset_index(name="Count")
    )
    counts["Share"] = counts["Count"] / counts["Count"].sum()
    return counts


def build_speaker_role_distribution(chunks: pd.DataFrame | None) -> pd.DataFrame | None:
    if chunks is None or "speaker_role" not in chunks.columns:
        return None

    counts = (
        chunks["speaker_role"]
        .value_counts(dropna=False)
        .rename_axis("Speaker role")
        .reset_index(name="Count")
    )
    counts["Share"] = counts["Count"] / counts["Count"].sum()
    return counts


def build_summary_table(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    vars_existing = [v for v in variables if v in df.columns]
    sample = df[vars_existing].dropna().copy()

    summary = sample.describe(percentiles=[0.25, 0.5, 0.75]).T
    summary = summary[["count", "mean", "std", "25%", "50%", "75%", "min", "max"]]
    summary = summary.rename(
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
    return summary


def build_correlation_table(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    vars_existing = [v for v in variables if v in df.columns]
    sample = df[vars_existing].dropna().copy()
    return sample.corr()


def print_title(title: str) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def print_df(df: pd.DataFrame, decimals: int = 4, index: bool = True) -> None:
    out = df.copy()
    float_cols = out.select_dtypes(include="number").columns
    out[float_cols] = out[float_cols].round(decimals)
    print(out.to_string(index=index))


def run_basic_checks(
    final_df: pd.DataFrame,
    section_distribution: pd.DataFrame | None,
    speaker_distribution: pd.DataFrame | None,
) -> None:
    print_title("Basic checks")

    checks = []

    obs = len(final_df)
    checks.append(("Final dataset observations > 0", obs > 0, obs))

    if "transcript_id" in final_df.columns:
        unique_calls = final_df["transcript_id"].nunique()
        checks.append(
            ("One row per call in final dataset", unique_calls == obs, f"{unique_calls} unique calls vs {obs} rows")
        )

    if "year_quarter" in final_df.columns:
        checks.append(
            ("Year-quarter count > 0", final_df["year_quarter"].nunique() > 0, final_df["year_quarter"].nunique())
        )

    if section_distribution is not None:
        share_sum = section_distribution["Share"].sum()
        checks.append(("Section shares sum to 1", abs(share_sum - 1.0) < 1e-8, share_sum))

    if speaker_distribution is not None:
        share_sum = speaker_distribution["Share"].sum()
        checks.append(("Speaker shares sum to 1", abs(share_sum - 1.0) < 1e-8, share_sum))

        if section_distribution is not None:
            sec = dict(zip(section_distribution["Section"], section_distribution["Count"]))
            spk = dict(zip(speaker_distribution["Speaker role"], speaker_distribution["Count"]))

            exec_expected = sec.get("Prepared", 0) + sec.get("A", 0)
            analyst_expected = sec.get("Q", 0)

            checks.append(
                ("Executive chunks = Prepared + A", spk.get("Executive", None) == exec_expected, f"{spk.get('Executive', None)} vs {exec_expected}")
            )
            checks.append(
                ("Analyst chunks = Q", spk.get("Analyst", None) == analyst_expected, f"{spk.get('Analyst', None)} vs {analyst_expected}")
            )

    check_df = pd.DataFrame(checks, columns=["Check", "Pass", "Value"])
    print_df(check_df, decimals=6, index=False)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    final_df = load_final_data()
    chunk_df = load_chunk_data()

    sample_overview, firm_quarter_distribution = build_sample_overview(final_df)
    section_distribution = build_section_distribution(chunk_df)
    speaker_distribution = build_speaker_role_distribution(chunk_df)
    summary_table = build_summary_table(final_df, SUMMARY_VARS)
    corr_table = build_correlation_table(final_df, CORR_VARS)

    # Console output
    print_title("Table 0 - Sample overview")
    print_df(sample_overview, decimals=2, index=False)

    if firm_quarter_distribution is not None:
        print_title("Table 0D - Firm-quarter coverage")
        print_df(firm_quarter_distribution, decimals=2, index=True)

    if section_distribution is not None:
        print_title("Table 0B - Section distribution")
        print_df(section_distribution, decimals=4, index=False)

    if speaker_distribution is not None:
        print_title("Table 0C - Speaker role distribution")
        print_df(speaker_distribution, decimals=4, index=False)

    print_title("Table 1 - Summary statistics")
    print_df(summary_table, decimals=4, index=True)

    print_title("Table 2 - Correlation matrix")
    print_df(corr_table, decimals=4, index=True)

    run_basic_checks(
        final_df=final_df,
        section_distribution=section_distribution,
        speaker_distribution=speaker_distribution,
    )

    # File export
    output_lines: list[str] = []

    output_lines.append("% TABLE 0: Sample overview")
    output_lines.append(sample_overview.to_latex(index=False))

    if firm_quarter_distribution is not None:
        output_lines.append("\n\n% TABLE 0D: Firm-quarter coverage")
        output_lines.append(firm_quarter_distribution.to_latex(float_format="%.2f"))

    if section_distribution is not None:
        output_lines.append("\n\n% TABLE 0B: Section distribution")
        output_lines.append(section_distribution.to_latex(index=False, float_format="%.4f"))

    if speaker_distribution is not None:
        output_lines.append("\n\n% TABLE 0C: Speaker role distribution")
        output_lines.append(speaker_distribution.to_latex(index=False, float_format="%.4f"))

    output_lines.append("\n\n% TABLE 1: Summary statistics")
    output_lines.append(summary_table.to_latex(float_format="%.4f"))

    output_lines.append("\n\n% TABLE 2: Correlation matrix")
    output_lines.append(corr_table.to_latex(float_format="%.4f"))

    output_path = RESULTS_DIR / "descriptive_statistics.txt"
    output_path.write_text("\n".join(output_lines), encoding="utf-8")

    print()
    print(f"Saved descriptive statistics to: {output_path}")


if __name__ == "__main__":
    main()