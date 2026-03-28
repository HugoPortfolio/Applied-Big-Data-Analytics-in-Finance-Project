from pathlib import Path

import pandas as pd

# PATHS
FILTERS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILTERS_DIR.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"

SP500_CONSTITUENTS_PATH = EXTERNAL_DIR / "sp500_constituents.csv"
SEGMENTS_INPUT_PATH = PROCESSED_DIR / "koyfin_speaker_segments.parquet"
CHUNKS_INPUT_PATH = PROCESSED_DIR / "koyfin_chunks.parquet"

SEGMENTS_SP500_OUTPUT_PATH = PROCESSED_DIR / "koyfin_speaker_segments_sp500.parquet"
CHUNKS_SP500_OUTPUT_PATH = PROCESSED_DIR / "koyfin_chunks_sp500.parquet"


def normalize_ticker(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .str.replace(".", "-", regex=False)
    )


def load_sp500_tickers(path: Path) -> set[str]:
    df = pd.read_csv(path)
    return set(normalize_ticker(df["Symbol"]).dropna())


def filter_to_sp500(df: pd.DataFrame, sp500_tickers: set[str]) -> pd.DataFrame:
    out = df.copy()
    out["_ticker_norm"] = normalize_ticker(out["ticker"])
    out = out[out["_ticker_norm"].isin(sp500_tickers)].copy()
    return out.drop(columns="_ticker_norm")


def print_stats(name: str, df_in: pd.DataFrame, df_out: pd.DataFrame) -> None:
    n_companies = df_out["ticker"].dropna().nunique()
    n_transcripts = df_out["transcript_id"].dropna().nunique()
    share = 100 * len(df_out) / len(df_in) if len(df_in) else 0

    print(f"{name} input rows              : {len(df_in):,}")
    print(f"{name} S&P500 rows             : {len(df_out):,} ({share:.2f}%)")
    print(f"{name} matched unique companies: {n_companies:,}")
    print(f"{name} matched transcripts      : {n_transcripts:,}")


def process_file(input_path: Path, output_path: Path, sp500_tickers: set[str], name: str) -> None:
    df_in = pd.read_parquet(input_path)
    df_out = filter_to_sp500(df_in, sp500_tickers)
    df_out.to_parquet(output_path, index=False)
    print_stats(name, df_in, df_out)


def main() -> None:
    sp500_tickers = load_sp500_tickers(SP500_CONSTITUENTS_PATH)
    print(f"S&P 500 tickers loaded       : {len(sp500_tickers):,}")

    process_file(
        input_path=SEGMENTS_INPUT_PATH,
        output_path=SEGMENTS_SP500_OUTPUT_PATH,
        sp500_tickers=sp500_tickers,
        name="Segments",
    )

    process_file(
        input_path=CHUNKS_INPUT_PATH,
        output_path=CHUNKS_SP500_OUTPUT_PATH,
        sp500_tickers=sp500_tickers,
        name="Chunks",
    )


if __name__ == "__main__":
    main()
