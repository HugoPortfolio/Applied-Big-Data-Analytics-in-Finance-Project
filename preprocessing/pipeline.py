from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    INPUT_DIR,
    PARQUET_PATTERN,
    TICKER_METADATA_PATH,
    SEGMENTS_OUTPUT_PATH,
    CHUNKS_OUTPUT_PATH,
    VALIDATION_OUTPUT_PATH,
    BAD_PARQUET_REPORT_PATH,
)
from parsing import build_segments_and_trace
from labeling import add_section_labels
from enrichment import enrich_with_ticker_metadata
from chunking import FinBERTChunker
from validation import build_validation_df
from utils.logger import get_logger

logger = get_logger(__name__)


def load_koyfin_parquets(
    input_dir: Path,
    pattern: str,
) -> tuple[pd.DataFrame, list[tuple[Path, str]]]:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {input_dir} with pattern {pattern}")

    good_dfs, bad_files = [], []

    for file_path in files:
        try:
            good_dfs.append(pd.read_parquet(file_path))
        except Exception as exc:
            bad_files.append((file_path, str(exc)))

    if not good_dfs:
        raise ValueError("No valid parquet files could be loaded.")

    return pd.concat(good_dfs, ignore_index=True), bad_files


def load_ticker_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Ticker metadata file not found: {path}")
    return pd.read_parquet(path)


def save_parquet(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def build_bad_file_report(bad_files: list[tuple[Path, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"file_path": str(path), "error": err} for path, err in bad_files]
    ) if bad_files else pd.DataFrame(columns=["file_path", "error"])


def load_chunk_summary(chunks_path: Path) -> pd.DataFrame:
    return (
        pd.read_parquet(chunks_path, columns=["chunk_token_count"])
        if chunks_path.exists()
        else pd.DataFrame(columns=["chunk_token_count"])
    )


def fmt_int(x: float | int) -> str:
    return f"{int(x):,}"


def stats_line(series: pd.Series, pct: bool = False) -> str:
    d = series.describe()
    f = (lambda x: f"{100 * float(x):,.2f}%") if pct else (lambda x: f"{float(x):,.2f}")
    return (
        f"count={fmt_int(d['count'])} | mean={f(d['mean'])} | std={f(d['std'])} | "
        f"min={f(d['min'])} | p25={f(d['25%'])} | median={f(d['50%'])} | "
        f"p75={f(d['75%'])} | max={f(d['max'])}"
    )


def log_segment_summary(df_segments: pd.DataFrame) -> None:
    if df_segments.empty:
        return

    counts = df_segments["section"].value_counts().to_dict()
    logger.info(
        "Section counts      | Prepared=%s | Q=%s | A=%s | O=%s",
        fmt_int(counts.get("Prepared", 0)),
        fmt_int(counts.get("Q", 0)),
        fmt_int(counts.get("A", 0)),
        fmt_int(counts.get("O", 0)),
    )

    if "ticker" not in df_segments.columns:
        return

    matched_segments = int(df_segments["ticker"].notna().sum())
    company_stats = (
        df_segments[["company_name", "ticker"]]
        .dropna(subset=["company_name"])
        .drop_duplicates()
        .assign(has_ticker=lambda x: x["ticker"].notna())
        .groupby("company_name", as_index=False)["has_ticker"]
        .max()
    )

    total = len(company_stats)
    matched = int(company_stats["has_ticker"].sum())

    logger.info("Metadata match      | segment_matches=%s", fmt_int(matched_segments))
    logger.info(
        "Company coverage    | total=%s | matched=%s | missing=%s",
        fmt_int(total),
        fmt_int(matched),
        fmt_int(total - matched),
    )


def log_chunk_summary(chunk_stats: dict[str, Any], df_chunk_summary: pd.DataFrame) -> None:
    logger.info("Chunk-level rows    | %s", fmt_int(chunk_stats["n_chunks"]))

    if not df_chunk_summary.empty:
        logger.info(
            "Chunk token summary | %s",
            stats_line(df_chunk_summary["chunk_token_count"]),
        )


def log_validation_summary(df_validation: pd.DataFrame) -> None:
    if df_validation.empty:
        logger.warning("Validation dataframe is empty.")
        return

    logger.info(
        "Drop rate summary   | %s",
        stats_line(df_validation["drop_rate"], pct=True),
    )

    bad_drop = df_validation.loc[~df_validation["drop_rate_ok"]]
    logger.info(
        "Drop rate alerts    | transcripts_with_drop_rate_ge_5pct=%s",
        fmt_int(len(bad_drop)),
    )

    qa = df_validation.loc[
        (df_validation["n_q"] > 0) | (df_validation["n_a"] > 0),
        ["n_q", "n_a", "qa_balance_gap", "qa_balance_ratio"],
    ]

    if qa.empty:
        logger.info("Q/A balance         | no transcripts with Q/A detected")
        return

    logger.info("Q/A transcripts     | %s", fmt_int(len(qa)))
    logger.info("Q count summary     | %s", stats_line(qa["n_q"]))
    logger.info("A count summary     | %s", stats_line(qa["n_a"]))
    logger.info("Q/A gap summary     | %s", stats_line(qa["qa_balance_gap"]))
    logger.info("Q/A ratio summary   | %s", stats_line(qa["qa_balance_ratio"]))
    logger.info(
        "Q/A imbalance flags | ratio_lt_0.50=%s | ratio_lt_0.25=%s | zero_answer_or_zero_question=%s",
        fmt_int((qa["qa_balance_ratio"] < 0.50).sum()),
        fmt_int((qa["qa_balance_ratio"] < 0.25).sum()),
        fmt_int(((qa["n_q"] == 0) | (qa["n_a"] == 0)).sum()),
    )


class PreprocessingPipeline:
    def __init__(self, chunker: FinBERTChunker | None = None) -> None:
        self.chunker = chunker or FinBERTChunker()

    def run(self) -> dict[str, Any]:
        logger.info("Preprocessing run started")

        logger.info("Loading curated transcript parquet...")
        df_raw, bad_files = load_koyfin_parquets(INPUT_DIR, PARQUET_PATTERN)

        logger.info("Building speaker-level segments...")
        df_segments, df_trace = build_segments_and_trace(df_raw)

        logger.info("Applying section labels...")
        df_segments = add_section_labels(df_segments)

        logger.info("Loading ticker metadata...")
        df_metadata = load_ticker_metadata(TICKER_METADATA_PATH)

        logger.info("Merging company metadata...")
        df_segments = enrich_with_ticker_metadata(df_segments, df_metadata)

        logger.info("Running validation...")
        df_validation = build_validation_df(df_segments, df_trace)

        logger.info("Saving segments...")
        save_parquet(df_segments, SEGMENTS_OUTPUT_PATH)

        logger.info("Chunking segments for FinBERT and writing chunks by batch...")
        chunk_stats = self.chunker.transform_to_parquet(df_segments, CHUNKS_OUTPUT_PATH)

        logger.info("Saving validation...")
        save_csv(df_validation, VALIDATION_OUTPUT_PATH)

        if bad_files:
            logger.info("Saving bad parquet report...")
            save_csv(build_bad_file_report(bad_files), BAD_PARQUET_REPORT_PATH)

        logger.info("Loading lightweight chunk summary...")
        df_chunk_summary = load_chunk_summary(CHUNKS_OUTPUT_PATH)

        logger.info("Preprocessing run completed successfully")
        logger.info("Raw transcripts     | %s", fmt_int(len(df_raw)))
        logger.info("Speaker-level rows  | %s", fmt_int(len(df_segments)))
        logger.info("Saved segments      | %s", SEGMENTS_OUTPUT_PATH)
        logger.info("Saved chunks        | %s", CHUNKS_OUTPUT_PATH)
        logger.info("Saved validation    | %s", VALIDATION_OUTPUT_PATH)

        if bad_files:
            logger.warning("Unreadable parquet files skipped: %s", len(bad_files))

        log_segment_summary(df_segments)
        log_chunk_summary(chunk_stats, df_chunk_summary)
        log_validation_summary(df_validation)

        return {
            "raw_count": len(df_raw),
            "segments_rows": len(df_segments),
            "chunks_rows": chunk_stats["n_chunks"],
            "validation_rows": len(df_validation),
            "bad_files": bad_files,
            "segments_output_path": SEGMENTS_OUTPUT_PATH,
            "chunks_output_path": CHUNKS_OUTPUT_PATH,
            "validation_output_path": VALIDATION_OUTPUT_PATH,
        }