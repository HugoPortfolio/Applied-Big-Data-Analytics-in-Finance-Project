import logging

from config import (
    INPUT_DIR,
    PARQUET_PATTERN,
    SEGMENTS_OUTPUT_PATH,
    VALIDATION_OUTPUT_PATH,
    TICKER_METADATA_PATH,
)
from data_io import (
    load_koyfin_parquets,
    load_ticker_metadata,
    save_segments,
    save_validation,
)
from parsing import build_segments_and_trace
from labeling import add_section_labels
from enrichment import enrich_with_ticker_metadata
from validation import build_validation_df


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


def print_summary(raw_count, df_segments, df_validation, bad_file):
    logger.info("Raw transcripts    : %s", raw_count)
    logger.info("Speaker-level rows : %s", len(df_segments))
    logger.info("Saved parquet      : %s", SEGMENTS_OUTPUT_PATH)
    logger.info("Saved validation   : %s", VALIDATION_OUTPUT_PATH)

    if bad_file is not None:
        logger.warning("One unreadable parquet file was skipped.")

    if not df_segments.empty:
        logger.info(
            "\nSection counts:\n%s",
            df_segments["section"].value_counts(dropna=False).to_string(),
        )

        # Segment-level metadata match count
        matched_segments = df_segments["ticker"].notna().sum() if "ticker" in df_segments.columns else 0
        logger.info("Segments with ticker metadata match: %s", matched_segments)

        # Company-level metadata match count
        company_merge_stats = (
            df_segments[["company_name", "ticker"]]
            .dropna(subset=["company_name"])
            .drop_duplicates()
            .assign(has_ticker=lambda x: x["ticker"].notna())
            .groupby("company_name", as_index=False)["has_ticker"]
            .max()
        )

        n_companies_total = len(company_merge_stats)
        n_companies_matched = int(company_merge_stats["has_ticker"].sum())
        n_companies_missing = n_companies_total - n_companies_matched

        logger.info("Distinct companies          : %s", n_companies_total)
        logger.info("Companies matched to ticker : %s", n_companies_matched)
        logger.info("Companies missing ticker    : %s", n_companies_missing)

        missing_companies = (
            company_merge_stats.loc[~company_merge_stats["has_ticker"], "company_name"]
            .sort_values()
        )

        if not missing_companies.empty:
            logger.info(
                "\nSample missing companies:\n%s",
                missing_companies.head(20).to_string(index=False),
            )

    if df_validation.empty:
        logger.warning("Validation dataframe is empty.")
        return

    logger.info(
        "\nDrop rate summary:\n%s",
        df_validation["drop_rate"].describe().to_string(),
    )

    bad_drop = df_validation.loc[~df_validation["drop_rate_ok"]]
    logger.info("Transcripts with drop_rate >= 5%%: %s", len(bad_drop))

    if not bad_drop.empty:
        logger.info(
            "\nWorst drop-rate cases:\n%s",
            bad_drop[
                ["transcript_id", "company_name", "drop_rate"]
            ].sort_values("drop_rate", ascending=False).head(20).to_string(index=False),
        )

    qa_only = df_validation.loc[
        (df_validation["n_q"] > 0) | (df_validation["n_a"] > 0),
        ["n_q", "n_a", "qa_balance_gap", "qa_balance_ratio"],
    ]

    if qa_only.empty:
        logger.info("No transcripts with Q/A detected.")
        return

    logger.info(
        "\nQ/A balance summary:\n%s",
        qa_only.describe().to_string(),
    )


def main():
    logger.info("Loading curated transcript parquet...")
    df_raw, bad_file = load_koyfin_parquets(INPUT_DIR, PARQUET_PATTERN)

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

    logger.info("Saving outputs...")
    save_segments(df_segments, SEGMENTS_OUTPUT_PATH)
    save_validation(df_validation, VALIDATION_OUTPUT_PATH)

    print_summary(
        raw_count=len(df_raw),
        df_segments=df_segments,
        df_validation=df_validation,
        bad_file=bad_file,
    )


if __name__ == "__main__":
    main()