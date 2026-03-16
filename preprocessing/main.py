import logging

from config import (
    INPUT_DIR,
    PARQUET_PATTERN,
    SEGMENTS_OUTPUT_PATH,
    VALIDATION_OUTPUT_PATH,
)
from data_io import (
    load_koyfin_parquets,
    save_segments,
    save_validation,
)
from parsing import build_segments_and_trace
from labeling import add_section_labels
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
    logger.info("Loading raw parquet shards...")
    df_raw, bad_file = load_koyfin_parquets(INPUT_DIR, PARQUET_PATTERN)

    logger.info("Building speaker-level segments...")
    df_segments, df_trace = build_segments_and_trace(df_raw)

    logger.info("Applying section labels...")
    df_segments = add_section_labels(df_segments)

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
