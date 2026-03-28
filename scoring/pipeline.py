from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scoring.config import (
    CHUNKS_INPUT_PATH,
    SCORED_CHUNKS_OUTPUT_PATH,
    DROP_OPERATOR,
    KEEP_COLS,
)
from scoring.finbert_scorer import FinBERTScorer
from utils.logger import get_logger

logger = get_logger(__name__)


def load_chunks(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Chunks input file not found: {path}")
    return pd.read_parquet(path)


def prepare_chunks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    keep_cols = [col for col in KEEP_COLS if col in df.columns]
    df = df[keep_cols].copy()

    if DROP_OPERATOR and "section" in df.columns:
        before = len(df)
        df = df[df["section"] != "O"].copy()
        logger.info("Dropped Operator    | removed=%s", f"{before - len(df):,}")

    if "chunk_token_count" in df.columns:
        df = df.sort_values("chunk_token_count").reset_index(drop=True)
        logger.info("Sorted by length    | using chunk_token_count")

    return df.reset_index(drop=True)


def iter_batches(df: pd.DataFrame, batch_size: int):
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        yield start, end, df.iloc[start:end].copy()


def write_batch(df_batch: pd.DataFrame, output_path: Path, writer):
    if df_batch.empty:
        return writer

    table = pa.Table.from_pandas(df_batch, preserve_index=False)

    if writer is None:
        writer = pq.ParquetWriter(output_path, table.schema)

    writer.write_table(table)
    return writer


class ScoringPipeline:
    def __init__(self, scorer: FinBERTScorer | None = None):
        self.scorer = scorer or FinBERTScorer()

    def run(self) -> dict:
        logger.info("Scoring run started")

        logger.info("Loading chunks      | %s", CHUNKS_INPUT_PATH)
        df_chunks = load_chunks(CHUNKS_INPUT_PATH)
        logger.info("Loaded chunks       | rows=%s", f"{len(df_chunks):,}")

        logger.info("Preparing chunks for inference...")
        df_chunks = prepare_chunks(df_chunks)
        logger.info("Prepared chunks     | rows=%s", f"{len(df_chunks):,}")

        total_rows = len(df_chunks)
        batch_size = self.scorer.batch_size
        total_batches = (total_rows + batch_size - 1) // batch_size if total_rows > 0 else 0

        logger.info(
            "Scoring plan        | rows=%s | batch_size=%s | n_batches=%s",
            f"{total_rows:,}",
            batch_size,
            f"{total_batches:,}",
        )

        output_path = SCORED_CHUNKS_OUTPUT_PATH
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()

        writer = None
        total_scored = 0

        for batch_idx, (_, _, df_batch) in enumerate(iter_batches(df_chunks, batch_size), start=1):
            logger.info(
                "Processing batch    | %s/%s | rows=%s",
                f"{batch_idx:,}",
                f"{total_batches:,}",
                f"{len(df_batch):,}",
            )

            df_scored_batch = self.scorer.score_batch(df_batch)
            writer = write_batch(df_scored_batch, output_path, writer)

            total_scored += len(df_scored_batch)
            logger.info(
                "Written batch       | %s/%s | total_rows=%s",
                f"{batch_idx:,}",
                f"{total_batches:,}",
                f"{total_scored:,}",
            )

        if writer is not None:
            writer.close()
        else:
            pd.DataFrame().to_parquet(output_path, index=False)

        logger.info("Scoring completed successfully")
        logger.info("Scored rows         | %s", f"{total_scored:,}")
        logger.info("Saved scored chunks | %s", output_path)

        return {
            "scored_rows": total_scored,
            "output_path": output_path,
        }
