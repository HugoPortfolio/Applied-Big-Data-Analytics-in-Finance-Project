import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from transformers import BertTokenizer

from config import (
    CHUNK_MODEL_NAME,
    CHUNK_MAX_TOKENS,
    CHUNK_BLOCK_SIZE,
    CHUNK_N_JOBS,
    CHUNK_WRITE_BATCH_BLOCKS,
)
from logger import get_logger

logger = get_logger(__name__)


class FinBERTChunker:
    SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        model_name=CHUNK_MODEL_NAME,
        max_text_tokens=CHUNK_MAX_TOKENS,
        block_size=CHUNK_BLOCK_SIZE,
        n_jobs=CHUNK_N_JOBS,
        write_batch_blocks=CHUNK_WRITE_BATCH_BLOCKS,
    ):
        self.model_name = model_name
        self.max_text_tokens = max_text_tokens
        self.block_size = block_size
        self.n_jobs = n_jobs
        self.write_batch_blocks = write_batch_blocks
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = BertTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def split_sentences(self, text):
        if pd.isna(text):
            return []

        text = str(text).strip()
        if not text:
            return []

        return self.SENTENCE_SPLIT_REGEX.split(text)

    def token_count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def chunk_text(self, text):
        if pd.isna(text):
            return []

        text = str(text).strip()
        if not text:
            return []

        full_len = self.token_count(text)
        if full_len <= self.max_text_tokens:
            return [text]

        sentences = self.split_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_sentences = []
        current_len = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            sent_ids = self.tokenizer.encode(sent, add_special_tokens=False)
            sent_len = len(sent_ids)

            if sent_len > self.max_text_tokens:
                if current_sentences:
                    chunks.append(" ".join(current_sentences).strip())
                    current_sentences = []
                    current_len = 0

                for i in range(0, sent_len, self.max_text_tokens):
                    sub_ids = sent_ids[i:i + self.max_text_tokens]
                    sub_text = self.tokenizer.decode(
                        sub_ids,
                        skip_special_tokens=True,
                    ).strip()
                    if sub_text:
                        chunks.append(sub_text)
                continue

            if current_len + sent_len > self.max_text_tokens:
                if current_sentences:
                    chunks.append(" ".join(current_sentences).strip())
                current_sentences = [sent]
                current_len = sent_len
            else:
                current_sentences.append(sent)
                current_len += sent_len

        if current_sentences:
            chunks.append(" ".join(current_sentences).strip())

        return [chunk for chunk in chunks if chunk]

    def _prepare_segments(self, df_segments):
        keep_cols = [
            "transcript_id",
            "segment_id",
            "company_name",
            "ticker",
            "date",
            "title",
            "speaker_name",
            "speaker_role",
            "section",
            "content",
            "country",
            "industry",
            "sector",
            "size_class",
            "exchange",
            "trading_region",
            "last_price",
        ]
        keep_cols = [col for col in keep_cols if col in df_segments.columns]
        return df_segments[keep_cols].copy()

    def _make_blocks(self, df_segments):
        return [
            df_segments.iloc[i:i + self.block_size].copy()
            for i in range(0, len(df_segments), self.block_size)
        ]

    def _make_block_batches(self, blocks):
        return [
            blocks[i:i + self.write_batch_blocks]
            for i in range(0, len(blocks), self.write_batch_blocks)
        ]

    def _process_block(self, df_block, block_id, total_blocks):
        logger.info(
            "Chunking block %s/%s | rows=%s",
            block_id,
            total_blocks,
            f"{len(df_block):,}",
        )

        rows = []

        for row in df_block.itertuples(index=False):
            text = getattr(row, "content", "")
            chunks = self.chunk_text(text)

            if not chunks:
                continue

            row_dict = row._asdict()
            row_dict.pop("content", None)

            for chunk_order, chunk_text in enumerate(chunks, start=1):
                chunk_tokens = self.token_count(chunk_text)
                if chunk_tokens <= 0:
                    continue

                out = row_dict.copy()
                out["chunk_order"] = chunk_order
                out["chunk_text"] = chunk_text
                out["chunk_token_count"] = chunk_tokens
                rows.append(out)

        if not rows:
            return pd.DataFrame()

        df_chunks = pd.DataFrame(rows)
        df_chunks["chunk_id"] = (
            df_chunks["transcript_id"].astype(str)
            + "_"
            + df_chunks["segment_id"].astype(str)
            + "_"
            + df_chunks["chunk_order"].astype(str)
        )

        logger.info(
            "Finished block %s/%s | chunks=%s",
            block_id,
            total_blocks,
            f"{len(df_chunks):,}",
        )
        return df_chunks

    @staticmethod
    def _write_batch(df_batch, output_path, writer):
        if df_batch.empty:
            return writer

        table = pa.Table.from_pandas(df_batch, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)

        writer.write_table(table)
        return writer

    @staticmethod
    def _build_chunk_stats(total_chunks):
        return {"n_chunks": int(total_chunks)}

    def transform_to_parquet(self, df_segments, output_path: Path):
        if df_segments.empty:
            empty_df = pd.DataFrame(
                columns=[
                    "transcript_id",
                    "segment_id",
                    "company_name",
                    "ticker",
                    "date",
                    "title",
                    "speaker_name",
                    "speaker_role",
                    "section",
                    "country",
                    "industry",
                    "sector",
                    "size_class",
                    "exchange",
                    "trading_region",
                    "last_price",
                    "chunk_order",
                    "chunk_text",
                    "chunk_token_count",
                    "chunk_id",
                ]
            )
            empty_df.to_parquet(output_path, index=False)
            return self._build_chunk_stats(total_chunks=0)

        df_segments = self._prepare_segments(df_segments)
        blocks = self._make_blocks(df_segments)
        block_batches = self._make_block_batches(blocks)

        total_blocks = len(blocks)
        logger.info(
            "Chunking %s segments in %s blocks (%s write batches)...",
            f"{len(df_segments):,}",
            f"{total_blocks:,}",
            f"{len(block_batches):,}",
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()

        writer = None
        total_chunks = 0

        for batch_idx, block_batch in enumerate(block_batches, start=1):
            start_block = (batch_idx - 1) * self.write_batch_blocks + 1
            end_block = start_block + len(block_batch) - 1

            logger.info(
                "Processing write batch %s/%s | blocks=%s-%s",
                batch_idx,
                len(block_batches),
                start_block,
                end_block,
            )

            chunked_blocks = Parallel(
                n_jobs=self.n_jobs,
                backend="loky",
            )(
                delayed(self._process_block)(
                    block,
                    block_id=start_block + i,
                    total_blocks=total_blocks,
                )
                for i, block in enumerate(block_batch)
            )

            chunked_blocks = [df for df in chunked_blocks if not df.empty]
            if not chunked_blocks:
                continue

            df_batch = pd.concat(chunked_blocks, ignore_index=True)
            df_batch = df_batch[df_batch["chunk_token_count"] > 0].reset_index(drop=True)

            total_chunks += len(df_batch)
            writer = self._write_batch(df_batch, output_path, writer)

            logger.info(
                "Written write batch %s/%s | batch_chunks=%s | total_chunks=%s",
                batch_idx,
                len(block_batches),
                f"{len(df_batch):,}",
                f"{total_chunks:,}",
            )

        if writer is not None:
            writer.close()

        logger.info("Chunking completed | total chunks=%s", f"{total_chunks:,}")
        return self._build_chunk_stats(total_chunks=total_chunks)