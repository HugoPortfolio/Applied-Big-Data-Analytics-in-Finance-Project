from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from config import (
    ANNOTATION_SAMPLE_PATH,
    CHUNKS_INPUT_PATH,
    DROP_OPERATOR_SECTION,
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
    OPTIONAL_SCORED_INPUT_PATH,
    RANDOM_SEED,
    SECTION_WEIGHTS,
    TARGET_SAMPLE_SIZE,
)

KEEP_COLS = [
    "chunk_id",
    "transcript_id",
    "segment_id",
    "chunk_order",
    "company_name",
    "ticker",
    "date",
    "title",
    "speaker_name",
    "speaker_role",
    "section",
    "chunk_text",
    "chunk_token_count",
]

EXTREME_SCORE_COLS = [
    "p_negative",
    "p_neutral",
    "p_positive",
    "pred_label",
    "neg_score",
]


def load_chunks(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")
    return pd.read_parquet(path)


def prepare_chunks(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [c for c in KEEP_COLS if c in df.columns]
    out = df[keep_cols].copy()

    if DROP_OPERATOR_SECTION and "section" in out.columns:
        out = out[out["section"] != "O"].copy()

    if "chunk_token_count" in out.columns:
        out = out[
            out["chunk_token_count"].between(MIN_CHUNK_TOKENS, MAX_CHUNK_TOKENS)
        ].copy()

    out["chunk_text"] = out["chunk_text"].astype(str).str.strip()
    out = out[out["chunk_text"] != ""].copy()

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    sort_cols = [
        c for c in ["date", "transcript_id", "segment_id", "chunk_order"]
        if c in out.columns
    ]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)

    return out


def attach_optional_teacher_scores(
    df: pd.DataFrame,
    scored_path: Path | None,
) -> pd.DataFrame:
    if scored_path is None:
        print("[dataset_builder] OPTIONAL_SCORED_INPUT_PATH is None. Skipping teacher scores.")
        return df.copy()

    scored_path = Path(scored_path)

    if not scored_path.exists():
        print(f"[dataset_builder] No optional scored parquet found at {scored_path}. Skipping teacher scores.")
        return df.copy()

    try:
        scored = pd.read_parquet(scored_path)
    except Exception as e:
        print(
            f"[dataset_builder] Could not read optional scored parquet at {scored_path}. "
            f"Skipping teacher scores. Error: {e}"
        )
        return df.copy()

    cols = [c for c in ["chunk_id", *EXTREME_SCORE_COLS] if c in scored.columns]
    if "chunk_id" not in cols or len(cols) == 1:
        print("[dataset_builder] Optional scored parquet has no usable score columns. Skipping teacher scores.")
        return df.copy()

    scored = scored[cols].drop_duplicates(subset=["chunk_id"])
    out = df.merge(scored, on="chunk_id", how="left")

    if "teacher_neg_score" not in out.columns:
        if "neg_score" in out.columns:
            out["teacher_neg_score"] = pd.to_numeric(out["neg_score"], errors="coerce")
        elif {"p_negative", "p_positive"}.issubset(out.columns):
            out["teacher_neg_score"] = (
                pd.to_numeric(out["p_negative"], errors="coerce")
                - pd.to_numeric(out["p_positive"], errors="coerce")
            )

    return out


def _sample_extremes(df_sec: pd.DataFrame, target_n: int) -> pd.DataFrame:
    if target_n <= 0 or df_sec.empty or "teacher_neg_score" not in df_sec.columns:
        return pd.DataFrame(columns=df_sec.columns)

    valid = df_sec.dropna(subset=["teacher_neg_score"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=df_sec.columns)

    ranked = valid.sort_values("teacher_neg_score")
    low_n = target_n // 2
    high_n = target_n - low_n

    low = ranked.head(low_n)
    high = ranked.tail(high_n)
    out = pd.concat([low, high], ignore_index=False)
    return out.drop_duplicates(subset=["chunk_id"])


def _sample_uniform(df_sec: pd.DataFrame, target_n: int) -> pd.DataFrame:
    if target_n <= 0 or df_sec.empty:
        return pd.DataFrame(columns=df_sec.columns)

    n = min(target_n, len(df_sec))
    return df_sec.sample(n=n, random_state=RANDOM_SEED)


def stratified_sample(df: pd.DataFrame, sample_size: int = TARGET_SAMPLE_SIZE) -> pd.DataFrame:
    sampled_parts: list[pd.DataFrame] = []

    for section, weight in SECTION_WEIGHTS.items():
        if "section" not in df.columns:
            continue

        df_sec = df[df["section"] == section].copy()
        if df_sec.empty:
            continue

        n_target = min(len(df_sec), int(math.floor(sample_size * weight)))
        has_teacher = (
            "teacher_neg_score" in df_sec.columns
            and df_sec["teacher_neg_score"].notna().any()
        )

        n_extreme = int(round(n_target * 0.30)) if has_teacher else 0
        n_uniform = max(0, n_target - n_extreme)

        part_extreme = _sample_extremes(df_sec, n_extreme)
        remaining = df_sec[~df_sec["chunk_id"].isin(part_extreme["chunk_id"])]
        part_uniform = _sample_uniform(remaining, n_uniform)

        sampled_parts.append(pd.concat([part_extreme, part_uniform], ignore_index=False))

    if sampled_parts:
        out = pd.concat(sampled_parts, ignore_index=True).drop_duplicates(subset=["chunk_id"])
    else:
        out = pd.DataFrame(columns=df.columns)

    if len(out) < sample_size:
        remaining = df[~df["chunk_id"].isin(out["chunk_id"])]
        extra_n = min(sample_size - len(out), len(remaining))
        if extra_n > 0:
            extra = remaining.sample(n=extra_n, random_state=RANDOM_SEED)
            out = pd.concat([out, extra], ignore_index=True)

    out = out.drop_duplicates(subset=["chunk_id"]).reset_index(drop=True)
    return out


def finalize_annotation_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["label"] = pd.NA
    out["label_reason"] = pd.NA
    out["label_source"] = "llm"
    out["llm_model"] = pd.NA
    out["label_status"] = "pending"

    preferred_order = [
        "chunk_id",
        "transcript_id",
        "segment_id",
        "chunk_order",
        "company_name",
        "ticker",
        "date",
        "title",
        "speaker_name",
        "speaker_role",
        "section",
        "chunk_token_count",
        "chunk_text",
        "pred_label",
        "p_negative",
        "p_neutral",
        "p_positive",
        "neg_score",
        "teacher_neg_score",
        "label",
        "label_reason",
        "label_source",
        "llm_model",
        "label_status",
    ]
    cols = [c for c in preferred_order if c in out.columns] + [
        c for c in out.columns if c not in preferred_order
    ]
    return out[cols].copy()


def build_annotation_sample(
    chunks_path: Path = CHUNKS_INPUT_PATH,
    scored_path: Path | None = OPTIONAL_SCORED_INPUT_PATH,
    output_path: Path = ANNOTATION_SAMPLE_PATH,
    sample_size: int = TARGET_SAMPLE_SIZE,
) -> Path:
    df = load_chunks(chunks_path)
    df = prepare_chunks(df)
    df = attach_optional_teacher_scores(df, scored_path)
    df = stratified_sample(df, sample_size=sample_size)
    df = finalize_annotation_frame(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return output_path


if __name__ == "__main__":
    path = build_annotation_sample()
    print(f"Saved annotation sample to: {path}")