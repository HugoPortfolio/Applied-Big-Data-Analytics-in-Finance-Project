from __future__ import annotations

import pandas as pd

BASE_INFO_COLS = ["transcript_id", "company_name", "ticker", "date"]
SECTION_SPECS = [
    ("Prepared", "NegPrepared", "n_chunks_prepared", "n_tokens_prepared"),
    ("Q", "NegQ", "n_chunks_q", "n_tokens_q"),
    ("A", "NegA", "n_chunks_a", "n_tokens_a"),
    (["Q", "A"], "NegQA", "n_chunks_qa", "n_tokens_qa"),
]


def add_neg_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["neg_score"] = out["p_negative"] - out["p_positive"]
    return out


def build_base_info(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in BASE_INFO_COLS if c in df.columns]
    return df[cols].drop_duplicates(subset=["transcript_id"]).copy()


def aggregate_section(
    df: pd.DataFrame,
    sections: str | list[str],
    score_col: str,
    count_col: str,
    token_col: str,
) -> pd.DataFrame:
    if isinstance(sections, str):
        sections = [sections]

    tmp = df.loc[df["section"].isin(sections)].copy()
    if tmp.empty:
        return pd.DataFrame(columns=["transcript_id", score_col, count_col, token_col])

    tmp["weight"] = tmp["chunk_token_count"].clip(lower=1)
    tmp["weighted_score"] = tmp["neg_score"] * tmp["weight"]

    out = (
        tmp.groupby("transcript_id", as_index=False)
        .agg(
            weighted_score_sum=("weighted_score", "sum"),
            weight_sum=("weight", "sum"),
            **{
                count_col: ("chunk_id", "count"),
                token_col: ("chunk_token_count", "sum"),
            },
        )
    )

    out[score_col] = out["weighted_score_sum"] / out["weight_sum"]
    return out.drop(columns=["weighted_score_sum", "weight_sum"])


def build_transcript_features(df: pd.DataFrame) -> pd.DataFrame:
    out = build_base_info(df)

    for sections, score_col, count_col, token_col in SECTION_SPECS:
        out = out.merge(
            aggregate_section(df, sections, score_col, count_col, token_col),
            on="transcript_id",
            how="left",
        )

    out["NegGap"] = out["NegQA"] - out["NegPrepared"]
    return out
