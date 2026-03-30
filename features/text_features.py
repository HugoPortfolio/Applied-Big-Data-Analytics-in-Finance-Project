from __future__ import annotations

import pandas as pd

BASE_INFO_COLS = ["transcript_id", "company_name", "ticker", "date"]

SECTION_SPECS = [
    ("Prepared", "Prepared"),
    ("Q", "Q"),
    ("A", "A"),
    (["Q", "A"], "QA"),
]


def add_neg_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["neg_score"] = out["p_negative"] - out["p_positive"]
    return out


def build_base_info(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in BASE_INFO_COLS if c in df.columns]
    return df[cols].drop_duplicates(subset=["transcript_id"]).copy()


def build_segment_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct one score per original segment.

    segment_neg_score_mean:
        token-weighted average across chunks within segment,
        used only to undo the technical chunk split.

    segment_neg_score_maxchunk:
        most negative chunk within segment,
        used for the 'most negative portion of the response' robustness.
    """
    tmp = df.copy()
    tmp["weight"] = tmp["chunk_token_count"].clip(lower=1)
    tmp["weighted_score"] = tmp["neg_score"] * tmp["weight"]

    seg = (
        tmp.groupby(
            ["transcript_id", "segment_id", "section"],
            as_index=False,
        )
        .agg(
            weighted_score_sum=("weighted_score", "sum"),
            weight_sum=("weight", "sum"),
            n_tokens_segment=("chunk_token_count", "sum"),
            segment_neg_score_maxchunk=("neg_score", "max"),
        )
    )

    seg["segment_neg_score_mean"] = seg["weighted_score_sum"] / seg["weight_sum"]
    return seg.drop(columns=["weighted_score_sum", "weight_sum"])


def aggregate_section_equal_weight(
    seg_df: pd.DataFrame,
    sections: str | list[str],
    score_source_col: str,
    out_score_col: str,
    out_n_col: str,
    out_token_col: str,
) -> pd.DataFrame:
    """
    Equal weight across segments.
    """
    if isinstance(sections, str):
        sections = [sections]

    tmp = seg_df.loc[seg_df["section"].isin(sections)].copy()
    if tmp.empty:
        return pd.DataFrame(columns=["transcript_id", out_score_col, out_n_col, out_token_col])

    out = (
        tmp.groupby("transcript_id", as_index=False)
        .agg(
            **{
                out_score_col: (score_source_col, "mean"),
                out_n_col: ("segment_id", "count"),
                out_token_col: ("n_tokens_segment", "sum"),
            }
        )
    )
    return out


def aggregate_section_length_weighted(
    seg_df: pd.DataFrame,
    sections: str | list[str],
    score_source_col: str,
    out_score_col: str,
) -> pd.DataFrame:
    """
    Weight segment-level scores by total segment length.
    """
    if isinstance(sections, str):
        sections = [sections]

    tmp = seg_df.loc[seg_df["section"].isin(sections)].copy()
    if tmp.empty:
        return pd.DataFrame(columns=["transcript_id", out_score_col])

    tmp["weighted_segment_score"] = (
        tmp[score_source_col] * tmp["n_tokens_segment"].clip(lower=1)
    )

    out = (
        tmp.groupby("transcript_id", as_index=False)
        .agg(
            weighted_score_sum=("weighted_segment_score", "sum"),
            token_sum=("n_tokens_segment", "sum"),
        )
    )

    out[out_score_col] = out["weighted_score_sum"] / out["token_sum"]
    return out.drop(columns=["weighted_score_sum", "token_sum"])


def build_transcript_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline:
        - reconstruct segment score from chunks
        - equal weight across segments within each section

    Robustness A:
        - segment-length weighted aggregation

    Robustness B:
        - use the most negative chunk within each segment,
          then equal weight across segments
    """
    out = build_base_info(df)
    seg_df = build_segment_scores(df)

    for sections, short_name in SECTION_SPECS:
        short_lower = short_name.lower()

        # Baseline: equal weight across segments using mean segment score
        out = out.merge(
            aggregate_section_equal_weight(
                seg_df=seg_df,
                sections=sections,
                score_source_col="segment_neg_score_mean",
                out_score_col=f"Neg{short_name}",
                out_n_col=f"n_segments_{short_lower}",
                out_token_col=f"n_tokens_{short_lower}",
            ),
            on="transcript_id",
            how="left",
        )

        # Robustness A: segment-length weighted
        out = out.merge(
            aggregate_section_length_weighted(
                seg_df=seg_df,
                sections=sections,
                score_source_col="segment_neg_score_mean",
                out_score_col=f"Neg{short_name}_seglenw",
            ),
            on="transcript_id",
            how="left",
        )

        # Robustness B: most negative portion of the response
        out = out.merge(
            aggregate_section_equal_weight(
                seg_df=seg_df,
                sections=sections,
                score_source_col="segment_neg_score_maxchunk",
                out_score_col=f"Neg{short_name}_segmax",
                out_n_col=f"n_segments_{short_lower}_segmax",
                out_token_col=f"n_tokens_{short_lower}_segmax",
            ),
            on="transcript_id",
            how="left",
        )

    # Baseline
    out["NegGap"] = out["NegQA"] - out["NegPrepared"]

    # Robustness A: segment-length weighted
    out["NegGap_seglenw"] = out["NegQA_seglenw"] - out["NegPrepared_seglenw"]

    # Robustness B: most negative portion of the response
    out["NegGap_segmax"] = out["NegQA_segmax"] - out["NegPrepared_segmax"]

    return out
