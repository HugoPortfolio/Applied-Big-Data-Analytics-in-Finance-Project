import pandas as pd


SECTION_TO_COUNT_COLUMN = {
    "Prepared": "n_prepared",
    "Q": "n_q",
    "A": "n_a",
    "O": "n_o",
}


def build_validation_df(
    df_segments: pd.DataFrame,
    df_trace: pd.DataFrame,
) -> pd.DataFrame:
    if df_trace.empty:
        return pd.DataFrame()

    df_section_counts = _build_section_counts(df_segments)

    df_validation = df_trace.merge(
        df_section_counts,
        how="left",
        on="transcript_id",
    )

    for column in SECTION_TO_COUNT_COLUMN.values():
        if column not in df_validation.columns:
            df_validation[column] = 0

    df_validation[list(SECTION_TO_COUNT_COLUMN.values())] = (
        df_validation[list(SECTION_TO_COUNT_COLUMN.values())]
        .fillna(0)
        .astype(int)
    )

    df_validation["drop_rate"] = _compute_drop_rate(df_validation)
    df_validation["drop_rate_ok"] = df_validation["drop_rate"] < 0.05
    df_validation["qa_balance_gap"] = (
        df_validation["n_q"] - df_validation["n_a"]
    ).abs()
    df_validation["qa_balance_ratio"] = _compute_qa_balance_ratio(df_validation)

    return df_validation


def _build_section_counts(df_segments: pd.DataFrame) -> pd.DataFrame:
    if df_segments.empty:
        return pd.DataFrame(
            columns=["transcript_id", *SECTION_TO_COUNT_COLUMN.values()]
        )

    counts = (
        df_segments.assign(_count=1)
        .pivot_table(
            index="transcript_id",
            columns="section",
            values="_count",
            aggfunc="sum",
            fill_value=0,
        )
        .rename(columns=SECTION_TO_COUNT_COLUMN)
        .reset_index()
    )

    for column in SECTION_TO_COUNT_COLUMN.values():
        if column not in counts.columns:
            counts[column] = 0

    selected_columns = ["transcript_id", *SECTION_TO_COUNT_COLUMN.values()]
    return counts[selected_columns]


def _compute_drop_rate(df_validation: pd.DataFrame) -> pd.Series:
    raw_length = df_validation["raw_length"].clip(lower=0)
    segmented_length = df_validation["segmented_length"].clip(lower=0)

    drop_rate = pd.Series(0.0, index=df_validation.index, dtype="float64")

    valid_mask = raw_length > 0
    drop_rate.loc[valid_mask] = 1 - (
        segmented_length.loc[valid_mask] / raw_length.loc[valid_mask]
    )

    return drop_rate.clip(lower=0.0)


def _compute_qa_balance_ratio(df_validation: pd.DataFrame) -> pd.Series:
    max_qa = df_validation[["n_q", "n_a"]].max(axis=1)
    min_qa = df_validation[["n_q", "n_a"]].min(axis=1)

    ratio = pd.Series(pd.NA, index=df_validation.index, dtype="Float64")

    valid_mask = max_qa > 0
    ratio.loc[valid_mask] = (
        min_qa.loc[valid_mask] / max_qa.loc[valid_mask]
    ).astype("float64")

    return ratio