import pandas as pd


def add_section_labels(df_segments: pd.DataFrame) -> pd.DataFrame:
    if df_segments.empty:
        return df_segments

    df_labeled = df_segments.sort_values(
        ["transcript_id", "segment_id"]
    ).copy()

    df_labeled["qa_started"] = (
        df_labeled["speaker_role"]
        .eq("Analyst")
        .groupby(df_labeled["transcript_id"])
        .cumsum()
        > 0
    )

    def compute_section(row: pd.Series) -> str:
        role = row["speaker_role"]

        if role == "Analyst":
            return "Q"
        if role == "Executive" and not row["qa_started"]:
            return "Prepared"
        if role == "Executive" and row["qa_started"]:
            return "A"
        return "O"

    df_labeled["section"] = df_labeled.apply(compute_section, axis=1)

    return df_labeled
