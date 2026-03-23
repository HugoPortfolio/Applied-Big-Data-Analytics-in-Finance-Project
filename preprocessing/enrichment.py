import re
import unicodedata

import pandas as pd


KEEP_METADATA_COLS = [
    "ticker",
    "name",
    "country",
    "industry",
    "sector",
    "size class",
    "exchange",
    "trading region",
    "last price",
]

LEGAL_SUFFIXES = [
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "ltd",
    "limited",
    "llc",
    "plc",
    "sa",
    "spa",
    "ag",
    "nv",
    "se",
    "oyj",
    "ab",
    "asa",
    "holdings",
    "holding",
    "group",
]

RENAME_MAP = {
    "name": "metadata_name",
    "size class": "size_class",
    "trading region": "trading_region",
    "last price": "last_price",
}


def normalize_company_name(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text).strip().lower()

    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))

    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    parts = text.split()
    while parts and parts[-1] in LEGAL_SUFFIXES:
        parts.pop()

    return " ".join(parts).strip()


def prepare_ticker_metadata(df_metadata: pd.DataFrame) -> pd.DataFrame:
    right = df_metadata[KEEP_METADATA_COLS].copy()
    right = right.rename(columns=RENAME_MAP)

    right["metadata_name_key"] = right["metadata_name"].map(normalize_company_name)
    right = right.drop_duplicates(subset=["metadata_name_key"], keep="first")

    return right


def enrich_with_ticker_metadata(
    df_segments: pd.DataFrame,
    df_metadata: pd.DataFrame,
) -> pd.DataFrame:
    if df_segments.empty:
        return df_segments

    left = df_segments.copy()
    left["company_name_key"] = left["company_name"].map(normalize_company_name)

    right = prepare_ticker_metadata(df_metadata)

    df_enriched = left.merge(
        right,
        how="left",
        left_on="company_name_key",
        right_on="metadata_name_key",
    )

    df_enriched = df_enriched.drop(
        columns=["company_name_key", "metadata_name_key"],
        errors="ignore",
    )

    return df_enriched