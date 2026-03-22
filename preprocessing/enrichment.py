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

# Conservative suffix list
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


def normalize_company_name(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text).strip().lower()

    # Remove accents
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))

    # Standardize common symbols
    text = text.replace("&", " and ")

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse spaces
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    # Remove trailing legal suffixes only
    parts = text.split()
    while parts and parts[-1] in LEGAL_SUFFIXES:
        parts.pop()

    return " ".join(parts).strip()


def enrich_with_ticker_metadata(
    df_segments: pd.DataFrame,
    df_metadata: pd.DataFrame,
) -> pd.DataFrame:
    if df_segments.empty:
        return df_segments

    left = df_segments.copy()
    right = df_metadata[KEEP_METADATA_COLS].copy()

    # Build normalized merge keys
    left["company_name_key"] = left["company_name"].map(normalize_company_name)
    right["name_key"] = right["name"].map(normalize_company_name)

    # Keep one row per normalized company name to avoid exploding rows
    right = right.drop_duplicates(subset=["name_key"], keep="first")

    df_enriched = left.merge(
        right,
        how="left",
        left_on="company_name_key",
        right_on="name_key",
    )

    # Optional cleanup of helper columns
    df_enriched = df_enriched.drop(columns=["name_key"], errors="ignore")

    return df_enriched