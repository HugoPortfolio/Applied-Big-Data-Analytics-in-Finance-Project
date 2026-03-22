from pathlib import Path
import pandas as pd

# Root paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "curated"

# Parameters
SOURCE_PATTERN = "koyfin_transcripts_*.parquet"

DATE_COL = "subheader"
DATE_FORMAT = "%A, %B %d, %Y %I:%M %p"

MERGED_OUTPUT = OUTPUT_DIR / "koyfin_transcripts_merged.parquet"

# True: deduplicate on all columns
# False: deduplicate only on the subset defined below
DROP_DUPLICATES_ON_ALL_COLUMNS = True

# Used only if DROP_DUPLICATES_ON_ALL_COLUMNS = False
DEDUP_SUBSET = ["title", "subheader", "body"]


def is_source_parquet(path: Path) -> bool:
    name = path.name
    return (
        name.startswith("koyfin_transcripts_")
        and name.endswith(".parquet")
        and not name.endswith("_trimmed.parquet")
        and "merged" not in name
    )


files = sorted([p for p in INPUT_DIR.glob(SOURCE_PATTERN) if is_source_parquet(p)])

if not files:
    raise FileNotFoundError(
        f"No source parquet files found in {INPUT_DIR} with pattern {SOURCE_PATTERN}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cleaned_dfs = []
total_source_rows = 0
total_clean_rows = 0
total_unparsed = 0

print("STEP 1: REMOVE UNPARSED DATES IN MEMORY")

for input_path in files:
    df = pd.read_parquet(input_path)

    if df.empty:
        print(f"{input_path.name} | EMPTY -> skipped")
        continue

    if DATE_COL not in df.columns:
        print(f"{input_path.name} | missing column '{DATE_COL}' -> skipped")
        continue

    # Parse the date column using the known Koyfin subheader format
    parsed_dates = pd.to_datetime(
        df[DATE_COL],
        format=DATE_FORMAT,
        errors="coerce"
    )

    # Keep only rows with successfully parsed dates
    keep_mask = parsed_dates.notna()
    df_clean = df.loc[keep_mask].copy()

    n_source = len(df)
    n_clean = len(df_clean)
    n_unparsed = parsed_dates.isna().sum()

    total_source_rows += n_source
    total_clean_rows += n_clean
    total_unparsed += n_unparsed

    cleaned_dfs.append(df_clean)

    print(f"{input_path.name}")
    print(f"  source_rows   = {n_source}")
    print(f"  clean_rows    = {n_clean}")
    print(f"  unparsed_drop = {n_unparsed}")
    print()

if not cleaned_dfs:
    raise ValueError("No cleaned dataframes were created.")

print("STEP 2: MERGE CLEANED FILES")

merged = pd.concat(cleaned_dfs, ignore_index=True)
before_dedup = len(merged)

# Apply duplicate removal
if DROP_DUPLICATES_ON_ALL_COLUMNS:
    merged = merged.drop_duplicates().reset_index(drop=True)
else:
    merged = merged.drop_duplicates(subset=DEDUP_SUBSET).reset_index(drop=True)

# Sort final dataframe chronologically
merged = merged.assign(
    _parsed_date=pd.to_datetime(
        merged[DATE_COL],
        format=DATE_FORMAT,
        errors="coerce"
    )
).sort_values("_parsed_date").drop(columns="_parsed_date").reset_index(drop=True)

after_dedup = len(merged)
n_duplicates_removed = before_dedup - after_dedup

# Compute observed date range on final merged file
merged_dates = pd.to_datetime(
    merged[DATE_COL],
    format=DATE_FORMAT,
    errors="coerce"
)

first_observed_date = merged_dates.min()
last_observed_date = merged_dates.max()

# Save final merged parquet
MERGED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
merged.to_parquet(MERGED_OUTPUT, index=False)

print()
print("SUMMARY")
print(f"Source files processed     : {len(files)}")
print(f"Total source rows          : {total_source_rows}")
print(f"Total rows after unparsed  : {total_clean_rows}")
print(f"Total unparsed removed     : {total_unparsed}")
print(f"Merged rows before dedup   : {before_dedup}")
print(f"Duplicates removed         : {n_duplicates_removed}")
print(f"Merged rows after dedup    : {after_dedup}")
print(f"First observed date        : {first_observed_date}")
print(f"Last observed date         : {last_observed_date}")
print(f"Final output               : {MERGED_OUTPUT}")