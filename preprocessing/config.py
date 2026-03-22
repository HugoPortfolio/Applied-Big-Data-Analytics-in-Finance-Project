from pathlib import Path

PREPROCESSING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PREPROCESSING_DIR.parent

INPUT_DIR = PROJECT_ROOT / "data" / "curated"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_PATTERN = "koyfin_transcripts_merged.parquet"
TICKER_METADATA_PATH = EXTERNAL_DIR / "TickerMetadata.pq"

SEGMENTS_OUTPUT_PATH = OUTPUT_DIR / "koyfin_speaker_segments.parquet"
VALIDATION_OUTPUT_PATH = OUTPUT_DIR / "koyfin_preprocessing_validation.csv"
BAD_PARQUET_REPORT_PATH = OUTPUT_DIR / "koyfin_bad_parquet_files.csv"