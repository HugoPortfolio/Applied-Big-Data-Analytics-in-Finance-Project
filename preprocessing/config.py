from pathlib import Path

# Root paths
PREPROCESSING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PREPROCESSING_DIR.parent

INPUT_DIR = PROJECT_ROOT / "data" / "curated"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
LOG_DIR = PROJECT_ROOT / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Input files
PARQUET_PATTERN = "koyfin_transcripts_merged.parquet"
TICKER_METADATA_PATH = EXTERNAL_DIR / "tickerMetadata.pq"

# Output files
SEGMENTS_OUTPUT_PATH = OUTPUT_DIR / "koyfin_speaker_segments.parquet"
CHUNKS_OUTPUT_PATH = OUTPUT_DIR / "koyfin_chunks.parquet"
VALIDATION_OUTPUT_PATH = OUTPUT_DIR / "koyfin_preprocessing_validation.csv"
BAD_PARQUET_REPORT_PATH = OUTPUT_DIR / "koyfin_bad_parquet_files.csv"

# Logging
PREPROCESSING_LOG_PATH = LOG_DIR / "preprocessing.log"

# Chunking parameters
CHUNK_MODEL_NAME = "yiyanghkust/finbert-tone"
CHUNK_MAX_TOKENS = 510
CHUNK_BLOCK_SIZE = 50_000
CHUNK_N_JOBS = 4
CHUNK_WRITE_BATCH_BLOCKS = 8

# Parsing
DATE_FORMAT = "%A, %B %d, %Y %I:%M %p"