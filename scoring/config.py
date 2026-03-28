from pathlib import Path

# ROOT PATHS
SCORING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCORING_DIR.parent

INPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "scored"
LOG_DIR = PROJECT_ROOT / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# LOGGING
SCORING_LOG_PATH = LOG_DIR / "scoring.log"

LOGGER_CONFIG = {
    "log_path": SCORING_LOG_PATH,
    "use_root": True,
    "propagate": False,
    "rotating_file": True,
}

# INPUT / OUTPUT FILES
CHUNKS_INPUT_PATH = INPUT_DIR / "koyfin_chunks.parquet"
SCORED_CHUNKS_OUTPUT_PATH = OUTPUT_DIR / "koyfin_chunks_scored.parquet"

# MODEL
MODEL_NAME = "yiyanghkust/finbert-tone"

# INFERENCE
BATCH_SIZE = 400

# FILTERING
DROP_OPERATOR = True

KEEP_COLS = [
    "chunk_id",
    "transcript_id",
    "segment_id",
    "company_name",
    "ticker",
    "date",
    "section",
    "speaker_role",
    "chunk_order",
    "chunk_text",
    "chunk_token_count",
]
