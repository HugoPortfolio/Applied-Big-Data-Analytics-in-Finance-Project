from pathlib import Path

FEATURES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FEATURES_DIR.parent

SCORED_DIR = PROJECT_ROOT / "data" / "scored"
OUTPUT_DIR = PROJECT_ROOT / "data" / "features"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
LOG_DIR = PROJECT_ROOT / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_LOG_PATH = LOG_DIR / "features.log"

LOGGER_CONFIG = {
    "log_path": FEATURES_LOG_PATH,
    "use_root": True,
    "propagate": False,
    "rotating_file": True,
}

SCORED_CHUNKS_INPUT_PATH = SCORED_DIR / "koyfin_chunks_scored_sp500.parquet"

MARKET_DIR = EXTERNAL_DIR / "market"
MARKETCAP_DIR = EXTERNAL_DIR / "marketCap"
EARNING_DIR = EXTERNAL_DIR / "earning"

TRANSCRIPT_FEATURES_OUTPUT_PATH = OUTPUT_DIR / "koyfin_transcript_features.parquet"
REGRESSION_DATASET_OUTPUT_PATH = OUTPUT_DIR / "koyfin_regression_dataset.parquet"
