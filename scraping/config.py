from pathlib import Path




# ROOT PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOG_DIR = PROJECT_ROOT / "logs"


RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)




# CREDENTIALS
EMAIL = "ftd2026_user.fjifdidfdz@outlook.com" #adapt with your email and password
PWD = "ftd2026_pwd"




# URLS
LOGIN_URL = "https://app.koyfin.com/login?prevUrl=%2Fsearch%2Ftranscripts"
TARGET_URL = "https://app.koyfin.com/search/transcripts"




# GLOBAL PERIOD TO COVER
GLOBAL_START = "08/01/2017"
GLOBAL_END = "02/28/2026"


# SCRAPING BLOCK = 2 DAYS
WINDOW_DAYS = 1




# WAITS / RETRIES
DEFAULT_WAIT = 5
CLICK_WAIT = 1.4
READY_WAIT = 1.2
POLL = 0.01
RETRY_COUNT = 1




# PLACEHOLDER TEXTS
PLACEHOLDER_TEXTS = [
   "your document is on its way...",
   "your document is on its way",
]




# PARQUET WRITING
SHARD_SIZE = 30000
WRITE_BATCH_SIZE = 100
SHARD_PREFIX = "koyfin_transcripts"
SHARD_DIR = RAW_DATA_DIR
