from pathlib import Path

REGRESSIONS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REGRESSIONS_DIR.parent

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "regressions"

FINAL_DATASET = FEATURES_DIR / "koyfin_regression_dataset_finetuned_sp500.parquet"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)