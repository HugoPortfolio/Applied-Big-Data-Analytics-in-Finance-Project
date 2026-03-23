from pathlib import Path
import pandas as pd


def load_koyfin_parquets(input_dir, pattern):
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files found in {input_dir} with pattern {pattern}"
        )

    good_dfs = []
    bad_files = []

    for file_path in files:
        try:
            good_dfs.append(pd.read_parquet(file_path))
        except Exception as exc:
            bad_files.append((file_path, str(exc)))

    if not good_dfs:
        raise ValueError("No valid parquet files could be loaded.")

    return pd.concat(good_dfs, ignore_index=True), bad_files


def load_ticker_metadata(path):
    if not path.exists():
        raise FileNotFoundError(f"Ticker metadata file not found: {path}")
    return pd.read_parquet(path)


def save_parquet(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def save_csv(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def build_bad_file_report(bad_files):
    if not bad_files:
        return pd.DataFrame(columns=["file_path", "error"])

    return pd.DataFrame(
        [{"file_path": str(path), "error": err} for path, err in bad_files]
    )