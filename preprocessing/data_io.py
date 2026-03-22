import pandas as pd


def load_koyfin_parquets(input_dir, pattern):
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files found in {input_dir} with pattern {pattern}"
        )

    good_dfs = []
    bad_file = None

    for file_path in files:
        try:
            good_dfs.append(pd.read_parquet(file_path))
        except Exception as e:
            bad_file = (file_path, str(e))

    if bad_file is not None:
        path, err = bad_file
        print("\nBad parquet file detected:")
        print(f"- {path}")
        print(f"  error: {err}")

    if not good_dfs:
        raise ValueError("No valid parquet files could be loaded.")

    return pd.concat(good_dfs, ignore_index=True), bad_file


def load_ticker_metadata(path):
    if not path.exists():
        raise FileNotFoundError(f"Ticker metadata file not found: {path}")
    return pd.read_parquet(path)


def save_segments(df_segments, output_path):
    df_segments.to_parquet(output_path, index=False)


def save_validation(df_validation, output_path):
    df_validation.to_csv(output_path, index=False)