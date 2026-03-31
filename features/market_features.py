from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MAX_EARNINGS_GAP_DAYS = 30
MIN_ABS_EPS_EST = 0.01
MIN_ABS_REV_EST = 1_000_000

# Constant mean return model settings
ESTIMATION_START = -250
ESTIMATION_END = -50
MIN_ESTIMATION_OBS = 60


def normalize_ticker(value: str) -> str:
    return str(value).strip().upper().replace(".", "-")


def find_ticker_file(folder: Path, ticker: str) -> Path | None:
    ticker = normalize_ticker(ticker)

    for path in (folder / f"{ticker}.parquet", folder / f"{ticker}.pq"):
        if path.exists():
            return path

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".parquet", ".pq"}:
            if normalize_ticker(path.stem) == ticker:
                return path

    return None


def load_ticker_panel(folder: Path, ticker: str) -> pd.DataFrame:
    path = find_ticker_file(folder, ticker)
    if path is None:
        return pd.DataFrame()

    df = pd.read_parquet(path).copy()

    if "symbol" in df.columns and "ticker" not in df.columns:
        df["ticker"] = df["symbol"]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    if "ticker" in df.columns:
        df["ticker_norm"] = df["ticker"].map(normalize_ticker)
    else:
        df["ticker_norm"] = normalize_ticker(ticker)

    return df


def load_feature_base(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    df["ticker_norm"] = df["ticker"].map(normalize_ticker)
    df["call_date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    return df


def _prepare_market_panel(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.sort_values("date").copy()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out = out.dropna(subset=["date", "close"]).copy()

    if out.empty:
        return out

    out["ret"] = np.log(out["close"] / out["close"].shift(1))
    return out


def _resolve_event_position(df: pd.DataFrame, call_date: pd.Timestamp) -> int | None:
    exact_idx = df.index[df["date"] == call_date]
    if len(exact_idx) > 0:
        return int(df.index.get_loc(exact_idx[0]))

    next_idx = df.index[df["date"] > call_date]
    if len(next_idx) > 0:
        return int(df.index.get_loc(next_idx[0]))

    return None


def _get_value_at_offset(df: pd.DataFrame, pos: int, col: str, offset: int) -> float:
    j = pos + offset
    if 0 <= j < len(df):
        value = df.iloc[j][col]
        return float(value) if pd.notna(value) else np.nan
    return np.nan


def _get_window_by_pos(df: pd.DataFrame, pos: int, col: str, start: int, end: int) -> pd.Series:
    i0 = pos + start
    i1 = pos + end + 1

    if i1 <= 0 or i0 >= len(df):
        return pd.Series(dtype="float64")

    i0 = max(0, i0)
    i1 = min(len(df), i1)

    if i0 >= i1:
        return pd.Series(dtype="float64")

    return df.iloc[i0:i1][col]


def _sum_if_any(values: list[float]) -> float:
    s = pd.Series(values, dtype="float64")
    return float(s.sum()) if s.notna().any() else np.nan


def _compute_expected_return(df: pd.DataFrame, pos: int) -> float:
    est_window = _get_window_by_pos(df, pos, "ret", ESTIMATION_START, ESTIMATION_END).dropna()
    if len(est_window) < MIN_ESTIMATION_OBS:
        return np.nan
    return float(est_window.mean())


def _compute_event_returns(df: pd.DataFrame, pos: int) -> tuple[float, float, float]:
    ret_m1 = _get_value_at_offset(df, pos, "ret", -1)
    ret_0 = _get_value_at_offset(df, pos, "ret", 0)
    ret_1 = _get_value_at_offset(df, pos, "ret", 1)
    return ret_m1, ret_0, ret_1


def _compute_abnormal_returns(
    ret_m1: float,
    ret_0: float,
    ret_1: float,
    mean_ret_est: float,
) -> tuple[float, float, float]:
    if pd.isna(mean_ret_est):
        return np.nan, np.nan, np.nan

    ar_m1 = ret_m1 - mean_ret_est if pd.notna(ret_m1) else np.nan
    ar_0 = ret_0 - mean_ret_est if pd.notna(ret_0) else np.nan
    ar_1 = ret_1 - mean_ret_est if pd.notna(ret_1) else np.nan
    return ar_m1, ar_0, ar_1


def _compute_volume_features(df: pd.DataFrame, pos: int) -> tuple[float, float, float]:
    avgvol_m20_m1 = _get_window_by_pos(df, pos, "volume", -20, -1).mean()
    avgvol_0_p1 = _get_window_by_pos(df, pos, "volume", 0, 1).mean()

    abvol_0_p1 = (
        avgvol_0_p1 / avgvol_m20_m1
        if pd.notna(avgvol_m20_m1) and avgvol_m20_m1 != 0
        else np.nan
    )
    return avgvol_m20_m1, avgvol_0_p1, abvol_0_p1


def build_market_event_features(df: pd.DataFrame, call_date: pd.Timestamp) -> dict[str, float]:
    df = _prepare_market_panel(df)
    if df.empty:
        return {}

    pos = _resolve_event_position(df, call_date)
    if pos is None:
        return {}

    ret_m1, ret_0, ret_1 = _compute_event_returns(df, pos)
    mean_ret_est = _compute_expected_return(df, pos)
    ar_m1, ar_0, ar_1 = _compute_abnormal_returns(ret_m1, ret_0, ret_1, mean_ret_est)

    avgvol_m20_m1, avgvol_0_p1, abvol_0_p1 = _compute_volume_features(df, pos)
    volatility_0_p5 = _get_window_by_pos(df, pos, "ret", 0, 5).std(ddof=0)

    return {
        "CAR_m1_p1": _sum_if_any([ar_m1, ar_0, ar_1]),
        "Volatility_0_p5": volatility_0_p5,
        "AvgVolume_m20_m1": avgvol_m20_m1,
        "AvgVolume_0_p1": avgvol_0_p1,
        "AbVol_0_p1": abvol_0_p1,
    }


def build_marketcap_feature(df: pd.DataFrame, call_date: pd.Timestamp) -> dict[str, float]:
    if df.empty:
        return {}

    out = df.sort_values("date").copy()
    out["marketCap"] = pd.to_numeric(out["marketCap"], errors="coerce")
    out = out.dropna(subset=["date", "marketCap"])
    out = out[out["date"] <= call_date]

    if out.empty:
        return {}

    market_cap = float(out.iloc[-1]["marketCap"])
    return {
        "marketCap": market_cap,
        "log_marketCap": np.log(market_cap) if market_cap > 0 else np.nan,
    }


def build_earning_feature(df: pd.DataFrame, call_date: pd.Timestamp) -> dict[str, float]:
    if df.empty:
        return {}

    out = df.sort_values("date").copy()

    for col in ["epsEstimated", "epsActual", "revenueEstimated", "revenueActual"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["date"])
    out = out[out["date"] <= call_date]
    if out.empty:
        return {}

    row = out.iloc[-1]
    earnings_date = pd.to_datetime(row["date"]).normalize()

    if (call_date - earnings_date).days > MAX_EARNINGS_GAP_DAYS:
        return {
            "earnings_date": pd.NaT,
            "epsEstimated": np.nan,
            "epsActual": np.nan,
            "revenueEstimated": np.nan,
            "revenueActual": np.nan,
            "eps_surprise": np.nan,
            "revenue_surprise": np.nan,
        }

    eps_est = row.get("epsEstimated", np.nan)
    eps_act = row.get("epsActual", np.nan)
    rev_est = row.get("revenueEstimated", np.nan)
    rev_act = row.get("revenueActual", np.nan)

    eps_surprise = (
        (eps_act - eps_est) / abs(eps_est)
        if pd.notna(eps_est) and pd.notna(eps_act) and abs(eps_est) >= MIN_ABS_EPS_EST
        else np.nan
    )

    revenue_surprise = (
        (rev_act - rev_est) / abs(rev_est)
        if pd.notna(rev_est) and pd.notna(rev_act) and abs(rev_est) >= MIN_ABS_REV_EST
        else np.nan
    )

    return {
        "earnings_date": earnings_date,
        "epsEstimated": eps_est,
        "epsActual": eps_act,
        "revenueEstimated": rev_est,
        "revenueActual": rev_act,
        "eps_surprise": eps_surprise,
        "revenue_surprise": revenue_surprise,
    }


def build_regression_dataset(
    df_features: pd.DataFrame,
    market_dir: Path,
    marketcap_dir: Path,
    earning_dir: Path,
    logger,
) -> pd.DataFrame:
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    rows = []

    def get_panel(kind: str, folder: Path, ticker: str) -> pd.DataFrame:
        key = (kind, ticker)
        if key not in cache:
            cache[key] = load_ticker_panel(folder, ticker)
        return cache[key]

    total = len(df_features)

    for i, row in enumerate(df_features.itertuples(index=False), start=1):
        ticker = row.ticker_norm
        call_date = row.call_date
        out = row._asdict()

        out.update(build_market_event_features(get_panel("market", market_dir, ticker), call_date))
        out.update(build_marketcap_feature(get_panel("marketcap", marketcap_dir, ticker), call_date))
        out.update(build_earning_feature(get_panel("earning", earning_dir, ticker), call_date))
        rows.append(out)

        if i % 500 == 0 or i == total:
            logger.info("Build progress       | %s/%s", f"{i:,}", f"{total:,}")

    return pd.DataFrame(rows)