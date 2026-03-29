from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MAX_EARNINGS_GAP_DAYS = 30
MIN_ABS_EPS_EST = 0.01
MIN_ABS_REV_EST = 1_000_000


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
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    df["ticker_norm"] = df["ticker"].map(normalize_ticker)
    return df


def load_feature_base(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    df["ticker_norm"] = df["ticker"].map(normalize_ticker)
    df["call_date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def build_market_event_features(df: pd.DataFrame, call_date: pd.Timestamp) -> dict[str, float]:
    if df.empty:
        return {}

    df = df.sort_values("date").copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).copy()
    if df.empty:
        return {}

    df["ret"] = df["close"].pct_change()

    idx = df.index[df["date"] == call_date]
    if len(idx) == 0:
        idx = df.index[df["date"] > call_date]
        if len(idx) == 0:
            return {}

    pos = df.index.get_loc(idx[0])

    def get_ret(offset: int) -> float:
        j = pos + offset
        if 0 <= j < len(df):
            value = df.iloc[j]["ret"]
            return float(value) if pd.notna(value) else np.nan
        return np.nan

    def get_window(col: str, start: int, end: int) -> pd.Series:
        i0 = max(0, pos + start)
        i1 = min(len(df), pos + end + 1)
        return df.iloc[i0:i1][col]

    ret_m1, ret_0, ret_1 = get_ret(-1), get_ret(0), get_ret(1)
    avgvol_m20_m1 = get_window("volume", -20, -1).mean()
    avgvol_0_p1 = get_window("volume", 0, 1).mean()

    return {
        "ret_m1": ret_m1,
        "ret_0": ret_0,
        "ret_1": ret_1,
        "CAR_m1_p1": np.nansum([ret_m1, ret_0, ret_1]),
        "AbsRet_0_p1": np.nansum(np.abs([ret_0, ret_1])),
        "Volatility_0_p5": get_window("ret", 0, 5).std(ddof=0),
        "AvgVolume_m20_m1": avgvol_m20_m1,
        "AvgVolume_0_p1": avgvol_0_p1,
        "AbVol_0_p1": avgvol_0_p1 / avgvol_m20_m1
        if pd.notna(avgvol_m20_m1) and avgvol_m20_m1 != 0
        else np.nan,
    }


def build_marketcap_feature(df: pd.DataFrame, call_date: pd.Timestamp) -> dict[str, float]:
    if df.empty:
        return {}

    df = df.sort_values("date").copy()
    df["marketCap"] = pd.to_numeric(df["marketCap"], errors="coerce")
    df = df.dropna(subset=["date", "marketCap"])
    df = df[df["date"] <= call_date]

    if df.empty:
        return {}

    market_cap = float(df.iloc[-1]["marketCap"])
    return {
        "marketCap": market_cap,
        "log_marketCap": np.log(market_cap) if market_cap > 0 else np.nan,
    }


def build_earning_feature(df: pd.DataFrame, call_date: pd.Timestamp) -> dict[str, float]:
    if df.empty:
        return {}

    df = df.sort_values("date").copy()

    for col in ["epsEstimated", "epsActual", "revenueEstimated", "revenueActual"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date"])
    df = df[df["date"] <= call_date]
    if df.empty:
        return {}

    row = df.iloc[-1]
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

    eps_est, eps_act = row.get("epsEstimated", np.nan), row.get("epsActual", np.nan)
    rev_est, rev_act = row.get("revenueEstimated", np.nan), row.get("revenueActual", np.nan)

    return {
        "earnings_date": earnings_date,
        "epsEstimated": eps_est,
        "epsActual": eps_act,
        "revenueEstimated": rev_est,
        "revenueActual": rev_act,
        "eps_surprise": (eps_act - eps_est) / abs(eps_est)
        if pd.notna(eps_est) and pd.notna(eps_act) and abs(eps_est) >= MIN_ABS_EPS_EST
        else np.nan,
        "revenue_surprise": (rev_act - rev_est) / abs(rev_est)
        if pd.notna(rev_est) and pd.notna(rev_act) and abs(rev_est) >= MIN_ABS_REV_EST
        else np.nan,
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
        ticker, call_date = row.ticker_norm, row.call_date
        out = row._asdict()

        out.update(build_market_event_features(get_panel("market", market_dir, ticker), call_date))
        out.update(build_marketcap_feature(get_panel("marketcap", marketcap_dir, ticker), call_date))
        out.update(build_earning_feature(get_panel("earning", earning_dir, ticker), call_date))
        rows.append(out)

        if i % 500 == 0 or i == total:
            logger.info("Build progress       | %s/%s", f"{i:,}", f"{total:,}")

    return pd.DataFrame(rows)
