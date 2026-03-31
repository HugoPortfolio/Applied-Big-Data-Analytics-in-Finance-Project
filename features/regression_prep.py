from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLS = [
    "NegGap",
    "CAR_m1_p1",
    "eps_surprise",
    "revenue_surprise",
    "log_marketCap",
    "AvgVolume_m20_m1",
]

WINSOR_COLS = ["CAR_m1_p1","eps_surprise", "revenue_surprise", "AbVol_0_p1"]


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() == 0:
        return x
    return x.clip(x.quantile(lower), x.quantile(upper))


def safe_log(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return np.log(x.where(x > 0))


def prepare_regression_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["year_quarter"] = out["date"].dt.to_period("Q").astype(str)

    for col in WINSOR_COLS:
        if col in out.columns:
            out[col] = winsorize(out[col])

    if "AvgVolume_m20_m1" in out.columns:
        out["log_AvgVolume_m20_m1"] = safe_log(out["AvgVolume_m20_m1"])

    if "n_tokens_qa" in out.columns:
        out["log_n_tokens_qa"] = safe_log(out["n_tokens_qa"])

    return out


def filter_regression_sample(df: pd.DataFrame, logger) -> pd.DataFrame:
    cols = [c for c in REQUIRED_COLS if c in df.columns]
    n_before = len(df)

    out = df.dropna(subset=cols).copy()

    logger.info("Regression filter    | cols=%s", cols)
    logger.info(
        "Rows kept            | %s/%s (%.2f%%)",
        f"{len(out):,}",
        f"{n_before:,}",
        100 * len(out) / n_before if n_before else 0,
    )
    return out


