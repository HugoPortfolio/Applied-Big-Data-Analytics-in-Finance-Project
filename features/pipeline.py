from __future__ import annotations

import pandas as pd

from features.config import (
    EARNING_DIR,
    MARKET_DIR,
    MARKETCAP_DIR,
    REGRESSION_DATASET_OUTPUT_PATH,
    SCORED_CHUNKS_INPUT_PATH,
    TRANSCRIPT_FEATURES_OUTPUT_PATH,
)
from features.market_features import build_regression_dataset, load_feature_base
from features.regression_prep import filter_regression_sample, prepare_regression_dataset
from features.text_features import add_neg_score, build_transcript_features
from utils.logger import get_logger

logger = get_logger(__name__)


def load_scored_chunks(path) -> pd.DataFrame:
    return pd.read_parquet(path)


def log_non_null(df: pd.DataFrame, label: str, cols: list[str]) -> None:
    logger.info("%s rows        | %s", label, f"{len(df):,}")
    for col in cols:
        if col in df.columns:
            logger.info("%s non-null       | %s", col, f"{df[col].notna().sum():,}")


class FeaturesPipeline:
    def run(self) -> dict:
        logger.info("Features run started")

        df_scored = load_scored_chunks(SCORED_CHUNKS_INPUT_PATH)
        logger.info("Loaded scored chunks | rows=%s", f"{len(df_scored):,}")

        df_text = build_transcript_features(add_neg_score(df_scored))
        df_text.to_parquet(TRANSCRIPT_FEATURES_OUTPUT_PATH, index=False)
        log_non_null(df_text, "Transcript features", ["NegPrepared", "NegQ", "NegA", "NegQA", "NegGap"])

        df_base = load_feature_base(TRANSCRIPT_FEATURES_OUTPUT_PATH)
        df_reg = build_regression_dataset(
            df_features=df_base,
            market_dir=MARKET_DIR,
            marketcap_dir=MARKETCAP_DIR,
            earning_dir=EARNING_DIR,
            logger=logger,
        )

        df_reg = prepare_regression_dataset(df_reg)
        df_reg = filter_regression_sample(df_reg, logger)
        df_reg.to_parquet(REGRESSION_DATASET_OUTPUT_PATH, index=False)

        log_non_null(
            df_reg,
            "Regression dataset",
            [
                "CAR_m1_p1",
                "AbsRet_0_p1",
                "Volatility_0_p5",
                "AbVol_0_p1",
                "eps_surprise",
                "revenue_surprise",
                "log_marketCap",
            ],
        )

        logger.info("Features run completed successfully")
        return {
            "text_rows": len(df_text),
            "regression_rows": len(df_reg),
            "text_output_path": TRANSCRIPT_FEATURES_OUTPUT_PATH,
            "regression_output_path": REGRESSION_DATASET_OUTPUT_PATH,
        }
