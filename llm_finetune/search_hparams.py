from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from config import (
    BASE_MODEL_NAME,
    EVAL_BATCH_SIZE,
    FP16,
    GRAD_ACCUM_STEPS,
    ID2LABEL,
    LABEL2ID,
    LLM_LABELED_PATH,
    MAX_LENGTH,
    RANDOM_SEED,
    TRAIN_FRACTION,
    VALID_FRACTION,
    TEST_FRACTION,
    WARMUP_RATIO,
)


SEARCH_DIR = Path(__file__).resolve().parent.parent / "models" / "hparam_search"
SEARCH_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = SEARCH_DIR / "search_results.csv"
BEST_JSON = SEARCH_DIR / "best_config.json"

LEARNING_RATES = [1e-5, 1.5e-5, 2e-5]
TRAIN_EPOCHS_LIST = [2, 3]
WEIGHT_DECAYS = [0.01, 0.02]
TRAIN_BATCH_SIZES = [8, 16]
EARLY_STOPPING_PATIENCES = [1, 2]

MAX_RUNS = None  # mets un entier si tu veux limiter
SELECT_METRIC = "eval_f1"  # on choisit le meilleur run sur validation, pas sur test


@dataclass
class SplitData:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def load_labeled_data(path: Path = LLM_LABELED_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Labeled dataset not found: {path}")

    df = pd.read_parquet(path).copy()

    if "label_status" in df.columns:
        df = df[df["label_status"] == "ok"].copy()

    df = df[df["label"].isin(LABEL2ID)].copy()
    df["chunk_text"] = df["chunk_text"].astype(str).str.strip()
    df = df[df["chunk_text"] != ""].copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df.empty:
        raise ValueError("No valid labeled rows available after filtering.")

    return df.reset_index(drop=True)


def group_split_by_transcript(df: pd.DataFrame) -> SplitData:
    if "transcript_id" not in df.columns:
        raise ValueError("Column 'transcript_id' is required.")

    transcript_frame = (
        df[["transcript_id", "date"]]
        .drop_duplicates(subset=["transcript_id"])
        .sort_values(["date", "transcript_id"])
        .reset_index(drop=True)
    )

    n_groups = len(transcript_frame)
    n_train = int(round(n_groups * TRAIN_FRACTION))
    n_valid = int(round(n_groups * VALID_FRACTION))
    n_test = n_groups - n_train - n_valid

    if n_train <= 0 or n_valid <= 0 or n_test <= 0:
        raise ValueError(
            f"Invalid split sizes with {n_groups=} and fractions "
            f"{TRAIN_FRACTION=}, {VALID_FRACTION=}, {TEST_FRACTION=}"
        )

    train_ids = set(transcript_frame.iloc[:n_train]["transcript_id"])
    valid_ids = set(transcript_frame.iloc[n_train:n_train + n_valid]["transcript_id"])
    test_ids = set(transcript_frame.iloc[n_train + n_valid:]["transcript_id"])

    train_df = df[df["transcript_id"].isin(train_ids)].copy()
    valid_df = df[df["transcript_id"].isin(valid_ids)].copy()
    test_df = df[df["transcript_id"].isin(test_ids)].copy()

    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("One of train/valid/test splits is empty.")

    return SplitData(train=train_df, valid=valid_df, test=test_df)


def hf_dataset_from_pandas(df: pd.DataFrame) -> Dataset:
    temp = df[["chunk_text", "label", "chunk_id", "transcript_id"]].copy()
    temp = temp.rename(columns={"chunk_text": "text"})
    temp["labels"] = temp["label"].map(LABEL2ID).astype(int)

    return Dataset.from_pandas(
        temp[["text", "labels", "chunk_id", "transcript_id"]],
        preserve_index=False,
    )


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def build_tokenized_datasets(split: SplitData, tokenizer) -> tuple[Dataset, Dataset, Dataset]:
    train_ds = hf_dataset_from_pandas(split.train)
    valid_ds = hf_dataset_from_pandas(split.valid)
    test_ds = hf_dataset_from_pandas(split.test)

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_ds = train_ds.map(tokenize, batched=True)
    valid_ds = valid_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=cols)
    valid_ds.set_format(type="torch", columns=cols)
    test_ds.set_format(type="torch", columns=cols)

    return train_ds, valid_ds, test_ds


def run_one_experiment(
    split: SplitData,
    tokenizer,
    train_ds: Dataset,
    valid_ds: Dataset,
    test_ds: Dataset,
    learning_rate: float,
    train_epochs: int,
    weight_decay: float,
    train_batch_size: int,
    early_stopping_patience: int,
    run_idx: int,
) -> dict[str, Any]:
    run_name = (
        f"run_{run_idx:03d}"
        f"_lr_{learning_rate}"
        f"_ep_{train_epochs}"
        f"_wd_{weight_decay}"
        f"_bs_{train_batch_size}"
        f"_pat_{early_stopping_patience}"
    )
    output_dir = SEARCH_DIR / run_name

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = BertForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=train_epochs,
        weight_decay=weight_decay,
        warmup_ratio=WARMUP_RATIO,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        fp16=FP16,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=RANDOM_SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience
            )
        ],
    )

    trainer.train()

    valid_metrics = trainer.evaluate(valid_ds)
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")

    result = {
        "run_name": run_name,
        "learning_rate": learning_rate,
        "train_epochs": train_epochs,
        "weight_decay": weight_decay,
        "train_batch_size": train_batch_size,
        "early_stopping_patience": early_stopping_patience,
        "n_train": len(split.train),
        "n_valid": len(split.valid),
        "n_test": len(split.test),
    }

    for k, v in valid_metrics.items():
        if isinstance(v, (int, float)):
            result[f"eval_{k.replace('eval_', '')}"] = float(v)

    for k, v in test_metrics.items():
        if isinstance(v, (int, float)):
            result[k] = float(v)

    return result


def main() -> None:
    df = load_labeled_data()
    split = group_split_by_transcript(df)

    print(f"Train rows: {len(split.train)}")
    print(f"Valid rows: {len(split.valid)}")
    print(f"Test rows : {len(split.test)}")

    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL_NAME)
    train_ds, valid_ds, test_ds = build_tokenized_datasets(split, tokenizer)

    grid = list(
        product(
            LEARNING_RATES,
            TRAIN_EPOCHS_LIST,
            WEIGHT_DECAYS,
            TRAIN_BATCH_SIZES,
            EARLY_STOPPING_PATIENCES,
        )
    )

    if MAX_RUNS is not None:
        grid = grid[:MAX_RUNS]

    print(f"Total runs to execute: {len(grid)}")

    results = []

    for run_idx, (lr, epochs, wd, bs, patience) in enumerate(grid, start=1):
        print(
            f"\n[{run_idx}/{len(grid)}] "
            f"lr={lr} | epochs={epochs} | wd={wd} | bs={bs} | patience={patience}"
        )

        result = run_one_experiment(
            split=split,
            tokenizer=tokenizer,
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
            learning_rate=lr,
            train_epochs=epochs,
            weight_decay=wd,
            train_batch_size=bs,
            early_stopping_patience=patience,
            run_idx=run_idx,
        )

        results.append(result)

        print(
            f"eval_f1={result.get('eval_f1'):.4f} | "
            f"test_f1={result.get('test_f1'):.4f} | "
            f"test_accuracy={result.get('test_accuracy'):.4f}"
        )

        df_results = pd.DataFrame(results).sort_values(
            by=SELECT_METRIC,
            ascending=False,
        )
        df_results.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")

    df_results = pd.DataFrame(results).sort_values(
        by=SELECT_METRIC,
        ascending=False,
    ).reset_index(drop=True)

    best = df_results.iloc[0].to_dict()

    with open(BEST_JSON, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print("\n=== TOP RESULTS ===")
    print(
        df_results[
            [
                "run_name",
                "learning_rate",
                "train_epochs",
                "weight_decay",
                "train_batch_size",
                "early_stopping_patience",
                "eval_f1",
                "test_f1",
                "test_accuracy",
            ]
        ].head(10)
    )

    print("\nBest config saved to:")
    print(BEST_JSON)
    print("All results saved to:")
    print(RESULTS_CSV)


if __name__ == "__main__":
    main()