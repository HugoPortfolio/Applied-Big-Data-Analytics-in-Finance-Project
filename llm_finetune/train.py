from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
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
    EARLY_STOPPING_PATIENCE,
    EVAL_BATCH_SIZE,
    EVAL_PREDICTIONS_PATH,
    FP16,
    GRAD_ACCUM_STEPS,
    ID2LABEL,
    LABEL2ID,
    LEARNING_RATE,
    LLM_LABELED_PATH,
    LOGGING_STEPS,
    MAX_LENGTH,
    METRICS_PATH,
    MODEL_OUTPUT_DIR,
    RANDOM_SEED,
    SAVE_TOTAL_LIMIT,
    TEST_FRACTION,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_FRACTION,
    TRAIN_READY_PATH,
    VALID_FRACTION,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)


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


def save_train_ready_dataset(split: SplitData, path: Path = TRAIN_READY_PATH) -> Path:
    def _tag(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
        out = frame.copy()
        out["split"] = split_name
        return out

    train_ready = pd.concat(
        [
            _tag(split.train, "train"),
            _tag(split.valid, "valid"),
            _tag(split.test, "test"),
        ],
        ignore_index=True,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    train_ready.to_parquet(path, index=False)
    return path


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


def train_model(
    labeled_path: Path = LLM_LABELED_PATH,
    model_output_dir: Path = MODEL_OUTPUT_DIR,
) -> dict[str, Any]:
    df = load_labeled_data(labeled_path)
    split = group_split_by_transcript(df)
    save_train_ready_dataset(split)

    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

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

    if model_output_dir.exists():
        shutil.rmtree(model_output_dir)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(model_output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=TRAIN_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        fp16=FP16,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=SAVE_TOTAL_LIMIT,
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
                early_stopping_patience=EARLY_STOPPING_PATIENCE
            )
        ],
    )

    trainer.train()
    trainer.save_model(str(model_output_dir))
    tokenizer.save_pretrained(str(model_output_dir))

    test_output = trainer.predict(test_ds)
    test_logits = test_output.predictions
    test_probs = np.exp(test_logits - test_logits.max(axis=1, keepdims=True))
    test_probs = test_probs / test_probs.sum(axis=1, keepdims=True)
    test_pred_ids = np.argmax(test_probs, axis=1)

    eval_rows = (
        split.test[["chunk_id", "transcript_id", "chunk_text", "label"]]
        .copy()
        .reset_index(drop=True)
    )
    eval_rows["pred_label"] = [ID2LABEL[int(i)] for i in test_pred_ids]
    eval_rows["p_negative"] = test_probs[:, LABEL2ID["negative"]]
    eval_rows["p_neutral"] = test_probs[:, LABEL2ID["neutral"]]
    eval_rows["p_positive"] = test_probs[:, LABEL2ID["positive"]]
    eval_rows["neg_score"] = eval_rows["p_negative"] - eval_rows["p_positive"]

    EVAL_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    eval_rows.to_parquet(EVAL_PREDICTIONS_PATH, index=False)

    metrics = {
        "n_train": int(len(split.train)),
        "n_valid": int(len(split.valid)),
        "n_test": int(len(split.test)),
        **{
            k: float(v)
            for k, v in test_output.metrics.items()
            if isinstance(v, (int, float))
        },
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    metrics = train_model()
    print(json.dumps(metrics, indent=2))