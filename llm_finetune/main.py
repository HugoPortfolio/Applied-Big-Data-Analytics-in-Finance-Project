from __future__ import annotations

from pathlib import Path

from config import (
    ANNOTATION_SAMPLE_PATH,
    LLM_LABELED_PATH,
    RUN_BUILD_SAMPLE,
    RUN_LLM_LABELING,
    RUN_TRAINING,
)
from dataset_builder import build_annotation_sample
from llm_labeler import label_dataset
from train import train_model


def main() -> None:
    sample_path = ANNOTATION_SAMPLE_PATH
    labeled_path = LLM_LABELED_PATH

    if RUN_BUILD_SAMPLE:
        print("[1/3] Building annotation sample...")
        sample_path = build_annotation_sample()
        print(f"Saved sample to: {sample_path}")
    else:
        print("[1/3] Skipped annotation sample build.")
        if not Path(sample_path).exists():
            raise FileNotFoundError(
                f"Sample file not found: {sample_path}. "
                "Enable RUN_BUILD_SAMPLE or generate the file first."
            )

    if RUN_LLM_LABELING:
        print("[2/3] Labeling sample with LLM...")
        labeled_path = label_dataset()
        print(f"Saved labeled data to: {labeled_path}")
    else:
        print("[2/3] Skipped LLM labeling.")
        if not Path(labeled_path).exists():
            raise FileNotFoundError(
                f"Labeled dataset not found: {labeled_path}. "
                "Enable RUN_LLM_LABELING or generate the file first."
            )

    if RUN_TRAINING:
        print("[3/3] Fine-tuning model...")
        metrics = train_model()
        print("Training completed successfully.")
        print(metrics)
    else:
        print("[3/3] Skipped training.")


if __name__ == "__main__":
    main()