from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from config import (
    ANNOTATION_SAMPLE_PATH,
    LITELLM_API_KEY,
    LITELLM_TEMPERATURE,
    LITELLM_TIMEOUT,
    LLM_LABELED_PATH,
    MAX_PARALLEL_REQUESTS,
    MAX_RETRIES,
    SLEEP_BETWEEN_RETRIES,
    SYSTEM_PROMPT,
)

VALID_LABELS = {"negative", "neutral", "positive"}

INCEPTION_URL = "https://api.inceptionlabs.ai/v1/chat/completions"
INCEPTION_MODEL = "mercury-2"


class LabelingError(RuntimeError):
    pass


def _extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise LabelingError("Empty LLM response")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise LabelingError(f"Invalid JSON response: {text[:200]}")
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise LabelingError(f"Could not parse JSON response: {text[:200]}") from exc


def _normalize_label(label: str) -> str:
    label = (label or "").strip().lower()
    aliases = {
        "neg": "negative",
        "neu": "neutral",
        "pos": "positive",
    }
    label = aliases.get(label, label)
    if label not in VALID_LABELS:
        raise LabelingError(f"Invalid label: {label}")
    return label


def _safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _build_user_prompt(row: pd.Series) -> str:
    speaker = _safe_text(row.get("speaker_name", ""))
    role = _safe_text(row.get("speaker_role", ""))
    section = _safe_text(row.get("section", ""))
    company = _safe_text(row.get("company_name", ""))
    ticker = _safe_text(row.get("ticker", ""))
    date = _safe_text(row.get("date", ""))
    text = _safe_text(row.get("chunk_text", ""))

    return (
        "Classify the following earnings-call chunk.\n\n"
        f"Company: {company}\n"
        f"Ticker: {ticker}\n"
        f"Date: {date}\n"
        f"Section: {section}\n"
        f"Speaker: {speaker}\n"
        f"Speaker role: {role}\n\n"
        "Chunk:\n"
        f"{text}\n\n"
        'Return valid JSON only with this exact schema: '
        '{"label":"negative|neutral|positive","reason":"short explanation"}'
    )


def label_one_chunk(row: pd.Series) -> dict[str, Any]:
    if not LITELLM_API_KEY or LITELLM_API_KEY == "ta_vraie_cle_inception":
        raise RuntimeError("Missing Inception API key in config.py -> LITELLM_API_KEY")

    user_prompt = _build_user_prompt(row)
    last_error = None

    headers = {
        "Authorization": f"Bearer {LITELLM_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            payload = {
                "model": INCEPTION_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": max(0.5, float(LITELLM_TEMPERATURE)),
            }

            response = requests.post(
                INCEPTION_URL,
                headers=headers,
                json=payload,
                timeout=LITELLM_TIMEOUT,
            )

            if response.status_code != 200:
                raise LabelingError(
                    f"HTTP {response.status_code}: {response.text[:500]}"
                )

            data = response.json()
            raw_text = data["choices"][0]["message"]["content"]
            parsed = _extract_json_object(raw_text)

            label = _normalize_label(parsed.get("label", ""))
            reason = _safe_text(parsed.get("reason", ""))[:300]

            return {
                "chunk_id": row["chunk_id"],
                "label": label,
                "label_reason": reason,
                "label_source": "llm",
                "llm_model": INCEPTION_MODEL,
                "label_status": "ok",
                "llm_cost": None,
                "attempts": attempt,
                "raw_response": raw_text,
                "error": None,
            }

        except Exception as exc:
            last_error = str(exc)
            if attempt < MAX_RETRIES:
                time.sleep(SLEEP_BETWEEN_RETRIES * attempt)

    return {
        "chunk_id": row["chunk_id"],
        "label": None,
        "label_reason": None,
        "label_source": "llm",
        "llm_model": INCEPTION_MODEL,
        "label_status": "failed",
        "llm_cost": None,
        "attempts": MAX_RETRIES,
        "raw_response": None,
        "error": last_error,
    }


def load_annotation_sample(path: Path = ANNOTATION_SAMPLE_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Annotation sample not found: {path}")
    return pd.read_parquet(path)


def _safe_sort(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ["date", "transcript_id", "segment_id", "chunk_order"] if c in df.columns]
    if sort_cols:
        return df.sort_values(sort_cols).reset_index(drop=True)
    return df.reset_index(drop=True)


def label_dataset(
    input_path: Path = ANNOTATION_SAMPLE_PATH,
    output_path: Path = LLM_LABELED_PATH,
    max_rows: int | None = None,
) -> Path:
    df = load_annotation_sample(input_path)
    if max_rows is not None:
        df = df.head(max_rows).copy()

    rows = [row for _, row in df.iterrows()]
    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_REQUESTS) as ex:
        futures = {ex.submit(label_one_chunk, row): row["chunk_id"] for row in rows}

        for i, fut in enumerate(as_completed(futures), start=1):
            chunk_id = futures[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                results.append({
                    "chunk_id": chunk_id,
                    "label": None,
                    "label_reason": None,
                    "label_source": "llm",
                    "llm_model": INCEPTION_MODEL,
                    "label_status": "failed",
                    "llm_cost": None,
                    "attempts": 0,
                    "raw_response": None,
                    "error": str(exc),
                })

            if i % 25 == 0 or i == len(rows):
                ok_count = sum(r.get("label_status") == "ok" for r in results)
                failed_count = sum(r.get("label_status") == "failed" for r in results)
                print(f"[llm_labeler] progress {i}/{len(rows)} | ok={ok_count} | failed={failed_count}")

    df_labels = pd.DataFrame(results)

    base = df.drop(
        columns=[
            c for c in [
                "label",
                "label_reason",
                "label_source",
                "llm_model",
                "label_status",
                "llm_cost",
                "attempts",
                "raw_response",
                "error",
            ]
            if c in df.columns
        ],
        errors="ignore",
    )

    out = base.merge(df_labels, on="chunk_id", how="left")
    out = _safe_sort(out)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    ok_total = int((out["label_status"] == "ok").sum()) if "label_status" in out.columns else 0
    failed_total = int((out["label_status"] == "failed").sum()) if "label_status" in out.columns else 0

    print(f"[llm_labeler] saved -> {output_path}")
    print(f"[llm_labeler] final ok={ok_total} | failed={failed_total}")

    return output_path


if __name__ == "__main__":
    path = label_dataset()
    print(f"Saved LLM-labeled dataset to: {path}")