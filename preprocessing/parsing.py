import re
from typing import Any

import pandas as pd

from config import DATE_FORMAT


SPEAKER_PATTERN = re.compile(
    r"(?m)^(OperatorOperator|[^\W\d_][^\n]{1,100}?(?:Executive|Analyst))$",
    re.UNICODE,
)


def as_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def extract_company_name(row: Any) -> str:
    transcript_subheader = as_text(getattr(row, "transcript_subheader", ""))
    raw_title = as_text(getattr(row, "title", ""))

    lines = [line.strip() for line in transcript_subheader.split("\n") if line.strip()]

    if lines:
        return lines[0]

    if " - " in raw_title:
        return raw_title.split(" - ", maxsplit=1)[0].strip()

    return raw_title


def extract_title(row: Any) -> str:
    return as_text(getattr(row, "title", ""))


def extract_date(row: Any):
    raw_date = as_text(getattr(row, "subheader", ""))

    if not raw_date:
        return pd.NaT

    return pd.to_datetime(raw_date, format=DATE_FORMAT, errors="coerce")


def extract_transcript_metadata(row: Any) -> dict[str, Any]:
    return {
        "company_name": extract_company_name(row),
        "title": extract_title(row),
        "date": extract_date(row),
    }


def parse_speaker(raw_speaker: str) -> dict[str, str]:
    raw_speaker = as_text(raw_speaker)

    if raw_speaker == "OperatorOperator":
        return {
            "raw_speaker": raw_speaker,
            "speaker_name": "Operator",
            "speaker_role": "Operator",
        }

    if raw_speaker.endswith("Executive"):
        return {
            "raw_speaker": raw_speaker,
            "speaker_name": raw_speaker.removesuffix("Executive").strip(),
            "speaker_role": "Executive",
        }

    if raw_speaker.endswith("Analyst"):
        return {
            "raw_speaker": raw_speaker,
            "speaker_name": raw_speaker.removesuffix("Analyst").strip(),
            "speaker_role": "Analyst",
        }

    return {
        "raw_speaker": raw_speaker,
        "speaker_name": raw_speaker,
        "speaker_role": "Unknown",
    }


def split_by_speaker(body: str) -> list[dict[str, str]]:
    matches = list(SPEAKER_PATTERN.finditer(body))

    if not matches:
        return []

    segments: list[dict[str, str]] = []

    for idx, match in enumerate(matches):
        raw_speaker = match.group(1).strip()
        speaker_info = parse_speaker(raw_speaker)

        content_start = match.end()
        content_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        content = body[content_start:content_end].strip()

        if not content:
            continue

        segments.append(
            {
                "raw_speaker": speaker_info["raw_speaker"],
                "speaker_name": speaker_info["speaker_name"],
                "speaker_role": speaker_info["speaker_role"],
                "content": content,
            }
        )

    return segments


def merge_consecutive_same_speaker(
    segments: list[dict[str, str]]
) -> list[dict[str, str]]:
    if not segments:
        return []

    merged = [segments[0].copy()]

    for segment in segments[1:]:
        same_speaker = (
            segment["speaker_name"] == merged[-1]["speaker_name"]
            and segment["speaker_role"] == merged[-1]["speaker_role"]
        )

        if same_speaker:
            merged[-1]["content"] = (
                merged[-1]["content"].rstrip() + "\n" + segment["content"].lstrip()
            ).strip()
        else:
            merged.append(segment.copy())

    return merged


def reconstruct_raw_segment_text(raw_segments: list[dict[str, str]]) -> str:
    parts: list[str] = []

    for segment in raw_segments:
        parts.append(segment["raw_speaker"])
        parts.append(segment["content"])

    return "\n".join(parts).strip()


def build_segment_rows(
    transcript_id: int,
    metadata: dict[str, Any],
    merged_segments: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for segment_id, segment in enumerate(merged_segments):
        rows.append(
            {
                "transcript_id": transcript_id,
                "segment_id": segment_id,
                "company_name": metadata["company_name"],
                "date": metadata["date"],
                "title": metadata["title"],
                "speaker_name": segment["speaker_name"],
                "speaker_role": segment["speaker_role"],
                "content": segment["content"],
            }
        )

    return rows


def build_trace_row(
    transcript_id: int,
    metadata: dict[str, Any],
    body: str,
    raw_segments: list[dict[str, str]],
    merged_segments: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "transcript_id": transcript_id,
        "company_name": metadata["company_name"],
        "date": metadata["date"],
        "title": metadata["title"],
        "raw_length": len(body),
        "segmented_length": len(reconstruct_raw_segment_text(raw_segments)),
        "n_raw_segments": len(raw_segments),
        "n_merged_segments": len(merged_segments),
    }


def build_segments_and_trace(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    segment_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []

    for transcript_id, row in enumerate(df_raw.itertuples(index=False)):
        body = as_text(getattr(row, "body", ""))
        metadata = extract_transcript_metadata(row)

        raw_segments = split_by_speaker(body)
        merged_segments = merge_consecutive_same_speaker(raw_segments)

        segment_rows.extend(
            build_segment_rows(
                transcript_id=transcript_id,
                metadata=metadata,
                merged_segments=merged_segments,
            )
        )

        trace_rows.append(
            build_trace_row(
                transcript_id=transcript_id,
                metadata=metadata,
                body=body,
                raw_segments=raw_segments,
                merged_segments=merged_segments,
            )
        )

    return pd.DataFrame(segment_rows), pd.DataFrame(trace_rows)