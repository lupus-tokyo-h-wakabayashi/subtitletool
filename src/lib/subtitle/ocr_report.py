from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lib.subtitle.ocr_inspection import (
    OcrInspectionEntry,
    OcrInspectionReport,
    OcrInspectionSummary,
)

OCR_REPORT_VERSION = 1


def serialize_ocr_inspection_entry(
    entry: OcrInspectionEntry,
) -> dict[str, Any]:
    return {
        "subtitle_id": entry.subtitle_id,
        "timestamp": entry.timestamp,
        "raw_text": entry.raw_text,
        "speaker": entry.speaker,
        "parsed_text": entry.parsed_text,
        "cleaned_text": entry.cleaned_text,
        "noise_candidates": list(
            entry.noise_candidates
        ),
        "noise_applied_text": (
            entry.noise_applied_text
        ),
        "changed_steps": list(
            entry.changed_steps
        ),
    }


def serialize_ocr_inspection_summary(
    summary: OcrInspectionSummary,
) -> dict[str, int]:
    return {
        "subtitle_count": (
            summary.subtitle_count
        ),
        "speaker_detected_count": (
            summary.speaker_detected_count
        ),
        "cleanup_changed_count": (
            summary.cleanup_changed_count
        ),
        "noise_candidate_subtitle_count": (
            summary
            .noise_candidate_subtitle_count
        ),
        "noise_candidate_count": (
            summary.noise_candidate_count
        ),
        "noise_applied_count": (
            summary.noise_applied_count
        ),
        "changed_subtitle_count": (
            summary.changed_subtitle_count
        ),
    }


def serialize_ocr_inspection_report(
    report: OcrInspectionReport,
) -> dict[str, Any]:
    return {
        "version": OCR_REPORT_VERSION,
        "source_srt": str(
            report.source_srt
        ),
        "profile": report.profile_name,
        "summary": (
            serialize_ocr_inspection_summary(
                report.summary
            )
        ),
        "entries": [
            serialize_ocr_inspection_entry(
                entry
            )
            for entry in report.entries
        ],
    }


def write_ocr_json_report(
    output_path: str | Path,
    report: OcrInspectionReport,
) -> Path:
    path = (
        Path(output_path)
        .expanduser()
        .resolve()
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    payload = (
        serialize_ocr_inspection_report(
            report
        )
    )

    temporary_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(path)

    return path
