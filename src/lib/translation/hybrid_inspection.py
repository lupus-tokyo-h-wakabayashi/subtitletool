from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Mapping

from .hybrid_group import (
    HybridTranslationGroup,
)

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

HYBRID_DEBUG_DIR = (
    PROJECT_ROOT
    / "tmp"
)

HYBRID_REPORT_VERSION = 1


def build_hybrid_attempt_report(
    *,
    group: HybridTranslationGroup,
    model: str,
    attempt: int,
    prompt: str,
    response_schema: Mapping[str, object],
    response: str,
    ocr_lines: Mapping[str, list[str]],
    validation_stage: str,
    validation_valid: bool,
    validation_reasons: list[str],
    created_at: datetime,
) -> dict[str, object]:
    """
    Hybrid Recoveryの1試行を
    観測用JSONへ変換する。

    LLMへ送ったPrompt、Schema、
    返却されたレスポンス、
    検証結果を同じファイルへ保存する。
    """
    return {
        "version": HYBRID_REPORT_VERSION,
        "created_at": (
            created_at.isoformat(
                timespec="microseconds"
            )
        ),
        "model": model,
        "attempt": attempt,
        "target_ids": list(
            group.target_ids
        ),
        "failed_ids": sorted(
            group.failed_ids
        ),
        "source_blocks": [
            {
                "id": block.number,
                "timestamp": block.timestamp,
                "text": block.text,
            }
            for block in group.blocks
        ],
        "ocr_lines": {
            subtitle_id: list(lines)
            for subtitle_id, lines
            in ocr_lines.items()
        },
        "request": {
            "prompt": prompt,
            "response_format": dict(
                response_schema
            ),
        },
        "response": response,
        "validation": {
            "stage": validation_stage,
            "valid": validation_valid,
            "reasons": list(
                validation_reasons
            ),
        },
    }


def build_hybrid_attempt_filename(
    *,
    group: HybridTranslationGroup,
    attempt: int,
    created_at: datetime,
) -> str:
    """
    Hybrid観測ファイル名を生成する。
    """
    first_id = group.target_ids[0]
    last_id = group.target_ids[-1]

    timestamp = created_at.strftime(
        "%Y%m%d-%H%M%S-%f"
    )

    return (
        "hybrid-translation-"
        f"{first_id}-{last_id}-"
        f"attempt-{attempt}-"
        f"{timestamp}.json"
    )


def save_hybrid_attempt_report(
    *,
    group: HybridTranslationGroup,
    model: str,
    attempt: int,
    prompt: str,
    response_schema: Mapping[str, object],
    response: str,
    ocr_lines: Mapping[str, list[str]],
    validation_stage: str,
    validation_valid: bool,
    validation_reasons: list[str],
    created_at: datetime | None = None,
    output_directory: Path | None = None,
) -> Path:
    """
    Hybrid Recoveryの1試行を
    tmp配下へJSON保存する。
    """
    resolved_created_at = (
        created_at
        if created_at is not None
        else datetime.now()
    )

    resolved_output_directory = (
        output_directory
        if output_directory is not None
        else HYBRID_DEBUG_DIR
    )

    resolved_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = build_hybrid_attempt_report(
        group=group,
        model=model,
        attempt=attempt,
        prompt=prompt,
        response_schema=response_schema,
        response=response,
        ocr_lines=ocr_lines,
        validation_stage=validation_stage,
        validation_valid=validation_valid,
        validation_reasons=validation_reasons,
        created_at=resolved_created_at,
    )

    filename = build_hybrid_attempt_filename(
        group=group,
        attempt=attempt,
        created_at=resolved_created_at,
    )

    output_path = (
        resolved_output_directory
        / filename
    )

    output_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return output_path
