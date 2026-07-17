from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from lib.subtitle.srt import SrtBlock
from lib.translation.hybrid_group import (
    build_hybrid_translation_group,
)
from lib.translation.hybrid_inspection import (
    build_hybrid_attempt_filename,
    build_hybrid_attempt_report,
    save_hybrid_attempt_report,
)


def make_group():
    blocks = [
        SrtBlock(
            number="281",
            timestamp=(
                "00:17:10,988 --> "
                "00:17:12,865"
            ),
            text=(
                "It's under control,\n"
                "but as a precaution,"
            ),
        ),
        SrtBlock(
            number="282",
            timestamp=(
                "00:17:12,949 --> "
                "00:17:15,201"
            ),
            text=(
                "=) EWA eam CO ma = Ae lan\n"
                "to their quarters"
            ),
        ),
        SrtBlock(
            number="283",
            timestamp=(
                "00:17:15,284 --> "
                "00:17:18,079"
            ),
            text=(
                "and remain there\n"
                "until further notice."
            ),
        ),
    ]

    group = build_hybrid_translation_group(
        blocks,
        {
            "282",
        },
    )

    assert group is not None

    return group


def test_build_hybrid_attempt_filename(
) -> None:
    group = make_group()

    created_at = datetime(
        2026,
        7,
        17,
        18,
        30,
        45,
        123456,
    )

    filename = (
        build_hybrid_attempt_filename(
            group=group,
            attempt=2,
            created_at=created_at,
        )
    )

    assert filename == (
        "hybrid-translation-"
        "281-283-"
        "attempt-2-"
        "20260717-183045-123456.json"
    )


def test_build_hybrid_attempt_report(
) -> None:
    group = make_group()

    report = build_hybrid_attempt_report(
        group=group,
        model="test-model",
        attempt=1,
        prompt="Translate this group.",
        response_schema={
            "type": "object",
        },
        response='{"group": {}}',
        ocr_lines={
            "282": [
                "=) EWA eam CO ma = Ae lan",
            ],
        },
        validation_stage=(
            "hybrid_validation"
        ),
        validation_valid=False,
        validation_reasons=[
            "Invalid Hybrid response",
        ],
        created_at=datetime(
            2026,
            7,
            17,
            18,
            30,
            45,
        ),
    )

    assert report["version"] == 1
    assert report["model"] == "test-model"
    assert report["attempt"] == 1

    assert report["target_ids"] == [
        "281",
        "282",
        "283",
    ]

    assert report["failed_ids"] == [
        "282",
    ]

    assert report["validation"] == {
        "stage": "hybrid_validation",
        "valid": False,
        "reasons": [
            "Invalid Hybrid response",
        ],
    }


def test_save_hybrid_attempt_report(
    tmp_path: Path,
) -> None:
    group = make_group()

    output_path = (
        save_hybrid_attempt_report(
            group=group,
            model="test-model",
            attempt=1,
            prompt="Translate this group.",
            response_schema={
                "type": "object",
            },
            response=(
                '{"group": {'
                '"full_translation": '
                '"居住区へ戻ってください"'
                "}}"
            ),
            ocr_lines={
                "282": [
                    "=) EWA eam CO ma = Ae lan",
                ],
            },
            validation_stage="complete",
            validation_valid=True,
            validation_reasons=[],
            created_at=datetime(
                2026,
                7,
                17,
                18,
                30,
                45,
                123456,
            ),
            output_directory=tmp_path,
        )
    )

    assert output_path == (
        tmp_path
        / (
            "hybrid-translation-"
            "281-283-"
            "attempt-1-"
            "20260717-183045-123456.json"
        )
    )

    payload = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    assert payload["target_ids"] == [
        "281",
        "282",
        "283",
    ]

    assert payload["ocr_lines"] == {
        "282": [
            "=) EWA eam CO ma = Ae lan",
        ],
    }

    assert payload["validation"] == {
        "stage": "complete",
        "valid": True,
        "reasons": [],
    }

    raw_text = output_path.read_text(
        encoding="utf-8"
    )

    assert "居住区へ戻ってください" in raw_text
    assert "\\u5c45" not in raw_text


def test_try_save_hybrid_attempt_report_returns_none_on_io_error(
    tmp_path: Path,
    capsys,
) -> None:
    from lib.translation.hybrid_inspection import (
        try_save_hybrid_attempt_report,
    )

    group = make_group()

    invalid_output_directory = (
        tmp_path
        / "not-a-directory"
    )

    invalid_output_directory.write_text(
        "This path is a file.",
        encoding="utf-8",
    )

    result = try_save_hybrid_attempt_report(
        group=group,
        model="test-model",
        attempt=1,
        prompt="Translate this group.",
        response_schema={
            "type": "object",
        },
        response='{"group": {}}',
        ocr_lines={},
        validation_stage="complete",
        validation_valid=True,
        validation_reasons=[],
        created_at=datetime(
            2026,
            7,
            17,
            18,
            30,
            45,
        ),
        output_directory=(
            invalid_output_directory
        ),
    )

    assert result is None

    captured = capsys.readouterr()

    assert (
        "Warning: Hybrid report "
        "could not be saved:"
        in captured.out
    )
