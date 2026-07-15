import pytest

from lib.subtitle.srt import SrtBlock
from lib.translation.retry import (
    build_retry_instruction,
    build_structural_retry_instruction,
    has_structural_validation_error,
)


@pytest.mark.parametrize(
    "error",
    [
        "Invalid targets: expected object",
        (
            "Invalid target item: "
            "id='1', expected=object"
        ),
        (
            "Invalid target source: "
            "id='1', expected=object"
        ),
        (
            "Source speaker changed: "
            "subtitle_id='1'"
        ),
        (
            "Source text changed: "
            "subtitle_id='1'"
        ),
        "Missing translation IDs: ['2']",
        (
            "Invalid translation ID order: "
            "expected=['1', '2'], "
            "actual=['2', '1']"
        ),
    ],
)
def test_has_structural_validation_error_accepts_current_errors(
    error: str,
) -> None:
    assert has_structural_validation_error(
        [
            error,
        ]
    )


@pytest.mark.parametrize(
    "error",
    [
        "Glossary violation: subtitle_id='1'",
        (
            "Chinese-specific characters detected: "
            "subtitle_id='1'"
        ),
        (
            "Untranslated English sentence detected: "
            "subtitle_id='1'"
        ),
    ],
)
def test_has_structural_validation_error_rejects_content_errors(
    error: str,
) -> None:
    assert not has_structural_validation_error(
        [
            error,
        ]
    )


def test_build_retry_instruction_uses_targets_format(
) -> None:
    instruction = build_retry_instruction(
        [
            "Missing translation IDs: ['2']",
        ]
    )

    assert (
        "最上位キーはtargetsだけ"
        in instruction
    )

    assert (
        "targetsは配列ではなくオブジェクト"
        in instruction
    )

    assert (
        "各字幕オブジェクトのキーは"
        "sourceとtranslationだけ"
        in instruction
    )

    assert (
        "sourceのキーはspeakerとtextだけ"
        in instruction
    )

    assert (
        "最上位キーはtranslations"
        not in instruction
    )


def test_build_structural_retry_instruction_uses_targets_format(
) -> None:
    target_blocks = [
        SrtBlock(
            number="11",
            timestamp=(
                "00:00:01,000 --> "
                "00:00:02,000"
            ),
            text="First",
        ),
        SrtBlock(
            number="12",
            timestamp=(
                "00:00:02,000 --> "
                "00:00:03,000"
            ),
            text="Second",
        ),
    ]

    instruction = (
        build_structural_retry_instruction(
            target_blocks,
            [
                "Missing translation IDs: ['12']",
            ],
        )
    )

    assert '["11", "12"]' in instruction
    assert "targetsは必ず2件" in instruction

    assert (
        "targetsは配列ではなくオブジェクト"
        in instruction
    )

    assert (
        "各字幕オブジェクトのキーは"
        "sourceとtranslationだけ"
        in instruction
    )

    assert (
        "sourceのキーはspeakerとtextだけ"
        in instruction
    )

    assert (
        "translationsは配列"
        not in instruction
    )
