import pytest

from lib.subtitle.srt import (
    SrtBlock,
    parse_speaker_from_text,
)
from lib.translation.translation_prompt import (
    build_request_item,
)
from lib.translation.translation_session import (
    cleanup_block,
    cleanup_blocks,
    rebuild_speaker_text,
)


def build_test_block(
    text: str,
) -> SrtBlock:
    return SrtBlock(
        number="1",
        timestamp=(
            "00:00:01,000 --> "
            "00:00:03,000"
        ),
        text=text,
    )


@pytest.mark.parametrize(
    (
            "source_text",
            "expected_speaker",
            "expected_text",
    ),
    [
        (
                "DANIEL: This is the Stargate.",
                "DANIEL",
                "This is the Stargate.",
        ),
        (
                "Daniel: This is the Stargate.",
                "Daniel",
                "This is the Stargate.",
        ),
        (
                "[DANIEL] This is the Stargate.",
                "DANIEL",
                "This is the Stargate.",
        ),
        (
                "MAN 1: Move away.",
                "MAN 1",
                "Move away.",
        ),
        (
                "DR. RUSH: Wait here.",
                "DR. RUSH",
                "Wait here.",
        ),
    ],
)
def test_cleanup_block_preserves_speaker(
    source_text: str,
    expected_speaker: str,
    expected_text: str,
) -> None:
    result = cleanup_block(
        build_test_block(
            source_text
        )
    )

    parsed = parse_speaker_from_text(
        result.text
    )

    assert parsed.speaker == expected_speaker
    assert parsed.text == expected_text


def test_cleanup_block_normalizes_speaker_body(
) -> None:
    result = cleanup_block(
        build_test_block(
            "DANIEL:   This   is   the Stargate.  "
        )
    )

    assert result.text == (
        "[DANIEL] This is the Stargate."
    )


def test_cleanup_block_without_speaker_uses_existing_cleanup(
) -> None:
    result = cleanup_block(
        build_test_block(
            "  This   is   the Stargate.  "
        )
    )

    assert result.text == (
        "This is the Stargate."
    )


def test_cleanup_block_preserves_subtitle_metadata(
) -> None:
    block = SrtBlock(
        number="42",
        timestamp=(
            "00:01:02,000 --> "
            "00:01:04,000"
        ),
        text="DANIEL: Move.",
    )

    result = cleanup_block(
        block
    )

    assert result.number == "42"

    assert result.timestamp == (
        "00:01:02,000 --> "
        "00:01:04,000"
    )


def test_cleanup_blocks_preserves_block_order(
) -> None:
    blocks = [
        SrtBlock(
            number="1",
            timestamp=(
                "00:00:01,000 --> "
                "00:00:02,000"
            ),
            text="DANIEL: First.",
        ),
        SrtBlock(
            number="2",
            timestamp=(
                "00:00:02,000 --> "
                "00:00:03,000"
            ),
            text="SCOTT: Second.",
        ),
    ]

    result = cleanup_blocks(
        blocks
    )

    assert [
               block.number
               for block in result
           ] == [
               "1",
               "2",
           ]

    assert [
               block.text
               for block in result
           ] == [
               "[DANIEL] First.",
               "[SCOTT] Second.",
           ]


def test_cleanup_block_preserves_speaker_for_request_item(
) -> None:
    cleaned_block = cleanup_block(
        build_test_block(
            "DANIEL: This is the Stargate."
        )
    )

    result = build_request_item(
        cleaned_block
    )

    assert result == {
        "source": {
            "speaker": "DANIEL",
            "text": "This is the Stargate.",
        },
        "translation": "",
    }


def test_rebuild_speaker_text_without_speaker(
) -> None:
    assert rebuild_speaker_text(
        None,
        "This is the Stargate.",
    ) == "This is the Stargate."


def test_rebuild_speaker_text_with_speaker(
) -> None:
    assert rebuild_speaker_text(
        "DANIEL",
        "This is the Stargate.",
    ) == "[DANIEL] This is the Stargate."
