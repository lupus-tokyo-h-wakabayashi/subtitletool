import pytest
from lib.subtitle.srt import SrtBlock
from lib.translation.translation_resume import (
    load_resume_blocks,
)


def test_load_resume_blocks_without_output(
    tmp_path,
) -> None:
    source_blocks = [
        SrtBlock(
            number="1",
            timestamp=(
                "00:00:01,000 --> "
                "00:00:03,000"
            ),
            text="Test",
        ),
    ]

    output_path = (
        tmp_path
        / "not-found.ja.srt"
    )

    result = load_resume_blocks(
        source_blocks,
        output_path,
    )

    assert result == []


def test_load_resume_blocks_with_valid_output(
    tmp_path,
) -> None:
    source_blocks = [
        SrtBlock(
            number="1",
            timestamp=(
                "00:00:01,000 --> "
                "00:00:03,000"
            ),
            text="Test",
        ),
        SrtBlock(
            number="2",
            timestamp=(
                "00:00:04,000 --> "
                "00:00:06,000"
            ),
            text="Next",
        ),
    ]

    output_path = (
        tmp_path
        / "resume.ja.srt"
    )

    output_path.write_text(
        "\n".join(
            [
                "1",
                (
                    "00:00:01,000 --> "
                    "00:00:03,000"
                ),
                "テスト",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = load_resume_blocks(
        source_blocks,
        output_path,
    )

    assert len(result) == 1
    assert result[0].number == "1"
    assert result[0].timestamp == (
        "00:00:01,000 --> "
        "00:00:03,000"
    )


def test_load_resume_blocks_rejects_invalid_number(
    tmp_path,
) -> None:
    source_blocks = [
        SrtBlock(
            number="1",
            timestamp=(
                "00:00:01,000 --> "
                "00:00:03,000"
            ),
            text="Test",
        ),
    ]

    output_path = (
        tmp_path
        / "invalid.ja.srt"
    )

    output_path.write_text(
        "\n".join(
            [
                "9",
                (
                    "00:00:01,000 --> "
                    "00:00:03,000"
                ),
                "テスト",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "subtitle number mismatch"
        ),
    ):
        load_resume_blocks(
            source_blocks,
            output_path,
        )
