import pytest

from lib.subtitle.srt import SrtBlock
from lib.translation.translate import (
    filter_empty_source_blocks,
    resolve_requested_profile,
)


def test_resolve_requested_profile_with_profile() -> None:
    result = resolve_requested_profile(
        profile_name="stargate",
        style_name=None,
        glossary_name=None,
    )

    assert result == "stargate"


def test_resolve_requested_profile_with_legacy_options() -> None:
    result = resolve_requested_profile(
        profile_name=None,
        style_name="stargate",
        glossary_name="stargate",
    )

    assert result == "stargate"


def test_resolve_requested_profile_rejects_legacy_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Style and glossary profiles "
            "must match"
        ),
    ):
        resolve_requested_profile(
            profile_name=None,
            style_name="stargate",
            glossary_name="default",
        )


def test_resolve_requested_profile_rejects_conflict() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Profile conflicts with "
            "legacy options"
        ),
    ):
        resolve_requested_profile(
            profile_name="default",
            style_name="stargate",
            glossary_name="stargate",
        )


def test_filter_empty_source_blocks_removes_empty_text() -> None:
    blocks = [
        SrtBlock(
            number="216",
            timestamp=(
                "00:10:00,000 --> "
                "00:10:02,000"
            ),
            text="I need to talk to you.",
        ),
        SrtBlock(
            number="217",
            timestamp=(
                "00:10:02,000 --> "
                "00:10:04,000"
            ),
            text="",
        ),
        SrtBlock(
            number="218",
            timestamp=(
                "00:10:04,000 --> "
                "00:10:06,000"
            ),
            text="Alone. It's important.",
        ),
    ]

    (
        translation_blocks,
        skipped_blocks,
    ) = filter_empty_source_blocks(
        blocks
    )

    assert [
               block.number
               for block in translation_blocks
           ] == [
               "216",
               "218",
           ]

    assert [
               block.number
               for block in skipped_blocks
           ] == [
               "217",
           ]


def test_filter_empty_source_blocks_removes_whitespace_only_text() -> None:
    blocks = [
        SrtBlock(
            number="1",
            timestamp=(
                "00:00:01,000 --> "
                "00:00:02,000"
            ),
            text=" \n\t ",
        ),
    ]

    (
        translation_blocks,
        skipped_blocks,
    ) = filter_empty_source_blocks(
        blocks
    )

    assert translation_blocks == []

    assert [
               block.number
               for block in skipped_blocks
           ] == [
               "1",
           ]


def test_filter_empty_source_blocks_preserves_order() -> None:
    blocks = [
        SrtBlock(
            number="10",
            timestamp=(
                "00:00:10,000 --> "
                "00:00:11,000"
            ),
            text="First",
        ),
        SrtBlock(
            number="11",
            timestamp=(
                "00:00:11,000 --> "
                "00:00:12,000"
            ),
            text="",
        ),
        SrtBlock(
            number="12",
            timestamp=(
                "00:00:12,000 --> "
                "00:00:13,000"
            ),
            text="Second",
        ),
    ]

    (
        translation_blocks,
        skipped_blocks,
    ) = filter_empty_source_blocks(
        blocks
    )

    assert [
               block.number
               for block in translation_blocks
           ] == [
               "10",
               "12",
           ]

    assert [
               block.number
               for block in skipped_blocks
           ] == [
               "11",
           ]
