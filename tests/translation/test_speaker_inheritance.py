from lib.subtitle.srt import (
    SrtBlock,
)
from lib.translation.translation_session import (
    prepare_translation_source_blocks,
)


def make_block(
    number: str,
    text: str,
) -> SrtBlock:
    return SrtBlock(
        number=number,
        timestamp=(
            "00:00:00,000 --> "
            "00:00:01,000"
        ),
        text=text,
    )


def test_prepare_translation_source_blocks_does_not_inherit_speaker(
) -> None:
    source_blocks = [
        make_block(
            "1",
            "[RUSH] Listen to me.",
        ),
        make_block(
            "2",
            "This is important.",
        ),
        make_block(
            "3",
            "(CHUCKLES) I don't think so.",
        ),
        make_block(
            "4",
            "(RUMBLING)",
        ),
        make_block(
            "5",
            "[YOUNG] I've heard enough.",
        ),
        make_block(
            "6",
            "We're done.",
        ),
    ]

    prepared_blocks = (
        prepare_translation_source_blocks(
            source_blocks
        )
    )

    assert [
               block.text
               for block in prepared_blocks
           ] == [
               "[RUSH] Listen to me.",
               "This is important.",
               "(CHUCKLES) I don't think so.",
               "(RUMBLING)",
               "[YOUNG] I've heard enough.",
               "We're done.",
           ]


def test_prepare_translation_source_blocks_normalizes_only_explicit_speakers(
) -> None:
    source_blocks = [
        make_block(
            "1",
            (
                "RUSH: This is what Destiny\n"
                "intended from the moment."
            ),
        ),
        make_block(
            "2",
            "We entered the star system.",
        ),
        make_block(
            "3",
            (
                "YOUNG: I don't believe "
                "that."
            ),
        ),
        make_block(
            "4",
            "There is no other explanation.",
        ),
    ]

    prepared_blocks = (
        prepare_translation_source_blocks(
            source_blocks
        )
    )

    assert [
               block.text
               for block in prepared_blocks
           ] == [
               (
                   "[RUSH] This is what Destiny\n"
                   "intended from the moment."
               ),
               "We entered the star system.",
               (
                   "[YOUNG] I don't believe "
                   "that."
               ),
               "There is no other explanation.",
           ]
