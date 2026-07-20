from lib.subtitle.srt import (
    SrtBlock,
)
from lib.translation.translation_session import (
    inherit_missing_speakers,
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


def test_inherit_missing_speakers(
) -> None:
    blocks = [
        make_block(
            "1",
            "Opening narration.",
        ),
        make_block(
            "2",
            (
                "[RUSH] This is what Destiny\n"
                "intended from the moment."
            ),
        ),
        make_block(
            "3",
            "We entered the star system.",
        ),
        make_block(
            "4",
            "(RUMBLING)",
        ),
        make_block(
            "5",
            "There is no other explanation.",
        ),
        make_block(
            "6",
            "[YOUNG] I don't believe that.",
        ),
        make_block(
            "7",
            "You never do.",
        ),
    ]

    actual = inherit_missing_speakers(
        blocks
    )

    assert [
               block.text
               for block in actual
           ] == [
               "Opening narration.",
               (
                   "[RUSH] This is what Destiny\n"
                   "intended from the moment."
               ),
               "[RUSH] We entered the star system.",
               "(RUMBLING)",
               (
                   "[RUSH] "
                   "There is no other explanation."
               ),
               "[YOUNG] I don't believe that.",
               "[YOUNG] You never do.",
           ]


def test_sound_effect_does_not_clear_current_speaker(
) -> None:
    blocks = [
        make_block(
            "1",
            "[ELI] Are you there?",
        ),
        make_block(
            "2",
            "(STATIC)",
        ),
        make_block(
            "3",
            "(RADIO BEEPS)\n(STATIC)",
        ),
        make_block(
            "4",
            "Can you hear me?",
        ),
    ]

    actual = inherit_missing_speakers(
        blocks
    )

    assert [
               block.text
               for block in actual
           ] == [
               "[ELI] Are you there?",
               "(STATIC)",
               "(RADIO BEEPS)\n(STATIC)",
               "[ELI] Can you hear me?",
           ]


def test_mixed_sound_effect_and_dialogue_inherits_speaker(
) -> None:
    blocks = [
        make_block(
            "1",
            "[YOUNG] We need to leave.",
        ),
        make_block(
            "2",
            "(CHUCKLES) I don't think so.",
        ),
    ]

    actual = inherit_missing_speakers(
        blocks
    )

    assert actual[1].text == (
        "[YOUNG] "
        "(CHUCKLES) I don't think so."
    )


def test_explicit_speaker_replaces_inherited_speaker(
) -> None:
    blocks = [
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
            "[YOUNG] I've heard enough.",
        ),
        make_block(
            "4",
            "We're done.",
        ),
    ]

    actual = inherit_missing_speakers(
        blocks
    )

    assert [
               block.text
               for block in actual
           ] == [
               "[RUSH] Listen to me.",
               "[RUSH] This is important.",
               "[YOUNG] I've heard enough.",
               "[YOUNG] We're done.",
           ]


def test_empty_text_does_not_receive_speaker(
) -> None:
    blocks = [
        make_block(
            "1",
            "[WRAY] Wait here.",
        ),
        make_block(
            "2",
            "",
        ),
    ]

    actual = inherit_missing_speakers(
        blocks
    )

    assert actual[1].text == ""
