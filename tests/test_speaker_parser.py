import pytest

from lib.subtitle.srt import (
    ParsedSubtitleText,
    parse_speaker_from_text,
)


@pytest.mark.parametrize(
    (
            "source_text",
            "expected",
    ),
    [
        (
                (
                    "RUSH: This is what Destiny\n"
                    "intended from the moment"
                ),
                ParsedSubtitleText(
                    speaker="RUSH",
                    text=(
                        "This is what Destiny\n"
                        "intended from the moment"
                    ),
                ),
        ),
        (
                (
                    "JOHANSEN:\n"
                    "Look, I lost contact."
                ),
                ParsedSubtitleText(
                    speaker="JOHANSEN",
                    text="Look, I lost contact.",
                ),
        ),
        (
                (
                    "[YOUNG] That ship\n"
                    "is not our only problem."
                ),
                ParsedSubtitleText(
                    speaker="YOUNG",
                    text=(
                        "That ship\n"
                        "is not our only problem."
                    ),
                ),
        ),
        (
                (
                    "Daniel: This is the Stargate.\n"
                    "We need to leave now."
                ),
                ParsedSubtitleText(
                    speaker="Daniel",
                    text=(
                        "This is the Stargate.\n"
                        "We need to leave now."
                    ),
                ),
        ),
        (
                "WRAY: Eli.",
                ParsedSubtitleText(
                    speaker="WRAY",
                    text="Eli.",
                ),
        ),
    ],
)
def test_parse_speaker_from_multiline_text(
    source_text: str,
    expected: ParsedSubtitleText,
) -> None:
    assert (
        parse_speaker_from_text(
            source_text
        )
        == expected
    )


def test_parse_speaker_does_not_accept_empty_body(
) -> None:
    assert (
        parse_speaker_from_text(
            "JOHANSEN:"
        )
        == ParsedSubtitleText(
        speaker=None,
        text="JOHANSEN:",
    )
    )


def test_parse_speaker_preserves_unlabeled_multiline_text(
) -> None:
    source_text = (
        "This is what Destiny\n"
        "intended from the moment."
    )

    assert (
        parse_speaker_from_text(
            source_text
        )
        == ParsedSubtitleText(
        speaker=None,
        text=source_text,
    )
    )


def test_parse_speaker_preserves_sound_effect_text(
) -> None:
    assert (
        parse_speaker_from_text(
            "(RUMBLING)"
        )
        == ParsedSubtitleText(
        speaker=None,
        text="(RUMBLING)",
    )
    )
