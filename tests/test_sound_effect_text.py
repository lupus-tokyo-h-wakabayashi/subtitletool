from __future__ import annotations

import pytest
from lib.subtitle.text import (
    split_leading_sound_effects,
)


@pytest.mark.parametrize(
    (
            "source_text",
            "expected_sound_effects",
            "expected_remaining_text",
    ),
    [
        (
                "(CHUCKLING) Is that",
                ["(CHUCKLING)"],
                "Is that",
        ),
        (
                "(GASPS) (PANTING) Get out!",
                [
                    "(GASPS)",
                    "(PANTING)",
                ],
                "Get out!",
        ),
        (
                "(RUMBLING)",
                ["(RUMBLING)"],
                "",
        ),
        (
                "Is that",
                [],
                "Is that",
        ),
        (
                "(not a sound effect) Is that",
                [],
                "(not a sound effect) Is that",
        ),
        (
                "Tell me (CHUCKLING)",
                [],
                "Tell me (CHUCKLING)",
        ),
        (
                "",
                [],
                "",
        ),
        (
                "   (CHUCKLING)   Is that   ",
                ["(CHUCKLING)"],
                "Is that",
        ),
    ],
)
def test_split_leading_sound_effects(
    source_text: str,
    expected_sound_effects: list[str],
    expected_remaining_text: str,
) -> None:
    sound_effects, remaining_text = (
        split_leading_sound_effects(
            source_text
        )
    )

    assert (
        sound_effects
        == expected_sound_effects
    )

    assert (
        remaining_text
        == expected_remaining_text
    )
