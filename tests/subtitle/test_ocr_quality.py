from __future__ import annotations

from lib.subtitle.ocr_quality import (
    find_suspicious_short_uppercase_fragments,
)


def test_sst_a_is_detected() -> None:
    candidates = (
        find_suspicious_short_uppercase_fragments(
            "SST A"
        )
    )

    assert candidates == (
        "SST A",
    )


def test_ftl_is_not_detected() -> None:
    candidates = (
        find_suspicious_short_uppercase_fragments(
            "FTL"
        )
    )

    assert candidates == ()


def test_normal_dialogue_is_not_detected() -> None:
    candidates = (
        find_suspicious_short_uppercase_fragments(
            "Yes, sir."
        )
    )

    assert candidates == ()


def test_sound_effect_is_not_detected() -> None:
    candidates = (
        find_suspicious_short_uppercase_fragments(
            "(ALARMS BLARING)"
        )
    )

    assert candidates == ()


def test_candidate_is_found_in_multiline_text(
) -> None:
    candidates = (
        find_suspicious_short_uppercase_fragments(
            "SST A\n"
            "FTL in three, two, one."
        )
    )

    assert candidates == (
        "SST A",
    )


def test_duplicate_candidates_are_returned_once(
) -> None:
    candidates = (
        find_suspicious_short_uppercase_fragments(
            "SST A\n"
            "SST A"
        )
    )

    assert candidates == (
        "SST A",
    )


def test_extra_spaces_are_normalized() -> None:
    candidates = (
        find_suspicious_short_uppercase_fragments(
            "  SST   A  "
        )
    )

    assert candidates == (
        "SST A",
    )


def test_allowed_fragment_is_not_detected(
) -> None:
    candidates = (
        find_suspicious_short_uppercase_fragments(
            "PLAN B",
            allowed_fragments={
                "PLAN B",
            },
        )
    )

    assert candidates == ()
