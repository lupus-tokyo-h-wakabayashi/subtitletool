from __future__ import annotations

from lib.subtitle.text import (
    detect_simplified_chinese,
)


def test_japanese_inner_character_is_ambiguous(
) -> None:
    detection = detect_simplified_chinese(
        "範囲内"
    )

    assert detection.detected is False

    assert detection.characters == ()

    assert detection.ambiguous_characters == (
        "内",
    )

    assert (
        detection.simplified_to_traditional
        != detection.source_text
    )


def test_repeated_ambiguous_character_is_deduplicated(
) -> None:
    detection = detect_simplified_chinese(
        "範囲内と領域内"
    )

    assert detection.detected is False

    assert detection.characters == ()

    assert detection.ambiguous_characters == (
        "内",
    )


def test_high_confidence_simplified_chinese_is_detected(
) -> None:
    detection = detect_simplified_chinese(
        "这些人"
    )

    assert detection.detected is True

    assert "这" in detection.characters

    assert detection.ambiguous_characters == ()


def test_high_confidence_and_ambiguous_characters_are_separated(
) -> None:
    detection = detect_simplified_chinese(
        "範囲内に这些人がいる"
    )

    assert detection.detected is True

    assert "这" in detection.characters

    assert detection.ambiguous_characters == (
        "内",
    )


def test_e15_japanese_phrases_are_not_detected_as_chinese(
) -> None:
    source_texts = (
        (
            "仮に、我々がいる惑星の範囲内にある"
            "ゲートを考えてみよう。"
        ),
        (
            "この円は、デスティニーの範囲内にある"
            "ゲートを表している。"
        ),
        (
            "現在、幸いにも、それぞれの範囲内にある"
            "ゲートが存在することを願っている。"
        ),
    )

    for source_text in source_texts:
        detection = detect_simplified_chinese(
            source_text
        )

        assert detection.detected is False
        assert detection.characters == ()

        assert detection.ambiguous_characters == (
            "内",
        )
