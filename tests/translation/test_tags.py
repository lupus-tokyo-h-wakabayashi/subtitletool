import pytest

from lib.translation.translation_tags import (
    parse_translation_tags,
    process_translation_tags,
    render_translation_tags,
    strip_translation_tags,
)


def test_parse_translation_tags_returns_values(
) -> None:
    result = parse_translation_tags(
        (
            "惑星"
            "[5]P4X-351[/5]"
            "と"
            "[3]Destiny[/3]"
            "、"
            "[1]garbled text[/1]"
        )
    )

    assert result.errors == ()

    assert [
               (
                   tag.level,
                   tag.value,
               )
               for tag in result.tags
           ] == [
               (
                   5,
                   "P4X-351",
               ),
               (
                   3,
                   "Destiny",
               ),
               (
                   1,
                   "garbled text",
               ),
           ]


def test_strip_translation_tags_keeps_values(
) -> None:
    result = strip_translation_tags(
        (
            "惑星"
            "[5]P4X-351[/5]"
            "の"
            "[3]Destiny[/3]"
        )
    )

    assert result == (
        "惑星P4X-351のDestiny"
    )


@pytest.mark.parametrize(
    "level",
    [
        2,
        4,
        6,
    ],
)
def test_parse_translation_tags_rejects_unsupported_levels(
    level: int,
) -> None:
    result = parse_translation_tags(
        (
            f"[{level}]"
            "P4X-351"
            f"[/{level}]"
        )
    )

    assert result.tags == ()

    assert any(
        (
            "Unsupported translation tag level"
            in error
        )
        for error in result.errors
    )


def test_parse_translation_tags_detects_missing_closing_tag(
) -> None:
    result = parse_translation_tags(
        "[5]P4X-351"
    )

    assert result.tags == ()

    assert result.errors == (
        (
            "Missing translation closing tag: "
            "level=5, position=0"
        ),
    )


def test_parse_translation_tags_detects_mismatched_closing_tag(
) -> None:
    result = parse_translation_tags(
        "[5]P4X-351[/3]"
    )

    assert result.tags == ()

    assert any(
        (
            "Mismatched translation closing tag"
            in error
        )
        for error in result.errors
    )


def test_parse_translation_tags_detects_unexpected_closing_tag(
) -> None:
    result = parse_translation_tags(
        "P4X-351[/5]"
    )

    assert result.tags == ()

    assert any(
        (
            "Unexpected translation closing tag"
            in error
        )
        for error in result.errors
    )


def test_parse_translation_tags_detects_nested_tags(
) -> None:
    result = parse_translation_tags(
        (
            "[5]"
            "P4X-"
            "[3]351[/3]"
            "[/5]"
        )
    )

    assert any(
        (
            "Nested translation tag"
            in error
        )
        for error in result.errors
    )


@pytest.mark.parametrize(
    (
            "source",
            "expected_error",
    ),
    [
        (
                "[1][/1]",
                "Empty translation tag value",
        ),
        (
                "[5] P4X-351[/5]",
                (
                    "Translation tag value has "
                    "surrounding whitespace"
                ),
        ),
        (
                "[5]P4X-351 [/5]",
                (
                    "Translation tag value has "
                    "surrounding whitespace"
                ),
        ),
    ],
)
def test_parse_translation_tags_rejects_invalid_values(
    source: str,
    expected_error: str,
) -> None:
    result = parse_translation_tags(
        source
    )

    assert result.tags == ()

    assert any(
        expected_error in error
        for error in result.errors
    )


def test_strip_translation_tags_rejects_invalid_structure(
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Invalid translation tags"
        ),
    ):
        strip_translation_tags(
            "[5]P4X-351[/3]"
        )


def test_render_translation_tags_replaces_only_tagged_level_1_value(
) -> None:
    result = render_translation_tags(
        (
            "[1]noise[/1] "
            "noise"
        ),
        level_1_replacement=(
            "（判読不能）"
        ),
    )

    assert result == (
        "（判読不能） noise"
    )


def test_process_translation_tags_ignores_level_5_source_mismatch(
) -> None:
    result = process_translation_tags(
        translated_texts=[
            (
                "こちらは"
                "[5]SG-1[/5]"
                "です。"
            ),
        ],
        subtitle_ids=[
            "1",
        ],
        source_texts=[
            (
                "it's indicating malfunction.\n"
                "Others are failing."
            ),
        ],
    )

    assert result.errors == ()

    assert result.translated_texts == (
        "こちらはSG-1です。",
    )


def test_process_translation_tags_accepts_level_5_without_source_text(
) -> None:
    result = process_translation_tags(
        translated_texts=[
            "[5]SG-1[/5]からの通信です。",
        ],
        subtitle_ids=[
            "1",
        ],
        source_texts=None,
    )

    assert result.errors == ()

    assert result.translated_texts == (
        "SG-1からの通信です。",
    )


def test_process_translation_tags_requires_source_text_for_level_3(
) -> None:
    result = process_translation_tags(
        translated_texts=[
            "[3]Destiny[/3]",
        ],
        subtitle_ids=[
            "1",
        ],
        source_texts=None,
    )

    assert len(result.errors) == 1

    error = result.errors[0]

    assert error.subtitle_id == "1"

    assert (
        "Translation evaluation tags require "
        "source text"
        in error.message
    )
