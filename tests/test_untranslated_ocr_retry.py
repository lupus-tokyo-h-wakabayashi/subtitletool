from __future__ import annotations

import json
from pathlib import Path

import pytest
from lib.profile.noise import (
    NoiseDictionary,
)
from lib.subtitle.srt import SrtBlock
from lib.translation import translation_chunk
from lib.translation.ocr_retry import (
    apply_level_1_ocr_fallback,
    extract_untranslated_english_error_ids,
    find_probable_untranslated_ocr_lines,
    is_low_symbol_word_salad_ocr_source_line,
    is_probable_ocr_source_line,
    is_short_mixed_case_ocr_source_line,
    is_symbol_dense_ocr_source_line,
)
from lib.translation.retry import (
    build_untranslated_english_retry_instruction,
)
from lib.translation.translation_chunk import (
    try_level_1_ocr_fallback,
)
from lib.translation.translation_validation import (
    validate_translation_response,
)

SYMBOL_DENSE_OCR_LINE = (
    "hm )olt=)a-te mda omc) "
    "t= meh Yd (=) 00"
)

SYMBOL_DENSE_SOURCE_TEXT = (
    "This is what Destiny\n"
    "intended from the moment\n"
    f"{SYMBOL_DENSE_OCR_LINE}"
)

SYMBOL_DENSE_TRANSLATION = (
    "これはデスティニーが"
    "最初から意図したことであり、"
    f"{SYMBOL_DENSE_OCR_LINE}"
)

SYMBOL_DENSE_UNTRANSLATED_ERROR = (
    "Untranslated English sentence detected: "
    "subtitle_id='361', "
    f"text={SYMBOL_DENSE_TRANSLATION!r}"
)

OCR_LINE = "AV Cag are T"
NORMAL_LINE = "the wrong people!"

SOURCE_TEXT = (
    f"{OCR_LINE}\n"
    f"{NORMAL_LINE}"
)

UNTRANSLATED_RESULT = (
    f"{OCR_LINE}"
    "は間違った人々です！"
)

UNTRANSLATED_ERROR = (
    "Untranslated English sentence detected: "
    "subtitle_id='80', "
    f"text={UNTRANSLATED_RESULT!r}"
)


@pytest.fixture
def noise_dictionary(
    tmp_path: Path,
) -> NoiseDictionary:
    """
    実際のStargateプロファイルを変更しない、
    テスト専用の空Noise辞書を返す。
    """
    return NoiseDictionary(
        profile_name="test",
        entries={},
        official_path=(
            tmp_path
            / "noise.json"
        ),
        local_path=(
            tmp_path
            / "noise.local.json"
        ),
        local_loaded=False,
    )


@pytest.fixture
def target_block() -> SrtBlock:
    return SrtBlock(
        number="80",
        timestamp=(
            "00:00:00,000 --> "
            "00:00:01,000"
        ),
        text=SOURCE_TEXT,
    )


def test_extract_untranslated_english_error_ids() -> None:
    actual = (
        extract_untranslated_english_error_ids(
            [
                UNTRANSLATED_ERROR,
                (
                    "Glossary violation: "
                    "subtitle_id='79'"
                ),
            ]
        )
    )

    assert actual == {
        "80",
    }


def test_probable_ocr_source_line_is_detected(
    noise_dictionary: NoiseDictionary,
) -> None:
    actual = is_probable_ocr_source_line(
        OCR_LINE,
        noise_dictionary,
    )

    assert actual is True


@pytest.mark.parametrize(
    "source_line",
    [
        "the wrong people!",
        "We are the wrong people!",
        "I am ready.",
        "Plan B is ready.",
    ],
)
def test_normal_english_is_not_probable_ocr(
    source_line: str,
    noise_dictionary: NoiseDictionary,
) -> None:
    actual = is_probable_ocr_source_line(
        source_line,
        noise_dictionary,
    )

    assert actual is False


def test_find_probable_untranslated_ocr_lines(
    target_block: SrtBlock,
    noise_dictionary: NoiseDictionary,
) -> None:
    actual = (
        find_probable_untranslated_ocr_lines(
            target_blocks=[
                target_block,
            ],
            translated_texts=[
                UNTRANSLATED_RESULT,
            ],
            errors=[
                UNTRANSLATED_ERROR,
            ],
            noise_dictionary=(
                noise_dictionary
            ),
        )
    )

    assert actual == {
        "80": [
            OCR_LINE,
        ],
    }


def test_does_not_select_normal_source_line(
    target_block: SrtBlock,
    noise_dictionary: NoiseDictionary,
) -> None:
    actual = (
        find_probable_untranslated_ocr_lines(
            target_blocks=[
                target_block,
            ],
            translated_texts=[
                (
                    "the wrong people!"
                    "は間違った人々です！"
                ),
            ],
            errors=[
                (
                    "Untranslated English sentence "
                    "detected: subtitle_id='80', "
                    "text='the wrong people!"
                    "は間違った人々です！'"
                ),
            ],
            noise_dictionary=(
                noise_dictionary
            ),
        )
    )

    assert actual == {}


def test_apply_level_1_fallback_to_raw_text(
    target_block: SrtBlock,
) -> None:
    (
        corrected,
        applied,
    ) = apply_level_1_ocr_fallback(
        target_blocks=[
            target_block,
        ],
        translated_texts=[
            UNTRANSLATED_RESULT,
        ],
        probable_ocr_lines={
            "80": [
                OCR_LINE,
            ],
        },
    )

    assert corrected == [
        (
            f"[1]{OCR_LINE}[/1]"
            "は間違った人々です！"
        ),
    ]

    assert applied == {
        "80": [
            OCR_LINE,
        ],
    }


@pytest.mark.parametrize(
    "existing_level",
    [
        "3",
        "5",
    ],
)
def test_existing_tag_is_replaced_without_nesting(
    existing_level: str,
    target_block: SrtBlock,
) -> None:
    existing_translation = (
        f"[{existing_level}]"
        f"{OCR_LINE}"
        f"[/{existing_level}]"
        "／間違った人たちを！"
    )

    (
        corrected,
        applied,
    ) = apply_level_1_ocr_fallback(
        target_blocks=[
            target_block,
        ],
        translated_texts=[
            existing_translation,
        ],
        probable_ocr_lines={
            "80": [
                OCR_LINE,
            ],
        },
    )

    expected = (
        f"[1]{OCR_LINE}[/1]"
        "／間違った人たちを！"
    )

    assert corrected == [
        expected,
    ]

    assert applied == {
        "80": [
            OCR_LINE,
        ],
    }

    assert "[3][1]" not in corrected[0]
    assert "[5][1]" not in corrected[0]
    assert "[/1][/3]" not in corrected[0]
    assert "[/1][/5]" not in corrected[0]


def test_existing_level_1_tag_is_not_duplicated(
    target_block: SrtBlock,
) -> None:
    existing_translation = (
        f"[1]{OCR_LINE}[/1]"
        "／間違った人たちを！"
    )

    (
        corrected,
        applied,
    ) = apply_level_1_ocr_fallback(
        target_blocks=[
            target_block,
        ],
        translated_texts=[
            existing_translation,
        ],
        probable_ocr_lines={
            "80": [
                OCR_LINE,
            ],
        },
    )

    assert corrected == [
        existing_translation,
    ]

    assert applied == {}

    assert corrected[0].count(
        f"[1]{OCR_LINE}[/1]"
    ) == 1


def test_fallback_ignores_other_subtitle_ids(
    target_block: SrtBlock,
) -> None:
    (
        corrected,
        applied,
    ) = apply_level_1_ocr_fallback(
        target_blocks=[
            target_block,
        ],
        translated_texts=[
            UNTRANSLATED_RESULT,
        ],
        probable_ocr_lines={
            "79": [
                OCR_LINE,
            ],
        },
    )

    assert corrected == [
        UNTRANSLATED_RESULT,
    ]

    assert applied == {}


def test_retry_instruction_contains_required_level_1_tag() -> None:
    instruction = (
        build_untranslated_english_retry_instruction(
            errors=[
                UNTRANSLATED_ERROR,
            ],
            probable_ocr_lines={
                "80": [
                    OCR_LINE,
                ],
            },
        )
    )

    assert "字幕ID: 80" in instruction
    assert (
        f"原文行: {OCR_LINE}"
        in instruction
    )
    assert (
        f"必須形式: [1]{OCR_LINE}[/1]"
        in instruction
    )
    assert (
        "正常な英文まで[1]タグで囲まない"
        in instruction
    )


def test_retry_instruction_is_empty_without_error() -> None:
    instruction = (
        build_untranslated_english_retry_instruction(
            errors=[],
            probable_ocr_lines={
                "80": [
                    OCR_LINE,
                ],
            },
        )
    )

    assert instruction == ""


def test_validator_accepts_complete_level_1_source_line(
    target_block: SrtBlock,
    noise_dictionary: NoiseDictionary,
) -> None:
    response = json.dumps(
        {
            "targets": {
                "80": {
                    "source": {
                        "speaker": None,
                        "text": SOURCE_TEXT,
                    },
                    "translation": (
                        f"[1]{OCR_LINE}[/1]"
                        "／間違った人たちを！"
                    ),
                },
            },
        },
        ensure_ascii=False,
    )

    validation = (
        validate_translation_response(
            response,
            expected_ids=[
                "80",
            ],
            source_speakers=[
                None,
            ],
            source_texts=[
                SOURCE_TEXT,
            ],
            noise_dictionary=(
                noise_dictionary
            ),
            glossary_entries={},
        )
    )

    assert validation.valid is True
    assert validation.reasons == []

    assert validation.translated_texts == [
        (
            "（判読不能）"
            "／間違った人たちを！"
        ),
    ]

    assert validation.noise_candidates == [
        OCR_LINE,
    ]


def test_validator_rejects_partial_level_1_source_line(
    noise_dictionary: NoiseDictionary,
) -> None:
    source_text = (
        "You want a gold\n"
        "irc] m ce) carla"
    )

    response = json.dumps(
        {
            "targets": {
                "452": {
                    "source": {
                        "speaker": None,
                        "text": source_text,
                    },
                    "translation": (
                        "金色の"
                        "[1]irc] m ce)[/1]"
                        "カーラが欲しい？"
                    ),
                },
            },
        },
        ensure_ascii=False,
    )

    validation = (
        validate_translation_response(
            response,
            expected_ids=[
                "452",
            ],
            source_speakers=[
                None,
            ],
            source_texts=[
                source_text,
            ],
            noise_dictionary=(
                noise_dictionary
            ),
            glossary_entries={},
        )
    )

    assert validation.valid is False

    assert any(
        reason.startswith(
            "Level 1 translation tag must match "
            "a complete source line:"
        )
        for reason in validation.reasons
    )


def test_level_1_fallback_integration(
    monkeypatch: pytest.MonkeyPatch,
    target_block: SrtBlock,
    noise_dictionary: NoiseDictionary,
) -> None:
    saved_candidates: list[str] = []

    def fake_append_noise_candidates(
        dictionary: NoiseDictionary,
        candidates: list[str],
    ) -> list[str]:
        assert dictionary is noise_dictionary

        saved_candidates.extend(
            candidates
        )

        return list(
            candidates
        )

    monkeypatch.setattr(
        translation_chunk,
        "append_noise_candidates",
        fake_append_noise_candidates,
    )

    monkeypatch.setattr(
        translation_chunk,
        "print_saved_noise_candidates",
        lambda entries, dictionary: None,
    )

    result = try_level_1_ocr_fallback(
        target_blocks=[
            target_block,
        ],
        translated_texts=[
            UNTRANSLATED_RESULT,
        ],
        errors=[
            UNTRANSLATED_ERROR,
        ],
        probable_ocr_lines={
            "80": [
                OCR_LINE,
            ],
        },
        noise_dictionary=(
            noise_dictionary
        ),
        glossary_entries={},
    )

    assert result == [
        (
            "（判読不能）"
            "は間違った人々です！"
        ),
    ]

    assert saved_candidates == [
        OCR_LINE,
    ]


def test_level_1_fallback_does_not_run_with_other_errors(
    target_block: SrtBlock,
    noise_dictionary: NoiseDictionary,
) -> None:
    result = try_level_1_ocr_fallback(
        target_blocks=[
            target_block,
        ],
        translated_texts=[
            UNTRANSLATED_RESULT,
        ],
        errors=[
            UNTRANSLATED_ERROR,
            (
                "Glossary violation: "
                "subtitle_id='80'"
            ),
        ],
        probable_ocr_lines={
            "80": [
                OCR_LINE,
            ],
        },
        noise_dictionary=(
            noise_dictionary
        ),
        glossary_entries={},
    )

    assert result is None


def test_symbol_dense_ocr_source_line_is_detected(
) -> None:
    actual = is_symbol_dense_ocr_source_line(
        SYMBOL_DENSE_OCR_LINE
    )

    assert actual is True


def test_symbol_dense_ocr_source_line_is_detected(
) -> None:
    actual = is_symbol_dense_ocr_source_line(
        SYMBOL_DENSE_OCR_LINE
    )

    assert actual is True


def test_short_mixed_case_ocr_source_line_is_detected(
) -> None:
    actual = (
        is_short_mixed_case_ocr_source_line(
            "dam IAN el ESie"
        )
    )

    assert actual is True


@pytest.mark.parametrize(
    "source_line",
    [
        "How can I not",
        "I am not ready",
        "Plan B is ready",
        "NASA will send help",
        "This is my home",
        "This IS my HOME",
        "Colonel Young is here",
        "What do you mean",
        "I couldn't deal with it",
        "SG-1 is ready",
        "(CHIRPING)",
    ],
)
def test_normal_line_is_not_short_mixed_case_ocr(
    source_line: str,
) -> None:
    actual = (
        is_short_mixed_case_ocr_source_line(
            source_line
        )
    )

    assert actual is False


@pytest.mark.parametrize(
    "source_line",
    [
        "",
        "   ",
        "sa",
        "Ui maar i mele",
        "dam ian el esie",
        "DAM IAN EL ESIE",
        "one two three four five",
        "P4X-351",
        "x = 10",
    ],
)
def test_ambiguous_line_is_not_short_mixed_case_ocr(
    source_line: str,
) -> None:
    actual = (
        is_short_mixed_case_ocr_source_line(
            source_line
        )
    )

    assert actual is False


def test_low_symbol_word_salad_ocr_source_line_is_detected(
) -> None:
    actual = (
        is_low_symbol_word_salad_ocr_source_line(
            "Ui maar i mele aah ml iaa"
        )
    )

    assert actual is True


@pytest.mark.parametrize(
    "source_line",
    [
        "How can I not",
        'The "us" on that recording',
        "I couldn't deal with it",
        "Hopefully we have proven that",
        "We need to find a way home",
        "(CONSOLE BEEPS)",
        "SG-1 is ready",
        "Colonel Young is in command",
    ],
)
def test_normal_line_is_not_low_symbol_word_salad_ocr(
    source_line: str,
) -> None:
    actual = (
        is_low_symbol_word_salad_ocr_source_line(
            source_line
        )
    )

    assert actual is False


@pytest.mark.parametrize(
    "source_line",
    [
        "",
        "   ",
        "sa",
        "Ui maar i mele",
        "one two three four five",
        "P4X-351",
        "x = 10",
    ],
)
def test_ambiguous_line_is_not_low_symbol_word_salad_ocr(
    source_line: str,
) -> None:
    actual = (
        is_low_symbol_word_salad_ocr_source_line(
            source_line
        )
    )

    assert actual is False


@pytest.mark.parametrize(
    "source_line",
    [
        "This is what Destiny intended.",
        "What's going on?",
        "(CROWD CHATTERING)",
        "(LOW MECHANICAL HUM) (CONTINUES)",
        "SG-1",
        "F-302",
        "P4X-351",
        "x = 10",
        "sa",
    ],
)
def test_normal_or_ambiguous_line_is_not_symbol_dense_ocr(
    source_line: str,
) -> None:
    actual = is_symbol_dense_ocr_source_line(
        source_line
    )

    assert actual is False


def test_probable_ocr_source_line_detects_symbol_dense_text(
    noise_dictionary: NoiseDictionary,
) -> None:
    actual = is_probable_ocr_source_line(
        SYMBOL_DENSE_OCR_LINE,
        noise_dictionary,
    )

    assert actual is True


def test_find_symbol_dense_untranslated_ocr_line(
    noise_dictionary: NoiseDictionary,
) -> None:
    target_block = SrtBlock(
        number="361",
        timestamp=(
            "00:32:31,867 --> "
            "00:32:35,287"
        ),
        text=SYMBOL_DENSE_SOURCE_TEXT,
    )

    actual = (
        find_probable_untranslated_ocr_lines(
            target_blocks=[
                target_block,
            ],
            translated_texts=[
                SYMBOL_DENSE_TRANSLATION,
            ],
            errors=[
                SYMBOL_DENSE_UNTRANSLATED_ERROR,
            ],
            noise_dictionary=(
                noise_dictionary
            ),
        )
    )

    assert actual == {
        "361": [
            SYMBOL_DENSE_OCR_LINE,
        ],
    }


def test_symbol_dense_ocr_fallback_wraps_complete_line(
) -> None:
    target_block = SrtBlock(
        number="361",
        timestamp=(
            "00:32:31,867 --> "
            "00:32:35,287"
        ),
        text=SYMBOL_DENSE_SOURCE_TEXT,
    )

    (
        corrected,
        applied,
    ) = apply_level_1_ocr_fallback(
        target_blocks=[
            target_block,
        ],
        translated_texts=[
            SYMBOL_DENSE_TRANSLATION,
        ],
        probable_ocr_lines={
            "361": [
                SYMBOL_DENSE_OCR_LINE,
            ],
        },
    )

    assert corrected == [
        (
            "これはデスティニーが"
            "最初から意図したことであり、"
            f"[1]{SYMBOL_DENSE_OCR_LINE}[/1]"
        ),
    ]

    assert applied == {
        "361": [
            SYMBOL_DENSE_OCR_LINE,
        ],
    }


def test_short_ambiguous_line_is_not_high_confidence_ocr(
    noise_dictionary: NoiseDictionary,
) -> None:
    actual = is_probable_ocr_source_line(
        "sa",
        noise_dictionary,
    )

    assert actual is False
