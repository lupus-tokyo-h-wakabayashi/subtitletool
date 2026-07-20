from __future__ import annotations

import pytest
from lib.profile.glossary import (
    GlossaryEntries,
    GlossaryEntry,
)
from lib.profile.ocr_scoring import (
    OcrScoringConfig,
    load_ocr_scoring_config,
)
from lib.subtitle.srt import (
    SrtBlock,
)
from lib.translation.ocr_assessment import (
    assess_ocr_source_line,
    select_ocr_threshold,
)
from lib.translation.ocr_retry import (
    find_probable_untranslated_ocr_lines,
)


@pytest.fixture
def scoring_config() -> OcrScoringConfig:
    return load_ocr_scoring_config()


@pytest.fixture
def empty_glossary() -> GlossaryEntries:
    return GlossaryEntries(
        ()
    )


@pytest.fixture
def stargate_glossary() -> GlossaryEntries:
    return GlossaryEntries(
        (
            GlossaryEntry(
                source="Stargate",
                target="スターゲイト",
            ),
            GlossaryEntry(
                source="SG-1",
                target="SG-1",
                case_sensitive=True,
            ),
        )
    )


@pytest.mark.parametrize(
    "source_text",
    [
        "Hopefully, we've proven",
        "that's not our goal.",
        "I couldn't deal with it,",
        "I'm sorry.",
        "the thought of you",
    ],
)
def test_assess_protects_contractions_and_continuing_sentences(
    source_text: str,
    scoring_config: OcrScoringConfig,
    empty_glossary: GlossaryEntries,
) -> None:
    result = assess_ocr_source_line(
        source_text,
        empty_glossary,
        scoring_config,
        validation_failed=True,
        has_normal_sibling=True,
    )

    assert not result.probable_ocr

    assert not result.has_contribution(
        "invalid_single_letter"
    )


def build_untranslated_english_error(
    subtitle_id: str,
    text: str,
) -> str:
    return (
        "Untranslated English sentence detected: "
        f"subtitle_id={subtitle_id!r}, "
        f"text={text!r}"
    )


def test_select_ocr_threshold_uses_high_confidence(
    scoring_config: OcrScoringConfig,
) -> None:
    assert (
        select_ocr_threshold(
            scoring_config,
            validation_failed=False,
            has_normal_sibling=False,
        )
        == 12
    )


def test_select_ocr_threshold_uses_failed_subtitle(
    scoring_config: OcrScoringConfig,
) -> None:
    assert (
        select_ocr_threshold(
            scoring_config,
            validation_failed=True,
            has_normal_sibling=False,
        )
        == 8
    )


def test_select_ocr_threshold_uses_failed_with_normal_sibling(
    scoring_config: OcrScoringConfig,
) -> None:
    assert (
        select_ocr_threshold(
            scoring_config,
            validation_failed=True,
            has_normal_sibling=True,
        )
        == 6
    )


def test_assess_detects_unbalanced_symbol_damaged_ocr(
    scoring_config: OcrScoringConfig,
    empty_glossary: GlossaryEntries,
) -> None:
    result = assess_ocr_source_line(
        '"(CLR (=r 108',
        empty_glossary,
        scoring_config,
        validation_failed=True,
        has_normal_sibling=True,
    )

    assert result.probable_ocr
    assert result.score >= result.threshold

    assert result.has_contribution(
        "unbalanced_parentheses"
    )

    assert result.has_contribution(
        "unbalanced_double_quotes"
    )

    assert result.has_contribution(
        "multiple_unbalanced_delimiters"
    )

    assert result.has_contribution(
        "damaged_alphanumeric_structure"
    )


def test_assess_detects_symbol_dense_structure(
    scoring_config: OcrScoringConfig,
    empty_glossary: GlossaryEntries,
) -> None:
    result = assess_ocr_source_line(
        "AB (= CD | EF \\ GH",
        empty_glossary,
        scoring_config,
        validation_failed=True,
        has_normal_sibling=True,
    )

    assert result.probable_ocr

    assert result.has_contribution(
        "multiple_structural_symbols"
    )

    assert result.has_contribution(
        "dense_structural_symbols"
    )

    assert result.has_contribution(
        "strong_corruption_symbol"
    )

    assert result.has_contribution(
        "symbol_dense_structure"
    )


def test_assess_detects_low_symbol_word_salad(
    scoring_config: OcrScoringConfig,
    empty_glossary: GlossaryEntries,
) -> None:
    result = assess_ocr_source_line(
        "Ui maar i mele aah ml iaa",
        empty_glossary,
        scoring_config,
        validation_failed=True,
        has_normal_sibling=True,
    )

    assert result.probable_ocr

    assert result.has_contribution(
        "suspicious_tokens"
    )

    assert result.has_contribution(
        "low_symbol_word_salad"
    )


def test_assess_detects_short_mixed_case_ocr(
    scoring_config: OcrScoringConfig,
    empty_glossary: GlossaryEntries,
) -> None:
    result = assess_ocr_source_line(
        "dam IAN el ESie",
        empty_glossary,
        scoring_config,
        validation_failed=True,
        has_normal_sibling=True,
    )

    assert result.probable_ocr

    assert result.has_contribution(
        "irregular_mixed_case"
    )

    assert result.has_contribution(
        "short_mixed_case"
    )


def test_context_word_salad_requires_failed_subtitle(
    scoring_config: OcrScoringConfig,
    empty_glossary: GlossaryEntries,
) -> None:
    result = assess_ocr_source_line(
        "Ui maar i mele aah ml iaa",
        empty_glossary,
        scoring_config,
        validation_failed=False,
        has_normal_sibling=True,
    )

    assert result.threshold == 12
    assert not result.probable_ocr

    assert not result.has_contribution(
        "low_symbol_word_salad"
    )


def test_natural_sentence_protects_uppercase_acronym(
    scoring_config: OcrScoringConfig,
    empty_glossary: GlossaryEntries,
) -> None:
    result = assess_ocr_source_line(
        "dropped out of FTL",
        empty_glossary,
        scoring_config,
        validation_failed=True,
        has_normal_sibling=True,
    )

    assert result.threshold == 6
    assert not result.probable_ocr

    assert result.has_contribution(
        "natural_sentence"
    )


def test_short_mixed_case_requires_validation_context(
    scoring_config: OcrScoringConfig,
    empty_glossary: GlossaryEntries,
) -> None:
    normal_result = assess_ocr_source_line(
        "dam IAN el ESie",
        empty_glossary,
        scoring_config,
        validation_failed=False,
        has_normal_sibling=False,
    )

    failed_result = assess_ocr_source_line(
        "dam IAN el ESie",
        empty_glossary,
        scoring_config,
        validation_failed=True,
        has_normal_sibling=False,
    )

    sibling_result = assess_ocr_source_line(
        "dam IAN el ESie",
        empty_glossary,
        scoring_config,
        validation_failed=True,
        has_normal_sibling=True,
    )

    assert normal_result.threshold == 12
    assert not normal_result.probable_ocr

    assert failed_result.threshold == 12
    assert not failed_result.probable_ocr

    assert sibling_result.threshold == 6
    assert sibling_result.probable_ocr


def test_assess_protects_exact_short_glossary_term(
    scoring_config: OcrScoringConfig,
    stargate_glossary: GlossaryEntries,
) -> None:
    result = assess_ocr_source_line(
        "SG-1",
        stargate_glossary,
        scoring_config,
        validation_failed=True,
        has_normal_sibling=True,
    )

    assert not result.probable_ocr

    assert (
        result.glossary_match.exact_source
        == "SG-1"
    )

    assert (
        result.glossary_match.exact_protection
    )

    assert result.has_contribution(
        "glossary_exact_match"
    )

    assert result.has_contribution(
        "identifier_like"
    )


def test_assess_adds_similar_glossary_weight(
    scoring_config: OcrScoringConfig,
    stargate_glossary: GlossaryEntries,
) -> None:
    result = assess_ocr_source_line(
        "Stargte",
        stargate_glossary,
        scoring_config,
        validation_failed=True,
        has_normal_sibling=True,
    )

    assert (
        result.glossary_match.similar_source
        == "Stargate"
    )

    assert result.has_contribution(
        "glossary_similar_match"
    )

    assert not result.has_contribution(
        "damaged_without_glossary_match"
    )


def test_assess_adds_glossary_miss_only_after_damage_threshold(
    scoring_config: OcrScoringConfig,
    empty_glossary: GlossaryEntries,
) -> None:
    damaged_result = assess_ocr_source_line(
        '"(CLR (=r 108',
        empty_glossary,
        scoring_config,
        validation_failed=True,
        has_normal_sibling=True,
    )

    normal_result = assess_ocr_source_line(
        "ordinary dialogue",
        empty_glossary,
        scoring_config,
        validation_failed=True,
        has_normal_sibling=True,
    )

    assert (
        damaged_result.damage_score
        >= scoring_config.get_value(
        "limits",
        "minimum_damage_score_for_glossary_miss",
    )
    )

    assert damaged_result.has_contribution(
        "damaged_without_glossary_match"
    )

    assert not normal_result.has_contribution(
        "damaged_without_glossary_match"
    )


def test_assess_protects_identifier(
    scoring_config: OcrScoringConfig,
    empty_glossary: GlossaryEntries,
) -> None:
    result = assess_ocr_source_line(
        "SG-1",
        empty_glossary,
        scoring_config,
        validation_failed=True,
        has_normal_sibling=True,
    )

    assert not result.probable_ocr

    assert result.has_contribution(
        "identifier_like"
    )


def test_assess_does_not_protect_damaged_identifier(
    scoring_config: OcrScoringConfig,
    empty_glossary: GlossaryEntries,
) -> None:
    result = assess_ocr_source_line(
        "(SG-1",
        empty_glossary,
        scoring_config,
        validation_failed=True,
        has_normal_sibling=True,
    )

    assert result.has_contribution(
        "unbalanced_parentheses"
    )

    assert not result.has_contribution(
        "identifier_like"
    )


def test_assess_protects_equation(
    scoring_config: OcrScoringConfig,
    empty_glossary: GlossaryEntries,
) -> None:
    result = assess_ocr_source_line(
        "E=mc2",
        empty_glossary,
        scoring_config,
        validation_failed=True,
        has_normal_sibling=True,
    )

    assert not result.probable_ocr

    assert result.has_contribution(
        "equation_like"
    )


def test_assess_protects_time_expression(
    scoring_config: OcrScoringConfig,
    empty_glossary: GlossaryEntries,
) -> None:
    result = assess_ocr_source_line(
        "So, 7:00, then?",
        empty_glossary,
        scoring_config,
        validation_failed=True,
        has_normal_sibling=True,
    )

    assert not result.probable_ocr

    assert result.has_contribution(
        "time_like"
    )


def test_assess_protects_natural_sentence(
    scoring_config: OcrScoringConfig,
    empty_glossary: GlossaryEntries,
) -> None:
    result = assess_ocr_source_line(
        "Okay, when she returns home...",
        empty_glossary,
        scoring_config,
        validation_failed=True,
        has_normal_sibling=True,
    )

    assert not result.probable_ocr

    assert result.has_contribution(
        "natural_sentence"
    )


def test_assess_empty_text(
    scoring_config: OcrScoringConfig,
    empty_glossary: GlossaryEntries,
) -> None:
    result = assess_ocr_source_line(
        "   ",
        empty_glossary,
        scoring_config,
        validation_failed=True,
        has_normal_sibling=True,
    )

    assert result.damage_score == 0
    assert result.score == 0
    assert result.threshold == 6
    assert not result.probable_ocr
    assert result.contributions == ()


def test_assessment_contribution_uses_config_description(
    scoring_config: OcrScoringConfig,
    empty_glossary: GlossaryEntries,
) -> None:
    result = assess_ocr_source_line(
        "(damaged",
        empty_glossary,
        scoring_config,
        validation_failed=True,
        has_normal_sibling=True,
    )

    contribution = next(
        contribution
        for contribution in (
            result.contributions
        )
        if (
            contribution.name
            == "unbalanced_parentheses"
        )
    )

    assert (
        contribution.description
        == scoring_config.weights[
            "unbalanced_parentheses"
        ].description
    )


def test_standard_recovery_finds_structural_ocr_line(
    scoring_config: OcrScoringConfig,
    empty_glossary: GlossaryEntries,
) -> None:
    source_text = '"(CLR (=r 108'

    blocks = [
        SrtBlock(
            number="497",
            timestamp=(
                "00:25:22,563 --> "
                "00:25:24,773"
            ),
            text=source_text,
        ),
    ]

    actual = (
        find_probable_untranslated_ocr_lines(
            blocks,
            [
                source_text,
            ],
            [
                build_untranslated_english_error(
                    "497",
                    source_text,
                ),
            ],
            glossary_entries=(
                empty_glossary
            ),
            scoring_config=(
                scoring_config
            ),
        )
    )

    assert actual == {
        "497": [
            source_text,
        ],
    }


def test_standard_recovery_finds_only_damaged_mixed_line(
    scoring_config: OcrScoringConfig,
    empty_glossary: GlossaryEntries,
) -> None:
    damaged_line = (
        "Ui maar i mele aah ml iaa"
    )

    normal_line = (
        "seeing the old homestead again."
    )

    source_text = (
        f"{damaged_line}\n"
        f"{normal_line}"
    )

    blocks = [
        SrtBlock(
            number="98",
            timestamp=(
                "00:05:00,000 --> "
                "00:05:02,000"
            ),
            text=source_text,
        ),
    ]

    actual = (
        find_probable_untranslated_ocr_lines(
            blocks,
            [
                source_text,
            ],
            [
                build_untranslated_english_error(
                    "98",
                    source_text,
                ),
            ],
            glossary_entries=(
                empty_glossary
            ),
            scoring_config=(
                scoring_config
            ),
        )
    )

    assert actual == {
        "98": [
            damaged_line,
        ],
    }


def test_standard_recovery_ignores_source_not_copied_to_translation(
    scoring_config: OcrScoringConfig,
    empty_glossary: GlossaryEntries,
) -> None:
    source_text = '"(CLR (=r 108'

    blocks = [
        SrtBlock(
            number="497",
            timestamp=(
                "00:25:22,563 --> "
                "00:25:24,773"
            ),
            text=source_text,
        ),
    ]

    actual = (
        find_probable_untranslated_ocr_lines(
            blocks,
            [
                "判読できない文字列です。",
            ],
            [
                build_untranslated_english_error(
                    "497",
                    source_text,
                ),
            ],
            glossary_entries=(
                empty_glossary
            ),
            scoring_config=(
                scoring_config
            ),
        )
    )

    assert actual == {}


def test_standard_recovery_protects_glossary_identifier(
    scoring_config: OcrScoringConfig,
    stargate_glossary: GlossaryEntries,
) -> None:
    source_text = "SG-1"

    blocks = [
        SrtBlock(
            number="280",
            timestamp=(
                "00:14:41,506 --> "
                "00:14:42,799"
            ),
            text=source_text,
        ),
    ]

    actual = (
        find_probable_untranslated_ocr_lines(
            blocks,
            [
                source_text,
            ],
            [
                build_untranslated_english_error(
                    "280",
                    source_text,
                ),
            ],
            glossary_entries=(
                stargate_glossary
            ),
            scoring_config=(
                scoring_config
            ),
        )
    )

    assert actual == {}


@pytest.mark.parametrize(
    "source_text",
    [
        (
            "Hopefully, we've proven\n"
            "that's not our goal."
        ),
        (
            "I couldn't deal with it,\n"
            "the thought of you\n"
            "being trapped on that ship."
        ),
    ],
)
def test_standard_recovery_protects_natural_english(
    source_text: str,
    scoring_config: OcrScoringConfig,
    empty_glossary: GlossaryEntries,
) -> None:
    blocks = [
        SrtBlock(
            number="602",
            timestamp=(
                "00:00:01,100 --> "
                "00:00:03,000"
            ),
            text=source_text,
        ),
    ]

    actual = (
        find_probable_untranslated_ocr_lines(
            blocks,
            [
                source_text,
            ],
            [
                build_untranslated_english_error(
                    "602",
                    source_text,
                ),
            ],
            glossary_entries=(
                empty_glossary
            ),
            scoring_config=(
                scoring_config
            ),
        )
    )

    assert actual == {}


def test_standard_recovery_ignores_unrelated_validation_error(
    scoring_config: OcrScoringConfig,
    empty_glossary: GlossaryEntries,
) -> None:
    source_text = '"(CLR (=r 108'

    blocks = [
        SrtBlock(
            number="497",
            timestamp=(
                "00:25:22,563 --> "
                "00:25:24,773"
            ),
            text=source_text,
        ),
    ]

    actual = (
        find_probable_untranslated_ocr_lines(
            blocks,
            [
                source_text,
            ],
            [
                (
                    "Chinese-specific characters "
                    "detected: "
                    "subtitle_id='497'"
                ),
            ],
            glossary_entries=(
                empty_glossary
            ),
            scoring_config=(
                scoring_config
            ),
        )
    )

    assert actual == {}
