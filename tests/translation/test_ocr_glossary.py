from __future__ import annotations

import pytest
from lib.profile.glossary import (
    GlossaryEntries,
    GlossaryEntry,
)
from lib.profile.ocr_scoring import (
    load_ocr_scoring_config,
)
from lib.translation.ocr_glossary import (
    assess_ocr_glossary_match,
    calculate_glossary_similarity,
    find_similar_glossary_source,
    normalize_glossary_comparison_text,
)


def build_glossary_entries(
    *entries: GlossaryEntry,
) -> GlossaryEntries:
    return GlossaryEntries(
        tuple(entries)
    )


def test_normalize_glossary_comparison_text() -> None:
    assert (
        normalize_glossary_comparison_text(
            "SG-1 Gate",
            case_sensitive=False,
        )
        == "sg1gate"
    )


def test_normalize_glossary_comparison_text_preserves_case() -> None:
    assert (
        normalize_glossary_comparison_text(
            "SG-1",
            case_sensitive=True,
        )
        == "SG1"
    )


def test_assess_ocr_glossary_match_finds_case_insensitive_exact_match(
) -> None:
    entries = build_glossary_entries(
        GlossaryEntry(
            source="Stargate",
            target="スターゲイト",
        )
    )

    result = assess_ocr_glossary_match(
        "stargate",
        entries,
        load_ocr_scoring_config(),
    )

    assert (
        result.exact_source
        == "Stargate"
    )

    assert result.exact_protection
    assert result.has_exact_match

    assert (
        result.similar_source
        is None
    )

    assert not result.has_similar_match
    assert result.similarity == 1.0


def test_assess_ocr_glossary_match_respects_case_sensitive_exact_match(
) -> None:
    entries = build_glossary_entries(
        GlossaryEntry(
            source="SG-1",
            target="SG-1",
            case_sensitive=True,
        )
    )

    exact_result = (
        assess_ocr_glossary_match(
            "SG-1",
            entries,
            load_ocr_scoring_config(),
        )
    )

    different_case_result = (
        assess_ocr_glossary_match(
            "sg-1",
            entries,
            load_ocr_scoring_config(),
        )
    )

    assert (
        exact_result.exact_source
        == "SG-1"
    )

    assert exact_result.exact_protection

    assert (
        different_case_result.exact_source
        is None
    )

    assert (
        different_case_result.similar_source
        is None
    )


def test_assess_ocr_glossary_match_does_not_protect_long_exact_match(
) -> None:
    entries = build_glossary_entries(
        GlossaryEntry(
            source="AncientTechnology",
            target="古代技術",
        )
    )

    result = assess_ocr_glossary_match(
        "AncientTechnology",
        entries,
        load_ocr_scoring_config(),
    )

    assert (
        result.exact_source
        == "AncientTechnology"
    )

    assert result.has_exact_match
    assert not result.exact_protection
    assert result.similarity == 1.0


def test_assess_ocr_glossary_match_finds_similar_source(
) -> None:
    entries = build_glossary_entries(
        GlossaryEntry(
            source="Stargate",
            target="スターゲイト",
        )
    )

    result = assess_ocr_glossary_match(
        "Stargte",
        entries,
        load_ocr_scoring_config(),
    )

    assert result.exact_source is None
    assert not result.exact_protection

    assert (
        result.similar_source
        == "Stargate"
    )

    assert result.has_similar_match

    assert result.similarity == pytest.approx(
        0.9333333333333333
    )


def test_assess_ocr_glossary_match_rejects_low_similarity(
) -> None:
    entries = build_glossary_entries(
        GlossaryEntry(
            source="Stargate",
            target="スターゲイト",
        )
    )

    result = assess_ocr_glossary_match(
        "Starxxxx",
        entries,
        load_ocr_scoring_config(),
    )

    assert result.exact_source is None
    assert result.similar_source is None
    assert result.similarity == 0.0


def test_assess_ocr_glossary_match_ignores_short_glossary_source(
) -> None:
    entries = build_glossary_entries(
        GlossaryEntry(
            source="ABC",
            target="ABC",
        )
    )

    result = assess_ocr_glossary_match(
        "ABD",
        entries,
        load_ocr_scoring_config(),
    )

    assert result.exact_source is None
    assert result.similar_source is None
    assert result.similarity == 0.0


def test_assess_ocr_glossary_match_rejects_large_length_difference(
) -> None:
    entries = build_glossary_entries(
        GlossaryEntry(
            source="Stargate",
            target="スターゲイト",
        )
    )

    result = assess_ocr_glossary_match(
        "StargateXXX",
        entries,
        load_ocr_scoring_config(),
    )

    assert result.exact_source is None
    assert result.similar_source is None
    assert result.similarity == 0.0


def test_find_similar_glossary_source_selects_highest_similarity(
) -> None:
    entries = build_glossary_entries(
        GlossaryEntry(
            source="Stargate",
            target="スターゲイト",
        ),
        GlossaryEntry(
            source="Starship",
            target="宇宙船",
        ),
    )

    (
        source_term,
        similarity,
    ) = find_similar_glossary_source(
        "Stargte",
        entries,
        load_ocr_scoring_config(),
    )

    assert source_term == "Stargate"

    assert similarity == pytest.approx(
        0.9333333333333333
    )


def test_find_similar_glossary_source_preserves_order_for_tie(
) -> None:
    entries = build_glossary_entries(
        GlossaryEntry(
            source="Stargate",
            target="スターゲイト",
        ),
        GlossaryEntry(
            source="Stargaze",
            target="星を見る",
        ),
    )

    (
        source_term,
        similarity,
    ) = find_similar_glossary_source(
        "Stargabe",
        entries,
        load_ocr_scoring_config(),
    )

    assert source_term == "Stargate"

    assert similarity == pytest.approx(
        0.875
    )


def test_calculate_glossary_similarity_ignores_separators(
) -> None:
    similarity = (
        calculate_glossary_similarity(
            "SG 01",
            "SG-01",
            case_sensitive=False,
        )
    )

    assert similarity == 1.0


def test_assess_ocr_glossary_match_returns_no_match_for_unrelated_text(
) -> None:
    entries = build_glossary_entries(
        GlossaryEntry(
            source="Stargate",
            target="スターゲイト",
        ),
        GlossaryEntry(
            source="Destiny",
            target="デスティニー",
        ),
    )

    result = assess_ocr_glossary_match(
        "ordinary dialogue",
        entries,
        load_ocr_scoring_config(),
    )

    assert result.exact_source is None
    assert not result.exact_protection
    assert result.similar_source is None
    assert result.similarity == 0.0
    assert not result.has_exact_match
    assert not result.has_similar_match
