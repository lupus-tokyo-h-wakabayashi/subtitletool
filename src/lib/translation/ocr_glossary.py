from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Mapping

from lib.profile.glossary import (
    GlossaryEntries,
)
from lib.profile.ocr_scoring import (
    OcrScoringConfig,
)


@dataclass(frozen=True)
class OcrGlossaryMatch:
    """
    OCR候補とGlossary sourceの照合結果。

    exact_source:
        原文全体と完全一致したGlossary source。

    exact_protection:
        短い固有名詞や識別子として
        OCR破損スコアから保護できるか。

    similar_source:
        OCR破損の可能性がある
        最も類似したGlossary source。

    similarity:
        similar_sourceとの類似度。
        一致候補がない場合は0.0。
    """

    exact_source: str | None
    exact_protection: bool
    similar_source: str | None
    similarity: float

    @property
    def has_exact_match(
        self,
    ) -> bool:
        return (
            self.exact_source
            is not None
        )

    @property
    def has_similar_match(
        self,
    ) -> bool:
        return (
            self.similar_source
            is not None
        )


def normalize_glossary_comparison_text(
    text: str,
    *,
    case_sensitive: bool,
) -> str:
    """
    Glossary類似比較用に文字列を正規化する。

    空白や区切り記号のOCR揺れを吸収するため、
    英数字を含むUnicode英数文字だけを残す。

    case_sensitive=Falseの場合は、
    大文字・小文字も正規化する。
    """
    comparison_text = (
        text
        if case_sensitive
        else text.casefold()
    )

    return "".join(
        character
        for character in comparison_text
        if character.isalnum()
    )


def is_glossary_source_case_sensitive(
    glossary_entries: Mapping[
        str,
        str,
    ],
    source_term: str,
) -> bool:
    """
    Glossary sourceが大小文字区別対象か返す。

    通常のMappingが渡された場合は、
    従来どおり大小文字を区別しない。
    """
    if not isinstance(
        glossary_entries,
        GlossaryEntries,
    ):
        return False

    return (
        glossary_entries.is_case_sensitive(
            source_term
        )
    )


def is_exact_glossary_match(
    text: str,
    source_term: str,
    *,
    case_sensitive: bool,
) -> bool:
    """
    OCR候補全体とGlossary sourceの
    完全一致を判定する。

    部分一致ではなく字幕行全体を比較するため、
    通常の英文中に用語が含まれるだけでは
    完全一致保護の対象にならない。
    """
    candidate = text.strip()
    source = source_term.strip()

    if case_sensitive:
        return candidate == source

    return (
        candidate.casefold()
        == source.casefold()
    )


def calculate_glossary_similarity(
    text: str,
    source_term: str,
    *,
    case_sensitive: bool,
) -> float:
    """
    OCR候補とGlossary sourceの類似度を返す。

    比較前に空白と区切り記号を除外し、
    OCRによる記号位置の揺れを吸収する。
    """
    candidate = (
        normalize_glossary_comparison_text(
            text,
            case_sensitive=case_sensitive,
        )
    )

    source = (
        normalize_glossary_comparison_text(
            source_term,
            case_sensitive=case_sensitive,
        )
    )

    if not candidate or not source:
        return 0.0

    return SequenceMatcher(
        None,
        candidate,
        source,
        autojunk=False,
    ).ratio()


def find_exact_glossary_source(
    text: str,
    glossary_entries: Mapping[
        str,
        str,
    ],
) -> str | None:
    """
    OCR候補全体と完全一致する
    Glossary sourceを返す。
    """
    for source_term in (
        glossary_entries
    ):
        case_sensitive = (
            is_glossary_source_case_sensitive(
                glossary_entries,
                source_term,
            )
        )

        if is_exact_glossary_match(
            text,
            source_term,
            case_sensitive=case_sensitive,
        ):
            return source_term

    return None


def find_similar_glossary_source(
    text: str,
    glossary_entries: Mapping[
        str,
        str,
    ],
    scoring_config: OcrScoringConfig,
) -> tuple[str | None, float]:
    """
    設定された条件を満たす
    最も類似したGlossary sourceを返す。

    完全一致はこの関数の対象外とし、
    OCR破損候補の検出だけを担当する。
    """
    minimum_source_length = int(
        scoring_config.get_value(
            "limits",
            "minimum_glossary_source_length",
        )
    )

    maximum_length_difference = int(
        scoring_config.get_value(
            "limits",
            "maximum_glossary_length_difference",
        )
    )

    minimum_similarity = float(
        scoring_config.get_value(
            "limits",
            "minimum_glossary_similarity",
        )
    )

    best_source: str | None = None
    best_similarity = 0.0

    for source_term in (
        glossary_entries
    ):
        case_sensitive = (
            is_glossary_source_case_sensitive(
                glossary_entries,
                source_term,
            )
        )

        candidate_normalized = (
            normalize_glossary_comparison_text(
                text,
                case_sensitive=case_sensitive,
            )
        )

        source_normalized = (
            normalize_glossary_comparison_text(
                source_term,
                case_sensitive=case_sensitive,
            )
        )

        if (
            len(source_normalized)
            < minimum_source_length
        ):
            continue

        if (
            abs(
                len(candidate_normalized)
                - len(source_normalized)
            )
            > maximum_length_difference
        ):
            continue

        similarity = (
            calculate_glossary_similarity(
                text,
                source_term,
                case_sensitive=case_sensitive,
            )
        )

        if (
            similarity
            < minimum_similarity
        ):
            continue

        if (
            similarity
            <= best_similarity
        ):
            continue

        best_source = source_term
        best_similarity = similarity

    return (
        best_source,
        best_similarity,
    )


def assess_ocr_glossary_match(
    text: str,
    glossary_entries: Mapping[
        str,
        str,
    ],
    scoring_config: OcrScoringConfig,
) -> OcrGlossaryMatch:
    """
    OCR候補に対するGlossary照合結果を返す。

    完全一致した場合は類似一致を探索しない。
    """
    exact_source = (
        find_exact_glossary_source(
            text,
            glossary_entries,
        )
    )

    if exact_source is not None:
        normalized_source = (
            normalize_glossary_comparison_text(
                exact_source,
                case_sensitive=False,
            )
        )

        maximum_protection_length = int(
            scoring_config.get_value(
                "limits",
                "maximum_glossary_exact_protection",
            )
        )

        return OcrGlossaryMatch(
            exact_source=exact_source,
            exact_protection=(
                len(normalized_source)
                <= maximum_protection_length
            ),
            similar_source=None,
            similarity=1.0,
        )

    (
        similar_source,
        similarity,
    ) = find_similar_glossary_source(
        text,
        glossary_entries,
        scoring_config,
    )

    return OcrGlossaryMatch(
        exact_source=None,
        exact_protection=False,
        similar_source=similar_source,
        similarity=similarity,
    )
