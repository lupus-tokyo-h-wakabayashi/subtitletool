from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from lib.profile.ocr_scoring import (
    OcrScoringConfig,
)
from lib.translation.ocr_glossary import (
    OcrGlossaryMatch,
    assess_ocr_glossary_match,
)

ASCII_TOKEN_PATTERN = re.compile(
    r"[A-Za-z]+"
    r"(?:['’][A-Za-z]+)*"
)

TIME_PATTERN = re.compile(
    r"(?<!\d)"
    r"(?:[01]?\d|2[0-3])"
    r":[0-5]\d"
    r"(?!\d)"
)

STRUCTURAL_SYMBOLS = frozenset(
    "=()[]{}<>|~\\"
)

STRONG_CORRUPTION_SYMBOLS = frozenset(
    "|~\\"
)

VOWELS = frozenset(
    "aeiou"
)

DELIMITER_PAIRS = (
    (
        "parentheses",
        "(",
        ")",
        "unbalanced_parentheses",
    ),
    (
        "square_brackets",
        "[",
        "]",
        "unbalanced_square_brackets",
    ),
    (
        "braces",
        "{",
        "}",
        "unbalanced_braces",
    ),
    (
        "angle_brackets",
        "<",
        ">",
        "unbalanced_angle_brackets",
    ),
)


@dataclass(frozen=True)
class OcrScoreContribution:
    """
    OCRスコアへ適用した1項目。

    name:
        config/ocr-scoring.jsonのweights項目名。

    value:
        実際に加算または減算した値。

    description:
        設定ファイルに記載された用途。
    """

    name: str
    value: int
    description: str


@dataclass(frozen=True)
class OcrSourceAssessment:
    """
    字幕原文1行のOCR評価結果。

    damage_score:
        保護用の負数を適用する前に得られた
        OCR破損方向の正数スコア。

    score:
        glossaryと正常文字列保護を含む最終スコア。

    threshold:
        呼出時のValidation状態に応じて
        選択されたOCR採用閾値。

    probable_ocr:
        scoreがthreshold以上の場合にTrue。
    """

    text: str
    damage_score: int
    score: int
    threshold: int
    probable_ocr: bool
    contributions: tuple[
        OcrScoreContribution,
        ...,
    ]
    glossary_match: OcrGlossaryMatch

    def has_contribution(
        self,
        name: str,
    ) -> bool:
        return any(
            contribution.name == name
            for contribution in self.contributions
        )


def get_integer_value(
    scoring_config: OcrScoringConfig,
    group_name: str,
    item_name: str,
) -> int:
    value = scoring_config.get_value(
        group_name,
        item_name,
    )

    if type(value) is not int:
        raise TypeError(
            "OCR scoring value must be an integer: "
            f"group={group_name!r}, "
            f"item={item_name!r}, "
            f"value={value!r}"
        )

    return value


def get_float_value(
    scoring_config: OcrScoringConfig,
    group_name: str,
    item_name: str,
) -> float:
    value = scoring_config.get_value(
        group_name,
        item_name,
    )

    if type(value) not in {
        int,
        float,
    }:
        raise TypeError(
            "OCR scoring value must be numeric: "
            f"group={group_name!r}, "
            f"item={item_name!r}, "
            f"value={value!r}"
        )

    return float(
        value
    )


def select_ocr_threshold(
    scoring_config: OcrScoringConfig,
    *,
    validation_failed: bool,
    has_normal_sibling: bool,
) -> int:
    """
    OCR判定へ使用する閾値を選択する。

    通常時:
        high_confidence

    Validation失敗時:
        failed_subtitle

    Validation失敗かつ同じ字幕に正常行がある場合:
        failed_with_normal_sibling
    """
    if (
        validation_failed
        and has_normal_sibling
    ):
        threshold_name = (
            "failed_with_normal_sibling"
        )
    elif validation_failed:
        threshold_name = (
            "failed_subtitle"
        )
    else:
        threshold_name = (
            "high_confidence"
        )

    return get_integer_value(
        scoring_config,
        "thresholds",
        threshold_name,
    )


def extract_ascii_tokens(
    text: str,
) -> list[str]:
    return ASCII_TOKEN_PATTERN.findall(
        text
    )


def count_ascii_letters(
    text: str,
) -> int:
    return sum(
        character.isascii()
        and character.isalpha()
        for character in text
    )


def count_structural_symbols(
    text: str,
) -> int:
    return sum(
        character in STRUCTURAL_SYMBOLS
        for character in text
    )


def find_unbalanced_delimiters(
    text: str,
) -> tuple[
    tuple[str, str],
    ...,
]:
    """
    不均衡な区切り記号と対応するweight名を返す。
    """
    results: list[
        tuple[str, str]
    ] = []

    for (
            delimiter_name,
            opening,
            closing,
            weight_name,
    ) in DELIMITER_PAIRS:
        if (
            text.count(opening)
            == text.count(closing)
        ):
            continue

        results.append(
            (
                delimiter_name,
                weight_name,
            )
        )

    if (
        text.count('"')
        % 2
        != 0
    ):
        results.append(
            (
                "double_quotes",
                "unbalanced_double_quotes",
            )
        )

    return tuple(
        results
    )


def token_has_repeated_vowel(
    token: str,
) -> bool:
    normalized = token.casefold()

    return any(
        first == second
        and first in VOWELS
        for first, second in zip(
            normalized,
            normalized[1:],
        )
    )


def is_invalid_single_letter_token(
    token: str,
) -> bool:
    return (
        len(token) == 1
        and token.casefold()
        not in {
            "a",
            "i",
        }
    )


def is_vowelless_token(
    token: str,
) -> bool:
    return (
        len(token) >= 2
        and not any(
        character in VOWELS
        for character in (
            token.casefold()
        )
    )
    )


def is_vowelless_uppercase_token(
    token: str,
) -> bool:
    return (
        len(token) >= 2
        and token.isupper()
        and is_vowelless_token(
        token
    )
    )


def is_irregular_mixed_case_token(
    token: str,
) -> bool:
    return (
        len(token) >= 3
        and not token.islower()
        and not token.isupper()
        and not token.istitle()
    )


def is_suspicious_token(
    token: str,
) -> bool:
    return (
        is_invalid_single_letter_token(
            token
        )
        or is_vowelless_token(
        token
    )
        or token_has_repeated_vowel(
        token
    )
        or is_irregular_mixed_case_token(
        token
    )
    )


def is_identifier_like(
    text: str,
    *,
    has_letters: bool,
    has_digits: bool,
    unbalanced_delimiters: tuple[
        tuple[str, str],
        ...,
    ],
) -> bool:
    """
    正常な型番や識別子に近い文字列を判定する。

    空白を含む文章や括弧不均衡文字列は
    保護対象にしない。
    """
    normalized = text.strip()

    if not normalized:
        return False

    if not (
        has_letters
        and has_digits
    ):
        return False

    if any(
        character.isspace()
        for character in normalized
    ):
        return False

    if unbalanced_delimiters:
        return False

    return all(
        character.isalnum()
        or character in {
            "-",
            "_",
            ".",
            "/",
        }
        for character in normalized
    )


def is_equation_like(
    text: str,
    *,
    unbalanced_delimiters: tuple[
        tuple[str, str],
        ...,
    ],
) -> bool:
    """
    単純な数式に近い正常文字列を判定する。

    不均衡な括弧を含む場合は保護しない。
    """
    normalized = text.strip()

    if (
        not normalized
        or "=" not in normalized
        or unbalanced_delimiters
    ):
        return False

    if any(
        character.isspace()
        for character in normalized
    ):
        return False

    left, right = normalized.split(
        "=",
        maxsplit=1,
    )

    if not left or not right:
        return False

    allowed_symbols = {
        "+",
        "-",
        "*",
        "/",
        "^",
        ".",
        "(",
        ")",
        "=",
    }

    return all(
        character.isalnum()
        or character in allowed_symbols
        for character in normalized
    )


def is_time_like(
    text: str,
) -> bool:
    return (
        TIME_PATTERN.search(
            text
        )
        is not None
    )


def is_natural_sentence(
    text: str,
    tokens: list[str],
    scoring_config: OcrScoringConfig,
) -> bool:
    """
    一定数の自然な英語トークンを持つ英文を
    正常英文として保護する。

    字幕は文の途中で行分割されるため、
    文末記号を持つ完成文だけでなく、
    不自然なトークンや構造記号を含まない
    文中断片も保護対象にする。
    """
    minimum_tokens = get_integer_value(
        scoring_config,
        "limits",
        "minimum_natural_sentence_tokens",
    )

    if (
        len(tokens)
        < minimum_tokens
    ):
        return False

    normalized = text.strip().rstrip(
        "\"')]}〉》」』）】"
    )

    has_natural_ending = (
        normalized.endswith(
            (
                ".",
                "?",
                "!",
                "…",
                ",",
                ";",
                ":",
            )
        )
    )

    if has_natural_ending:
        return True

    if (
        count_structural_symbols(
            normalized
        )
        > 0
    ):
        return False

    if any(
        is_suspicious_token(
            token
        )
        for token in tokens
    ):
        return False

    return all(
        (
            token.islower()
            or token.istitle()
            or token.casefold()
            in {
                "a",
                "i",
            }
        )
        for token in tokens
    )


def assess_ocr_source_line(
    text: str,
    glossary_entries: Mapping[
        str,
        str,
    ],
    scoring_config: OcrScoringConfig,
    *,
    validation_failed: bool = False,
    has_normal_sibling: bool = False,
) -> OcrSourceAssessment:
    """
    字幕原文1行を汎用的な特徴量で評価する。

    エピソード固有文字列やNoise辞書は使用しない。
    """
    normalized = text.strip()

    glossary_match = (
        assess_ocr_glossary_match(
            normalized,
            glossary_entries,
            scoring_config,
        )
    )

    threshold = select_ocr_threshold(
        scoring_config,
        validation_failed=validation_failed,
        has_normal_sibling=(
            has_normal_sibling
        ),
    )

    contributions: list[
        OcrScoreContribution
    ] = []

    score = 0

    def apply_weight(
        weight_name: str,
    ) -> None:
        nonlocal score

        item = scoring_config.weights[
            weight_name
        ]

        value = get_integer_value(
            scoring_config,
            "weights",
            weight_name,
        )

        score += value

        contributions.append(
            OcrScoreContribution(
                name=weight_name,
                value=value,
                description=(
                    item.description
                ),
            )
        )

    if not normalized:
        return OcrSourceAssessment(
            text=text,
            damage_score=0,
            score=0,
            threshold=threshold,
            probable_ocr=False,
            contributions=(),
            glossary_match=glossary_match,
        )

    tokens = extract_ascii_tokens(
        normalized
    )

    ascii_letter_count = (
        count_ascii_letters(
            normalized
        )
    )

    has_letters = (
        ascii_letter_count
        > 0
    )

    has_digits = any(
        character.isdigit()
        for character in normalized
    )

    maximum_short_line_length = (
        get_integer_value(
            scoring_config,
            "limits",
            "maximum_short_line_length",
        )
    )

    minimum_ascii_letters = (
        get_integer_value(
            scoring_config,
            "limits",
            "minimum_ascii_letters",
        )
    )

    minimum_tokens = get_integer_value(
        scoring_config,
        "limits",
        "minimum_tokens",
    )

    minimum_many_tokens = (
        get_integer_value(
            scoring_config,
            "limits",
            "minimum_many_tokens",
        )
    )

    maximum_short_token_length = (
        get_integer_value(
            scoring_config,
            "limits",
            "maximum_short_token_length",
        )
    )

    minimum_short_token_ratio = (
        get_float_value(
            scoring_config,
            "limits",
            "minimum_short_token_ratio",
        )
    )

    minimum_high_short_token_ratio = (
        get_float_value(
            scoring_config,
            "limits",
            "minimum_high_short_token_ratio",
        )
    )

    minimum_suspicious_tokens = (
        get_integer_value(
            scoring_config,
            "limits",
            "minimum_suspicious_tokens",
        )
    )

    minimum_structural_symbols = (
        get_integer_value(
            scoring_config,
            "limits",
            "minimum_structural_symbols",
        )
    )

    minimum_dense_structural_symbols = (
        get_integer_value(
            scoring_config,
            "limits",
            "minimum_dense_structural_symbols",
        )
    )

    structural_symbol_count = (
        count_structural_symbols(
            normalized
        )
    )

    short_tokens = [
        token
        for token in tokens
        if (
            len(token)
            <= maximum_short_token_length
        )
    ]

    short_token_ratio = (
        len(short_tokens)
        / len(tokens)
        if tokens
        else 0.0
    )

    suspicious_tokens = [
        token
        for token in tokens
        if is_suspicious_token(
            token
        )
    ]

    invalid_single_letter_tokens = [
        token
        for token in tokens
        if is_invalid_single_letter_token(
            token
        )
    ]

    uppercase_tokens = [
        token
        for token in tokens
        if (
            len(token) >= 2
            and token.isupper()
        )
    ]

    vowelless_uppercase_tokens = [
        token
        for token in tokens
        if is_vowelless_uppercase_token(
            token
        )
    ]

    irregular_mixed_case_tokens = [
        token
        for token in tokens
        if is_irregular_mixed_case_token(
            token
        )
    ]

    unbalanced_delimiters = (
        find_unbalanced_delimiters(
            normalized
        )
    )

    if (
        len(normalized)
        <= maximum_short_line_length
    ):
        apply_weight(
            "short_line"
        )

    if (
        has_letters
        and has_digits
    ):
        apply_weight(
            "letters_and_digits"
        )

    if (
        ascii_letter_count
        >= minimum_ascii_letters
    ):
        apply_weight(
            "minimum_ascii_letters"
        )

    if (
        len(tokens)
        >= minimum_tokens
    ):
        apply_weight(
            "minimum_tokens"
        )

    if (
        len(tokens)
        >= minimum_many_tokens
    ):
        apply_weight(
            "many_tokens"
        )

    if (
        structural_symbol_count
        >= minimum_structural_symbols
    ):
        apply_weight(
            "multiple_structural_symbols"
        )

    if (
        structural_symbol_count
        >= minimum_dense_structural_symbols
    ):
        apply_weight(
            "dense_structural_symbols"
        )

    if any(
        character
        in STRONG_CORRUPTION_SYMBOLS
        for character in normalized
    ):
        apply_weight(
            "strong_corruption_symbol"
        )

    for (
            _,
            weight_name,
    ) in unbalanced_delimiters:
        apply_weight(
            weight_name
        )

    if (
        len(unbalanced_delimiters)
        >= 2
    ):
        apply_weight(
            "multiple_unbalanced_delimiters"
        )

    if (
        tokens
        and short_token_ratio
        >= minimum_short_token_ratio
    ):
        apply_weight(
            "short_token_ratio"
        )

    if (
        tokens
        and short_token_ratio
        >= minimum_high_short_token_ratio
    ):
        apply_weight(
            "high_short_token_ratio"
        )

    if (
        len(suspicious_tokens)
        >= minimum_suspicious_tokens
    ):
        apply_weight(
            "suspicious_tokens"
        )

    if invalid_single_letter_tokens:
        apply_weight(
            "invalid_single_letter"
        )

    if vowelless_uppercase_tokens:
        apply_weight(
            "vowelless_uppercase_token"
        )

    if irregular_mixed_case_tokens:
        apply_weight(
            "irregular_mixed_case"
        )

    symbol_dense_structure = (
        ascii_letter_count
        >= minimum_ascii_letters
        and len(tokens)
        >= minimum_tokens
        and structural_symbol_count
        >= minimum_dense_structural_symbols
        and short_token_ratio
        >= minimum_short_token_ratio
    )

    if symbol_dense_structure:
        apply_weight(
            "symbol_dense_structure"
        )

    low_symbol_word_salad = (
        len(tokens)
        >= minimum_many_tokens
        and structural_symbol_count
        < minimum_structural_symbols
        and short_token_ratio
        >= minimum_high_short_token_ratio
        and len(suspicious_tokens)
        >= minimum_suspicious_tokens
    )

    if low_symbol_word_salad:
        apply_weight(
            "low_symbol_word_salad"
        )

    short_mixed_case = (
        minimum_tokens
        <= len(tokens)
        < minimum_many_tokens
        and short_token_ratio
        >= minimum_high_short_token_ratio
        and bool(
        uppercase_tokens
    )
        and bool(
        irregular_mixed_case_tokens
    )
    )

    if short_mixed_case:
        apply_weight(
            "short_mixed_case"
        )

    damaged_alphanumeric_structure = (
        has_letters
        and has_digits
        and bool(
        unbalanced_delimiters
    )
    )

    if damaged_alphanumeric_structure:
        apply_weight(
            "damaged_alphanumeric_structure"
        )

    damage_score = score

    minimum_damage_score = (
        get_integer_value(
            scoring_config,
            "limits",
            "minimum_damage_score_for_glossary_miss",
        )
    )

    if glossary_match.exact_protection:
        apply_weight(
            "glossary_exact_match"
        )
    elif glossary_match.has_similar_match:
        apply_weight(
            "glossary_similar_match"
        )
    elif (
        damage_score
        >= minimum_damage_score
    ):
        apply_weight(
            "damaged_without_glossary_match"
        )

    if is_identifier_like(
        normalized,
        has_letters=has_letters,
        has_digits=has_digits,
        unbalanced_delimiters=(
            unbalanced_delimiters
        ),
    ):
        apply_weight(
            "identifier_like"
        )

    if is_equation_like(
        normalized,
        unbalanced_delimiters=(
            unbalanced_delimiters
        ),
    ):
        apply_weight(
            "equation_like"
        )

    if is_time_like(
        normalized
    ):
        apply_weight(
            "time_like"
        )

    if is_natural_sentence(
        normalized,
        tokens,
        scoring_config,
    ):
        apply_weight(
            "natural_sentence"
        )

    return OcrSourceAssessment(
        text=text,
        damage_score=damage_score,
        score=score,
        threshold=threshold,
        probable_ocr=(
            score
            >= threshold
        ),
        contributions=tuple(
            contributions
        ),
        glossary_match=glossary_match,
    )
