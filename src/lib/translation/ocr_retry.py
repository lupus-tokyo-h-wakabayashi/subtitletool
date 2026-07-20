from __future__ import annotations

import ast
import re
from collections.abc import Mapping

from lib.profile.noise import (
    NoiseDictionary,
    find_suspicious_latin_sequences,
)
from lib.profile.ocr_scoring import (
    OcrScoringConfig,
)
from lib.subtitle.srt import SrtBlock
from lib.subtitle.text import (
    DEFAULT_ALLOWED_LATIN_TERMS,
    mask_chinese_ocr_text,
    mask_suspicious_latin_sequences,
)
from .ocr_assessment import (
    assess_ocr_source_line,
)

MIN_SYMBOL_DENSE_OCR_ASCII_LETTERS = 8
MIN_SYMBOL_DENSE_OCR_TOKENS = 4
MIN_SYMBOL_DENSE_OCR_SHORT_TOKENS = 3
MIN_SYMBOL_DENSE_OCR_STRUCTURAL_SYMBOLS = 3
MIN_SYMBOL_DENSE_OCR_SHORT_TOKEN_RATIO = 0.6

MIN_LOW_SYMBOL_WORD_SALAD_TOKENS = 6
MAX_LOW_SYMBOL_WORD_SALAD_SHORT_TOKEN_LENGTH = 4
MIN_LOW_SYMBOL_WORD_SALAD_SHORT_TOKEN_RATIO = 0.75
MIN_LOW_SYMBOL_WORD_SALAD_SUSPICIOUS_TOKENS = 3

MIN_SHORT_MIXED_CASE_OCR_TOKENS = 4
MAX_SHORT_MIXED_CASE_OCR_TOKENS = 5
MAX_SHORT_MIXED_CASE_OCR_TOKEN_LENGTH = 4
MIN_SHORT_MIXED_CASE_OCR_SHORT_TOKEN_RATIO = 1.0

SYMBOL_DENSE_OCR_STRUCTURAL_PATTERN = re.compile(
    r"[=()\[\]{}<>|~\\]"
)

LOW_SYMBOL_WORD_SALAD_LINE_PATTERN = re.compile(
    r"^[A-Za-z]+(?:[ '-][A-Za-z]+)*$"
)

LOW_SYMBOL_WORD_SALAD_VOWEL_RUN_PATTERN = re.compile(
    r"([aeiou])\1",
    re.IGNORECASE,
)

LOW_SYMBOL_WORD_SALAD_VOWEL_PATTERN = re.compile(
    r"[aeiou]",
    re.IGNORECASE,
)

SOUND_EFFECT_ONLY_PATTERN = re.compile(
    r"^(?:"
    r"\([A-Z0-9 ,.'’!?-]+\)"
    r"\s*"
    r")+$"
)


def extract_chinese_error_ids(
    errors: list[str],
) -> set[str]:
    """
    中国語混入エラーから対象字幕IDを抽出する。
    """
    subtitle_ids: set[str] = set()

    pattern = re.compile(
        r"subtitle_id=(?P<quote>['\"])"
        r"(?P<id>.+?)"
        r"(?P=quote)"
    )

    for error in errors:
        if not error.startswith(
            "Chinese-specific characters detected:"
        ):
            continue

        match = pattern.search(error)

        if not match:
            continue

        subtitle_ids.add(
            match.group("id")
        )

    return subtitle_ids


def extract_garbled_latin_errors(
    errors: list[str],
) -> dict[str, list[str]]:
    """
    OCR英字破損エラーから字幕IDと文字列を抽出する。
    """
    results: dict[str, list[str]] = {}

    pattern = re.compile(
        r"subtitle_id=(?P<quote>['\"])"
        r"(?P<id>.+?)"
        r"(?P=quote), "
        r"sequences=(?P<sequences>\[.*?\]), "
        r"text="
    )

    for error in errors:
        if not error.startswith(
            "Garbled Latin text detected:"
        ):
            continue

        match = pattern.search(error)

        if not match:
            continue

        raw_sequences = match.group(
            "sequences"
        )

        try:
            sequences = ast.literal_eval(
                raw_sequences
            )
        except (
                SyntaxError,
                ValueError,
        ):
            continue

        if not isinstance(sequences, list):
            continue

        if not all(
            isinstance(sequence, str)
            for sequence in sequences
        ):
            continue

        results[match.group("id")] = (
            sequences
        )

    return results


def extract_garbled_latin_candidates(
    errors: list[str],
) -> list[str]:
    """
    OCR英字破損エラーから、
    noise辞書へ保存する候補文字列を抽出する。
    """
    error_details = (
        extract_garbled_latin_errors(
            errors
        )
    )

    candidates: list[str] = []

    for sequences in error_details.values():
        for sequence in sequences:
            if sequence in candidates:
                continue

            candidates.append(
                sequence
            )

    return candidates


def extract_untranslated_english_error_ids(
    errors: list[str],
) -> set[str]:
    """
    未翻訳英文エラーから対象字幕IDを抽出する。
    """
    subtitle_ids: set[str] = set()

    pattern = re.compile(
        r"subtitle_id=(?P<quote>['\"])"
        r"(?P<id>.+?)"
        r"(?P=quote)"
    )

    for error in errors:
        if not error.startswith(
            "Untranslated English sentence detected:"
        ):
            continue

        match = pattern.search(
            error
        )

        if not match:
            continue

        subtitle_ids.add(
            match.group("id")
        )

    return subtitle_ids


def is_symbol_dense_ocr_source_line(
    source_line: str,
) -> bool:
    """
    原文1行が、短い英字トークンと構造記号が密集した
    OCR破損文字列か判定する。

    この関数は単独で翻訳前字幕を削除するためには使用しない。

    呼出元で次を確認した後の、
    高確度OCR判定として使用する。

    - 未翻訳英文エラーの対象字幕IDである
    - 原文の完全な1行がtranslationへ残っている

    正常な効果音、型番、識別子、短い数式などを
    誤検出しないよう、複数条件を同時に要求する。
    """
    normalized = source_line.strip()

    if not normalized:
        return False

    if SOUND_EFFECT_ONLY_PATTERN.fullmatch(
        normalized
    ):
        return False

    ascii_letters = re.findall(
        r"[A-Za-z]",
        normalized,
    )

    if (
        len(ascii_letters)
        < MIN_SYMBOL_DENSE_OCR_ASCII_LETTERS
    ):
        return False

    tokens = re.findall(
        r"[A-Za-z]+",
        normalized,
    )

    if (
        len(tokens)
        < MIN_SYMBOL_DENSE_OCR_TOKENS
    ):
        return False

    short_tokens = [
        token
        for token in tokens
        if len(token) <= 3
    ]

    if (
        len(short_tokens)
        < MIN_SYMBOL_DENSE_OCR_SHORT_TOKENS
    ):
        return False

    structural_symbols = (
        SYMBOL_DENSE_OCR_STRUCTURAL_PATTERN.findall(
            normalized
        )
    )

    if (
        len(structural_symbols)
        < MIN_SYMBOL_DENSE_OCR_STRUCTURAL_SYMBOLS
    ):
        return False

    short_token_ratio = (
        len(short_tokens)
        / len(tokens)
    )

    if (
        short_token_ratio
        < MIN_SYMBOL_DENSE_OCR_SHORT_TOKEN_RATIO
    ):
        return False

    return True


def is_short_mixed_case_ocr_source_line(
    source_line: str,
) -> bool:
    """
    原文1行が、短い英字トークンで構成された
    大小文字混在型のOCR破損文字列か判定する。

    対象例:

        dam IAN el ESie

    既存の低記号ワードサラダ判定は、
    正常英文の誤検出を避けるため
    6トークン以上を要求している。

    この関数は4〜5トークンの短い行を対象にし、
    次の条件をすべて要求する。

    - 英字中心の1行である
    - 4〜5トークンである
    - すべてのトークンが短い
    - 2文字以上の全大文字トークンがある
    - 通常のlower/title/upperに該当しない
      不規則な大小文字トークンがある

    この関数だけで字幕を削除またはマスクしない。

    Hybrid Recoveryまたは通常翻訳の再試行で、
    未翻訳エラーになった複数行字幕の
    行分類へ使用する。
    """
    normalized = source_line.strip()

    if not normalized:
        return False

    if SOUND_EFFECT_ONLY_PATTERN.fullmatch(
        normalized
    ):
        return False

    if not LOW_SYMBOL_WORD_SALAD_LINE_PATTERN.fullmatch(
        normalized
    ):
        return False

    tokens = re.findall(
        r"[A-Za-z]+",
        normalized,
    )

    if not (
        MIN_SHORT_MIXED_CASE_OCR_TOKENS
        <= len(tokens)
        <= MAX_SHORT_MIXED_CASE_OCR_TOKENS
    ):
        return False

    short_tokens = [
        token
        for token in tokens
        if (
            len(token)
            <= MAX_SHORT_MIXED_CASE_OCR_TOKEN_LENGTH
        )
    ]

    short_token_ratio = (
        len(short_tokens)
        / len(tokens)
    )

    if (
        short_token_ratio
        < MIN_SHORT_MIXED_CASE_OCR_SHORT_TOKEN_RATIO
    ):
        return False

    uppercase_tokens = [
        token
        for token in tokens
        if (
            len(token) >= 2
            and token.isupper()
        )
    ]

    if not uppercase_tokens:
        return False

    irregular_mixed_case_tokens = [
        token
        for token in tokens
        if (
            len(token) >= 3
            and not token.islower()
            and not token.isupper()
            and not token.istitle()
        )
    ]

    if not irregular_mixed_case_tokens:
        return False

    return True


def is_low_symbol_word_salad_ocr_source_line(
    source_line: str,
) -> bool:
    """
    原文1行が、記号をほとんど含まない短い英字トークンの
    連続によるOCR破損文字列か判定する。

    記号密度を利用する既存判定では検出できない、
    次のような英字ワードサラダを対象とする。

        Ui maar i mele aah ml iaa

    正常な短文や固有名詞を誤検出しないよう、
    次の条件をすべて要求する。

    - 英字中心の1行である
    - 一定数以上のトークンがある
    - 短いトークンの割合が高い
    - 不自然なトークンが複数ある

    この関数は単独で字幕を削除またはマスクしない。
    Hybrid Recoveryで失敗した字幕の行を分類するための
    補助判定として使用する。
    """
    normalized = source_line.strip()

    if not normalized:
        return False

    if SOUND_EFFECT_ONLY_PATTERN.fullmatch(
        normalized
    ):
        return False

    if not LOW_SYMBOL_WORD_SALAD_LINE_PATTERN.fullmatch(
        normalized
    ):
        return False

    tokens = re.findall(
        r"[A-Za-z]+",
        normalized,
    )

    if (
        len(tokens)
        < MIN_LOW_SYMBOL_WORD_SALAD_TOKENS
    ):
        return False

    short_tokens = [
        token
        for token in tokens
        if (
            len(token)
            <= MAX_LOW_SYMBOL_WORD_SALAD_SHORT_TOKEN_LENGTH
        )
    ]

    short_token_ratio = (
        len(short_tokens)
        / len(tokens)
    )

    if (
        short_token_ratio
        < MIN_LOW_SYMBOL_WORD_SALAD_SHORT_TOKEN_RATIO
    ):
        return False

    suspicious_tokens: list[str] = []

    for token in tokens:
        normalized_token = token.lower()

        is_invalid_single_letter = (
            len(token) == 1
            and token not in {
                "a",
                "A",
                "I",
            }
        )

        has_no_vowel = (
            len(token) >= 2
            and not LOW_SYMBOL_WORD_SALAD_VOWEL_PATTERN.search(
            token
        )
        )

        has_long_vowel_run = (
            len(token) >= 3
            and LOW_SYMBOL_WORD_SALAD_VOWEL_RUN_PATTERN.search(
            normalized_token
        )
            is not None
        )

        if not (
            is_invalid_single_letter
            or has_no_vowel
            or has_long_vowel_run
        ):
            continue

        suspicious_tokens.append(
            token
        )

    if (
        len(suspicious_tokens)
        < MIN_LOW_SYMBOL_WORD_SALAD_SUSPICIOUS_TOKENS
    ):
        return False

    return True


def is_probable_ocr_source_line(
    source_line: str,
    noise_dictionary: NoiseDictionary,
) -> bool:
    """
    原文1行が高確度のOCR英字破損か判定する。

    次の順序で判定する。

    1. Noise辞書または既存ヒューリスティック
    2. 大文字トークンと不自然な1文字トークン
    3. 短い英字トークンと構造記号の密集

    正常な英文を誤ってOCR扱いしないため、
    単一の弱い条件だけではTrueにしない。
    """
    normalized = source_line.strip()

    if not normalized:
        return False

    existing_sequences = (
        find_suspicious_latin_sequences(
            normalized,
            noise_dictionary,
            allowed_terms=(
                DEFAULT_ALLOWED_LATIN_TERMS
            ),
        )
    )

    if existing_sequences:
        return True

    tokens = re.findall(
        r"[A-Za-z]+",
        normalized,
    )

    if len(tokens) >= 3:
        abnormal_single_letter_tokens = [
            token
            for token in tokens
            if (
                len(token) == 1
                and token.upper() not in {
                    "A",
                    "I",
                }
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

        has_mixed_case_pattern = any(
            (
                len(token) >= 3
                and token[0].isupper()
                and token[1:].islower()
            )
            for token in tokens
        )

        if (
            abnormal_single_letter_tokens
            and uppercase_tokens
            and has_mixed_case_pattern
        ):
            return True

    return is_symbol_dense_ocr_source_line(
        normalized
    )


def find_short_mixed_case_ocr_lines_in_source(
    source_text: str,
    noise_dictionary: NoiseDictionary,
) -> list[str]:
    """
    複数行の字幕原文から、
    正常行と混在する短い大小文字型OCR行を抽出する。

    次の条件をすべて満たす行だけを返す。

    - 空でない非効果音行が2行以上ある
    - 対象行が短い大小文字型OCR判定を通る
    - 同じ字幕内に正常と判断できる別行がある

    単独の短文や、すべての行がOCR候補になる字幕は
    誤検出防止のため対象にしない。

    この関数は字幕IDやValidation結果を判断しない。
    呼出側で失敗した字幕だけに適用する。
    """
    source_lines = [
        raw_line.strip()
        for raw_line in source_text.splitlines()
        if (
            raw_line.strip()
            and not SOUND_EFFECT_ONLY_PATTERN.fullmatch(
            raw_line.strip()
        )
        )
    ]

    if len(source_lines) < 2:
        return []

    results: list[str] = []

    for position, source_line in enumerate(
        source_lines
    ):
        if not is_short_mixed_case_ocr_source_line(
            source_line
        ):
            continue

        has_normal_sibling_line = any(
            (
                sibling_position
                != position
                and not is_probable_ocr_source_line(
                sibling_line,
                noise_dictionary,
            )
                and not (
                is_low_symbol_word_salad_ocr_source_line(
                    sibling_line
                )
            )
                and not (
                is_short_mixed_case_ocr_source_line(
                    sibling_line
                )
            )
            )
            for (
                sibling_position,
                sibling_line,
            ) in enumerate(
                source_lines
            )
        )

        if not has_normal_sibling_line:
            continue

        if source_line in results:
            continue

        results.append(
            source_line
        )

    return results


def find_assessed_ocr_lines_in_source(
    source_text: str,
    glossary_entries: Mapping[
        str,
        str,
    ],
    scoring_config: OcrScoringConfig,
    *,
    validation_failed: bool,
) -> list[str]:
    """
    字幕原文を行単位で統合OCR評価し、
    OCR破損候補だけを原文順で返す。

    Validation失敗済みの場合は、
    同じ字幕に正常行があるかを判定し、
    状況に応じた閾値を使用する。

    効果音だけの行はOCR候補にしない。
    """
    source_lines = [
        raw_line.strip()
        for raw_line in (
            source_text.splitlines()
        )
        if raw_line.strip()
    ]

    if not source_lines:
        return []

    base_assessments = [
        assess_ocr_source_line(
            source_line,
            glossary_entries,
            scoring_config,
            validation_failed=(
                validation_failed
            ),
            has_normal_sibling=False,
        )
        for source_line in source_lines
    ]

    results: list[str] = []

    for position, source_line in enumerate(
        source_lines
    ):
        if SOUND_EFFECT_ONLY_PATTERN.fullmatch(
            source_line
        ):
            continue

        has_normal_sibling = any(
            (
                sibling_position
                != position
                and not (
                sibling_assessment
                .probable_ocr
            )
            )
            for (
                sibling_position,
                sibling_assessment,
            ) in enumerate(
                base_assessments
            )
        )

        assessment = (
            assess_ocr_source_line(
                source_line,
                glossary_entries,
                scoring_config,
                validation_failed=(
                    validation_failed
                ),
                has_normal_sibling=(
                    has_normal_sibling
                ),
            )
        )

        if not assessment.probable_ocr:
            continue

        if source_line in results:
            continue

        results.append(
            source_line
        )

    return results


def find_probable_untranslated_ocr_lines(
    target_blocks: list[SrtBlock],
    translated_texts: list[str],
    errors: list[str],
    noise_dictionary: NoiseDictionary,
    *,
    glossary_entries: Mapping[
        str,
        str,
    ],
    scoring_config: OcrScoringConfig,
) -> dict[str, list[str]]:
    """
    未翻訳英文エラーになった字幕から、
    translationへ残ったOCR破損原文行を返す。

    OCR候補の判定には、
    Glossary対応の統合OCR評価器を使用する。

    noise_dictionaryは呼出API移行中の
    互換引数として保持する。
    """
    error_ids = (
        extract_untranslated_english_error_ids(
            errors
        )
    )

    if not error_ids:
        return {}

    probable_lines: dict[
        str,
        list[str],
    ] = {}

    for block, translated_text in zip(
        target_blocks,
        translated_texts,
        strict=True,
    ):
        if block.number not in error_ids:
            continue

        assessed_lines = (
            find_assessed_ocr_lines_in_source(
                block.text,
                glossary_entries,
                scoring_config,
                validation_failed=True,
            )
        )

        matched_lines: list[str] = []

        for source_line in assessed_lines:
            if (
                source_line
                not in translated_text
            ):
                continue

            if source_line in matched_lines:
                continue

            matched_lines.append(
                source_line
            )

        if not matched_lines:
            continue

        probable_lines[
            block.number
        ] = matched_lines

    return probable_lines


def apply_level_1_ocr_fallback(
    target_blocks: list[SrtBlock],
    translated_texts: list[str],
    probable_ocr_lines: dict[str, list[str]],
) -> tuple[
    list[str],
    dict[str, list[str]],
]:
    """
    最終再試行後も高確度OCR文字列が残った場合に、
    原文の完全な1行だけを[1]タグで囲む。

    既に[3]または[5]で囲まれている場合は、
    タグをネストせず、タグ全体を[1]へ置換する。

    正常英文や判定不能な文字列には適用しない。

    戻り値:
        corrected_texts:
            [1]タグを適用した翻訳一覧

        applied_lines:
            実際にタグを適用した字幕IDと原文行
    """
    corrected_texts: list[str] = []
    applied_lines: dict[
        str,
        list[str],
    ] = {}

    for block, translated_text in zip(
        target_blocks,
        translated_texts,
        strict=True,
    ):
        corrected_text = translated_text
        applied_for_block: list[str] = []

        for source_line in probable_ocr_lines.get(
            block.number,
            [],
        ):
            level_1_text = (
                f"[1]{source_line}[/1]"
            )

            if level_1_text in corrected_text:
                continue

            replaced_existing_tag = False

            for existing_level in (
                    "3",
                    "5",
            ):
                existing_tag = (
                    f"[{existing_level}]"
                    f"{source_line}"
                    f"[/{existing_level}]"
                )

                if existing_tag not in corrected_text:
                    continue

                corrected_text = (
                    corrected_text.replace(
                        existing_tag,
                        level_1_text,
                        1,
                    )
                )

                replaced_existing_tag = True
                break

            if replaced_existing_tag:
                applied_for_block.append(
                    source_line
                )
                continue

            if source_line not in corrected_text:
                continue

            corrected_text = (
                corrected_text.replace(
                    source_line,
                    level_1_text,
                    1,
                )
            )

            applied_for_block.append(
                source_line
            )

        corrected_texts.append(
            corrected_text
        )

        if applied_for_block:
            applied_lines[
                block.number
            ] = applied_for_block

    return (
        corrected_texts,
        applied_lines,
    )


def build_chinese_retry_blocks(
    target_blocks: list[SrtBlock],
    errors: list[str],
) -> list[SrtBlock]:
    """
    中国語混入エラーが出た字幕だけ、
    再試行用入力の中国語OCR文字列をマスクする。
    """
    error_ids = extract_chinese_error_ids(
        errors
    )

    if not error_ids:
        return target_blocks

    return [
        SrtBlock(
            number=block.number,
            timestamp=block.timestamp,
            text=(
                mask_chinese_ocr_text(block.text)
                if block.number in error_ids
                else block.text
            ),
        )
        for block in target_blocks
    ]


def mask_chinese_translation_errors(
    target_blocks: list[SrtBlock],
    translated_texts: list[str],
    errors: list[str],
) -> list[str]:
    """
    中国語混入エラーが出た翻訳結果だけ、
    中国語固有文字を判読不能表記へ置換する。
    """
    error_ids = extract_chinese_error_ids(
        errors
    )

    if not error_ids:
        return translated_texts

    return [
        (
            mask_chinese_ocr_text(
                translation
            )
            if block.number in error_ids
            else translation
        )
        for block, translation in zip(
            target_blocks,
            translated_texts,
            strict=True,
        )
    ]


def build_latin_ocr_retry_blocks(
    target_blocks: list[SrtBlock],
    errors: list[str],
) -> list[SrtBlock]:
    """
    OCR英字破損が出た字幕だけ、
    該当文字列を再試行入力でマスクする。
    """
    error_details = (
        extract_garbled_latin_errors(
            errors
        )
    )

    if not error_details:
        return target_blocks

    return [
        SrtBlock(
            number=block.number,
            timestamp=block.timestamp,
            text=(
                mask_suspicious_latin_sequences(
                    block.text,
                    sequences=error_details[
                        block.number
                    ],
                )
                if block.number in error_details
                else block.text
            ),
        )
        for block in target_blocks
    ]
