from __future__ import annotations

import ast
import re

from lib.profile.noise import (
    NoiseDictionary,
    find_suspicious_latin_sequences,
)
from lib.subtitle.srt import SrtBlock
from lib.subtitle.text import (
    DEFAULT_ALLOWED_LATIN_TERMS,
    mask_chinese_ocr_text,
    mask_suspicious_latin_sequences,
)

MIN_SYMBOL_DENSE_OCR_ASCII_LETTERS = 8
MIN_SYMBOL_DENSE_OCR_TOKENS = 4
MIN_SYMBOL_DENSE_OCR_SHORT_TOKENS = 3
MIN_SYMBOL_DENSE_OCR_STRUCTURAL_SYMBOLS = 3
MIN_SYMBOL_DENSE_OCR_SHORT_TOKEN_RATIO = 0.6

SYMBOL_DENSE_OCR_STRUCTURAL_PATTERN = re.compile(
    r"[=()\[\]{}<>|~]"
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


def find_probable_untranslated_ocr_lines(
    target_blocks: list[SrtBlock],
    translated_texts: list[str],
    errors: list[str],
    noise_dictionary: NoiseDictionary,
) -> dict[str, list[str]]:
    """
    未翻訳英文エラーになった字幕について、
    translationへそのままコピーされた原文行のうち、
    OCR破損の可能性が高い行を返す。

    戻り値:
        {
            "80": [
                "AV Cag are T",
            ],
        }
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

        matched_lines: list[str] = []

        for raw_line in block.text.splitlines():
            source_line = raw_line.strip()

            if not source_line:
                continue

            if source_line not in translated_text:
                continue

            if not is_probable_ocr_source_line(
                source_line,
                noise_dictionary,
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
