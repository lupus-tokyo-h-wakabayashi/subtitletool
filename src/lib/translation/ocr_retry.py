from __future__ import annotations

import ast
import re
from collections.abc import Mapping

from lib.profile.noise import (
    NoiseDictionary,
)
from lib.profile.ocr_scoring import (
    OcrScoringConfig,
)
from lib.subtitle.srt import SrtBlock
from lib.subtitle.text import (
    mask_chinese_ocr_text,
    mask_suspicious_latin_sequences,
)
from .ocr_assessment import (
    assess_ocr_source_line,
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
