from __future__ import annotations

import ast
import re

from lib.srt import SrtBlock
from lib.text import (
    mask_chinese_ocr_text,
    mask_suspicious_latin_sequences,
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
