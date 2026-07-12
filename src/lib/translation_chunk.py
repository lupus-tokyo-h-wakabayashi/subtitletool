from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from lib.noise import (
    NoiseDictionary,
    append_noise_candidates,
    find_suspicious_latin_sequences,
)
from lib.ocr_retry import (
    build_chinese_retry_blocks,
    build_latin_ocr_retry_blocks,
    extract_garbled_latin_candidates,
    mask_chinese_translation_errors,
)
from lib.ollama import generate
from lib.retry import (
    build_chinese_retry_instruction,
    build_glossary_retry_instruction,
    build_latin_ocr_retry_instruction,
    build_preserved_translations_instruction,
    build_required_glossary_instruction,
    build_retry_instruction,
    build_structural_retry_instruction,
    build_untranslated_english_retry_instruction,
    has_structural_validation_error,
)
from lib.srt import SrtBlock
from lib.text import (
    find_suspicious_latin_sequences,
)
from lib.translation_output import (
    print_saved_noise_candidates,
)
from lib.translation_prompt import (
    build_prompt,
)
from lib.translation_validation import (
    validate_translation_response,
)

MAX_TRANSLATION_ATTEMPTS = 3

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

TRANSLATION_DEBUG_DIR = (
    PROJECT_ROOT
    / "tmp"
    / "subtitletool"
)


def normalize_translation_text(
    text: str,
) -> str:
    """
    翻訳文内の字幕改行記号を空白へ変換する。

    半角スラッシュは前後に空白がある場合だけ対象とし、
    24/7、km/h、URLなどは維持する。
    """
    normalized = re.sub(
        r"(?:\s+/\s+|\s*／\s*)",
        " ",
        text,
    )

    normalized = re.sub(
        r"[ \t]+",
        " ",
        normalized,
    )

    return normalized.strip()


def normalize_translation_texts(
    translated_texts: list[str],
) -> list[str]:
    """
    翻訳済み字幕をSRT保存用に一括正規化する。
    """
    return [
        normalize_translation_text(text)
        for text in translated_texts
    ]


def extract_noise_candidates_from_blocks(
    blocks: list[SrtBlock],
    noise_dictionary: NoiseDictionary,
) -> list[str]:
    """
    翻訳前字幕からOCR英字破損候補を抽出する。

    同じ候補は1件にまとめ、
    字幕の出現順を維持する。
    """
    candidates: list[str] = []

    for block in blocks:
        sequences = (
            find_suspicious_latin_sequences(
                block.text,
                noise_dictionary,
            )
        )

        for sequence in sequences:
            if sequence in candidates:
                continue

            candidates.append(
                sequence
            )

    return candidates


def save_failed_translation_response(
    response: str,
    *,
    chunk_start: int,
    chunk_end: int,
    attempt: int,
) -> Path:
    TRANSLATION_DEBUG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d-%H%M%S-%f"
    )

    output_path = TRANSLATION_DEBUG_DIR / (
        "failed-translation-"
        f"{chunk_start}-{chunk_end}-"
        f"attempt-{attempt}-"
        f"{timestamp}.txt"
    )

    output_path.write_text(
        response,
        encoding="utf-8",
    )

    return output_path


def translate_chunk(
    before_context: list[SrtBlock],
    target_blocks: list[SrtBlock],
    after_context: list[SrtBlock],
    model: str,
    *,
    chunk_start: int,
    chunk_end: int,
    glossary_entries: dict[str, str],
    noise_dictionary: NoiseDictionary,
    profile_name: str,
) -> list[str]:
    last_errors: list[str] = []
    last_translated_texts: list[str] = []

    input_noise_candidates = (
        extract_noise_candidates_from_blocks(
            target_blocks,
            noise_dictionary,
        )
    )

    saved_input_noise_entries = (
        append_noise_candidates(
            noise_dictionary,
            input_noise_candidates,
        )
    )

    print_saved_noise_candidates(
        saved_input_noise_entries,
        noise_dictionary,
    )

    glossary_instruction = (
        build_required_glossary_instruction(
            target_blocks,
            glossary_entries,
        )
    )

    for attempt in range(
        1,
        MAX_TRANSLATION_ATTEMPTS + 1,
    ):
        retry_target_blocks = target_blocks

        if attempt > 1:
            retry_target_blocks = (
                build_chinese_retry_blocks(
                    target_blocks,
                    last_errors,
                )
            )

            retry_target_blocks = (
                build_latin_ocr_retry_blocks(
                    retry_target_blocks,
                    last_errors,
                )
            )

        prompt = build_prompt(
            before_context,
            retry_target_blocks,
            after_context,
            profile_name=profile_name,
        )

        prompt += glossary_instruction

        if attempt > 1:
            prompt += build_retry_instruction(
                last_errors
            )

            prompt += build_structural_retry_instruction(
                target_blocks,
                last_errors,
            )

            if not has_structural_validation_error(
                last_errors
            ):
                prompt += build_chinese_retry_instruction(
                    last_errors
                )

                prompt += build_latin_ocr_retry_instruction(
                    last_errors
                )

                prompt += (
                    build_untranslated_english_retry_instruction(
                        last_errors
                    )
                )

                prompt += build_glossary_retry_instruction(
                    last_errors
                )

                prompt += (
                    build_preserved_translations_instruction(
                        target_blocks,
                        last_translated_texts,
                        last_errors,
                    )
                )

        response = generate(
            prompt,
            model=model,
        )

        display_response = "\n".join(
            normalize_translation_text(line)
            for line in response.splitlines()
        )

        print("=" * 80)
        print(display_response)
        print("=" * 80)

        validation = validate_translation_response(
            response,
            expected_ids=[
                block.number
                for block in target_blocks
            ],
            noise_dictionary=noise_dictionary,
            source_texts=[
                block.text
                for block in target_blocks
            ],
            glossary_entries=glossary_entries,
        )

        if validation.warnings:
            print("Validation Warnings:")

            for warning in validation.warnings:
                print(f"  - {warning}")

        if validation.valid:
            return normalize_translation_texts(
                validation.translated_texts
            )

        failed_path = save_failed_translation_response(
            response,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            attempt=attempt,
        )

        last_errors = validation.reasons

        noise_candidates = (
            extract_garbled_latin_candidates(
                last_errors
            )
        )

        added_noise_entries = append_noise_candidates(
            noise_dictionary,
            noise_candidates,
        )

        print_saved_noise_candidates(
            added_noise_entries,
            noise_dictionary,
        )

        if (
            len(validation.translated_texts)
            == len(target_blocks)
        ):
            last_translated_texts = (
                validation.translated_texts
            )
        else:
            last_translated_texts = []

        print(
            "Translation validation failed "
            f"(attempt {attempt}/"
            f"{MAX_TRANSLATION_ATTEMPTS})"
        )

        print("Validation Errors:")

        for reason in last_errors:
            print(f"  - {reason}")

        print(f"Saved response: {failed_path}")

        if attempt < MAX_TRANSLATION_ATTEMPTS:
            print("Retrying translation...")

        chinese_only_errors = (
            bool(last_errors)
            and all(
            error.startswith(
                "Chinese-specific characters detected:"
            )
            for error in last_errors
        )
        )

    if (
        chinese_only_errors
        and len(last_translated_texts)
        == len(target_blocks)
    ):
        corrected_texts = (
            mask_chinese_translation_errors(
                target_blocks,
                last_translated_texts,
                last_errors,
            )
        )

        corrected_response = json.dumps(
            {
                "translations": [
                    {
                        "id": block.number,
                        "translation": translation,
                    }
                    for block, translation in zip(
                        target_blocks,
                        corrected_texts,
                        strict=True,
                    )
                ],
            },
            ensure_ascii=False,
        )

        corrected_validation = (
            validate_translation_response(
                corrected_response,
                expected_ids=[
                    block.number
                    for block in target_blocks
                ],
                noise_dictionary=noise_dictionary,
                source_texts=[
                    block.text
                    for block in target_blocks
                ],
                glossary_entries=glossary_entries,
            )
        )

        if corrected_validation.valid:
            print(
                "Chinese translation fallback applied:"
            )

            for error in last_errors:
                print(f"  - {error}")

            return normalize_translation_texts(
                corrected_validation.translated_texts
            )

    raise RuntimeError(
        "Translation failed after "
        f"{MAX_TRANSLATION_ATTEMPTS} attempts "
        f"for subtitles "
        f"{chunk_start}-{chunk_end}: "
        + "; ".join(last_errors)
    )
