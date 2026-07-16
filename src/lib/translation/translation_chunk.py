from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from lib.infrastructure.ollama import (
    build_generate_payload,
    generate,
)
from lib.profile.noise import (
    NoiseDictionary,
    append_noise_candidates,
    find_suspicious_latin_sequences,
)
from lib.subtitle.srt import (
    SrtBlock,
    parse_speaker_from_text,
)
from .ocr_retry import (
    build_chinese_retry_blocks,
    build_latin_ocr_retry_blocks,
    extract_garbled_latin_candidates,
    mask_chinese_translation_errors,
)
from .retry import (
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
from .translation_output import (
    print_saved_noise_candidates,
)
from .translation_prompt import (
    build_ocr_noise_instruction,
    build_prompt,
)
from .translation_schema import (
    build_translation_response_schema,
)
from .translation_validation import (
    validate_translation_response,
)

MAX_TRANSLATION_ATTEMPTS = 3

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

TRANSLATION_DEBUG_DIR = (
    PROJECT_ROOT
    / "tmp"
)


def build_initial_translation_prompt(
    before_context: list[SrtBlock],
    target_blocks: list[SrtBlock],
    after_context: list[SrtBlock],
    *,
    profile_name: str,
    ocr_noise_instruction: str,
    glossary_instruction: str,
) -> str:
    """
    初回翻訳リクエストで使用するPromptを生成する。

    通常翻訳とリクエスト確認機能の両方が
    この関数を使用し、送信内容の差異を防ぐ。
    """
    prompt = build_prompt(
        before_context,
        target_blocks,
        after_context,
        profile_name=profile_name,
        ocr_noise_instruction=(
            ocr_noise_instruction
        ),
    )

    prompt += glossary_instruction

    return prompt


def build_initial_translation_request_payload(
    before_context: list[SrtBlock],
    target_blocks: list[SrtBlock],
    after_context: list[SrtBlock],
    model: str,
    *,
    glossary_entries: dict[str, str],
    noise_dictionary: NoiseDictionary,
    profile_name: str,
) -> dict[str, object]:
    """
    次の初回翻訳リクエストと同一のPayloadを生成する。

    Ollamaへの接続・送信やNoise辞書の更新は行わない。
    """
    suspicious_ids = find_noise_candidate_ids(
        target_blocks,
        noise_dictionary,
    )

    ocr_noise_instruction = (
        build_ocr_noise_instruction(
            suspicious_ids
        )
    )

    glossary_instruction = (
        build_required_glossary_instruction(
            target_blocks,
            glossary_entries,
        )
    )

    prompt = build_initial_translation_prompt(
        before_context,
        target_blocks,
        after_context,
        profile_name=profile_name,
        ocr_noise_instruction=(
            ocr_noise_instruction
        ),
        glossary_instruction=(
            glossary_instruction
        ),
    )

    response_schema = (
        build_translation_response_schema(
            target_blocks
        )
    )

    return build_generate_payload(
        prompt=prompt,
        model=model,
        temperature=0.2,
        top_p=0.9,
        response_format=response_schema,
    )


def save_translation_request_payload(
    payload: dict[str, object],
    *,
    chunk_start: int,
    chunk_end: int,
) -> Path:
    """
    Ollamaへ送信予定のPayloadを確認用JSONへ保存する。
    """
    TRANSLATION_DEBUG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = TRANSLATION_DEBUG_DIR / (
        "translation-request-"
        f"{chunk_start}-{chunk_end}.json"
    )

    output_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return output_path


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


def build_expected_source_metadata(
    blocks: list[SrtBlock],
) -> tuple[
    list[str | None],
    list[str],
]:
    """
    Validationで使用する読み取り専用source情報を生成する。

    Prompt生成時と同じspeaker解析を使用し、
    source.speakerとsource.textの期待値を返す。
    """
    source_speakers: list[str | None] = []
    source_texts: list[str] = []

    for block in blocks:
        parsed = parse_speaker_from_text(
            block.text
        )

        source_speakers.append(
            parsed.speaker
        )

        source_texts.append(
            parsed.text
        )

    return (
        source_speakers,
        source_texts,
    )


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


def find_noise_candidate_ids(
    blocks: list[SrtBlock],
    noise_dictionary: NoiseDictionary,
) -> list[str]:
    """
    OCR破損候補を含む字幕IDを、
    字幕の出現順で返す。
    """
    suspicious_ids: list[str] = []

    for block in blocks:
        if find_suspicious_latin_sequences(
            block.text,
            noise_dictionary,
        ):
            suspicious_ids.append(
                block.number
            )

    return suspicious_ids


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


def generate_translation_response(
    prompt: str,
    model: str,
    target_blocks: list[SrtBlock],
) -> str:
    """
    翻訳対象ブロックからレスポンスSchemaを生成し、
    OllamaへSchema付き生成リクエストを送信する。
    """
    response_schema = (
        build_translation_response_schema(
            target_blocks
        )
    )

    return generate(
        prompt,
        model=model,
        response_format=response_schema,
    )


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

    (
        original_source_speakers,
        original_source_texts,
    ) = build_expected_source_metadata(
        target_blocks
    )

    input_noise_candidates = (
        extract_noise_candidates_from_blocks(
            target_blocks,
            noise_dictionary,
        )
    )

    suspicious_ids = find_noise_candidate_ids(
        target_blocks,
        noise_dictionary,
    )

    ocr_noise_instruction = (
        build_ocr_noise_instruction(
            suspicious_ids
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

        (
            response_source_speakers,
            response_source_texts,
        ) = build_expected_source_metadata(
            retry_target_blocks
        )

        prompt = build_initial_translation_prompt(
            before_context,
            retry_target_blocks,
            after_context,
            profile_name=profile_name,
            ocr_noise_instruction=(
                ocr_noise_instruction
            ),
            glossary_instruction=(
                glossary_instruction
            ),
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

        response = generate_translation_response(
            prompt,
            model,
            retry_target_blocks,
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
            source_speakers=(
                original_source_speakers
            ),
            source_texts=(
                original_source_texts
            ),
            response_source_speakers=(
                response_source_speakers
            ),
            response_source_texts=(
                response_source_texts
            ),
            noise_dictionary=noise_dictionary,
            glossary_entries=glossary_entries,
        )

        if (
            validation.valid
            and validation.noise_candidates
        ):
            saved_entries = (
                append_noise_candidates(
                    noise_dictionary,
                    validation.noise_candidates,
                )
            )

            print_saved_noise_candidates(
                saved_entries,
                noise_dictionary,
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
                "targets": {
                    block.number: {
                        "source": {
                            "speaker": source_speaker,
                            "text": source_text,
                        },
                        "translation": translation,
                    }
                    for (
                        block,
                        source_speaker,
                        source_text,
                        translation,
                    ) in zip(
                        target_blocks,
                        original_source_speakers,
                        original_source_texts,
                        corrected_texts,
                        strict=True,
                    )
                },
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
                source_speakers=(
                    original_source_speakers
                ),
                source_texts=(
                    original_source_texts
                ),
                noise_dictionary=noise_dictionary,
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
