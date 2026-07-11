#!/usr/bin/env python3
import json
import re
import time
from datetime import datetime
from pathlib import Path

from lib.config import (
    resolve_profile_config,
)
from lib.noise import (
    NoiseDictionary,
    append_noise_candidates,
    apply_noise_dictionary_to_text,
    load_noise_dictionary,
)
from lib.ocr_retry import (
    build_chinese_retry_blocks,
    build_latin_ocr_retry_blocks,
    extract_garbled_latin_candidates,
)
from lib.ollama import generate
from lib.progress import (
    ProgressTracker,
)
from lib.prompt import (
    build_translation_prompt,
    load_glossary_entries,
)
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
from lib.srt import (
    SrtBlock,
    apply_translations,
    parse_speaker_from_text,
    parse_srt,
    write_structured_srt,
)
from lib.text import (
    cleanup_ocr_text,
    find_suspicious_latin_sequences,
    is_suspicious_ocr_text,
)
from lib.translation_output import (
    print_chunk_start,
    print_saved_noise_candidates,
    print_translation_already_complete,
    print_translation_complete,
    print_translation_progress,
    print_translation_start,
)
from lib.translation_validation import (
    validate_translation_response,
)

MODEL = "qwen3:14b"

# 再試行用
MAX_TRANSLATION_ATTEMPTS = 3
TRANSLATION_DEBUG_DIR = (
    Path("~/tmp/subtitletool")
    .expanduser()
)

# 実際に翻訳する字幕数
CHUNK_SIZE = 10

# 翻訳対象の前後に参考として渡す字幕数
CONTEXT_SIZE = 15


def cleanup_blocks(blocks: list[SrtBlock]) -> list[SrtBlock]:
    """
    字幕番号とタイムコードを維持し、
    AIへ渡す本文だけOCR前処理する。
    """
    return [
        SrtBlock(
            number=block.number,
            timestamp=block.timestamp,
            text=cleanup_ocr_text(block.text),
        )
        for block in blocks
    ]


def apply_noise_to_blocks(
    blocks: list[SrtBlock],
    noise_dictionary: NoiseDictionary,
) -> list[SrtBlock]:
    """
    字幕番号とタイムコードを維持し、
    本文だけへnoise辞書を適用する。
    """
    return [
        SrtBlock(
            number=block.number,
            timestamp=block.timestamp,
            text=apply_noise_dictionary_to_text(
                block.text,
                noise_dictionary,
            ),
        )
        for block in blocks
    ]


def build_request_item(
    block: SrtBlock,
) -> dict[str, str | None]:
    """
    SRTブロックをLLMリクエスト用JSON要素へ変換する。

    話者が明示されている場合だけspeakerへ設定し、
    本文から話者表記を除去する。
    """
    parsed = parse_speaker_from_text(
        block.text
    )

    return {
        "id": block.number,
        "speaker": parsed.speaker,
        "text": parsed.text,
    }


def build_context_item(
    block: SrtBlock,
) -> dict[str, str | None]:
    """
    参考文脈をLLMリクエスト用JSONへ変換する。

    contextは出力対象ではないため、
    targetと混同されないようidを含めない。
    """
    parsed = parse_speaker_from_text(
        block.text
    )

    return {
        "speaker": parsed.speaker,
        "text": parsed.text,
    }


def build_translation_request_json(
    before_context: list[SrtBlock],
    target_blocks: list[SrtBlock],
    after_context: list[SrtBlock],
) -> str:
    """
    前後文脈と翻訳対象をJSON文字列へ変換する。
    """
    payload = {
        "context_before": [
            build_context_item(block)
            for block in before_context
        ],
        "target": [
            build_request_item(block)
            for block in target_blocks
        ],
        "context_after": [
            build_context_item(block)
            for block in after_context
        ],
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
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
                block.text
            )
        )

        for sequence in sequences:
            if sequence in candidates:
                continue

            candidates.append(
                sequence
            )

    return candidates


def build_ocr_noise_instruction(
    target_blocks: list[SrtBlock],
) -> str:
    """
    翻訳対象内のOCR破損候補を検出し、
    チャンク内番号でLLMへ通知する。
    """
    suspicious_ids = [
        block.number
        for block in target_blocks
        if is_suspicious_ocr_text(
            block.text
        )
    ]

    if not suspicious_ids:
        return ""

    ids = ", ".join(
        suspicious_ids
    )

    print(
        "OCR Noise IDs: "
        f"{ids}"
    )

    return f"""

【OCR破損の可能性がある字幕】

対象ID: {ids}

これらの字幕にはOCRで壊れた英字列が含まれる可能性がある。

* 壊れた英字列を人名、地名、セリフとして推測しない
* 意味不明な文字列をカタカナへ音写しない
* 理解できる部分だけ翻訳する
* 判読できない部分は「（判読不能）」とする
* 原文にない意味を追加しない
"""


def build_prompt(
    before_context: list[SrtBlock],
    target_blocks: list[SrtBlock],
    after_context: list[SrtBlock],
    profile_name: str,
) -> str:
    request_json = build_translation_request_json(
        before_context,
        target_blocks,
        after_context,
    )

    base_prompt = build_translation_prompt(
        target_count=len(target_blocks),
        request_json=request_json,
        profile_name=profile_name,
    )

    ocr_noise_instruction = (
        build_ocr_noise_instruction(
            target_blocks
        )
    )

    return (
        base_prompt
        + ocr_noise_instruction
    )


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
            target_blocks
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

    raise RuntimeError(
        "Translation failed after "
        f"{MAX_TRANSLATION_ATTEMPTS} attempts "
        f"for subtitles "
        f"{chunk_start}-{chunk_end}: "
        + "; ".join(last_errors)
    )


def validate_resume_blocks(
    source_blocks: list[SrtBlock],
    translated_blocks: list[SrtBlock],
) -> None:
    """
    途中保存されたSRTが、入力SRTの先頭部分と一致するか確認する。

    本文は翻訳後なので比較しない。
    字幕番号とタイムコードだけを比較する。
    """
    if len(translated_blocks) > len(source_blocks):
        raise RuntimeError(
            "Resume failed: "
            "output SRT contains more subtitles than input SRT. "
            f"input={len(source_blocks)}, "
            f"output={len(translated_blocks)}"
        )

    for index, translated_block in enumerate(
        translated_blocks
    ):
        source_block = source_blocks[index]

        if translated_block.number != source_block.number:
            raise RuntimeError(
                "Resume failed: subtitle number mismatch "
                f"at position {index + 1}. "
                f"input={source_block.number}, "
                f"output={translated_block.number}"
            )

        if translated_block.timestamp != source_block.timestamp:
            raise RuntimeError(
                "Resume failed: timestamp mismatch "
                f"at subtitle {source_block.number}. "
                f"input={source_block.timestamp!r}, "
                f"output={translated_block.timestamp!r}"
            )


def translate_srt(
    input_srt: str | Path,
    output_srt: str | Path,
    model: str = MODEL,
    chunk_size: int = CHUNK_SIZE,
    context_size: int = CONTEXT_SIZE,
    profile_name: str | None = None,
    style_name: str | None = None,
    glossary_name: str | None = None,
) -> Path:
    input_path = (
        Path(input_srt)
        .expanduser()
        .resolve()
    )

    output_path = (
        Path(output_srt)
        .expanduser()
        .resolve()
    )

    if input_path == output_path:
        raise ValueError(
            "Input and output SRT paths must be different: "
            f"{input_path}"
        )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"SRT not found: {input_path}"
        )

    requested_profile = profile_name

    legacy_profile_specified = (
        style_name is not None
        or glossary_name is not None
    )

    if legacy_profile_specified:
        if style_name != glossary_name:
            raise ValueError(
                "Style and glossary profiles must match "
                "during migration: "
                f"style={style_name!r}, "
                f"glossary={glossary_name!r}"
            )

        legacy_profile = style_name

        if (
            requested_profile is not None
            and legacy_profile is not None
            and requested_profile != legacy_profile
        ):
            raise ValueError(
                "Profile conflicts with legacy options: "
                f"profile={requested_profile!r}, "
                f"style={style_name!r}, "
                f"glossary={glossary_name!r}"
            )

        if requested_profile is None:
            requested_profile = legacy_profile

    profile_config = resolve_profile_config(
        requested_profile
    )

    resolved_profile = (
        profile_config.resolved_profile
    )

    noise_dictionary = load_noise_dictionary(
        profile_config
    )

    source_blocks = parse_srt(
        input_path
    )

    if not source_blocks:
        raise RuntimeError(
            "No valid subtitle blocks: "
            f"{input_path}"
        )

    translated_blocks_all: list[SrtBlock] = []

    if output_path.exists():
        translated_blocks_all = parse_srt(
            output_path
        )

        if not translated_blocks_all:
            raise RuntimeError(
                "Resume failed: output SRT exists "
                "but contains no valid subtitle blocks: "
                f"{output_path}"
            )

        validate_resume_blocks(
            source_blocks,
            translated_blocks_all,
        )

    total_blocks = len(source_blocks)
    resume_start = len(
        translated_blocks_all
    )

    if resume_start == total_blocks:
        print_translation_already_complete(
            requested_profile=(
                profile_config.requested_profile
            ),
            resolved_profile=resolved_profile,
            fallback_used=(
                profile_config.fallback_used
            ),
            noise_dictionary=noise_dictionary,
            subtitle_count=total_blocks,
            output_path=output_path,
        )

        return output_path

    glossary_entries = load_glossary_entries(
        resolved_profile
    )

    remaining_blocks = (
        total_blocks - resume_start
    )

    remaining_chunks = (
        remaining_blocks
        + chunk_size
        - 1
    )

    translation_started_at = (
        time.monotonic()
    )

    progress = ProgressTracker(
        total_chunks=remaining_chunks
    )

    print_translation_start(
        model=model,
        requested_profile=(
            profile_config.requested_profile
        ),
        resolved_profile=resolved_profile,
        fallback_used=(
            profile_config.fallback_used
        ),
        noise_dictionary=noise_dictionary,
        total_blocks=total_blocks,
        chunk_size=chunk_size,
        context_size=context_size,
        resume_start=resume_start,
        remaining_blocks=remaining_blocks,
        remaining_chunks=remaining_chunks,
    )

    chunk_starts = range(
        resume_start,
        total_blocks,
        chunk_size,
    )

    for chunk_number, start in enumerate(
        chunk_starts,
        start=1,
    ):
        end = min(
            start + chunk_size,
            total_blocks,
        )

        before_start = max(
            0,
            start - context_size,
        )

        after_end = min(
            total_blocks,
            end + context_size,
        )

        # タイムコードと元の字幕構造を維持するため、
        # OCR前処理前の翻訳対象を保持する。
        source_target_blocks = (
            source_blocks[start:end]
        )

        # AIへ渡す字幕本文だけOCR前処理する。
        before_context = cleanup_blocks(
            source_blocks[
                before_start:start
            ]
        )

        target_blocks = apply_noise_to_blocks(
            cleanup_blocks(
                source_target_blocks
            ),
            noise_dictionary,
        )

        after_context = cleanup_blocks(
            source_blocks[
                end:after_end
            ]
        )

        chunk_started_at = time.monotonic()

        print_chunk_start(
            chunk_number=chunk_number,
            remaining_chunks=remaining_chunks,
            start=start,
            end=end,
            total_blocks=total_blocks,
            before_context_count=len(
                before_context
            ),
            after_context_count=len(
                after_context
            ),
        )

        translated_texts = translate_chunk(
            before_context,
            target_blocks,
            after_context,
            model,
            chunk_start=start + 1,
            chunk_end=end,
            glossary_entries=glossary_entries,
            noise_dictionary=noise_dictionary,
            profile_name=resolved_profile,
        )

        translated_chunk_blocks = (
            apply_translations(
                source_target_blocks,
                translated_texts,
            )
        )

        translated_blocks_all.extend(
            translated_chunk_blocks
        )

        # 各チャンク終了時に途中保存する。
        write_structured_srt(
            output_path,
            translated_blocks_all,
        )

        chunk_elapsed = (
            time.monotonic()
            - chunk_started_at
        )

        progress.add(
            chunk_elapsed
        )

        elapsed = (
            time.monotonic()
            - translation_started_at
        )

        translated_count = len(
            translated_blocks_all
        )

        print_translation_progress(
            progress=progress,
            translated_count=translated_count,
            total_blocks=total_blocks,
            chunk_elapsed=chunk_elapsed,
            elapsed=elapsed,
        )

    total_elapsed = (
        time.monotonic()
        - translation_started_at
    )

    translated_count = len(
        translated_blocks_all
    )

    if translated_count != total_blocks:
        raise RuntimeError(
            "Subtitle count mismatch: "
            f"source={total_blocks}, "
            f"translated={translated_count}"
        )

    print_translation_complete(
        translated_count=translated_count,
        progress=progress,
        total_elapsed=total_elapsed,
        output_path=output_path,
    )

    return output_path
