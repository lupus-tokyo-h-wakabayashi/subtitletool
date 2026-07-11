#!/usr/bin/env python3
import json
import re
import time
from pathlib import Path

from lib.config import (
    resolve_profile_config,
)
from lib.noise import (
    NoiseDictionary,
    apply_noise_dictionary_to_text,
    load_noise_dictionary,
)
from lib.progress import (
    ProgressTracker,
)
from lib.prompt import (
    load_glossary_entries,
)
from lib.srt import (
    SrtBlock,
    apply_translations,
    parse_srt,
    write_structured_srt,
)
from lib.text import (
    cleanup_ocr_text,
)
from lib.translation_output import (
    print_chunk_start,
    print_translation_already_complete,
    print_translation_complete,
    print_translation_progress,
    print_translation_start,
)

MODEL = "qwen3:14b"

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
