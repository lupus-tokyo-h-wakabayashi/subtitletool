from __future__ import annotations

import time
from pathlib import Path

from lib.profile.config import ProfileConfig
from lib.profile.noise import (
    NoiseDictionary,
    apply_noise_dictionary_to_text,
)
from lib.profile.prompt import load_glossary_entries
from lib.progress import ProgressTracker
from lib.srt import (
    SrtBlock,
    apply_translations,
    write_structured_srt,
)
from lib.text import cleanup_ocr_text
from lib.translation.translation_chunk import translate_chunk
from lib.translation.translation_output import (
    print_chunk_start,
    print_translation_complete,
    print_translation_progress,
    print_translation_start,
)


def cleanup_blocks(
    blocks: list[SrtBlock],
) -> list[SrtBlock]:
    """
    字幕番号とタイムコードを維持したまま、
    AIへ渡す字幕本文だけOCR前処理する。
    """
    return [
        SrtBlock(
            number=block.number,
            timestamp=block.timestamp,
            text=cleanup_ocr_text(
                block.text
            ),
        )
        for block in blocks
    ]


def apply_noise_to_blocks(
    blocks: list[SrtBlock],
    noise_dictionary: NoiseDictionary,
) -> list[SrtBlock]:
    """
    字幕番号とタイムコードを維持したまま、
    翻訳対象の本文へNoise辞書を適用する。
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


def run_translation_session(
    *,
    source_blocks: list[SrtBlock],
    translated_blocks_all: list[SrtBlock],
    output_path: Path,
    model: str,
    chunk_size: int,
    context_size: int,
    profile_config: ProfileConfig,
    noise_dictionary: NoiseDictionary,
) -> None:
    """
    未翻訳部分をチャンク単位で翻訳し、
    各チャンク終了時に途中保存する。
    """
    resolved_profile = (
        profile_config.resolved_profile
    )

    glossary_entries = load_glossary_entries(
        resolved_profile
    )

    total_blocks = len(
        source_blocks
    )

    resume_start = len(
        translated_blocks_all
    )

    remaining_blocks = (
        total_blocks - resume_start
    )

    remaining_chunks = (
                           remaining_blocks + chunk_size - 1
                       ) // chunk_size

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

        chunk_started_at = (
            time.monotonic()
        )

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
