from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from lib.infrastructure.progress import ProgressTracker
from lib.profile.config import ProfileConfig
from lib.profile.noise import (
    NoiseDictionary,
    apply_noise_dictionary_to_text,
)
from lib.profile.prompt import load_glossary_entries
from lib.subtitle.srt import (
    SrtBlock,
    apply_translations,
    parse_speaker_from_text,
    write_structured_srt,
)
from lib.subtitle.text import cleanup_ocr_text
from .translation_chunk import (
    build_initial_translation_request_payload,
    save_translation_request_payload,
    translate_chunk,
)
from .translation_metrics import (
    TranslationChunkMetric,
    TranslationSessionMetric,
)
from .translation_metrics_inspection import (
    try_save_translation_metrics_reports,
)
from .translation_output import (
    print_chunk_start,
    print_translation_complete,
    print_translation_progress,
    print_translation_start,
)
from .translation_policy import (
    build_adaptive_translation_decision,
    resolve_adaptive_chunk_size,
)


def rebuild_speaker_text(
    speaker: str | None,
    text: str,
) -> str:
    """
    OCR前処理後の本文へ、
    抽出済みの話者情報を内部形式で戻す。

    話者あり:
        [DANIEL] This is the Stargate.

    話者なし:
        This is the Stargate.
    """
    if speaker is None:
        return text

    return f"[{speaker}] {text}"


def cleanup_block(
    block: SrtBlock,
) -> SrtBlock:
    """
    字幕番号とタイムコードを維持したまま、
    話者情報を失わずにOCR前処理する。

    明示的な話者ラベルがある場合は、
    話者と本文を先に分離する。

    OCR前処理は本文だけへ適用し、
    処理後に話者を内部形式へ戻す。
    """
    parsed = parse_speaker_from_text(
        block.text
    )

    if parsed.speaker is None:
        cleaned_text = cleanup_ocr_text(
            block.text
        )
    else:
        cleaned_body = cleanup_ocr_text(
            parsed.text
        )

        cleaned_text = rebuild_speaker_text(
            parsed.speaker,
            cleaned_body,
        )

    return SrtBlock(
        number=block.number,
        timestamp=block.timestamp,
        text=cleaned_text,
    )


def cleanup_blocks(
    blocks: list[SrtBlock],
) -> list[SrtBlock]:
    """
    字幕番号とタイムコードを維持したまま、
    AIへ渡す字幕本文をOCR前処理する。

    明示された話者情報は内部形式で維持する。
    """
    return [
        cleanup_block(block)
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
    inspect_request: bool = False,
) -> Path | None:
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

    # Phase 1-8：翻訳セッション計測を開始する
    session_metrics = TranslationSessionMetric(
        model=model,
        profile_name=resolved_profile,
        output_name=output_path.name,
        chunk_size=chunk_size,
        context_size=context_size,
        total_blocks=total_blocks,
        resume_start=resume_start,
        started_at=datetime.now(),
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

    # Phase 2-3：チャンクサイズを
    # 直前の処理結果から動的に変更する
    start = resume_start
    chunk_number = 1
    current_chunk_size = chunk_size

    while start < total_blocks:
        current_remaining_blocks = (
            total_blocks - start
        )

        remaining_chunks = (
            (
                current_remaining_blocks
                + current_chunk_size
                - 1
            )
            // current_chunk_size
        )

        progress.total_chunks = (
            progress.completed_chunks
            + remaining_chunks
        )

        end = min(
            start + current_chunk_size,
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

        if inspect_request:
            request_payload = (
                build_initial_translation_request_payload(
                    before_context,
                    target_blocks,
                    after_context,
                    model,
                    glossary_entries=glossary_entries,
                    noise_dictionary=noise_dictionary,
                    profile_name=resolved_profile,
                )
            )

            inspection_path = (
                save_translation_request_payload(
                    request_payload,
                    chunk_start=start + 1,
                    chunk_end=end,
                )
            )

            print()
            print(
                "Translation request inspection saved:"
            )
            print(f"  {inspection_path}")
            print(
                "Ollama request was not sent."
            )

            return inspection_path

        # Phase 1-8：現在のチャンク計測を開始する
        chunk_metrics = TranslationChunkMetric(
            chunk_number=chunk_number,
            chunk_start=start + 1,
            chunk_end=end,
            target_ids=tuple(
                block.number
                for block in target_blocks
            ),
            started_at=datetime.now(),
        )

        session_metrics.add_chunk(
            chunk_metrics
        )

        try:
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
                metrics=chunk_metrics,
            )
        except Exception:
            session_metrics.complete(
                elapsed_seconds=(
                    time.monotonic()
                    - translation_started_at
                ),
            )

            try_save_translation_metrics_reports(
                session=session_metrics,
                chunk=chunk_metrics,
            )

            raise

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

        # Phase 1-8：チャンクとセッション計測を保存する
        session_metrics.complete(
            elapsed_seconds=elapsed,
        )

        try_save_translation_metrics_reports(
            session=session_metrics,
            chunk=chunk_metrics,
        )

        # Phase 2-3：完了チャンクから
        # 次チャンクのサイズを決定する
        adaptive_decision = (
            build_adaptive_translation_decision(
                chunk_metrics
            )
        )

        current_chunk_size = (
            resolve_adaptive_chunk_size(
                adaptive_decision,
                configured_chunk_size=(
                    chunk_size
                ),
            )
        )

        start = end
        chunk_number += 1

    total_elapsed = (
        time.monotonic()
        - translation_started_at
    )

    translated_count = len(
        translated_blocks_all
    )

    session_metrics.complete(
        elapsed_seconds=total_elapsed,
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

    return None
