from __future__ import annotations

from pathlib import Path

from lib.infrastructure.progress import (
    ProgressTracker,
    format_duration,
)
from lib.profile.config import (
    DEFAULT_PROFILE_NAME,
)
from lib.profile.noise import (
    NoiseDictionary,
    NoiseEntry,
)
from .translation_policy import (
    ADAPTIVE_TRIGGER_NONE,
    AdaptiveTranslationDecision,
)


def print_profile_resolution(
    requested_profile: str | None,
    resolved_profile: str,
    fallback_used: bool,
) -> None:
    """
    profile解決結果を表示する。
    """
    requested_text = (
        requested_profile
        if requested_profile is not None
        else DEFAULT_PROFILE_NAME
    )

    print(
        f"Profile Req : {requested_text}"
    )
    print(
        f"Profile Use : {resolved_profile}"
    )

    if fallback_used:
        print(
            "Warning     : "
            f"Profile {requested_text!r} was not found. "
            f"Using {resolved_profile!r}."
        )


def print_saved_noise_candidates(
    entries: list[NoiseEntry],
    noise_dictionary: NoiseDictionary,
) -> None:
    """
    今回noise.local.jsonへ保存した候補を表示する。
    """
    if not entries:
        return

    print("Noise Candidates Saved:")

    for entry in entries:
        print(
            f"  - {entry.source}"
        )

    print(
        "Noise Candidate File: "
        f"{noise_dictionary.local_path}"
    )


def print_noise_dictionary_summary(
    noise_dictionary: NoiseDictionary,
) -> None:
    """
    読み込んだnoise辞書の概要を表示する。
    """
    print(
        "Noise       : "
        f"{len(noise_dictionary.entries)} entries"
    )
    print(
        "Noise Local : "
        f"{'Yes' if noise_dictionary.local_loaded else 'No'}"
    )


def print_translation_already_complete(
    *,
    requested_profile: str | None,
    resolved_profile: str,
    fallback_used: bool,
    noise_dictionary: NoiseDictionary,
    subtitle_count: int,
    output_path: Path,
) -> None:
    """
    翻訳済み出力が全件揃っている場合の結果を表示する。
    """
    print()
    print("========================================")
    print("Translation Already Complete")
    print("========================================")

    print_profile_resolution(
        requested_profile,
        resolved_profile,
        fallback_used,
    )

    print_noise_dictionary_summary(
        noise_dictionary
    )

    print(f"Subtitles   : {subtitle_count}")
    print(f"Output      : {output_path}")
    print("========================================")


def print_translation_start(
    *,
    model: str,
    requested_profile: str | None,
    resolved_profile: str,
    fallback_used: bool,
    noise_dictionary: NoiseDictionary,
    total_blocks: int,
    chunk_size: int,
    context_size: int,
    resume_start: int,
    remaining_blocks: int,
    remaining_chunks: int,
) -> None:
    """
    翻訳開始時の設定・進捗概要を表示する。
    """
    print()
    print("========================================")
    print("Translation Start")
    print("========================================")
    print(f"Model       : {model}")

    print_profile_resolution(
        requested_profile,
        resolved_profile,
        fallback_used,
    )

    print_noise_dictionary_summary(
        noise_dictionary
    )

    print(f"Profile     : {resolved_profile}")
    print(f"Subtitles   : {total_blocks}")
    print(f"Chunk Size  : {chunk_size}")
    print(
        "Context     : "
        f"{context_size} before / after"
    )
    print(
        "Resume      : "
        f"{'Yes' if resume_start else 'No'}"
    )
    print(f"Completed   : {resume_start}")
    print(f"Remaining   : {remaining_blocks}")
    print(f"Chunks Left : {remaining_chunks}")
    print("========================================")


def print_chunk_start(
    *,
    chunk_number: int,
    total_chunks: int,
    start: int,
    end: int,
    total_blocks: int,
    before_context_count: int,
    after_context_count: int,
) -> None:
    """
    各翻訳チャンク開始時の対象範囲を表示する。

    チャンク番号と総チャンク数は、
    成功済みチャンク数と現在の処理計画から表示する。
    """
    print()
    print(
        f"[{chunk_number}/{total_chunks}] "
        f"Translating "
        f"{start + 1}-{end} / {total_blocks} "
        f"(context: "
        f"{before_context_count} + "
        f"{after_context_count})"
    )


def print_translation_progress(
    *,
    progress: ProgressTracker,
    translated_count: int,
    total_blocks: int,
    chunk_elapsed: float,
    elapsed: float,
) -> None:
    """
    チャンク完了後の進捗と残り時間を表示する。
    """
    overall_progress = (
        translated_count
        / total_blocks
        * 100
    )

    print(
        "Session     : "
        f"{progress.progress_percent:5.1f}%"
    )
    print(
        "Progress    : "
        f"{overall_progress:5.1f}% "
        f"({translated_count}/{total_blocks})"
    )
    print(
        "Chunk Time  : "
        f"{format_duration(chunk_elapsed)}"
    )
    print(
        "Average     : "
        f"{progress.average_seconds:.1f} "
        "sec/chunk"
    )
    print(
        "Elapsed     : "
        f"{format_duration(elapsed)}"
    )
    print(
        "ETA         : "
        f"{format_duration(progress.eta_seconds)}"
    )


def print_adaptive_translation_decision(
    *,
    decision: AdaptiveTranslationDecision,
    next_chunk_size: int,
) -> None:
    """
    次チャンクへ適用する適応制御を表示する。

    通常戦略を維持する場合は表示しない。
    """
    if decision.trigger == (
        ADAPTIVE_TRIGGER_NONE
    ):
        return

    trigger_codes = (
        ", ".join(
            decision.trigger_codes
        )
        if decision.trigger_codes
        else "-"
    )

    source_chunk_number = (
        str(
            decision.source_chunk_number
        )
        if (
            decision.source_chunk_number
            is not None
        )
        else "-"
    )

    print()
    print(
        "Adaptive Translation:"
    )
    print(
        "  Source Chunk   : "
        f"{source_chunk_number}"
    )
    print(
        "  Strategy       : "
        f"{decision.strategy}"
    )
    print(
        "  Trigger        : "
        f"{decision.trigger}"
    )
    print(
        "  Next Chunk Size: "
        f"{next_chunk_size}"
    )
    print(
        "  Trigger Codes  : "
        f"{trigger_codes}"
    )


def print_translation_complete(
    *,
    session_result: str,
    translated_count: int,
    progress: ProgressTracker,
    total_elapsed: float,
    output_path: Path,
) -> None:
    """
    翻訳完了時の結果を表示する。
    """
    print()
    print("========================================")
    print("Translation Complete")
    print("========================================")
    print(f"Result      : {session_result}")
    print(f"Subtitles   : {translated_count}")
    print(
        "Chunks      : "
        f"{progress.completed_chunks}"
    )
    print(
        "Total Time  : "
        f"{format_duration(total_elapsed)}"
    )
    print(
        "Average     : "
        f"{progress.average_seconds:.1f} "
        "sec/chunk"
    )
    print(
        "Fastest     : "
        f"{progress.fastest_seconds:.1f} sec"
    )
    print(
        "Slowest     : "
        f"{progress.slowest_seconds:.1f} sec"
    )
    print(f"Output      : {output_path}")
    print("========================================")
