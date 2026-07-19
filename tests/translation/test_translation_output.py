from __future__ import annotations

from pathlib import Path

import pytest
from lib.infrastructure.progress import (
    ProgressTracker,
)
from lib.translation.translation_metrics_inspection import (
    TRANSLATION_SESSION_RESULT_COMPLETED_WITH_RECOVERY,
    TRANSLATION_SESSION_RESULT_FAILED,
)
from lib.translation.translation_output import (
    print_adaptive_translation_decision,
    print_translation_complete,
    print_translation_failed,
)
from lib.translation.translation_policy import (
    ADAPTIVE_STRATEGY_REDUCED_CHUNK,
    ADAPTIVE_STRATEGY_STANDARD,
    ADAPTIVE_TRIGGER_NONE,
    ADAPTIVE_TRIGGER_STANDARD_RETRY,
    AdaptiveTranslationDecision,
)


# 通常再試行による縮小表示
def test_print_adaptive_translation_decision_displays_reduced_chunk(
    capsys: pytest.CaptureFixture[str],
) -> None:
    decision = AdaptiveTranslationDecision(
        strategy=(
            ADAPTIVE_STRATEGY_REDUCED_CHUNK
        ),
        trigger=(
            ADAPTIVE_TRIGGER_STANDARD_RETRY
        ),
        source_chunk_number=1,
        trigger_codes=(
            "glossary_violation",
        ),
    )

    print_adaptive_translation_decision(
        decision=decision,
        next_chunk_size=5,
    )

    captured = capsys.readouterr()

    assert (
        "Adaptive Translation:"
        in captured.out
    )

    assert (
        "Source Chunk   : 1"
        in captured.out
    )

    assert (
        "Strategy       : reduced_chunk"
        in captured.out
    )

    assert (
        "Trigger        : standard_retry"
        in captured.out
    )

    assert (
        "Next Chunk Size: 5"
        in captured.out
    )

    assert (
        "Trigger Codes  : glossary_violation"
        in captured.out
    )


# 通常戦略維持時は表示しない
def test_print_adaptive_translation_decision_skips_standard_strategy(
    capsys: pytest.CaptureFixture[str],
) -> None:
    decision = AdaptiveTranslationDecision(
        strategy=(
            ADAPTIVE_STRATEGY_STANDARD
        ),
        trigger=(
            ADAPTIVE_TRIGGER_NONE
        ),
        source_chunk_number=1,
    )

    print_adaptive_translation_decision(
        decision=decision,
        next_chunk_size=10,
    )

    captured = capsys.readouterr()

    assert captured.out == ""


# 発火理由コードがない場合
def test_print_adaptive_translation_decision_displays_empty_codes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    decision = AdaptiveTranslationDecision(
        strategy=(
            ADAPTIVE_STRATEGY_REDUCED_CHUNK
        ),
        trigger=(
            ADAPTIVE_TRIGGER_STANDARD_RETRY
        ),
        source_chunk_number=2,
    )

    print_adaptive_translation_decision(
        decision=decision,
        next_chunk_size=5,
    )

    captured = capsys.readouterr()

    assert (
        "Trigger Codes  : -"
        in captured.out
    )


# 適応回復後の翻訳完了表示
def test_print_translation_complete_displays_session_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    progress = ProgressTracker(
        total_chunks=3
    )

    progress.add(
        1.0
    )
    progress.add(
        2.0
    )
    progress.add(
        3.0
    )

    print_translation_complete(
        session_result=(
            TRANSLATION_SESSION_RESULT_COMPLETED_WITH_RECOVERY
        ),
        translated_count=20,
        progress=progress,
        total_elapsed=6.0,
        output_path=Path(
            "output.srt"
        ),
    )

    captured = capsys.readouterr()

    assert (
        "Translation Complete"
        in captured.out
    )

    assert (
        "Result      : "
        "completed_with_recovery"
        in captured.out
    )

    assert (
        "Subtitles   : 20"
        in captured.out
    )

    assert (
        "Chunks      : 3"
        in captured.out
    )

    assert (
        "Output      : output.srt"
        in captured.out
    )


# 回復不能な翻訳失敗表示
def test_print_translation_failed_displays_terminal_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_translation_failed(
        session_result=(
            TRANSLATION_SESSION_RESULT_FAILED
        ),
        translated_count=1,
        total_blocks=4,
        failed_ids=(
            "2",
        ),
        error=RuntimeError(
            "single subtitle failed"
        ),
        output_path=Path(
            "output.srt"
        ),
    )

    captured = capsys.readouterr()

    assert (
        "Translation Failed"
        in captured.out
    )

    assert (
        "Result      : failed"
        in captured.out
    )

    assert (
        "Completed   : 1 / 4"
        in captured.out
    )

    assert (
        "Failed IDs  : 2"
        in captured.out
    )

    assert (
        "Error       : RuntimeError: "
        "single subtitle failed"
        in captured.out
    )

    assert (
        "Partial     : output.srt"
        in captured.out
    )
