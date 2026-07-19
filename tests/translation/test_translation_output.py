from __future__ import annotations

import pytest
from lib.translation.translation_output import (
    print_adaptive_translation_decision,
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
