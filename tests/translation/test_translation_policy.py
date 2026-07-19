from __future__ import annotations

from datetime import datetime

import pytest
from lib.translation.translation_metrics import (
    TRANSLATION_RESULT_FAILED,
    TRANSLATION_RESULT_HYBRID_SUCCESS,
    TRANSLATION_RESULT_STANDARD_SUCCESS,
    TranslationAttemptMetric,
    TranslationChunkMetric,
)
from lib.translation.translation_policy import (
    ADAPTIVE_STRATEGY_REDUCED_CHUNK,
    ADAPTIVE_STRATEGY_SINGLE_SUBTITLE,
    ADAPTIVE_STRATEGY_STANDARD,
    AdaptiveTranslationDecision,
    AdaptiveTranslationStrategy,
    ADAPTIVE_TRIGGER_FAILED,
    ADAPTIVE_TRIGGER_HYBRID,
    ADAPTIVE_TRIGGER_NONE,
    ADAPTIVE_TRIGGER_STANDARD_RETRY,
    build_adaptive_translation_decision,
    resolve_adaptive_chunk_size,
)


def build_chunk_metrics(
) -> TranslationChunkMetric:
    return TranslationChunkMetric(
        chunk_number=1,
        chunk_start=1,
        chunk_end=10,
        target_ids=tuple(
            str(number)
            for number in range(
                1,
                11,
            )
        ),
        started_at=datetime(
            2026,
            7,
            19,
            12,
            0,
            0,
        ),
    )


def build_standard_attempt(
    *,
    attempt: int,
    valid: bool,
    reason_codes: tuple[str, ...] = (),
) -> TranslationAttemptMetric:
    return TranslationAttemptMetric(
        pipeline="standard",
        attempt=attempt,
        target_ids=tuple(
            str(number)
            for number in range(
                1,
                11,
            )
        ),
        elapsed_seconds=1.0,
        response_received=True,
        validation_stage=(
            "standard_validation"
        ),
        validation_valid=valid,
        reason_codes=reason_codes,
    )


# 1回目の通常翻訳成功
def test_standard_success_keeps_standard_strategy(
) -> None:
    metrics = build_chunk_metrics()

    metrics.add_standard_attempt(
        build_standard_attempt(
            attempt=1,
            valid=True,
        )
    )

    metrics.complete(
        final_result=(
            TRANSLATION_RESULT_STANDARD_SUCCESS
        ),
        elapsed_seconds=1.0,
    )

    decision = (
        build_adaptive_translation_decision(
            metrics
        )
    )

    assert (
        decision.strategy
        == ADAPTIVE_STRATEGY_STANDARD
    )

    assert (
        decision.trigger
        == ADAPTIVE_TRIGGER_NONE
    )

    assert decision.source_chunk_number == 1
    assert decision.trigger_codes == ()


# 通常翻訳の再試行
def test_standard_retry_selects_reduced_chunk_strategy(
) -> None:
    metrics = build_chunk_metrics()

    metrics.add_standard_attempt(
        build_standard_attempt(
            attempt=1,
            valid=False,
            reason_codes=(
                "glossary_violation",
            ),
        )
    )

    metrics.add_standard_attempt(
        build_standard_attempt(
            attempt=2,
            valid=True,
        )
    )

    metrics.complete(
        final_result=(
            TRANSLATION_RESULT_STANDARD_SUCCESS
        ),
        elapsed_seconds=2.0,
    )

    decision = (
        build_adaptive_translation_decision(
            metrics
        )
    )

    assert (
        decision.strategy
        == ADAPTIVE_STRATEGY_REDUCED_CHUNK
    )

    assert (
        decision.trigger
        == ADAPTIVE_TRIGGER_STANDARD_RETRY
    )

    assert decision.trigger_codes == (
        "glossary_violation",
    )


# 再試行理由コードの重複除去
def test_standard_retry_keeps_unique_trigger_codes_in_order(
) -> None:
    metrics = build_chunk_metrics()

    metrics.add_standard_attempt(
        build_standard_attempt(
            attempt=1,
            valid=False,
            reason_codes=(
                "glossary_violation",
                "untranslated_english_sentence_detected",
            ),
        )
    )

    metrics.add_standard_attempt(
        build_standard_attempt(
            attempt=2,
            valid=False,
            reason_codes=(
                "glossary_violation",
            ),
        )
    )

    metrics.add_standard_attempt(
        build_standard_attempt(
            attempt=3,
            valid=True,
        )
    )

    metrics.complete(
        final_result=(
            TRANSLATION_RESULT_STANDARD_SUCCESS
        ),
        elapsed_seconds=3.0,
    )

    decision = (
        build_adaptive_translation_decision(
            metrics
        )
    )

    assert decision.trigger_codes == (
        "glossary_violation",
        "untranslated_english_sentence_detected",
    )


# Hybrid移行
def test_hybrid_trigger_selects_reduced_chunk_strategy(
) -> None:
    metrics = build_chunk_metrics()

    metrics.add_standard_attempt(
        build_standard_attempt(
            attempt=1,
            valid=False,
            reason_codes=(
                "untranslated_english_sentence_detected",
            ),
        )
    )

    metrics.trigger_hybrid(
        [
            (
                "Untranslated English sentence "
                "detected: subtitle_id='1'"
            ),
        ]
    )

    metrics.complete(
        final_result=(
            TRANSLATION_RESULT_HYBRID_SUCCESS
        ),
        elapsed_seconds=2.0,
    )

    decision = (
        build_adaptive_translation_decision(
            metrics
        )
    )

    assert (
        decision.strategy
        == ADAPTIVE_STRATEGY_REDUCED_CHUNK
    )

    assert (
        decision.trigger
        == ADAPTIVE_TRIGGER_HYBRID
    )

    assert decision.trigger_codes == (
        "untranslated_english_sentence_detected",
    )


# Hybridは通常再試行より優先する
def test_hybrid_trigger_has_priority_over_standard_retry(
) -> None:
    metrics = build_chunk_metrics()

    metrics.add_standard_attempt(
        build_standard_attempt(
            attempt=1,
            valid=False,
            reason_codes=(
                "glossary_violation",
            ),
        )
    )

    metrics.add_standard_attempt(
        build_standard_attempt(
            attempt=2,
            valid=False,
            reason_codes=(
                "untranslated_english_sentence_detected",
            ),
        )
    )

    metrics.trigger_hybrid(
        [
            (
                "Untranslated English sentence "
                "detected: subtitle_id='1'"
            ),
        ]
    )

    metrics.complete(
        final_result=(
            TRANSLATION_RESULT_HYBRID_SUCCESS
        ),
        elapsed_seconds=3.0,
    )

    decision = (
        build_adaptive_translation_decision(
            metrics
        )
    )

    assert (
        decision.strategy
        == ADAPTIVE_STRATEGY_REDUCED_CHUNK
    )

    assert (
        decision.trigger
        == ADAPTIVE_TRIGGER_HYBRID
    )


# 最終失敗
def test_failed_chunk_selects_single_subtitle_strategy(
) -> None:
    metrics = build_chunk_metrics()

    metrics.add_standard_attempt(
        build_standard_attempt(
            attempt=1,
            valid=False,
            reason_codes=(
                "invalid_json_response",
            ),
        )
    )

    metrics.complete(
        final_result=(
            TRANSLATION_RESULT_FAILED
        ),
        elapsed_seconds=1.0,
        failed_ids=(
            "1",
            "2",
        ),
    )

    decision = (
        build_adaptive_translation_decision(
            metrics
        )
    )

    assert (
        decision.strategy
        == ADAPTIVE_STRATEGY_SINGLE_SUBTITLE
    )

    assert (
        decision.trigger
        == ADAPTIVE_TRIGGER_FAILED
    )

    assert decision.trigger_codes == (
        "translation_failed",
    )


# 最終失敗はHybrid移行より優先する
def test_failed_chunk_has_priority_over_hybrid_trigger(
) -> None:
    metrics = build_chunk_metrics()

    metrics.trigger_hybrid(
        [
            (
                "Untranslated English sentence "
                "detected: subtitle_id='1'"
            ),
        ]
    )

    metrics.complete(
        final_result=(
            TRANSLATION_RESULT_FAILED
        ),
        elapsed_seconds=2.0,
        failed_ids=(
            "1",
        ),
    )

    decision = (
        build_adaptive_translation_decision(
            metrics
        )
    )

    assert (
        decision.strategy
        == ADAPTIVE_STRATEGY_SINGLE_SUBTITLE
    )

    assert (
        decision.trigger
        == ADAPTIVE_TRIGGER_FAILED
    )


def build_test_decision(
    strategy: AdaptiveTranslationStrategy,
) -> AdaptiveTranslationDecision:
    return AdaptiveTranslationDecision(
        strategy=strategy,
        trigger=ADAPTIVE_TRIGGER_NONE,
        source_chunk_number=1,
    )


# 通常戦略のチャンクサイズ
def test_standard_strategy_keeps_configured_chunk_size(
) -> None:
    decision = build_test_decision(
        ADAPTIVE_STRATEGY_STANDARD
    )

    actual = resolve_adaptive_chunk_size(
        decision,
        configured_chunk_size=10,
    )

    assert actual == 10


# 縮小戦略の偶数チャンクサイズ
def test_reduced_strategy_halves_even_chunk_size(
) -> None:
    decision = build_test_decision(
        ADAPTIVE_STRATEGY_REDUCED_CHUNK
    )

    actual = resolve_adaptive_chunk_size(
        decision,
        configured_chunk_size=10,
    )

    assert actual == 5


# 縮小戦略の奇数チャンクサイズ
def test_reduced_strategy_rounds_odd_chunk_size_up(
) -> None:
    decision = build_test_decision(
        ADAPTIVE_STRATEGY_REDUCED_CHUNK
    )

    actual = resolve_adaptive_chunk_size(
        decision,
        configured_chunk_size=9,
    )

    assert actual == 5


# 縮小後の最小チャンクサイズ
def test_reduced_strategy_does_not_return_zero(
) -> None:
    decision = build_test_decision(
        ADAPTIVE_STRATEGY_REDUCED_CHUNK
    )

    actual = resolve_adaptive_chunk_size(
        decision,
        configured_chunk_size=1,
    )

    assert actual == 1


# 単一字幕戦略のチャンクサイズ
def test_single_subtitle_strategy_returns_one(
) -> None:
    decision = build_test_decision(
        ADAPTIVE_STRATEGY_SINGLE_SUBTITLE
    )

    actual = resolve_adaptive_chunk_size(
        decision,
        configured_chunk_size=10,
    )

    assert actual == 1


# 不正な設定チャンクサイズ
@pytest.mark.parametrize(
    "configured_chunk_size",
    [
        0,
        -1,
    ],
)
def test_adaptive_chunk_size_rejects_non_positive_value(
    configured_chunk_size: int,
) -> None:
    decision = build_test_decision(
        ADAPTIVE_STRATEGY_STANDARD
    )

    with pytest.raises(
        ValueError,
        match=(
            "Configured chunk size must be "
            "greater than zero"
        ),
    ):
        resolve_adaptive_chunk_size(
            decision,
            configured_chunk_size=(
                configured_chunk_size
            ),
        )
