from __future__ import annotations

from datetime import datetime

import pytest
from lib.translation.translation_metrics import (
    AdaptiveChunkMetric,
    HybridGroupMetric,
    TRANSLATION_RESULT_CHINESE_FALLBACK_SUCCESS,
    TRANSLATION_RESULT_FAILED,
    TRANSLATION_RESULT_HYBRID_SUCCESS,
    TRANSLATION_RESULT_LEVEL_1_FALLBACK_SUCCESS,
    TRANSLATION_RESULT_PENDING,
    TRANSLATION_RESULT_STANDARD_SUCCESS,
    TranslationAttemptMetric,
    TranslationChunkMetric,
    TranslationSessionMetric,
    build_validation_reason_code,
    build_validation_reason_codes,
)


# Validation理由のコード変換
def test_build_validation_reason_code_removes_details(
) -> None:
    reason = (
        "Chinese-specific characters detected: "
        "subtitle_id='83', characters='这'"
    )

    actual = build_validation_reason_code(
        reason
    )

    assert actual == (
        "chinese_specific_characters_detected"
    )


def test_build_validation_reason_code_handles_level_3_error(
) -> None:
    reason = (
        "Translation evaluation tag value "
        "not found in source: "
        "subtitle_id='316', level=3"
    )

    actual = build_validation_reason_code(
        reason
    )

    assert actual == (
        "translation_evaluation_tag_value_"
        "not_found_in_source"
    )


def test_build_validation_reason_code_handles_empty_reason(
) -> None:
    actual = build_validation_reason_code(
        "   "
    )

    assert actual == (
        "unknown_validation_error"
    )


def test_build_validation_reason_codes_preserves_order_and_deduplicates(
) -> None:
    reasons = [
        (
            "Chinese-specific characters detected: "
            "subtitle_id='83'"
        ),
        (
            "Glossary violation: "
            "subtitle_id='84'"
        ),
        (
            "Chinese-specific characters detected: "
            "subtitle_id='85'"
        ),
    ]

    actual = build_validation_reason_codes(
        reasons
    )

    assert actual == (
        "chinese_specific_characters_detected",
        "glossary_violation",
    )


def test_build_validation_reason_codes_accepts_empty_reasons(
) -> None:
    actual = build_validation_reason_codes(
        []
    )

    assert actual == ()


# 通常翻訳試行の保持
def test_translation_attempt_metric_holds_standard_result(
) -> None:
    attempt = TranslationAttemptMetric(
        pipeline="standard",
        attempt=1,
        target_ids=(
            "1",
            "2",
        ),
        elapsed_seconds=12.5,
        response_received=True,
        validation_stage=(
            "standard_validation"
        ),
        validation_valid=True,
    )

    assert attempt.pipeline == "standard"
    assert attempt.attempt == 1

    assert attempt.target_ids == (
        "1",
        "2",
    )

    assert attempt.elapsed_seconds == 12.5
    assert attempt.response_received is True

    assert (
        attempt.validation_stage
        == "standard_validation"
    )

    assert attempt.validation_valid is True
    assert attempt.validation_reasons == ()
    assert attempt.reason_codes == ()
    assert attempt.exception_type is None
    assert attempt.exception_message is None


def test_translation_attempt_metric_holds_generation_exception(
) -> None:
    attempt = TranslationAttemptMetric(
        pipeline="hybrid",
        attempt=2,
        target_ids=(
            "316",
        ),
        elapsed_seconds=3.25,
        response_received=False,
        validation_stage=(
            "generation_exception"
        ),
        validation_valid=None,
        exception_type="RuntimeError",
        exception_message=(
            "Ollama request failed"
        ),
    )

    assert attempt.pipeline == "hybrid"
    assert attempt.response_received is False

    assert (
        attempt.validation_stage
        == "generation_exception"
    )

    assert attempt.validation_valid is None

    assert (
        attempt.exception_type
        == "RuntimeError"
    )

    assert (
        attempt.exception_message
        == "Ollama request failed"
    )


# Hybridグループの試行と結果
def test_hybrid_group_metric_accepts_hybrid_attempt(
) -> None:
    group = HybridGroupMetric(
        group_number=1,
        target_ids=(
            "315",
            "316",
        ),
        failed_ids=(
            "316",
        ),
    )

    attempt = TranslationAttemptMetric(
        pipeline="hybrid",
        attempt=1,
        target_ids=(
            "315",
            "316",
        ),
        elapsed_seconds=8.0,
        response_received=True,
        validation_stage=(
            "complete"
        ),
        validation_valid=True,
    )

    group.add_attempt(
        attempt
    )
    group.mark_success()

    assert group.attempts == [
        attempt,
    ]

    assert group.result == "success"


def test_hybrid_group_metric_rejects_standard_attempt(
) -> None:
    group = HybridGroupMetric(
        group_number=1,
        target_ids=(
            "316",
        ),
        failed_ids=(
            "316",
        ),
    )

    attempt = TranslationAttemptMetric(
        pipeline="standard",
        attempt=1,
        target_ids=(
            "316",
        ),
        elapsed_seconds=5.0,
        response_received=True,
        validation_stage=(
            "standard_validation"
        ),
        validation_valid=False,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Hybrid group only accepts "
            "hybrid attempts"
        ),
    ):
        group.add_attempt(
            attempt
        )

    assert group.attempts == []
    assert group.result == "pending"


def test_hybrid_group_metric_marks_failed(
) -> None:
    group = HybridGroupMetric(
        group_number=2,
        target_ids=(
            "320",
        ),
        failed_ids=(
            "320",
        ),
    )

    group.mark_failed()

    assert group.result == "failed"


# チャンクへの通常翻訳試行追加
def test_translation_chunk_metric_accepts_standard_attempt(
) -> None:
    chunk = TranslationChunkMetric(
        chunk_number=1,
        chunk_start=1,
        chunk_end=10,
        target_ids=(
            "1",
            "2",
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

    attempt = TranslationAttemptMetric(
        pipeline="standard",
        attempt=1,
        target_ids=(
            "1",
            "2",
        ),
        elapsed_seconds=10.0,
        response_received=True,
        validation_stage=(
            "standard_validation"
        ),
        validation_valid=True,
    )

    chunk.add_standard_attempt(
        attempt
    )

    assert chunk.standard_attempts == [
        attempt,
    ]

    assert (
        chunk.final_result
        == TRANSLATION_RESULT_PENDING
    )


def test_translation_chunk_metric_rejects_hybrid_as_standard_attempt(
) -> None:
    chunk = TranslationChunkMetric(
        chunk_number=1,
        chunk_start=1,
        chunk_end=10,
        target_ids=(
            "1",
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

    attempt = TranslationAttemptMetric(
        pipeline="hybrid",
        attempt=1,
        target_ids=(
            "1",
        ),
        elapsed_seconds=10.0,
        response_received=True,
        validation_stage=(
            "complete"
        ),
        validation_valid=True,
    )

    with pytest.raises(
        ValueError,
        match="Standard attempt required",
    ):
        chunk.add_standard_attempt(
            attempt
        )

    assert chunk.standard_attempts == []


# Hybrid移行理由の保持
def test_translation_chunk_metric_records_hybrid_trigger(
) -> None:
    chunk = TranslationChunkMetric(
        chunk_number=2,
        chunk_start=11,
        chunk_end=20,
        target_ids=(
            "11",
            "12",
        ),
        started_at=datetime(
            2026,
            7,
            19,
            12,
            1,
            0,
        ),
    )

    reasons = [
        (
            "Chinese-specific characters detected: "
            "subtitle_id='11'"
        ),
        (
            "Glossary violation: "
            "subtitle_id='12'"
        ),
    ]

    chunk.trigger_hybrid(
        reasons
    )

    assert chunk.hybrid_triggered is True

    assert chunk.hybrid_trigger_reasons == (
        reasons[0],
        reasons[1],
    )

    assert chunk.hybrid_trigger_codes == (
        "chinese_specific_characters_detected",
        "glossary_violation",
    )


# Hybridグループの追加と検索
def test_translation_chunk_metric_adds_and_finds_hybrid_group(
) -> None:
    chunk = TranslationChunkMetric(
        chunk_number=2,
        chunk_start=11,
        chunk_end=20,
        target_ids=(
            "11",
            "12",
        ),
        started_at=datetime(
            2026,
            7,
            19,
            12,
            1,
            0,
        ),
    )

    first_group = HybridGroupMetric(
        group_number=1,
        target_ids=(
            "11",
        ),
        failed_ids=(
            "11",
        ),
    )

    second_group = HybridGroupMetric(
        group_number=2,
        target_ids=(
            "12",
        ),
        failed_ids=(
            "12",
        ),
    )

    chunk.add_hybrid_group(
        first_group
    )
    chunk.add_hybrid_group(
        second_group
    )

    assert (
        chunk.find_hybrid_group(1)
        is first_group
    )

    assert (
        chunk.find_hybrid_group(2)
        is second_group
    )


def test_translation_chunk_metric_rejects_missing_hybrid_group(
) -> None:
    chunk = TranslationChunkMetric(
        chunk_number=1,
        chunk_start=1,
        chunk_end=10,
        target_ids=(
            "1",
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

    with pytest.raises(
        ValueError,
        match=(
            "Hybrid metrics group not found: "
            "group_number=3"
        ),
    ):
        chunk.find_hybrid_group(
            3
        )


# チャンクの最終結果
def test_translation_chunk_metric_completes_standard_success(
) -> None:
    chunk = TranslationChunkMetric(
        chunk_number=1,
        chunk_start=1,
        chunk_end=10,
        target_ids=(
            "1",
            "2",
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

    chunk.complete(
        final_result=(
            TRANSLATION_RESULT_STANDARD_SUCCESS
        ),
        elapsed_seconds=15.5,
    )

    assert (
        chunk.final_result
        == TRANSLATION_RESULT_STANDARD_SUCCESS
    )

    assert chunk.elapsed_seconds == 15.5
    assert chunk.failed_ids == ()


def test_translation_chunk_metric_accepts_fallback_and_hybrid_results(
) -> None:
    final_results = (
        (
            TRANSLATION_RESULT_LEVEL_1_FALLBACK_SUCCESS
        ),
        (
            TRANSLATION_RESULT_CHINESE_FALLBACK_SUCCESS
        ),
        TRANSLATION_RESULT_HYBRID_SUCCESS,
        TRANSLATION_RESULT_FAILED,
    )

    for final_result in final_results:
        chunk = TranslationChunkMetric(
            chunk_number=1,
            chunk_start=1,
            chunk_end=10,
            target_ids=(
                "1",
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

        chunk.complete(
            final_result=final_result,
            elapsed_seconds=20.0,
            failed_ids=(
                "1",
            ),
        )

        assert (
            chunk.final_result
            == final_result
        )

        assert chunk.elapsed_seconds == 20.0

        assert chunk.failed_ids == (
            "1",
        )


def test_translation_chunk_metric_rejects_pending_completion(
) -> None:
    chunk = TranslationChunkMetric(
        chunk_number=1,
        chunk_start=1,
        chunk_end=10,
        target_ids=(
            "1",
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

    with pytest.raises(
        ValueError,
        match=(
            "Pending result cannot complete "
            "translation chunk metrics"
        ),
    ):
        chunk.complete(
            final_result=(
                TRANSLATION_RESULT_PENDING
            ),
            elapsed_seconds=1.0,
        )

    assert (
        chunk.final_result
        == TRANSLATION_RESULT_PENDING
    )

    assert chunk.elapsed_seconds is None


def test_translation_chunk_metric_records_exception(
) -> None:
    chunk = TranslationChunkMetric(
        chunk_number=3,
        chunk_start=21,
        chunk_end=30,
        target_ids=(
            "21",
            "22",
        ),
        started_at=datetime(
            2026,
            7,
            19,
            12,
            2,
            0,
        ),
    )

    error = RuntimeError(
        "Ollama request failed"
    )

    chunk.fail_with_exception(
        error,
        elapsed_seconds=4.5,
        failed_ids=(
            "21",
            "22",
        ),
    )

    assert (
        chunk.final_result
        == TRANSLATION_RESULT_FAILED
    )

    assert chunk.elapsed_seconds == 4.5

    assert chunk.failed_ids == (
        "21",
        "22",
    )

    assert (
        chunk.exception_type
        == "RuntimeError"
    )

    assert (
        chunk.exception_message
        == "Ollama request failed"
    )


# 翻訳セッションのチャンクと処理時間
def test_translation_session_metric_adds_chunk_and_completes(
) -> None:
    session = TranslationSessionMetric(
        model="qwen3:14b",
        profile_name="stargate",
        output_name="episode.ja.srt",
        chunk_size=10,
        context_size=15,
        total_blocks=20,
        resume_start=0,
        started_at=datetime(
            2026,
            7,
            19,
            12,
            0,
            0,
        ),
    )

    chunk = TranslationChunkMetric(
        chunk_number=1,
        chunk_start=1,
        chunk_end=10,
        target_ids=(
            "1",
            "2",
        ),
        started_at=datetime(
            2026,
            7,
            19,
            12,
            0,
            1,
        ),
    )

    session.add_chunk(
        chunk
    )
    session.complete(
        elapsed_seconds=120.0
    )

    assert session.chunks == [
        chunk,
    ]

    assert session.elapsed_seconds == 120.0


# 適応チャンク計測の保持
def test_adaptive_chunk_metric_holds_applied_strategy(
) -> None:
    adaptive = AdaptiveChunkMetric(
        strategy="reduced_chunk",
        trigger="standard_retry",
        source_chunk_number=1,
        configured_chunk_size=10,
        applied_chunk_size=5,
        trigger_codes=(
            "glossary_violation",
        ),
    )

    assert adaptive.strategy == (
        "reduced_chunk"
    )

    assert adaptive.trigger == (
        "standard_retry"
    )

    assert adaptive.source_chunk_number == 1
    assert adaptive.configured_chunk_size == 10
    assert adaptive.applied_chunk_size == 5

    assert adaptive.trigger_codes == (
        "glossary_violation",
    )


# 初回チャンクの適応制御
def test_adaptive_chunk_metric_accepts_initial_chunk(
) -> None:
    adaptive = AdaptiveChunkMetric(
        strategy="standard",
        trigger="none",
        source_chunk_number=None,
        configured_chunk_size=10,
        applied_chunk_size=10,
    )

    assert adaptive.strategy == "standard"
    assert adaptive.trigger == "none"

    assert (
        adaptive.source_chunk_number
        is None
    )

    assert adaptive.trigger_codes == ()


# チャンク計測への適応制御記録
def test_translation_chunk_metric_records_adaptive_chunk(
) -> None:
    chunk = TranslationChunkMetric(
        chunk_number=2,
        chunk_start=11,
        chunk_end=15,
        target_ids=(
            "11",
            "12",
            "13",
            "14",
            "15",
        ),
        started_at=datetime(
            2026,
            7,
            19,
            12,
            1,
            0,
        ),
    )

    adaptive = AdaptiveChunkMetric(
        strategy="reduced_chunk",
        trigger="hybrid",
        source_chunk_number=1,
        configured_chunk_size=10,
        applied_chunk_size=5,
        trigger_codes=(
            "untranslated_english_sentence_detected",
        ),
    )

    assert chunk.adaptive is None

    chunk.record_adaptive_chunk(
        adaptive
    )

    assert chunk.adaptive is adaptive

    assert (
        chunk.adaptive.strategy
        == "reduced_chunk"
    )

    assert (
        chunk.adaptive.applied_chunk_size
        == 5
    )
