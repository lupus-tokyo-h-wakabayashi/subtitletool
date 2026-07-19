from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from lib.subtitle.srt import SrtBlock
from lib.translation import translation_chunk
from lib.translation.translation_metrics import (
    TRANSLATION_RESULT_CHINESE_FALLBACK_SUCCESS,
    TRANSLATION_RESULT_FAILED,
    TRANSLATION_RESULT_HYBRID_SUCCESS,
    TRANSLATION_RESULT_LEVEL_1_FALLBACK_SUCCESS,
    TRANSLATION_RESULT_PENDING,
    TRANSLATION_RESULT_STANDARD_SUCCESS,
    TranslationChunkMetric,
)
from lib.translation.translation_validation import (
    ValidationResult,
)
from .helpers import (
    build_test_noise_dictionary,
)


def build_target_block(
) -> SrtBlock:
    return SrtBlock(
        number="1",
        timestamp=(
            "00:00:01,000 --> "
            "00:00:02,000"
        ),
        text="Original text.",
    )


def build_metrics(
) -> TranslationChunkMetric:
    return TranslationChunkMetric(
        chunk_number=1,
        chunk_start=1,
        chunk_end=1,
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


def patch_common_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        translation_chunk,
        "extract_noise_candidates_from_blocks",
        lambda *args, **kwargs: [],
    )

    monkeypatch.setattr(
        translation_chunk,
        "find_noise_candidate_ids",
        lambda *args, **kwargs: [],
    )

    monkeypatch.setattr(
        translation_chunk,
        "append_noise_candidates",
        lambda *args, **kwargs: [],
    )

    monkeypatch.setattr(
        translation_chunk,
        "print_saved_noise_candidates",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        translation_chunk,
        "build_ocr_noise_instruction",
        lambda *args, **kwargs: "",
    )

    monkeypatch.setattr(
        translation_chunk,
        "build_required_glossary_instruction",
        lambda *args, **kwargs: "",
    )

    monkeypatch.setattr(
        translation_chunk,
        "build_initial_translation_prompt",
        lambda *args, **kwargs: "prompt",
    )

    monkeypatch.setattr(
        translation_chunk,
        "save_failed_translation_response",
        lambda *args, **kwargs: Path(
            "failed-response.txt"
        ),
    )

    monkeypatch.setattr(
        translation_chunk,
        "build_chinese_retry_blocks",
        lambda blocks, errors: blocks,
    )

    monkeypatch.setattr(
        translation_chunk,
        "build_latin_ocr_retry_blocks",
        lambda blocks, errors: blocks,
    )

    monkeypatch.setattr(
        translation_chunk,
        "build_retry_instruction",
        lambda *args, **kwargs: "",
    )

    monkeypatch.setattr(
        translation_chunk,
        "build_structural_retry_instruction",
        lambda *args, **kwargs: "",
    )

    monkeypatch.setattr(
        translation_chunk,
        "build_chinese_retry_instruction",
        lambda *args, **kwargs: "",
    )

    monkeypatch.setattr(
        translation_chunk,
        "build_latin_ocr_retry_instruction",
        lambda *args, **kwargs: "",
    )

    monkeypatch.setattr(
        translation_chunk,
        "build_untranslated_english_retry_instruction",
        lambda *args, **kwargs: "",
    )

    monkeypatch.setattr(
        translation_chunk,
        "build_glossary_retry_instruction",
        lambda *args, **kwargs: "",
    )

    monkeypatch.setattr(
        translation_chunk,
        "build_preserved_translations_instruction",
        lambda *args, **kwargs: "",
    )

    monkeypatch.setattr(
        translation_chunk,
        "find_probable_untranslated_ocr_lines",
        lambda *args, **kwargs: {},
    )

    monkeypatch.setattr(
        translation_chunk,
        "try_level_1_ocr_fallback",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        translation_chunk,
        "recover_translation_with_hybrid",
        lambda *args, **kwargs: None,
    )


def run_translate_chunk(
    *,
    noise_dictionary,
    metrics: TranslationChunkMetric | None,
) -> list[str]:
    return translation_chunk.translate_chunk(
        before_context=[],
        target_blocks=[
            build_target_block(),
        ],
        after_context=[],
        model="test-model",
        chunk_start=1,
        chunk_end=1,
        glossary_entries={},
        noise_dictionary=noise_dictionary,
        profile_name="test",
        metrics=metrics,
    )


# 1回目の通常翻訳成功
def test_standard_success_records_one_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_common_dependencies(
        monkeypatch
    )

    monkeypatch.setattr(
        translation_chunk,
        "generate_translation_response",
        lambda *args, **kwargs: "response",
    )

    monkeypatch.setattr(
        translation_chunk,
        "validate_translation_response",
        lambda *args, **kwargs: (
            ValidationResult(
                valid=True,
                translated_texts=[
                    "翻訳結果です。",
                ],
            )
        ),
    )

    metrics = build_metrics()

    actual = run_translate_chunk(
        noise_dictionary=(
            build_test_noise_dictionary(
                []
            )
        ),
        metrics=metrics,
    )

    assert actual == [
        "翻訳結果です。",
    ]

    assert len(
        metrics.standard_attempts
    ) == 1

    attempt = metrics.standard_attempts[0]

    assert attempt.pipeline == "standard"
    assert attempt.attempt == 1
    assert attempt.target_ids == ("1",)
    assert attempt.response_received is True
    assert attempt.validation_valid is True
    assert attempt.validation_reasons == ()
    assert attempt.reason_codes == ()
    assert attempt.elapsed_seconds >= 0

    assert (
        metrics.final_result
        == TRANSLATION_RESULT_STANDARD_SUCCESS
    )

    assert metrics.elapsed_seconds is not None
    assert metrics.elapsed_seconds >= 0


# 再試行後の通常翻訳成功
def test_standard_retry_records_each_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_common_dependencies(
        monkeypatch
    )

    monkeypatch.setattr(
        translation_chunk,
        "MAX_TRANSLATION_ATTEMPTS",
        2,
    )

    responses = iter(
        [
            "failed response",
            "successful response",
        ]
    )

    monkeypatch.setattr(
        translation_chunk,
        "generate_translation_response",
        lambda *args, **kwargs: next(
            responses
        ),
    )

    validations = iter(
        [
            ValidationResult(
                valid=False,
                reasons=[
                    (
                        "Invalid JSON response: "
                        "message=test"
                    ),
                ],
                translated_texts=[],
            ),
            ValidationResult(
                valid=True,
                translated_texts=[
                    "再試行で成功しました。",
                ],
            ),
        ]
    )

    monkeypatch.setattr(
        translation_chunk,
        "validate_translation_response",
        lambda *args, **kwargs: next(
            validations
        ),
    )

    metrics = build_metrics()

    actual = run_translate_chunk(
        noise_dictionary=(
            build_test_noise_dictionary(
                []
            )
        ),
        metrics=metrics,
    )

    assert actual == [
        "再試行で成功しました。",
    ]

    assert len(
        metrics.standard_attempts
    ) == 2

    first_attempt = (
        metrics.standard_attempts[0]
    )
    second_attempt = (
        metrics.standard_attempts[1]
    )

    assert first_attempt.attempt == 1
    assert first_attempt.validation_valid is False

    assert first_attempt.reason_codes == (
        "invalid_json_response",
    )

    assert second_attempt.attempt == 2
    assert second_attempt.validation_valid is True

    assert (
        metrics.final_result
        == TRANSLATION_RESULT_STANDARD_SUCCESS
    )


# LLM生成例外
def test_generation_exception_is_recorded_and_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_common_dependencies(
        monkeypatch
    )

    def raise_generation_error(
        *args,
        **kwargs,
    ) -> str:
        raise RuntimeError(
            "Ollama request failed"
        )

    monkeypatch.setattr(
        translation_chunk,
        "generate_translation_response",
        raise_generation_error,
    )

    metrics = build_metrics()

    with pytest.raises(
        RuntimeError,
        match="Ollama request failed",
    ):
        run_translate_chunk(
            noise_dictionary=(
                build_test_noise_dictionary(
                    []
                )
            ),
            metrics=metrics,
        )

    assert len(
        metrics.standard_attempts
    ) == 1

    attempt = metrics.standard_attempts[0]

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

    assert (
        metrics.final_result
        == TRANSLATION_RESULT_PENDING
    )


# Level 1 Fallback成功
def test_level_1_fallback_success_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_common_dependencies(
        monkeypatch
    )

    monkeypatch.setattr(
        translation_chunk,
        "MAX_TRANSLATION_ATTEMPTS",
        1,
    )

    monkeypatch.setattr(
        translation_chunk,
        "generate_translation_response",
        lambda *args, **kwargs: "response",
    )

    monkeypatch.setattr(
        translation_chunk,
        "validate_translation_response",
        lambda *args, **kwargs: (
            ValidationResult(
                valid=False,
                reasons=[
                    (
                        "Untranslated English sentence "
                        "detected: subtitle_id='1'"
                    ),
                ],
                translated_texts=[
                    "English remains.",
                ],
            )
        ),
    )

    monkeypatch.setattr(
        translation_chunk,
        "try_level_1_ocr_fallback",
        lambda *args, **kwargs: [
            "（判読不能）",
        ],
    )

    metrics = build_metrics()

    actual = run_translate_chunk(
        noise_dictionary=(
            build_test_noise_dictionary(
                []
            )
        ),
        metrics=metrics,
    )

    assert actual == [
        "（判読不能）",
    ]

    assert (
        metrics.final_result
        == TRANSLATION_RESULT_LEVEL_1_FALLBACK_SUCCESS
    )


# 中国語Fallback成功
def test_chinese_fallback_success_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_common_dependencies(
        monkeypatch
    )

    monkeypatch.setattr(
        translation_chunk,
        "MAX_TRANSLATION_ATTEMPTS",
        1,
    )

    monkeypatch.setattr(
        translation_chunk,
        "generate_translation_response",
        lambda *args, **kwargs: "response",
    )

    validations = iter(
        [
            ValidationResult(
                valid=False,
                reasons=[
                    (
                        "Chinese-specific characters "
                        "detected: subtitle_id='1'"
                    ),
                ],
                translated_texts=[
                    "这些人です。",
                ],
            ),
            ValidationResult(
                valid=True,
                translated_texts=[
                    "判読不能です。",
                ],
            ),
        ]
    )

    monkeypatch.setattr(
        translation_chunk,
        "validate_translation_response",
        lambda *args, **kwargs: next(
            validations
        ),
    )

    monkeypatch.setattr(
        translation_chunk,
        "mask_chinese_translation_errors",
        lambda *args, **kwargs: [
            "判読不能です。",
        ],
    )

    metrics = build_metrics()

    actual = run_translate_chunk(
        noise_dictionary=(
            build_test_noise_dictionary(
                []
            )
        ),
        metrics=metrics,
    )

    assert actual == [
        "判読不能です。",
    ]

    assert (
        metrics.final_result
        == TRANSLATION_RESULT_CHINESE_FALLBACK_SUCCESS
    )


# Hybrid成功
def test_hybrid_success_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_common_dependencies(
        monkeypatch
    )

    monkeypatch.setattr(
        translation_chunk,
        "MAX_TRANSLATION_ATTEMPTS",
        1,
    )

    monkeypatch.setattr(
        translation_chunk,
        "generate_translation_response",
        lambda *args, **kwargs: "response",
    )

    monkeypatch.setattr(
        translation_chunk,
        "validate_translation_response",
        lambda *args, **kwargs: (
            ValidationResult(
                valid=False,
                reasons=[
                    (
                        "Glossary violation: "
                        "subtitle_id='1'"
                    ),
                ],
                translated_texts=[
                    "誤訳です。",
                ],
            )
        ),
    )

    monkeypatch.setattr(
        translation_chunk,
        "recover_translation_with_hybrid",
        lambda *args, **kwargs: [
            "Hybrid翻訳です。",
        ],
    )

    metrics = build_metrics()

    actual = run_translate_chunk(
        noise_dictionary=(
            build_test_noise_dictionary(
                []
            )
        ),
        metrics=metrics,
    )

    assert actual == [
        "Hybrid翻訳です。",
    ]

    assert (
        metrics.final_result
        == TRANSLATION_RESULT_HYBRID_SUCCESS
    )

    # Hybridの実試行計測はPhase 1-7で追加する
    assert metrics.hybrid_triggered is False
    assert metrics.hybrid_groups == []


# Hybridを開始できない場合の最終失敗
def test_final_failure_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_common_dependencies(
        monkeypatch
    )

    monkeypatch.setattr(
        translation_chunk,
        "MAX_TRANSLATION_ATTEMPTS",
        1,
    )

    monkeypatch.setattr(
        translation_chunk,
        "generate_translation_response",
        lambda *args, **kwargs: "response",
    )

    monkeypatch.setattr(
        translation_chunk,
        "validate_translation_response",
        lambda *args, **kwargs: (
            ValidationResult(
                valid=False,
                reasons=[
                    (
                        "Glossary violation: "
                        "subtitle_id='1'"
                    ),
                ],
                translated_texts=[
                    "誤訳です。",
                ],
            )
        ),
    )

    metrics = build_metrics()

    with pytest.raises(
        RuntimeError,
        match=(
            "Translation failed after "
            "1 attempts"
        ),
    ):
        run_translate_chunk(
            noise_dictionary=(
                build_test_noise_dictionary(
                    []
                )
            ),
            metrics=metrics,
        )

    assert (
        metrics.final_result
        == TRANSLATION_RESULT_FAILED
    )

    assert metrics.failed_ids == (
        "1",
    )


# Hybrid回復例外
def test_hybrid_exception_is_recorded_and_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_common_dependencies(
        monkeypatch
    )

    monkeypatch.setattr(
        translation_chunk,
        "MAX_TRANSLATION_ATTEMPTS",
        1,
    )

    monkeypatch.setattr(
        translation_chunk,
        "generate_translation_response",
        lambda *args, **kwargs: "response",
    )

    monkeypatch.setattr(
        translation_chunk,
        "validate_translation_response",
        lambda *args, **kwargs: (
            ValidationResult(
                valid=False,
                reasons=[
                    (
                        "Glossary violation: "
                        "subtitle_id='1'"
                    ),
                ],
                translated_texts=[
                    "誤訳です。",
                ],
            )
        ),
    )

    def raise_hybrid_error(
        *args,
        **kwargs,
    ) -> list[str]:
        raise RuntimeError(
            "Hybrid recovery failed"
        )

    monkeypatch.setattr(
        translation_chunk,
        "recover_translation_with_hybrid",
        raise_hybrid_error,
    )

    metrics = build_metrics()

    with pytest.raises(
        RuntimeError,
        match="Hybrid recovery failed",
    ):
        run_translate_chunk(
            noise_dictionary=(
                build_test_noise_dictionary(
                    []
                )
            ),
            metrics=metrics,
        )

    assert (
        metrics.final_result
        == TRANSLATION_RESULT_FAILED
    )

    assert metrics.failed_ids == (
        "1",
    )

    assert (
        metrics.exception_type
        == "RuntimeError"
    )

    assert (
        metrics.exception_message
        == "Hybrid recovery failed"
    )


# metrics未指定時の後方互換
def test_translate_chunk_accepts_missing_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_common_dependencies(
        monkeypatch
    )

    monkeypatch.setattr(
        translation_chunk,
        "generate_translation_response",
        lambda *args, **kwargs: "response",
    )

    monkeypatch.setattr(
        translation_chunk,
        "validate_translation_response",
        lambda *args, **kwargs: (
            ValidationResult(
                valid=True,
                translated_texts=[
                    "翻訳結果です。",
                ],
            )
        ),
    )

    actual = run_translate_chunk(
        noise_dictionary=(
            build_test_noise_dictionary(
                []
            )
        ),
        metrics=None,
    )

    assert actual == [
        "翻訳結果です。",
    ]
