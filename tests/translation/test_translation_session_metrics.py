from __future__ import annotations

from pathlib import Path

import pytest
from lib.profile.config import ProfileConfig
from lib.subtitle.srt import SrtBlock
from lib.translation import translation_session
from lib.translation.translation_metrics import (
    TRANSLATION_RESULT_FAILED,
    TRANSLATION_RESULT_STANDARD_SUCCESS,
    TranslationAttemptMetric,
    TranslationChunkMetric,
    TranslationSessionMetric,
)
from lib.translation.translation_policy import (
    AdaptiveTranslationDecision,
)
from .helpers import (
    build_test_noise_dictionary,
)


def build_source_blocks(
) -> list[SrtBlock]:
    return [
        SrtBlock(
            number="1",
            timestamp=(
                "00:00:01,000 --> "
                "00:00:02,000"
            ),
            text="First subtitle.",
        ),
        SrtBlock(
            number="2",
            timestamp=(
                "00:00:02,100 --> "
                "00:00:03,000"
            ),
            text="Second subtitle.",
        ),
    ]


def build_profile_config(
    tmp_path: Path,
) -> ProfileConfig:
    profile_directory = (
        tmp_path
        / "profile"
    )

    return ProfileConfig(
        requested_profile="test",
        resolved_profile="test",
        profile_dir=profile_directory,
        prompt_path=(
            profile_directory
            / "prompt.txt"
        ),
        glossary_path=(
            profile_directory
            / "glossary.json"
        ),
        style_path=(
            profile_directory
            / "style.json"
        ),
        noise_path=(
            profile_directory
            / "noise.json"
        ),
        noise_local_path=(
            profile_directory
            / "noise.local.json"
        ),
        fallback_used=False,
    )


def patch_session_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # セッション計測テストでは
    # 翻訳以外の副作用を止める
    monkeypatch.setattr(
        translation_session,
        "load_glossary_entries",
        lambda *args, **kwargs: {},
    )

    monkeypatch.setattr(
        translation_session,
        "cleanup_blocks",
        lambda blocks: list(
            blocks
        ),
    )

    monkeypatch.setattr(
        translation_session,
        "apply_noise_to_blocks",
        lambda blocks, noise_dictionary: (
            list(
                blocks
            )
        ),
    )

    monkeypatch.setattr(
        translation_session,
        "write_structured_srt",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        translation_session,
        "print_translation_start",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        translation_session,
        "print_chunk_start",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        translation_session,
        "print_translation_progress",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        translation_session,
        "print_adaptive_translation_decision",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        translation_session,
        "print_translation_complete",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        translation_session,
        "print_translation_failed",
        lambda *args, **kwargs: None,
    )


# 正常終了チャンクの計測保存
def test_run_translation_session_saves_success_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_session_dependencies(
        monkeypatch
    )

    received_chunk_metrics: list[
        TranslationChunkMetric
    ] = []

    saved_metrics: list[
        tuple[
            TranslationSessionMetric,
            TranslationChunkMetric,
        ]
    ] = []

    displayed_failures: list[
        tuple[
            str,
            int,
            int,
            tuple[str, ...],
            str,
            str,
            Path | None,
        ]
    ] = []

    def fake_translate_chunk(
        *args: object,
        metrics: (
            TranslationChunkMetric
            | None
        ) = None,
        **kwargs: object,
    ) -> list[str]:
        assert metrics is not None

        received_chunk_metrics.append(
            metrics
        )

        metrics.complete(
            final_result=(
                TRANSLATION_RESULT_STANDARD_SUCCESS
            ),
            elapsed_seconds=1.0,
        )

        return [
            "1番目の翻訳です。",
            "2番目の翻訳です。",
        ]

    def fake_save_metrics(
        *,
        session: TranslationSessionMetric,
        chunk: TranslationChunkMetric,
        output_directory: Path | None = None,
    ) -> tuple[Path, Path]:
        del output_directory

        saved_metrics.append(
            (
                session,
                chunk,
            )
        )

        return (
            Path(
                "chunk-000001-000002.json"
            ),
            Path(
                "summary.json"
            ),
        )

    def fake_print_translation_failed(
        *,
        session_result: str,
        translated_count: int,
        total_blocks: int,
        failed_ids: tuple[str, ...],
        error: Exception,
        partial_output_path: Path | None,
    ) -> None:
        displayed_failures.append(
            (
                session_result,
                translated_count,
                total_blocks,
                failed_ids,
                type(error).__name__,
                str(
                    error
                ),
                partial_output_path,
            )
        )

    monkeypatch.setattr(
        translation_session,
        "translate_chunk",
        fake_translate_chunk,
    )

    monkeypatch.setattr(
        translation_session,
        "try_save_translation_metrics_reports",
        fake_save_metrics,
    )

    source_blocks = build_source_blocks()

    result = (
        translation_session.run_translation_session(
            source_blocks=source_blocks,
            translated_blocks_all=[],
            output_path=(
                tmp_path
                / "output.srt"
            ),
            model="test-model",
            chunk_size=2,
            context_size=1,
            profile_config=(
                build_profile_config(
                    tmp_path
                )
            ),
            noise_dictionary=(
                build_test_noise_dictionary(
                    []
                )
            ),
            inspect_request=False,
        )
    )

    assert result is None

    assert len(
        received_chunk_metrics
    ) == 1

    assert len(
        saved_metrics
    ) == 1

    session_metrics, chunk_metrics = (
        saved_metrics[0]
    )

    assert isinstance(
        session_metrics,
        TranslationSessionMetric,
    )

    assert isinstance(
        chunk_metrics,
        TranslationChunkMetric,
    )

    assert session_metrics.model == (
        "test-model"
    )

    assert (
        session_metrics.profile_name
        == "test"
    )

    assert (
        session_metrics.output_name
        == "output.srt"
    )

    assert session_metrics.chunk_size == 2
    assert session_metrics.context_size == 1
    assert session_metrics.total_blocks == 2
    assert session_metrics.resume_start == 0

    assert (
        session_metrics.elapsed_seconds
        is not None
    )

    assert (
        session_metrics.elapsed_seconds
        >= 0
    )

    assert len(
        session_metrics.chunks
    ) == 1

    assert (
        session_metrics.chunks[0]
        is chunk_metrics
    )

    assert chunk_metrics.chunk_number == 1
    assert chunk_metrics.chunk_start == 1
    assert chunk_metrics.chunk_end == 2

    assert chunk_metrics.target_ids == (
        "1",
        "2",
    )

    assert (
        chunk_metrics.final_result
        == TRANSLATION_RESULT_STANDARD_SUCCESS
    )

    assert (
        received_chunk_metrics[0]
        is chunk_metrics
    )


# 単一字幕の翻訳例外は計測保存後に再送出する
def test_run_translation_session_reraises_single_subtitle_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_session_dependencies(
        monkeypatch
    )

    saved_metrics: list[
        tuple[
            TranslationSessionMetric,
            TranslationChunkMetric,
        ]
    ] = []

    displayed_failures: list[
        tuple[
            str,
            int,
            int,
            tuple[str, ...],
            str,
            str,
            Path | None,
        ]
    ] = []

    def fake_translate_chunk(
        *args: object,
        metrics: (
            TranslationChunkMetric
            | None
        ) = None,
        **kwargs: object,
    ) -> list[str]:
        assert metrics is not None

        error = RuntimeError(
            "translation failed"
        )

        metrics.fail_with_exception(
            error,
            elapsed_seconds=1.0,
            failed_ids=(
                "1",
            ),
        )

        raise error

    def fake_save_metrics(
        *,
        session: TranslationSessionMetric,
        chunk: TranslationChunkMetric,
        output_directory: Path | None = None,
    ) -> tuple[Path, Path]:
        del output_directory

        saved_metrics.append(
            (
                session,
                chunk,
            )
        )

        return (
            Path(
                "chunk-000001-000002.json"
            ),
            Path(
                "summary.json"
            ),
        )

    def fake_print_translation_failed(
        *,
        session_result: str,
        translated_count: int,
        total_blocks: int,
        failed_ids: tuple[str, ...],
        error: Exception,
        partial_output_path: Path | None,
    ) -> None:
        displayed_failures.append(
            (
                session_result,
                translated_count,
                total_blocks,
                failed_ids,
                type(error).__name__,
                str(
                    error
                ),
                partial_output_path,
            )
        )

    monkeypatch.setattr(
        translation_session,
        "translate_chunk",
        fake_translate_chunk,
    )

    monkeypatch.setattr(
        translation_session,
        "try_save_translation_metrics_reports",
        fake_save_metrics,
    )

    monkeypatch.setattr(
        translation_session,
        "print_translation_failed",
        fake_print_translation_failed,
    )

    source_blocks = [
        build_source_blocks()[0],
    ]

    with pytest.raises(
        RuntimeError,
        match="translation failed",
    ):
        translation_session.run_translation_session(
            source_blocks=source_blocks,
            translated_blocks_all=[],
            output_path=(
                tmp_path
                / "output.srt"
            ),
            model="test-model",
            chunk_size=2,
            context_size=1,
            profile_config=(
                build_profile_config(
                    tmp_path
                )
            ),
            noise_dictionary=(
                build_test_noise_dictionary(
                    []
                )
            ),
            inspect_request=False,
        )

    assert len(
        saved_metrics
    ) == 1

    session_metrics, chunk_metrics = (
        saved_metrics[0]
    )

    assert (
        session_metrics.elapsed_seconds
        is not None
    )

    assert (
        session_metrics.elapsed_seconds
        >= 0
    )

    assert len(
        session_metrics.chunks
    ) == 1

    assert (
        session_metrics.chunks[0]
        is chunk_metrics
    )

    assert chunk_metrics.chunk_number == 1
    assert chunk_metrics.chunk_start == 1
    assert chunk_metrics.chunk_end == 1

    assert chunk_metrics.target_ids == (
        "1",
    )

    assert (
        chunk_metrics.final_result
        == TRANSLATION_RESULT_FAILED
    )

    assert chunk_metrics.failed_ids == (
        "1",
    )

    assert (
        chunk_metrics.exception_type
        == "RuntimeError"
    )

    assert (
        chunk_metrics.exception_message
        == "translation failed"
    )

    adaptive = chunk_metrics.adaptive

    assert adaptive is not None
    assert adaptive.strategy == "standard"
    assert adaptive.trigger == "none"

    assert (
        adaptive.source_chunk_number
        is None
    )

    assert adaptive.configured_chunk_size == 2
    assert adaptive.applied_chunk_size == 1
    assert adaptive.trigger_codes == ()

    assert displayed_failures == [
        (
            "failed",
            0,
            1,
            (
                "1",
            ),
            "RuntimeError",
            "translation failed",
            None,
        ),
    ]


# 複数字幕の失敗後は先頭から単一字幕で再処理する
def test_run_translation_session_retries_failed_group_individually(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_session_dependencies(
        monkeypatch
    )

    source_blocks = [
        SrtBlock(
            number=str(number),
            timestamp=(
                "00:00:01,000 --> "
                "00:00:02,000"
            ),
            text=(
                f"Subtitle {number}."
            ),
        )
        for number in range(
            1,
            7,
        )
    ]

    received_target_ids: list[
        tuple[str, ...]
    ] = []

    saved_chunks: list[
        TranslationChunkMetric
    ] = []

    displayed_chunks: list[
        tuple[
            int,
            int,
            int,
            int,
        ]
    ] = []

    def fake_translate_chunk(
        *args: object,
        metrics: (
            TranslationChunkMetric
            | None
        ) = None,
        **kwargs: object,
    ) -> list[str]:
        assert metrics is not None

        received_target_ids.append(
            metrics.target_ids
        )

        if metrics.chunk_number == 1:
            error = RuntimeError(
                "group translation failed"
            )

            metrics.fail_with_exception(
                error,
                elapsed_seconds=1.0,
                failed_ids=(
                    "1",
                    "2",
                    "3",
                    "4",
                ),
            )

            raise error

        metrics.add_standard_attempt(
            TranslationAttemptMetric(
                pipeline="standard",
                attempt=1,
                target_ids=(
                    metrics.target_ids
                ),
                elapsed_seconds=1.0,
                response_received=True,
                validation_stage=(
                    "standard_validation"
                ),
                validation_valid=True,
            )
        )

        metrics.complete(
            final_result=(
                TRANSLATION_RESULT_STANDARD_SUCCESS
            ),
            elapsed_seconds=1.0,
        )

        return [
            f"翻訳結果 {subtitle_id}"
            for subtitle_id
            in metrics.target_ids
        ]

    def fake_save_metrics(
        *,
        session: TranslationSessionMetric,
        chunk: TranslationChunkMetric,
        output_directory: Path | None = None,
    ) -> tuple[Path, Path]:
        del session
        del output_directory

        saved_chunks.append(
            chunk
        )

        return (
            Path(
                f"chunk-{chunk.chunk_number}.json"
            ),
            Path(
                "summary.json"
            ),
        )

    def fake_print_chunk_start(
        *,
        chunk_number: int,
        total_chunks: int,
        start: int,
        end: int,
        total_blocks: int,
        before_context_count: int,
        after_context_count: int,
    ) -> None:
        del total_blocks
        del before_context_count
        del after_context_count

        displayed_chunks.append(
            (
                chunk_number,
                total_chunks,
                start + 1,
                end,
            )
        )

    monkeypatch.setattr(
        translation_session,
        "translate_chunk",
        fake_translate_chunk,
    )

    monkeypatch.setattr(
        translation_session,
        "try_save_translation_metrics_reports",
        fake_save_metrics,
    )

    monkeypatch.setattr(
        translation_session,
        "print_chunk_start",
        fake_print_chunk_start,
    )

    translated_blocks: list[
        SrtBlock
    ] = []

    result = (
        translation_session.run_translation_session(
            source_blocks=source_blocks,
            translated_blocks_all=(
                translated_blocks
            ),
            output_path=(
                tmp_path
                / "output.srt"
            ),
            model="test-model",
            chunk_size=4,
            context_size=1,
            profile_config=(
                build_profile_config(
                    tmp_path
                )
            ),
            noise_dictionary=(
                build_test_noise_dictionary(
                    []
                )
            ),
            inspect_request=False,
        )
    )

    assert result is None

    assert received_target_ids == [
        (
            "1",
            "2",
            "3",
            "4",
        ),
        (
            "1",
        ),
        (
            "2",
        ),
        (
            "3",
        ),
        (
            "4",
        ),
        (
            "5",
            "6",
        ),
    ]

    assert [
               block.number
               for block in translated_blocks
           ] == [
               "1",
               "2",
               "3",
               "4",
               "5",
               "6",
           ]

    assert len(
        saved_chunks
    ) == 6

    failed_chunk = saved_chunks[0]

    retry_chunks = (
        saved_chunks[1:5]
    )

    resumed_chunk = saved_chunks[5]

    assert (
        failed_chunk.final_result
        == TRANSLATION_RESULT_FAILED
    )

    assert failed_chunk.target_ids == (
        "1",
        "2",
        "3",
        "4",
    )

    assert (
        failed_chunk.exception_message
        == "group translation failed"
    )

    assert [
               chunk.target_ids
               for chunk in retry_chunks
           ] == [
               (
                   "1",
               ),
               (
                   "2",
               ),
               (
                   "3",
               ),
               (
                   "4",
               ),
           ]

    for retry_chunk in retry_chunks:
        retry_adaptive = (
            retry_chunk.adaptive
        )

        assert retry_adaptive is not None

        assert (
            retry_adaptive.strategy
            == "single_subtitle"
        )

        assert (
            retry_adaptive.trigger
            == "failed"
        )

        assert (
            retry_adaptive
            .source_chunk_number
            == 1
        )

        assert (
            retry_adaptive.applied_chunk_size
            == 1
        )

        assert (
            retry_adaptive.trigger_codes
            == (
                "translation_failed",
            )
        )

    assert resumed_chunk.target_ids == (
        "5",
        "6",
    )

    resumed_adaptive = (
        resumed_chunk.adaptive
    )

    assert resumed_adaptive is not None

    assert (
        resumed_adaptive.strategy
        == "standard"
    )

    assert (
        resumed_adaptive.trigger
        == "none"
    )

    assert (
        resumed_adaptive
        .source_chunk_number
        == 5
    )

    assert (
        resumed_adaptive.configured_chunk_size
        == 4
    )

    assert (
        resumed_adaptive.applied_chunk_size
        == 2
    )

    # 最初は設定チャンクサイズ4により
    # 全6字幕を2チャンクで処理する予定
    assert displayed_chunks[0] == (
        1,
        2,
        1,
        4,
    )

    # 1〜4の失敗後は同じ開始位置から
    # 単一字幕6チャンクの計画へ変更する
    assert displayed_chunks[1] == (
        1,
        6,
        1,
        1,
    )

    assert displayed_chunks[2] == (
        2,
        6,
        2,
        2,
    )

    assert displayed_chunks[3] == (
        3,
        6,
        3,
        3,
    )

    assert displayed_chunks[4] == (
        4,
        6,
        4,
        4,
    )

    # 元の失敗範囲を処理した後は
    # 設定チャンクサイズへ戻る
    assert displayed_chunks[5] == (
        5,
        5,
        5,
        6,
    )

    assert all(
        chunk_number <= total_chunks
        for (
            chunk_number,
            total_chunks,
            _,
            _,
        ) in displayed_chunks
    )


# 単一字幕回復中の再失敗は
# 成功済み字幕を保存したまま再送出する
def test_run_translation_session_preserves_partial_recovery_before_reraising(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_session_dependencies(
        monkeypatch
    )

    source_blocks = [
        SrtBlock(
            number=str(number),
            timestamp=(
                "00:00:01,000 --> "
                "00:00:02,000"
            ),
            text=(
                f"Subtitle {number}."
            ),
        )
        for number in range(
            1,
            5,
        )
    ]

    received_target_ids: list[
        tuple[str, ...]
    ] = []

    saved_chunks: list[
        TranslationChunkMetric
    ] = []

    written_target_ids: list[
        tuple[str, ...]
    ] = []

    displayed_partial_outputs: list[
        Path | None
        ] = []

    def fake_translate_chunk(
        *args: object,
        metrics: (
            TranslationChunkMetric
            | None
        ) = None,
        **kwargs: object,
    ) -> list[str]:
        assert metrics is not None

        received_target_ids.append(
            metrics.target_ids
        )

        if metrics.chunk_number == 1:
            error = RuntimeError(
                "group translation failed"
            )

            metrics.fail_with_exception(
                error,
                elapsed_seconds=1.0,
                failed_ids=(
                    "1",
                    "2",
                    "3",
                    "4",
                ),
            )

            raise error

        if metrics.target_ids == (
                "2",
        ):
            error = RuntimeError(
                "single subtitle failed"
            )

            metrics.fail_with_exception(
                error,
                elapsed_seconds=1.0,
                failed_ids=(
                    "2",
                ),
            )

            raise error

        metrics.add_standard_attempt(
            TranslationAttemptMetric(
                pipeline="standard",
                attempt=1,
                target_ids=(
                    metrics.target_ids
                ),
                elapsed_seconds=1.0,
                response_received=True,
                validation_stage=(
                    "standard_validation"
                ),
                validation_valid=True,
            )
        )

        metrics.complete(
            final_result=(
                TRANSLATION_RESULT_STANDARD_SUCCESS
            ),
            elapsed_seconds=1.0,
        )

        return [
            f"翻訳結果 {subtitle_id}"
            for subtitle_id
            in metrics.target_ids
        ]

    def fake_save_metrics(
        *,
        session: TranslationSessionMetric,
        chunk: TranslationChunkMetric,
        output_directory: Path | None = None,
    ) -> tuple[Path, Path]:
        del session
        del output_directory

        saved_chunks.append(
            chunk
        )

        return (
            Path(
                f"chunk-{chunk.chunk_number}.json"
            ),
            Path(
                "summary.json"
            ),
        )

    def fake_write_structured_srt(
        output_path: Path,
        blocks: list[SrtBlock],
    ) -> None:
        written_target_ids.append(
            tuple(
                block.number
                for block in blocks
            )
        )

        output_path.write_text(
            "partial translation\n",
            encoding="utf-8",
        )

    def fake_print_translation_failed(
        *,
        session_result: str,
        translated_count: int,
        total_blocks: int,
        failed_ids: tuple[str, ...],
        error: Exception,
        partial_output_path: Path | None,
    ) -> None:
        del session_result
        del translated_count
        del total_blocks
        del failed_ids
        del error

        displayed_partial_outputs.append(
            partial_output_path
        )

    monkeypatch.setattr(
        translation_session,
        "translate_chunk",
        fake_translate_chunk,
    )

    monkeypatch.setattr(
        translation_session,
        "try_save_translation_metrics_reports",
        fake_save_metrics,
    )

    monkeypatch.setattr(
        translation_session,
        "write_structured_srt",
        fake_write_structured_srt,
    )

    monkeypatch.setattr(
        translation_session,
        "print_translation_failed",
        fake_print_translation_failed,
    )

    translated_blocks: list[
        SrtBlock
    ] = []

    with pytest.raises(
        RuntimeError,
        match="single subtitle failed",
    ):
        translation_session.run_translation_session(
            source_blocks=source_blocks,
            translated_blocks_all=(
                translated_blocks
            ),
            output_path=(
                tmp_path
                / "output.srt"
            ),
            model="test-model",
            chunk_size=4,
            context_size=1,
            profile_config=(
                build_profile_config(
                    tmp_path
                )
            ),
            noise_dictionary=(
                build_test_noise_dictionary(
                    []
                )
            ),
            inspect_request=False,
        )

    # 元チャンク失敗後、字幕1は成功し、
    # 字幕2の単一処理で停止する
    assert received_target_ids == [
        (
            "1",
            "2",
            "3",
            "4",
        ),
        (
            "1",
        ),
        (
            "2",
        ),
    ]

    # 成功済みの字幕1は
    # セッション内の結果へ残る
    assert [
               block.number
               for block in translated_blocks
           ] == [
               "1",
           ]

    # 字幕1の成功直後に
    # 途中保存が実行されている
    assert written_target_ids == [
        (
            "1",
        ),
    ]

    assert displayed_partial_outputs == [
        (
            tmp_path
            / "output.srt"
        ),
    ]

    assert (
        tmp_path
        / "output.srt"
    ).exists()

    # 元チャンク失敗、字幕1成功、
    # 字幕2失敗の3件が保存される
    assert len(
        saved_chunks
    ) == 3

    group_failure = saved_chunks[0]
    first_recovery = saved_chunks[1]
    single_failure = saved_chunks[2]

    assert (
        group_failure.final_result
        == TRANSLATION_RESULT_FAILED
    )

    assert group_failure.target_ids == (
        "1",
        "2",
        "3",
        "4",
    )

    assert (
        first_recovery.final_result
        == TRANSLATION_RESULT_STANDARD_SUCCESS
    )

    assert first_recovery.target_ids == (
        "1",
    )

    assert (
        single_failure.final_result
        == TRANSLATION_RESULT_FAILED
    )

    assert single_failure.target_ids == (
        "2",
    )

    assert single_failure.failed_ids == (
        "2",
    )

    assert (
        single_failure.exception_type
        == "RuntimeError"
    )

    assert (
        single_failure.exception_message
        == "single subtitle failed"
    )

    single_failure_adaptive = (
        single_failure.adaptive
    )

    assert single_failure_adaptive is not None

    assert (
        single_failure_adaptive.strategy
        == "single_subtitle"
    )

    assert (
        single_failure_adaptive.trigger
        == "failed"
    )

    assert (
        single_failure_adaptive
        .source_chunk_number
        == 1
    )

    assert (
        single_failure_adaptive
        .configured_chunk_size
        == 4
    )

    assert (
        single_failure_adaptive
        .applied_chunk_size
        == 1
    )

    assert (
        single_failure_adaptive.trigger_codes
        == (
            "translation_failed",
        )
    )


# 複数チャンクの計測保存
def test_run_translation_session_saves_each_chunk_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_session_dependencies(
        monkeypatch
    )

    source_blocks = [
        SrtBlock(
            number=str(number),
            timestamp=(
                "00:00:01,000 --> "
                "00:00:02,000"
            ),
            text=(
                f"Subtitle {number}."
            ),
        )
        for number in range(
            1,
            5,
        )
    ]

    received_metrics: list[
        TranslationChunkMetric
    ] = []

    saved_chunks: list[
        TranslationChunkMetric
    ] = []

    saved_sessions: list[
        TranslationSessionMetric
    ] = []

    def fake_translate_chunk(
        *args: object,
        metrics: (
            TranslationChunkMetric
            | None
        ) = None,
        **kwargs: object,
    ) -> list[str]:
        assert metrics is not None

        received_metrics.append(
            metrics
        )

        metrics.complete(
            final_result=(
                TRANSLATION_RESULT_STANDARD_SUCCESS
            ),
            elapsed_seconds=1.0,
        )

        return [
            f"翻訳結果 {subtitle_id}"
            for subtitle_id
            in metrics.target_ids
        ]

    def fake_save_metrics(
        *,
        session: TranslationSessionMetric,
        chunk: TranslationChunkMetric,
        output_directory: Path | None = None,
    ) -> tuple[Path, Path]:
        del output_directory

        saved_sessions.append(
            session
        )

        saved_chunks.append(
            chunk
        )

        return (
            Path(
                f"chunk-{chunk.chunk_number}.json"
            ),
            Path(
                "summary.json"
            ),
        )

    monkeypatch.setattr(
        translation_session,
        "translate_chunk",
        fake_translate_chunk,
    )

    monkeypatch.setattr(
        translation_session,
        "try_save_translation_metrics_reports",
        fake_save_metrics,
    )

    result = (
        translation_session.run_translation_session(
            source_blocks=source_blocks,
            translated_blocks_all=[],
            output_path=(
                tmp_path
                / "output.srt"
            ),
            model="test-model",
            chunk_size=2,
            context_size=1,
            profile_config=(
                build_profile_config(
                    tmp_path
                )
            ),
            noise_dictionary=(
                build_test_noise_dictionary(
                    []
                )
            ),
            inspect_request=False,
        )
    )

    assert result is None

    assert len(
        received_metrics
    ) == 2

    assert len(
        saved_chunks
    ) == 2

    assert len(
        saved_sessions
    ) == 2

    first_chunk = saved_chunks[0]
    second_chunk = saved_chunks[1]

    assert first_chunk.chunk_number == 1
    assert first_chunk.chunk_start == 1
    assert first_chunk.chunk_end == 2

    assert first_chunk.target_ids == (
        "1",
        "2",
    )

    assert second_chunk.chunk_number == 2
    assert second_chunk.chunk_start == 3
    assert second_chunk.chunk_end == 4

    assert second_chunk.target_ids == (
        "3",
        "4",
    )

    assert (
        saved_sessions[0]
        is saved_sessions[1]
    )

    session_metrics = saved_sessions[0]

    assert len(
        session_metrics.chunks
    ) == 2

    assert (
        session_metrics.chunks[0]
        is first_chunk
    )

    assert (
        session_metrics.chunks[1]
        is second_chunk
    )


# 再開位置からのチャンク計測
def test_run_translation_session_records_resume_position(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_session_dependencies(
        monkeypatch
    )

    source_blocks = [
        SrtBlock(
            number=str(number),
            timestamp=(
                "00:00:01,000 --> "
                "00:00:02,000"
            ),
            text=(
                f"Subtitle {number}."
            ),
        )
        for number in range(
            1,
            5,
        )
    ]

    translated_blocks_all = [
        SrtBlock(
            number="1",
            timestamp=(
                "00:00:01,000 --> "
                "00:00:02,000"
            ),
            text="翻訳済み1",
        ),
        SrtBlock(
            number="2",
            timestamp=(
                "00:00:01,000 --> "
                "00:00:02,000"
            ),
            text="翻訳済み2",
        ),
    ]

    saved_metrics: list[
        tuple[
            TranslationSessionMetric,
            TranslationChunkMetric,
        ]
    ] = []

    def fake_translate_chunk(
        *args: object,
        metrics: (
            TranslationChunkMetric
            | None
        ) = None,
        **kwargs: object,
    ) -> list[str]:
        assert metrics is not None

        metrics.complete(
            final_result=(
                TRANSLATION_RESULT_STANDARD_SUCCESS
            ),
            elapsed_seconds=1.0,
        )

        return [
            "翻訳結果3",
            "翻訳結果4",
        ]

    def fake_save_metrics(
        *,
        session: TranslationSessionMetric,
        chunk: TranslationChunkMetric,
        output_directory: Path | None = None,
    ) -> tuple[Path, Path]:
        del output_directory

        saved_metrics.append(
            (
                session,
                chunk,
            )
        )

        return (
            Path(
                "chunk-000003-000004.json"
            ),
            Path(
                "summary.json"
            ),
        )

    monkeypatch.setattr(
        translation_session,
        "translate_chunk",
        fake_translate_chunk,
    )

    monkeypatch.setattr(
        translation_session,
        "try_save_translation_metrics_reports",
        fake_save_metrics,
    )

    result = (
        translation_session.run_translation_session(
            source_blocks=source_blocks,
            translated_blocks_all=(
                translated_blocks_all
            ),
            output_path=(
                tmp_path
                / "output.srt"
            ),
            model="test-model",
            chunk_size=2,
            context_size=1,
            profile_config=(
                build_profile_config(
                    tmp_path
                )
            ),
            noise_dictionary=(
                build_test_noise_dictionary(
                    []
                )
            ),
            inspect_request=False,
        )
    )

    assert result is None

    assert len(
        saved_metrics
    ) == 1

    session_metrics, chunk_metrics = (
        saved_metrics[0]
    )

    assert session_metrics.resume_start == 2

    assert chunk_metrics.chunk_number == 1
    assert chunk_metrics.chunk_start == 3
    assert chunk_metrics.chunk_end == 4

    assert chunk_metrics.target_ids == (
        "3",
        "4",
    )


# 通常再試行後に次チャンクを縮小する
def test_run_translation_session_applies_adaptive_chunk_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_session_dependencies(
        monkeypatch
    )

    source_blocks = [
        SrtBlock(
            number=str(number),
            timestamp=(
                "00:00:01,000 --> "
                "00:00:02,000"
            ),
            text=(
                f"Subtitle {number}."
            ),
        )
        for number in range(
            1,
            9,
        )
    ]

    received_target_ids: list[
        tuple[str, ...]
    ] = []

    saved_chunks: list[
        TranslationChunkMetric
    ] = []

    displayed_decisions: list[
        tuple[
            str,
            str,
            int | None,
            int,
            tuple[str, ...],
        ]
    ] = []

    def fake_translate_chunk(
        *args: object,
        metrics: (
            TranslationChunkMetric
            | None
        ) = None,
        **kwargs: object,
    ) -> list[str]:
        assert metrics is not None

        received_target_ids.append(
            metrics.target_ids
        )

        if metrics.chunk_number == 1:
            metrics.add_standard_attempt(
                TranslationAttemptMetric(
                    pipeline="standard",
                    attempt=1,
                    target_ids=(
                        metrics.target_ids
                    ),
                    elapsed_seconds=1.0,
                    response_received=True,
                    validation_stage=(
                        "standard_validation"
                    ),
                    validation_valid=False,
                    validation_reasons=(
                        (
                            "Glossary violation: "
                            "subtitle_id='1'"
                        ),
                    ),
                    reason_codes=(
                        "glossary_violation",
                    ),
                )
            )

            metrics.add_standard_attempt(
                TranslationAttemptMetric(
                    pipeline="standard",
                    attempt=2,
                    target_ids=(
                        metrics.target_ids
                    ),
                    elapsed_seconds=1.0,
                    response_received=True,
                    validation_stage=(
                        "standard_validation"
                    ),
                    validation_valid=True,
                )
            )
        else:
            metrics.add_standard_attempt(
                TranslationAttemptMetric(
                    pipeline="standard",
                    attempt=1,
                    target_ids=(
                        metrics.target_ids
                    ),
                    elapsed_seconds=1.0,
                    response_received=True,
                    validation_stage=(
                        "standard_validation"
                    ),
                    validation_valid=True,
                )
            )

        metrics.complete(
            final_result=(
                TRANSLATION_RESULT_STANDARD_SUCCESS
            ),
            elapsed_seconds=1.0,
        )

        return [
            f"翻訳結果 {subtitle_id}"
            for subtitle_id
            in metrics.target_ids
        ]

    def fake_save_metrics(
        *,
        session: TranslationSessionMetric,
        chunk: TranslationChunkMetric,
        output_directory: Path | None = None,
    ) -> tuple[Path, Path]:
        del session
        del output_directory

        saved_chunks.append(
            chunk
        )

        return (
            Path(
                f"chunk-{chunk.chunk_number}.json"
            ),
            Path(
                "summary.json"
            ),
        )

    def fake_print_adaptive_translation_decision(
        *,
        decision: AdaptiveTranslationDecision,
        next_chunk_size: int,
    ) -> None:
        displayed_decisions.append(
            (
                decision.strategy,
                decision.trigger,
                decision.source_chunk_number,
                next_chunk_size,
                decision.trigger_codes,
            )
        )

    monkeypatch.setattr(
        translation_session,
        "translate_chunk",
        fake_translate_chunk,
    )

    monkeypatch.setattr(
        translation_session,
        "try_save_translation_metrics_reports",
        fake_save_metrics,
    )

    monkeypatch.setattr(
        translation_session,
        "print_adaptive_translation_decision",
        fake_print_adaptive_translation_decision,
    )

    result = (
        translation_session.run_translation_session(
            source_blocks=source_blocks,
            translated_blocks_all=[],
            output_path=(
                tmp_path
                / "output.srt"
            ),
            model="test-model",
            chunk_size=4,
            context_size=1,
            profile_config=(
                build_profile_config(
                    tmp_path
                )
            ),
            noise_dictionary=(
                build_test_noise_dictionary(
                    []
                )
            ),
            inspect_request=False,
        )
    )

    assert result is None

    assert received_target_ids == [
        (
            "1",
            "2",
            "3",
            "4",
        ),
        (
            "5",
            "6",
        ),
        (
            "7",
            "8",
        ),
    ]

    assert len(
        saved_chunks
    ) == 3

    assert [
               chunk.chunk_number
               for chunk in saved_chunks
           ] == [
               1,
               2,
               3,
           ]

    assert [
               (
                   chunk.chunk_start,
                   chunk.chunk_end,
               )
               for chunk in saved_chunks
           ] == [
               (
                   1,
                   4,
               ),
               (
                   5,
                   6,
               ),
               (
                   7,
                   8,
               ),
           ]

    first_adaptive = (
        saved_chunks[0].adaptive
    )

    second_adaptive = (
        saved_chunks[1].adaptive
    )

    third_adaptive = (
        saved_chunks[2].adaptive
    )

    assert first_adaptive is not None
    assert second_adaptive is not None
    assert third_adaptive is not None

    # 初回チャンクは設定値を使用する
    assert first_adaptive.strategy == (
        "standard"
    )

    assert first_adaptive.trigger == (
        "none"
    )

    assert (
        first_adaptive.source_chunk_number
        is None
    )

    assert (
        first_adaptive.configured_chunk_size
        == 4
    )

    assert (
        first_adaptive.applied_chunk_size
        == 4
    )

    assert first_adaptive.trigger_codes == ()

    # 1チャンク目の再試行により縮小する
    assert second_adaptive.strategy == (
        "reduced_chunk"
    )

    assert second_adaptive.trigger == (
        "standard_retry"
    )

    assert (
        second_adaptive.source_chunk_number
        == 1
    )

    assert (
        second_adaptive.configured_chunk_size
        == 4
    )

    assert (
        second_adaptive.applied_chunk_size
        == 2
    )

    assert second_adaptive.trigger_codes == (
        "glossary_violation",
    )

    # 2チャンク目の通常成功により
    # 設定サイズへ戻る
    assert third_adaptive.strategy == (
        "standard"
    )

    assert third_adaptive.trigger == (
        "none"
    )

    assert (
        third_adaptive.source_chunk_number
        == 2
    )

    assert (
        third_adaptive.configured_chunk_size
        == 4
    )

    # 設定サイズは4だが、
    # 残り字幕数は2件
    assert (
        third_adaptive.applied_chunk_size
        == 2
    )

    assert third_adaptive.trigger_codes == ()

    # 1チャンク目の再試行後は
    # 次チャンクの縮小を表示する
    assert displayed_decisions[0] == (
        "reduced_chunk",
        "standard_retry",
        1,
        2,
        (
            "glossary_violation",
        ),
    )

    # 2チャンク目の通常成功後は
    # 設定サイズへ戻る判断を渡す
    assert displayed_decisions[1] == (
        "standard",
        "none",
        2,
        4,
        (),
    )

    # 最終チャンク終了後は
    # 次チャンクの表示を呼び出さない
    assert len(
        displayed_decisions
    ) == 2
