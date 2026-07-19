from __future__ import annotations

from pathlib import Path

import pytest
from lib.profile.config import ProfileConfig
from lib.subtitle.srt import SrtBlock
from lib.translation import translation_session
from lib.translation.translation_metrics import (
    TRANSLATION_RESULT_FAILED,
    TRANSLATION_RESULT_STANDARD_SUCCESS,
    TranslationChunkMetric,
    TranslationSessionMetric,
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
        "print_translation_complete",
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


# 翻訳例外時の計測保存
def test_run_translation_session_saves_failed_metrics_before_reraising(
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
                "2",
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
    assert chunk_metrics.chunk_end == 2

    assert chunk_metrics.target_ids == (
        "1",
        "2",
    )

    assert (
        chunk_metrics.final_result
        == TRANSLATION_RESULT_FAILED
    )

    assert chunk_metrics.failed_ids == (
        "1",
        "2",
    )

    assert (
        chunk_metrics.exception_type
        == "RuntimeError"
    )

    assert (
        chunk_metrics.exception_message
        == "translation failed"
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
