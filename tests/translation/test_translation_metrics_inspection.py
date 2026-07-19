from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from lib.translation import (
    translation_metrics_inspection,
)
from lib.translation.translation_metrics import (
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
)
from lib.translation.translation_metrics_inspection import (
    build_attempt_metrics_report,
    build_chunk_metrics_filename,
    build_chunk_metrics_report,
    build_hybrid_group_metrics_report,
    build_result_counts,
    build_session_metrics_report,
    build_translation_metrics_directory_name,
    build_translation_metrics_output_directory,
    build_validation_reason_counts,
    normalize_metrics_output_name,
    save_translation_metrics_reports,
    try_save_translation_metrics_reports,
)

STARTED_AT = datetime(
    2026,
    7,
    19,
    12,
    34,
    56,
    123456,
)


def make_standard_attempt(
    *,
    attempt: int = 1,
    validation_valid: bool = True,
    validation_reasons: tuple[
        str,
        ...
    ] = (),
    reason_codes: tuple[
        str,
        ...
    ] = (),
) -> TranslationAttemptMetric:
    return TranslationAttemptMetric(
        pipeline="standard",
        attempt=attempt,
        target_ids=(
            "1",
            "2",
        ),
        elapsed_seconds=10.5,
        response_received=True,
        validation_stage=(
            "standard_validation"
        ),
        validation_valid=(
            validation_valid
        ),
        validation_reasons=(
            validation_reasons
        ),
        reason_codes=reason_codes,
    )


def make_hybrid_attempt(
    *,
    attempt: int = 1,
    validation_stage: str = "complete",
    validation_valid: bool = True,
    validation_reasons: tuple[
        str,
        ...
    ] = (),
    reason_codes: tuple[
        str,
        ...
    ] = (),
) -> TranslationAttemptMetric:
    return TranslationAttemptMetric(
        pipeline="hybrid",
        attempt=attempt,
        target_ids=(
            "2",
        ),
        elapsed_seconds=7.25,
        response_received=True,
        validation_stage=validation_stage,
        validation_valid=(
            validation_valid
        ),
        validation_reasons=(
            validation_reasons
        ),
        reason_codes=reason_codes,
    )


def make_chunk(
    *,
    chunk_number: int = 1,
    chunk_start: int = 1,
    chunk_end: int = 10,
    final_result: str = (
        TRANSLATION_RESULT_STANDARD_SUCCESS
    ),
) -> TranslationChunkMetric:
    chunk = TranslationChunkMetric(
        chunk_number=chunk_number,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        target_ids=(
            str(chunk_start),
            str(chunk_start + 1),
        ),
        started_at=STARTED_AT,
    )

    chunk.complete(
        final_result=final_result,
        elapsed_seconds=20.5,
    )

    return chunk


def make_session(
) -> TranslationSessionMetric:
    return TranslationSessionMetric(
        model="qwen3:14b",
        profile_name="stargate",
        output_name="episode.ja.srt",
        chunk_size=10,
        context_size=15,
        total_blocks=20,
        resume_start=0,
        started_at=STARTED_AT,
        elapsed_seconds=120.5,
    )


# 出力名とディレクトリ名
def test_normalize_metrics_output_name(
) -> None:
    assert (
        normalize_metrics_output_name(
            "episode.ja.srt"
        )
        == "episode.ja"
    )

    assert (
        normalize_metrics_output_name(
            "episode name.ja.srt"
        )
        == "episode-name.ja"
    )

    assert (
        normalize_metrics_output_name(
            "日本語字幕.srt"
        )
        == "translation"
    )


def test_build_translation_metrics_directory_name(
) -> None:
    session = make_session()

    actual = (
        build_translation_metrics_directory_name(
            session
        )
    )

    assert actual == (
        "20260719-123456-123456-"
        "episode.ja"
    )


def test_build_translation_metrics_output_directory(
    tmp_path: Path,
) -> None:
    session = make_session()

    actual = (
        build_translation_metrics_output_directory(
            session,
            base_directory=tmp_path,
        )
    )

    assert actual == (
        tmp_path
        / (
            "20260719-123456-123456-"
            "episode.ja"
        )
    )

    assert not actual.exists()


# 試行レポート
def test_build_attempt_metrics_report(
) -> None:
    reason = (
        "Glossary violation: "
        "expected='デスティニー'"
    )

    attempt = make_standard_attempt(
        validation_valid=False,
        validation_reasons=(
            reason,
        ),
        reason_codes=(
            "glossary_violation",
        ),
    )

    report = build_attempt_metrics_report(
        attempt
    )

    assert report == {
        "pipeline": "standard",
        "attempt": 1,
        "target_ids": [
            "1",
            "2",
        ],
        "elapsed_seconds": 10.5,
        "response_received": True,
        "validation": {
            "stage": (
                "standard_validation"
            ),
            "valid": False,
            "reasons": [
                reason,
            ],
            "reason_codes": [
                "glossary_violation",
            ],
        },
        "exception": {
            "type": None,
            "message": None,
        },
    }


# Hybridグループレポート
def test_build_hybrid_group_metrics_report(
) -> None:
    group = HybridGroupMetric(
        group_number=1,
        target_ids=(
            "2",
        ),
        failed_ids=(
            "2",
        ),
    )

    attempt = make_hybrid_attempt()

    group.add_attempt(
        attempt
    )
    group.mark_success()

    report = (
        build_hybrid_group_metrics_report(
            group
        )
    )

    assert report["group_number"] == 1

    assert report["target_ids"] == [
        "2",
    ]

    assert report["failed_ids"] == [
        "2",
    ]

    assert report["result"] == "success"
    assert report["attempt_count"] == 1

    assert report["attempts"] == [
        build_attempt_metrics_report(
            attempt
        ),
    ]


# チャンクレポート
def test_build_chunk_metrics_report(
) -> None:
    chunk = make_chunk(
        final_result=(
            TRANSLATION_RESULT_HYBRID_SUCCESS
        )
    )

    standard_attempt = (
        make_standard_attempt(
            validation_valid=False,
            validation_reasons=(
                (
                    "Chinese-specific characters "
                    "detected: subtitle_id='2'"
                ),
            ),
            reason_codes=(
                (
                    "chinese_specific_"
                    "characters_detected"
                ),
            ),
        )
    )

    chunk.add_standard_attempt(
        standard_attempt
    )

    chunk.trigger_hybrid(
        standard_attempt.validation_reasons
    )

    group = HybridGroupMetric(
        group_number=1,
        target_ids=(
            "2",
        ),
        failed_ids=(
            "2",
        ),
    )

    group.add_attempt(
        make_hybrid_attempt()
    )
    group.mark_success()

    chunk.add_hybrid_group(
        group
    )

    report = build_chunk_metrics_report(
        chunk
    )

    assert report["version"] == 1

    assert report["chunk"] == {
        "number": 1,
        "start": 1,
        "end": 10,
        "target_ids": [
            "1",
            "2",
        ],
        "started_at": (
            "2026-07-19T12:34:56.123456"
        ),
        "elapsed_seconds": 20.5,
    }

    standard_report = report[
        "standard"
    ]

    assert isinstance(
        standard_report,
        dict,
    )

    assert (
        standard_report["attempt_count"]
        == 1
    )

    hybrid_report = report[
        "hybrid"
    ]

    assert isinstance(
        hybrid_report,
        dict,
    )

    assert hybrid_report["triggered"] is True
    assert hybrid_report["group_count"] == 1
    assert hybrid_report["attempt_count"] == 1

    result_report = report[
        "result"
    ]

    assert isinstance(
        result_report,
        dict,
    )

    assert (
        result_report["final_result"]
        == TRANSLATION_RESULT_HYBRID_SUCCESS
    )


# 最終結果別件数
def test_build_result_counts_includes_zero_results(
) -> None:
    session = make_session()

    session.add_chunk(
        make_chunk(
            final_result=(
                TRANSLATION_RESULT_STANDARD_SUCCESS
            )
        )
    )

    session.add_chunk(
        make_chunk(
            chunk_number=2,
            chunk_start=11,
            chunk_end=20,
            final_result=(
                TRANSLATION_RESULT_HYBRID_SUCCESS
            ),
        )
    )

    actual = build_result_counts(
        session
    )

    assert actual == {
        TRANSLATION_RESULT_PENDING: 0,
        TRANSLATION_RESULT_STANDARD_SUCCESS: 1,
        (
            TRANSLATION_RESULT_LEVEL_1_FALLBACK_SUCCESS
        ): 0,
        (
            TRANSLATION_RESULT_CHINESE_FALLBACK_SUCCESS
        ): 0,
        TRANSLATION_RESULT_HYBRID_SUCCESS: 1,
        TRANSLATION_RESULT_FAILED: 0,
    }


# Validation理由の集計
def test_build_validation_reason_counts_does_not_duplicate_hybrid_trigger(
) -> None:
    session = make_session()

    chunk = make_chunk(
        final_result=(
            TRANSLATION_RESULT_HYBRID_SUCCESS
        )
    )

    standard_attempt = (
        make_standard_attempt(
            validation_valid=False,
            validation_reasons=(
                (
                    "Chinese-specific characters "
                    "detected: subtitle_id='2'"
                ),
            ),
            reason_codes=(
                (
                    "chinese_specific_"
                    "characters_detected"
                ),
            ),
        )
    )

    chunk.add_standard_attempt(
        standard_attempt
    )

    chunk.trigger_hybrid(
        standard_attempt.validation_reasons
    )

    group = HybridGroupMetric(
        group_number=1,
        target_ids=(
            "2",
        ),
        failed_ids=(
            "2",
        ),
    )

    group.add_attempt(
        make_hybrid_attempt(
            validation_stage=(
                "hybrid_validation"
            ),
            validation_valid=False,
            validation_reasons=(
                "Glossary violation: id='2'",
            ),
            reason_codes=(
                "glossary_violation",
            ),
        )
    )

    group.add_attempt(
        make_hybrid_attempt(
            attempt=2,
        )
    )

    group.mark_success()

    chunk.add_hybrid_group(
        group
    )
    session.add_chunk(
        chunk
    )

    actual = (
        build_validation_reason_counts(
            session
        )
    )

    assert actual == {
        (
            "chinese_specific_"
            "characters_detected"
        ): 1,
        "glossary_violation": 1,
    }


# セッションサマリー
def test_build_session_metrics_report(
) -> None:
    session = make_session()

    standard_chunk = make_chunk(
        final_result=(
            TRANSLATION_RESULT_STANDARD_SUCCESS
        )
    )

    standard_chunk.add_standard_attempt(
        make_standard_attempt()
    )

    hybrid_chunk = make_chunk(
        chunk_number=2,
        chunk_start=11,
        chunk_end=20,
        final_result=(
            TRANSLATION_RESULT_HYBRID_SUCCESS
        ),
    )

    hybrid_chunk.add_standard_attempt(
        make_standard_attempt(
            validation_valid=False,
            reason_codes=(
                "glossary_violation",
            ),
        )
    )

    hybrid_chunk.trigger_hybrid(
        [
            (
                "Glossary violation: "
                "subtitle_id='12'"
            ),
        ]
    )

    group = HybridGroupMetric(
        group_number=1,
        target_ids=(
            "12",
        ),
        failed_ids=(
            "12",
        ),
    )

    group.add_attempt(
        make_hybrid_attempt()
    )
    group.mark_success()

    hybrid_chunk.add_hybrid_group(
        group
    )

    session.add_chunk(
        standard_chunk
    )
    session.add_chunk(
        hybrid_chunk
    )

    report = build_session_metrics_report(
        session
    )

    assert report["version"] == 1

    session_report = report[
        "session"
    ]

    assert isinstance(
        session_report,
        dict,
    )

    assert session_report["model"] == (
        "qwen3:14b"
    )

    assert (
        session_report["profile_name"]
        == "stargate"
    )

    assert (
        session_report["elapsed_seconds"]
        == 120.5
    )

    summary = report[
        "summary"
    ]

    assert isinstance(
        summary,
        dict,
    )

    assert summary[
               "total_chunk_count"
           ] == 2

    assert summary[
               "completed_chunk_count"
           ] == 2

    assert summary[
               "standard_attempt_count"
           ] == 2

    assert summary[
               "hybrid_triggered_chunk_count"
           ] == 1

    assert summary[
               "hybrid_group_count"
           ] == 1

    assert summary[
               "hybrid_attempt_count"
           ] == 1

    assert len(
        report["chunks"]
    ) == 2


# チャンクファイル名
def test_build_chunk_metrics_filename(
) -> None:
    chunk = make_chunk(
        chunk_start=1,
        chunk_end=10,
    )

    actual = build_chunk_metrics_filename(
        chunk
    )

    assert actual == (
        "chunk-000001-000010.json"
    )


# JSONファイル保存
def test_save_translation_metrics_reports(
    tmp_path: Path,
) -> None:
    session = make_session()

    chunk = make_chunk()
    chunk.add_standard_attempt(
        make_standard_attempt(
            validation_reasons=(
                (
                    "Glossary violation: "
                    "expected='デスティニー'"
                ),
            ),
            reason_codes=(
                "glossary_violation",
            ),
        )
    )

    session.add_chunk(
        chunk
    )

    output_directory = (
        tmp_path
        / "metrics"
    )

    saved_paths = (
        save_translation_metrics_reports(
            session=session,
            chunk=chunk,
            output_directory=(
                output_directory
            ),
        )
    )

    chunk_path, summary_path = (
        saved_paths
    )

    assert chunk_path == (
        output_directory
        / "chunk-000001-000010.json"
    )

    assert summary_path == (
        output_directory
        / "summary.json"
    )

    assert chunk_path.exists()
    assert summary_path.exists()

    chunk_report = json.loads(
        chunk_path.read_text(
            encoding="utf-8"
        )
    )

    summary_report = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        chunk_report["result"][
            "final_result"
        ]
        == TRANSLATION_RESULT_STANDARD_SUCCESS
    )

    assert (
        "デスティニー"
        in chunk_path.read_text(
        encoding="utf-8"
    )
    )

    assert (
        summary_report["summary"][
            "total_chunk_count"
        ]
        == 1
    )


# best-effort保存
def test_try_save_translation_metrics_reports_returns_none_on_io_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session = make_session()
    chunk = make_chunk()

    def raise_write_error(
        output_path: Path,
        report: dict[str, object],
    ) -> None:
        raise OSError(
            "write failed"
        )

    monkeypatch.setattr(
        translation_metrics_inspection,
        "write_metrics_report",
        raise_write_error,
    )

    result = (
        try_save_translation_metrics_reports(
            session=session,
            chunk=chunk,
        )
    )

    assert result is None

    captured = capsys.readouterr()

    assert (
        "Warning: Translation metrics "
        "could not be saved:"
        in captured.out
    )

    assert (
        "OSError: write failed"
        in captured.out
    )
