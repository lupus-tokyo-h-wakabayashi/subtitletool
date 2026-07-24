from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .translation_artifacts import (
    TranslationArtifactRegistry,
)
from .translation_metrics import (
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
)
from .translation_policy import (
    ADAPTIVE_STRATEGY_REDUCED_CHUNK,
    ADAPTIVE_STRATEGY_SINGLE_SUBTITLE,
    ADAPTIVE_STRATEGY_STANDARD,
    ADAPTIVE_TRIGGER_FAILED,
    ADAPTIVE_TRIGGER_HYBRID,
    ADAPTIVE_TRIGGER_NONE,
    ADAPTIVE_TRIGGER_STANDARD_RETRY,
)

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

TRANSLATION_METRICS_DIRECTORY = (
    PROJECT_ROOT
    / "tmp"
    / "translation-metrics"
)

TRANSLATION_METRICS_VERSION = 5

TRANSLATION_SESSION_RESULT_IN_PROGRESS = (
    "in_progress"
)

TRANSLATION_SESSION_RESULT_COMPLETED = (
    "completed"
)

TRANSLATION_SESSION_RESULT_COMPLETED_WITH_RECOVERY = (
    "completed_with_recovery"
)

TRANSLATION_SESSION_RESULT_FAILED = (
    "failed"
)


# 出力SRT名を計測ディレクトリ名へ使用できる形にする
def normalize_metrics_output_name(
    output_name: str,
) -> str:
    """
    出力SRT名から、計測ディレクトリへ使用する
    安全な名前を生成する。
    """
    output_stem = Path(
        output_name
    ).stem

    normalized_name = re.sub(
        r"[^A-Za-z0-9._-]+",
        "-",
        output_stem,
    ).strip(
        "-._"
    )

    if normalized_name:
        return normalized_name

    return "translation"


def build_translation_metrics_directory_name(
    session: TranslationSessionMetric,
) -> str:
    """
    セッション開始日時と出力名から
    実行単位のディレクトリ名を生成する。
    """
    timestamp = session.started_at.strftime(
        "%Y%m%d-%H%M%S-%f"
    )

    output_name = normalize_metrics_output_name(
        session.output_name
    )

    return (
        f"{timestamp}-"
        f"{output_name}"
    )


def build_translation_metrics_output_directory(
    session: TranslationSessionMetric,
    *,
    base_directory: Path | None = None,
) -> Path:
    """
    セッションの計測出力先を生成する。

    ディレクトリ作成は行わない。
    """
    resolved_base_directory = (
        base_directory
        if base_directory is not None
        else TRANSLATION_METRICS_DIRECTORY
    )

    return (
        resolved_base_directory
        / build_translation_metrics_directory_name(
        session
    )
    )


# 1試行をJSON形式へ変換する
def build_attempt_metrics_report(
    attempt: TranslationAttemptMetric,
) -> dict[str, object]:
    """
    通常翻訳またはHybridの1試行を
    JSON保存可能な辞書へ変換する。
    """
    return {
        "pipeline": attempt.pipeline,
        "attempt": attempt.attempt,
        "target_ids": list(
            attempt.target_ids
        ),
        "elapsed_seconds": (
            attempt.elapsed_seconds
        ),
        "response_received": (
            attempt.response_received
        ),
        "validation": {
            "stage": (
                attempt.validation_stage
            ),
            "valid": (
                attempt.validation_valid
            ),
            "reasons": list(
                attempt.validation_reasons
            ),
            "reason_codes": list(
                attempt.reason_codes
            ),
        },
        "exception": {
            "type": attempt.exception_type,
            "message": (
                attempt.exception_message
            ),
        },
    }


# HybridグループをJSON形式へ変換する
def build_hybrid_group_metrics_report(
    group: HybridGroupMetric,
) -> dict[str, object]:
    """
    Hybridグループと試行一覧を
    JSON保存可能な辞書へ変換する。
    """
    return {
        "group_number": (
            group.group_number
        ),
        "target_ids": list(
            group.target_ids
        ),
        "failed_ids": list(
            group.failed_ids
        ),
        "result": group.result,
        "attempt_count": len(
            group.attempts
        ),
        "attempts": [
            build_attempt_metrics_report(
                attempt
            )
            for attempt in group.attempts
        ],
    }


# 適応チャンクをJSON形式へ変換する
def build_adaptive_chunk_metrics_report(
    adaptive: AdaptiveChunkMetric,
) -> dict[str, object]:
    """
    現在のチャンクへ適用した
    適応制御をJSON保存可能な辞書へ変換する。
    """
    return {
        "strategy": adaptive.strategy,
        "trigger": adaptive.trigger,
        "source_chunk_number": (
            adaptive.source_chunk_number
        ),
        "configured_chunk_size": (
            adaptive.configured_chunk_size
        ),
        "applied_chunk_size": (
            adaptive.applied_chunk_size
        ),
        "trigger_codes": list(
            adaptive.trigger_codes
        ),
    }


# 1チャンクをJSON形式へ変換する
def build_chunk_metrics_report(
    chunk: TranslationChunkMetric,
) -> dict[str, object]:
    """
    1チャンクの通常翻訳・Fallback・Hybrid経路を
    JSON保存可能な辞書へ変換する。
    """
    hybrid_attempt_count = sum(
        len(group.attempts)
        for group in chunk.hybrid_groups
    )

    return {
        "version": (
            TRANSLATION_METRICS_VERSION
        ),
        "chunk": {
            "number": (
                chunk.chunk_number
            ),
            "start": chunk.chunk_start,
            "end": chunk.chunk_end,
            "target_ids": list(
                chunk.target_ids
            ),
            "started_at": (
                chunk.started_at.isoformat(
                    timespec="microseconds"
                )
            ),
            "elapsed_seconds": (
                chunk.elapsed_seconds
            ),
        },
        "adaptive": (
            build_adaptive_chunk_metrics_report(
                chunk.adaptive
            )
            if chunk.adaptive is not None
            else None
        ),
        "standard": {
            "attempt_count": len(
                chunk.standard_attempts
            ),
            "attempts": [
                build_attempt_metrics_report(
                    attempt
                )
                for attempt
                in chunk.standard_attempts
            ],
        },
        "hybrid": {
            "triggered": (
                chunk.hybrid_triggered
            ),
            "trigger_reasons": list(
                chunk.hybrid_trigger_reasons
            ),
            "trigger_codes": list(
                chunk.hybrid_trigger_codes
            ),
            "group_count": len(
                chunk.hybrid_groups
            ),
            "attempt_count": (
                hybrid_attempt_count
            ),
            "groups": [
                build_hybrid_group_metrics_report(
                    group
                )
                for group
                in chunk.hybrid_groups
            ],
        },
        "result": {
            "final_result": (
                chunk.final_result
            ),
            "failed_ids": list(
                chunk.failed_ids
            ),
        },
        "exception": {
            "type": chunk.exception_type,
            "message": (
                chunk.exception_message
            ),
        },
    }


def build_result_counts(
    session: TranslationSessionMetric,
) -> dict[str, int]:
    """
    チャンクの最終結果別件数を集計する。

    0件の結果もキーとして出力する。
    """
    result_counts = Counter(
        chunk.final_result
        for chunk in session.chunks
    )

    return {
        TRANSLATION_RESULT_PENDING: (
            result_counts[
                TRANSLATION_RESULT_PENDING
            ]
        ),
        TRANSLATION_RESULT_STANDARD_SUCCESS: (
            result_counts[
                TRANSLATION_RESULT_STANDARD_SUCCESS
            ]
        ),
        TRANSLATION_RESULT_LEVEL_1_FALLBACK_SUCCESS: (
            result_counts[
                TRANSLATION_RESULT_LEVEL_1_FALLBACK_SUCCESS
            ]
        ),
        TRANSLATION_RESULT_CHINESE_FALLBACK_SUCCESS: (
            result_counts[
                TRANSLATION_RESULT_CHINESE_FALLBACK_SUCCESS
            ]
        ),
        TRANSLATION_RESULT_HYBRID_SUCCESS: (
            result_counts[
                TRANSLATION_RESULT_HYBRID_SUCCESS
            ]
        ),
        TRANSLATION_RESULT_FAILED: (
            result_counts[
                TRANSLATION_RESULT_FAILED
            ]
        ),
    }


def build_validation_reason_counts(
    session: TranslationSessionMetric,
) -> dict[str, int]:
    """
    通常翻訳とHybridの全試行から、
    Validation理由コード別件数を集計する。

    Hybrid移行理由は最後の通常翻訳試行と
    重複するため、別途加算しない。
    """
    reason_counts: Counter[str] = (
        Counter()
    )

    for chunk in session.chunks:
        for attempt in (
            chunk.standard_attempts
        ):
            reason_counts.update(
                attempt.reason_codes
            )

        for group in chunk.hybrid_groups:
            for attempt in group.attempts:
                reason_counts.update(
                    attempt.reason_codes
                )

    return dict(
        sorted(
            reason_counts.items()
        )
    )


def build_adaptive_strategy_counts(
    session: TranslationSessionMetric,
) -> dict[str, int]:
    """
    適応情報が記録されたチャンクを
    適用戦略ごとに集計する。

    0件の戦略もキーとして出力する。
    """
    strategy_counts = Counter(
        chunk.adaptive.strategy
        for chunk in session.chunks
        if chunk.adaptive is not None
    )

    return {
        ADAPTIVE_STRATEGY_STANDARD: (
            strategy_counts[
                ADAPTIVE_STRATEGY_STANDARD
            ]
        ),
        ADAPTIVE_STRATEGY_REDUCED_CHUNK: (
            strategy_counts[
                ADAPTIVE_STRATEGY_REDUCED_CHUNK
            ]
        ),
        ADAPTIVE_STRATEGY_SINGLE_SUBTITLE: (
            strategy_counts[
                ADAPTIVE_STRATEGY_SINGLE_SUBTITLE
            ]
        ),
    }


def build_adaptive_trigger_counts(
    session: TranslationSessionMetric,
) -> dict[str, int]:
    """
    適応情報が記録されたチャンクを
    適応制御の発火理由ごとに集計する。

    0件の発火理由もキーとして出力する。
    """
    trigger_counts = Counter(
        chunk.adaptive.trigger
        for chunk in session.chunks
        if chunk.adaptive is not None
    )

    return {
        ADAPTIVE_TRIGGER_NONE: (
            trigger_counts[
                ADAPTIVE_TRIGGER_NONE
            ]
        ),
        ADAPTIVE_TRIGGER_STANDARD_RETRY: (
            trigger_counts[
                ADAPTIVE_TRIGGER_STANDARD_RETRY
            ]
        ),
        ADAPTIVE_TRIGGER_HYBRID: (
            trigger_counts[
                ADAPTIVE_TRIGGER_HYBRID
            ]
        ),
        ADAPTIVE_TRIGGER_FAILED: (
            trigger_counts[
                ADAPTIVE_TRIGGER_FAILED
            ]
        ),
    }


def build_translated_block_count(
    session: TranslationSessionMetric,
) -> int:
    """
    再開時点の翻訳済み字幕と、
    現在のセッションで成功した字幕を集計する。

    失敗または処理中のチャンクは、
    翻訳済み字幕数へ含めない。
    """
    translated_in_session = sum(
        len(
            chunk.target_ids
        )
        for chunk in session.chunks
        if chunk.final_result not in (
            TRANSLATION_RESULT_PENDING,
            TRANSLATION_RESULT_FAILED,
        )
    )

    return (
        session.resume_start
        + translated_in_session
    )


def translation_session_used_recovery(
    session: TranslationSessionMetric,
) -> bool:
    """
    セッション内で通常1回成功以外の
    回復経路が使用されたかを判定する。
    """
    recovery_results = (
        TRANSLATION_RESULT_LEVEL_1_FALLBACK_SUCCESS,
        TRANSLATION_RESULT_CHINESE_FALLBACK_SUCCESS,
        TRANSLATION_RESULT_HYBRID_SUCCESS,
        TRANSLATION_RESULT_FAILED,
    )

    return any(
        (
            chunk.final_result
            in recovery_results
        )
        or len(
            chunk.standard_attempts
        ) >= 2
        for chunk in session.chunks
    )


def build_translation_session_result(
    session: TranslationSessionMetric,
) -> str:
    """
    チャンク履歴と翻訳済み字幕数から、
    翻訳セッション全体の結果を判定する。

    最後のチャンクが失敗:
        セッションは失敗終了。

    未処理字幕が残っている:
        セッションは処理途中。

    全件処理済みで失敗履歴がある:
        適応回復を経て完了。

    全件処理済みで失敗履歴がない:
        通常完了。
    """
    if (
        session.chunks
        and session.chunks[-1].final_result
        == TRANSLATION_RESULT_FAILED
    ):
        return (
            TRANSLATION_SESSION_RESULT_FAILED
        )

    translated_block_count = (
        build_translated_block_count(
            session
        )
    )

    if (
        translated_block_count
        < session.total_blocks
    ):
        return (
            TRANSLATION_SESSION_RESULT_IN_PROGRESS
        )

    if translation_session_used_recovery(
        session
    ):
        return (
            TRANSLATION_SESSION_RESULT_COMPLETED_WITH_RECOVERY
        )

    return (
        TRANSLATION_SESSION_RESULT_COMPLETED
    )


def build_session_metrics_report(
    session: TranslationSessionMetric,
) -> dict[str, object]:
    """
    翻訳セッション全体を集計し、
    JSON保存可能な辞書へ変換する。
    """
    standard_attempt_count = sum(
        len(chunk.standard_attempts)
        for chunk in session.chunks
    )

    hybrid_group_count = sum(
        len(chunk.hybrid_groups)
        for chunk in session.chunks
    )

    hybrid_attempt_count = sum(
        len(group.attempts)
        for chunk in session.chunks
        for group in chunk.hybrid_groups
    )

    hybrid_triggered_chunk_count = sum(
        1
        for chunk in session.chunks
        if chunk.hybrid_triggered
    )

    completed_chunk_count = sum(
        1
        for chunk in session.chunks
        if (
            chunk.final_result
            != TRANSLATION_RESULT_PENDING
        )
    )

    adaptive_recorded_chunk_count = sum(
        1
        for chunk in session.chunks
        if chunk.adaptive is not None
    )

    return {
        "version": (
            TRANSLATION_METRICS_VERSION
        ),
        "session": {
            "model": session.model,
            "profile_name": (
                session.profile_name
            ),
            "output_name": (
                session.output_name
            ),
            "chunk_size": (
                session.chunk_size
            ),
            "context_size": (
                session.context_size
            ),
            "total_blocks": (
                session.total_blocks
            ),
            "resume_start": (
                session.resume_start
            ),
            "started_at": (
                session.started_at.isoformat(
                    timespec="microseconds"
                )
            ),
            "elapsed_seconds": (
                session.elapsed_seconds
            ),
        },
        "summary": {
            "session_result": (
                build_translation_session_result(
                    session
                )
            ),
            "translated_block_count": (
                build_translated_block_count(
                    session
                )
            ),
            "total_chunk_count": len(
                session.chunks
            ),
            "completed_chunk_count": (
                completed_chunk_count
            ),
            "standard_attempt_count": (
                standard_attempt_count
            ),
            "hybrid_triggered_chunk_count": (
                hybrid_triggered_chunk_count
            ),
            "hybrid_group_count": (
                hybrid_group_count
            ),
            "hybrid_attempt_count": (
                hybrid_attempt_count
            ),
            "adaptive_recorded_chunk_count": (
                adaptive_recorded_chunk_count
            ),
            "adaptive_strategy_counts": (
                build_adaptive_strategy_counts(
                    session
                )
            ),
            "adaptive_trigger_counts": (
                build_adaptive_trigger_counts(
                    session
                )
            ),
            "result_counts": (
                build_result_counts(
                    session
                )
            ),
            "validation_reason_counts": (
                build_validation_reason_counts(
                    session
                )
            ),
        },
        "chunks": [
            build_chunk_metrics_report(
                chunk
            )
            for chunk in session.chunks
        ],
    }


def build_chunk_metrics_filename(
    chunk: TranslationChunkMetric,
) -> str:
    """
    チャンクの字幕範囲から
    計測ファイル名を生成する。
    """
    return (
        "chunk-"
        f"{chunk.chunk_start:06d}-"
        f"{chunk.chunk_end:06d}.json"
    )


def write_metrics_report(
    output_path: Path,
    report: dict[str, object],
) -> None:
    """
    計測レポートをUTF-8 JSONとして保存する。
    """
    output_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def save_chunk_metrics_report(
    *,
    session: TranslationSessionMetric,
    chunk: TranslationChunkMetric,
    output_directory: Path | None = None,
    artifact_registry: (
        TranslationArtifactRegistry
        | None
    ) = None,
) -> Path:
    """
    1チャンクの計測レポートを保存する。
    """
    resolved_output_directory = (
        output_directory
        if output_directory is not None
        else (
            build_translation_metrics_output_directory(
                session
            )
        )
    )

    resolved_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if artifact_registry is not None:
        artifact_registry.register_directory(
            resolved_output_directory
        )

    output_path = (
        resolved_output_directory
        / build_chunk_metrics_filename(
        chunk
    )
    )

    write_metrics_report(
        output_path,
        build_chunk_metrics_report(
            chunk
        ),
    )

    if artifact_registry is not None:
        artifact_registry.register_file(
            output_path
        )

    return output_path


def save_session_metrics_report(
    *,
    session: TranslationSessionMetric,
    output_directory: Path | None = None,
    artifact_registry: (
        TranslationArtifactRegistry
        | None
    ) = None,
) -> Path:
    """
    セッションのsummary.jsonを保存する。
    """
    resolved_output_directory = (
        output_directory
        if output_directory is not None
        else (
            build_translation_metrics_output_directory(
                session
            )
        )
    )

    resolved_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if artifact_registry is not None:
        artifact_registry.register_directory(
            resolved_output_directory
        )

    output_path = (
        resolved_output_directory
        / "summary.json"
    )

    write_metrics_report(
        output_path,
        build_session_metrics_report(
            session
        ),
    )

    if artifact_registry is not None:
        artifact_registry.register_file(
            output_path
        )

    return output_path


def save_translation_metrics_reports(
    *,
    session: TranslationSessionMetric,
    chunk: TranslationChunkMetric,
    output_directory: Path | None = None,
    artifact_registry: (
        TranslationArtifactRegistry
        | None
    ) = None,
) -> tuple[Path, Path]:
    """
    現在のチャンクとセッションサマリーを保存する。
    """
    chunk_path = save_chunk_metrics_report(
        session=session,
        chunk=chunk,
        output_directory=output_directory,
        artifact_registry=artifact_registry,
    )

    summary_path = (
        save_session_metrics_report(
            session=session,
            output_directory=(
                output_directory
            ),
            artifact_registry=(
                artifact_registry
            ),
        )
    )

    return (
        chunk_path,
        summary_path,
    )


def try_save_translation_metrics_reports(
    *,
    session: TranslationSessionMetric,
    chunk: TranslationChunkMetric,
    output_directory: Path | None = None,
    artifact_registry: (
        TranslationArtifactRegistry
        | None
    ) = None,
) -> tuple[Path, Path] | None:
    """
    計測レポートをbest-effortで保存する。

    保存失敗は翻訳結果や再試行制御へ
    影響させない。
    """
    try:
        saved_paths = (
            save_translation_metrics_reports(
                session=session,
                chunk=chunk,
                output_directory=(
                    output_directory
                ),
                artifact_registry=(
                    artifact_registry
                ),
            )
        )
    except (
            OSError,
            TypeError,
            ValueError,
    ) as error:
        print(
            "Warning: Translation metrics "
            "could not be saved:"
        )
        print(
            "  "
            f"{type(error).__name__}: "
            f"{error}"
        )

        return None

    print(
        "Translation metrics saved:"
    )

    for saved_path in saved_paths:
        print(f"  {saved_path}")

    return saved_paths
