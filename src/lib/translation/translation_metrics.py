from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

# チャンクの最終処理結果
TRANSLATION_RESULT_PENDING = "pending"
TRANSLATION_RESULT_STANDARD_SUCCESS = (
    "standard_success"
)
TRANSLATION_RESULT_LEVEL_1_FALLBACK_SUCCESS = (
    "level_1_fallback_success"
)
TRANSLATION_RESULT_CHINESE_FALLBACK_SUCCESS = (
    "chinese_fallback_success"
)
TRANSLATION_RESULT_HYBRID_SUCCESS = (
    "hybrid_success"
)
TRANSLATION_RESULT_FAILED = "failed"

TranslationFinalResult = Literal[
    "pending",
    "standard_success",
    "level_1_fallback_success",
    "chinese_fallback_success",
    "hybrid_success",
    "failed",
]

TranslationPipeline = Literal[
    "standard",
    "hybrid",
]

TranslationValidationStage = Literal[
    "standard_validation",
    "hybrid_validation",
    "complete",
    "generation_exception",
]


# Validation理由を集計用コードへ変換する
def build_validation_reason_code(
    reason: str,
) -> str:
    """
    Validation理由を集計に使用できる
    snake_caseのコードへ変換する。

    詳細値は最初のコロン以降に含まれるため、
    エラー種別として先頭部分だけを使用する。
    """
    reason_name = (
        reason.split(
            ":",
            maxsplit=1,
        )[0]
        .strip()
        .lower()
    )

    reason_code = re.sub(
        r"[^a-z0-9]+",
        "_",
        reason_name,
    ).strip(
        "_"
    )

    if reason_code:
        return reason_code

    return "unknown_validation_error"


def build_validation_reason_codes(
    reasons: Sequence[str],
) -> tuple[str, ...]:
    """
    Validation理由一覧から、
    重複しない集計用コードを生成する。

    最初に出現した順序は維持する。
    """
    reason_codes: list[str] = []

    for reason in reasons:
        reason_code = (
            build_validation_reason_code(
                reason
            )
        )

        if reason_code in reason_codes:
            continue

        reason_codes.append(
            reason_code
        )

    return tuple(
        reason_codes
    )


# 通常翻訳またはHybridの1試行
@dataclass(frozen=True)
class TranslationAttemptMetric:
    """
    1回のLLM生成と検証結果を保持する。
    """

    pipeline: TranslationPipeline
    attempt: int
    target_ids: tuple[str, ...]
    elapsed_seconds: float
    response_received: bool
    validation_stage: TranslationValidationStage
    validation_valid: bool | None

    validation_reasons: tuple[
        str,
        ...
    ] = ()

    reason_codes: tuple[
        str,
        ...
    ] = ()

    exception_type: str | None = None
    exception_message: str | None = None


# Hybrid回復の1グループ
@dataclass
class HybridGroupMetric:
    """
    1つのHybrid回復グループと、
    そのグループに対する試行を保持する。
    """

    group_number: int
    target_ids: tuple[str, ...]
    failed_ids: tuple[str, ...]

    result: Literal[
        "pending",
        "success",
        "failed",
    ] = "pending"

    attempts: list[
        TranslationAttemptMetric
    ] = field(
        default_factory=list
    )

    def add_attempt(
        self,
        attempt: TranslationAttemptMetric,
    ) -> None:
        """
        Hybrid試行を追加する。
        """
        if attempt.pipeline != "hybrid":
            raise ValueError(
                "Hybrid group only accepts "
                "hybrid attempts: "
                f"pipeline={attempt.pipeline!r}"
            )

        self.attempts.append(
            attempt
        )

    def mark_success(self) -> None:
        """
        グループの回復成功を記録する。
        """
        self.result = "success"

    def mark_failed(self) -> None:
        """
        グループの回復失敗を記録する。
        """
        self.result = "failed"


# 1チャンクへ適用した適応制御
@dataclass(frozen=True)
class AdaptiveChunkMetric:
    """
    現在のチャンクへ適用した
    適応制御の内容を保持する。
    """

    strategy: str
    trigger: str
    source_chunk_number: int | None
    configured_chunk_size: int
    applied_chunk_size: int

    trigger_codes: tuple[
        str,
        ...
    ] = ()


# 1チャンクの翻訳経路
@dataclass
class TranslationChunkMetric:
    """
    通常翻訳開始からFallback・Hybridを含む
    1チャンクの処理経路を保持する。
    """

    chunk_number: int
    chunk_start: int
    chunk_end: int
    target_ids: tuple[str, ...]
    started_at: datetime

    elapsed_seconds: float | None = None

    adaptive: AdaptiveChunkMetric | None = None

    standard_attempts: list[
        TranslationAttemptMetric
    ] = field(
        default_factory=list
    )

    hybrid_triggered: bool = False

    hybrid_trigger_reasons: tuple[
        str,
        ...
    ] = ()

    hybrid_trigger_codes: tuple[
        str,
        ...
    ] = ()

    hybrid_groups: list[
        HybridGroupMetric
    ] = field(
        default_factory=list
    )

    final_result: TranslationFinalResult = (
        TRANSLATION_RESULT_PENDING
    )

    failed_ids: tuple[
        str,
        ...
    ] = ()

    exception_type: str | None = None
    exception_message: str | None = None

    def record_adaptive_chunk(
        self,
        adaptive: AdaptiveChunkMetric,
    ) -> None:
        """
        現在のチャンクへ適用した
        適応制御の内容を記録する。
        """
        self.adaptive = adaptive

    def add_standard_attempt(
        self,
        attempt: TranslationAttemptMetric,
    ) -> None:
        """
        通常翻訳の試行を追加する。
        """
        if attempt.pipeline != "standard":
            raise ValueError(
                "Standard attempt required: "
                f"pipeline={attempt.pipeline!r}"
            )

        self.standard_attempts.append(
            attempt
        )

    def trigger_hybrid(
        self,
        reasons: Sequence[str],
    ) -> None:
        """
        Hybridへ移行した事実と理由を記録する。
        """
        self.hybrid_triggered = True
        self.hybrid_trigger_reasons = tuple(
            reasons
        )
        self.hybrid_trigger_codes = (
            build_validation_reason_codes(
                reasons
            )
        )

    def add_hybrid_group(
        self,
        group: HybridGroupMetric,
    ) -> None:
        """
        Hybridグループを追加する。
        """
        self.hybrid_groups.append(
            group
        )

    def find_hybrid_group(
        self,
        group_number: int,
    ) -> HybridGroupMetric:
        """
        グループ番号に対応する
        Hybrid計測データを返す。
        """
        for group in self.hybrid_groups:
            if (
                group.group_number
                == group_number
            ):
                return group

        raise ValueError(
            "Hybrid metrics group not found: "
            f"group_number={group_number}"
        )

    def complete(
        self,
        *,
        final_result: TranslationFinalResult,
        elapsed_seconds: float,
        failed_ids: Sequence[str] = (),
    ) -> None:
        """
        チャンクの最終結果を確定する。
        """
        if final_result == (
            TRANSLATION_RESULT_PENDING
        ):
            raise ValueError(
                "Pending result cannot complete "
                "translation chunk metrics"
            )

        self.final_result = final_result
        self.elapsed_seconds = (
            elapsed_seconds
        )
        self.failed_ids = tuple(
            failed_ids
        )

    def fail_with_exception(
        self,
        error: Exception,
        *,
        elapsed_seconds: float,
        failed_ids: Sequence[str] = (),
    ) -> None:
        """
        例外で終了したチャンクの状態を記録する。
        """
        self.final_result = (
            TRANSLATION_RESULT_FAILED
        )
        self.elapsed_seconds = (
            elapsed_seconds
        )
        self.failed_ids = tuple(
            failed_ids
        )
        self.exception_type = (
            type(error).__name__
        )
        self.exception_message = str(
            error
        )


# 翻訳セッション全体
@dataclass
class TranslationSessionMetric:
    """
    1回のtranslate実行に含まれる
    全チャンクの計測結果を保持する。
    """

    model: str
    profile_name: str
    output_name: str
    chunk_size: int
    context_size: int
    total_blocks: int
    resume_start: int
    started_at: datetime

    elapsed_seconds: float | None = None

    chunks: list[
        TranslationChunkMetric
    ] = field(
        default_factory=list
    )

    def add_chunk(
        self,
        chunk: TranslationChunkMetric,
    ) -> None:
        """
        チャンク計測結果を追加する。
        """
        self.chunks.append(
            chunk
        )

    def complete(
        self,
        *,
        elapsed_seconds: float,
    ) -> None:
        """
        セッションの総処理時間を確定する。
        """
        self.elapsed_seconds = (
            elapsed_seconds
        )
