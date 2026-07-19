from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .translation_metrics import (
    TRANSLATION_RESULT_FAILED,
    TranslationChunkMetric,
)

ADAPTIVE_STRATEGY_STANDARD = (
    "standard"
)

ADAPTIVE_STRATEGY_REDUCED_CHUNK = (
    "reduced_chunk"
)

ADAPTIVE_STRATEGY_SINGLE_SUBTITLE = (
    "single_subtitle"
)

ADAPTIVE_TRIGGER_NONE = (
    "none"
)

ADAPTIVE_TRIGGER_STANDARD_RETRY = (
    "standard_retry"
)

ADAPTIVE_TRIGGER_HYBRID = (
    "hybrid"
)

ADAPTIVE_TRIGGER_FAILED = (
    "failed"
)

AdaptiveTranslationStrategy = Literal[
    "standard",
    "reduced_chunk",
    "single_subtitle",
]

AdaptiveTranslationTrigger = Literal[
    "none",
    "standard_retry",
    "hybrid",
    "failed",
]


@dataclass(frozen=True)
class AdaptiveTranslationDecision:
    """
    直前チャンクの計測結果から決定した、
    次チャンクの翻訳戦略を保持する。
    """

    strategy: AdaptiveTranslationStrategy
    trigger: AdaptiveTranslationTrigger
    source_chunk_number: int

    trigger_codes: tuple[
        str,
        ...
    ] = ()


def build_adaptive_translation_decision(
    chunk: TranslationChunkMetric,
) -> AdaptiveTranslationDecision:
    """
    直前チャンクの計測結果から、
    次チャンクの翻訳戦略を決定する。

    判定優先順位:

    1. 最終失敗
    2. Hybrid移行
    3. 通常翻訳の複数回試行
    4. 通常戦略を維持

    この関数は判定だけを行い、
    チャンクサイズや翻訳処理を変更しない。
    """
    if (
        chunk.final_result
        == TRANSLATION_RESULT_FAILED
    ):
        return AdaptiveTranslationDecision(
            strategy=(
                ADAPTIVE_STRATEGY_SINGLE_SUBTITLE
            ),
            trigger=(
                ADAPTIVE_TRIGGER_FAILED
            ),
            source_chunk_number=(
                chunk.chunk_number
            ),
            trigger_codes=(
                (
                    "translation_failed",
                )
            ),
        )

    if chunk.hybrid_triggered:
        return AdaptiveTranslationDecision(
            strategy=(
                ADAPTIVE_STRATEGY_REDUCED_CHUNK
            ),
            trigger=(
                ADAPTIVE_TRIGGER_HYBRID
            ),
            source_chunk_number=(
                chunk.chunk_number
            ),
            trigger_codes=(
                chunk.hybrid_trigger_codes
            ),
        )

    if len(
        chunk.standard_attempts
    ) >= 2:
        retry_codes: list[str] = []

        for attempt in chunk.standard_attempts:
            for reason_code in (
                attempt.reason_codes
            ):
                if reason_code in retry_codes:
                    continue

                retry_codes.append(
                    reason_code
                )

        return AdaptiveTranslationDecision(
            strategy=(
                ADAPTIVE_STRATEGY_REDUCED_CHUNK
            ),
            trigger=(
                ADAPTIVE_TRIGGER_STANDARD_RETRY
            ),
            source_chunk_number=(
                chunk.chunk_number
            ),
            trigger_codes=tuple(
                retry_codes
            ),
        )

    return AdaptiveTranslationDecision(
        strategy=(
            ADAPTIVE_STRATEGY_STANDARD
        ),
        trigger=(
            ADAPTIVE_TRIGGER_NONE
        ),
        source_chunk_number=(
            chunk.chunk_number
        ),
    )


def resolve_adaptive_chunk_size(
    decision: AdaptiveTranslationDecision,
    *,
    configured_chunk_size: int,
) -> int:
    """
    適応制御の判定結果から、
    次チャンクで使用する字幕数を決定する。

    standard:
        設定されたチャンクサイズを維持する。

    reduced_chunk:
        設定値の半分へ縮小する。
        奇数の場合は端数を切り上げる。

    single_subtitle:
        字幕1件だけを対象にする。
    """
    if configured_chunk_size <= 0:
        raise ValueError(
            "Configured chunk size must be "
            "greater than zero: "
            f"configured_chunk_size="
            f"{configured_chunk_size}"
        )

    if decision.strategy == (
        ADAPTIVE_STRATEGY_SINGLE_SUBTITLE
    ):
        return 1

    if decision.strategy == (
        ADAPTIVE_STRATEGY_REDUCED_CHUNK
    ):
        return max(
            1,
            (
                configured_chunk_size
                + 1
            )
            // 2,
        )

    return configured_chunk_size
