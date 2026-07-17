from __future__ import annotations

import re
from dataclasses import dataclass

from lib.subtitle.srt import SrtBlock

DEFAULT_MAX_HYBRID_GROUP_BLOCKS = 6

SENTENCE_END_PATTERN = re.compile(
    r"""[.!?]["'”’)]*\s*$"""
)


@dataclass(frozen=True)
class HybridTranslationGroup:
    """
    Hybrid Recoveryで全文翻訳する字幕グループ。

    positions:
        現在のチャンク内での位置。

    blocks:
        グループに含まれる字幕。

    failed_ids:
        通常翻訳でValidationに失敗した字幕ID。
    """

    positions: tuple[int, ...]
    blocks: tuple[SrtBlock, ...]
    failed_ids: frozenset[str]

    @property
    def target_ids(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            block.number
            for block in self.blocks
        )


def source_text_ends_sentence(
    text: str,
) -> bool:
    """
    原文が明確な英文末記号で終了しているか判定する。
    """
    normalized = text.strip()

    if not normalized:
        return False

    return bool(
        SENTENCE_END_PATTERN.search(
            normalized
        )
    )


def build_hybrid_translation_group(
    target_blocks: list[SrtBlock],
    failed_ids: set[str],
    *,
    maximum_blocks: int = (
        DEFAULT_MAX_HYBRID_GROUP_BLOCKS
    ),
) -> HybridTranslationGroup | None:
    """
    失敗字幕を含む連続文グループを構築する。

    Phase 1では現在のチャンク内だけを対象にする。

    複数の失敗IDがある場合は、
    それらをすべて含む1グループを作る。

    最大件数を超える場合や、
    対象IDが見つからない場合はNoneを返す。
    """
    if not target_blocks:
        return None

    if not failed_ids:
        return None

    if maximum_blocks < 1:
        raise ValueError(
            "maximum_blocks must be at least 1"
        )

    failed_positions = [
        position
        for position, block in enumerate(
            target_blocks
        )
        if block.number in failed_ids
    ]

    if not failed_positions:
        return None

    start = min(
        failed_positions
    )

    end = max(
        failed_positions
    )

    if (
        end - start + 1
        > maximum_blocks
    ):
        return None

    while start > 0:
        previous_position = start - 1

        if source_text_ends_sentence(
            target_blocks[
                previous_position
            ].text
        ):
            break

        if (
            end - previous_position + 1
            > maximum_blocks
        ):
            break

        start = previous_position

    while end + 1 < len(
        target_blocks
    ):
        if source_text_ends_sentence(
            target_blocks[end].text
        ):
            break

        next_position = end + 1

        if (
            next_position - start + 1
            > maximum_blocks
        ):
            break

        end = next_position

    positions = tuple(
        range(
            start,
            end + 1,
        )
    )

    blocks = tuple(
        target_blocks[position]
        for position in positions
    )

    return HybridTranslationGroup(
        positions=positions,
        blocks=blocks,
        failed_ids=frozenset(
            failed_ids
        ),
    )
