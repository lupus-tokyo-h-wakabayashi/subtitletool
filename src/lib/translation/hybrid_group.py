from __future__ import annotations

import re
from dataclasses import dataclass

from lib.subtitle.srt import SrtBlock

DEFAULT_MAX_HYBRID_GROUP_BLOCKS = 6

DEFAULT_MAX_HYBRID_GAP_MILLISECONDS = 1_500

SENTENCE_END_PATTERN = re.compile(
    r"""[.!?]["'”’)]*\s*$"""
)

SRT_TIMESTAMP_PATTERN = re.compile(
    r"""
    ^
    (?P<start_hour>\d{2}):
    (?P<start_minute>\d{2}):
    (?P<start_second>\d{2}),
    (?P<start_millisecond>\d{3})
    \s+-->\s+
    (?P<end_hour>\d{2}):
    (?P<end_minute>\d{2}):
    (?P<end_second>\d{2}),
    (?P<end_millisecond>\d{3})
    $
    """,
    re.VERBOSE,
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


def timestamp_part_to_milliseconds(
    *,
    hour: str,
    minute: str,
    second: str,
    millisecond: str,
) -> int:
    """
    SRTタイムスタンプの各要素を
    ミリ秒へ変換する。
    """
    return (
        int(hour) * 60 * 60 * 1_000
        + int(minute) * 60 * 1_000
        + int(second) * 1_000
        + int(millisecond)
    )


def parse_srt_timestamp_range(
    timestamp: str,
) -> tuple[int, int] | None:
    """
    SRTタイムスタンプを
    開始・終了ミリ秒へ変換する。

    解析できない場合や、
    終了時刻が開始時刻より前の場合は
    Noneを返す。
    """
    match = SRT_TIMESTAMP_PATTERN.fullmatch(
        timestamp.strip()
    )

    if match is None:
        return None

    values = match.groupdict()

    start = timestamp_part_to_milliseconds(
        hour=values["start_hour"],
        minute=values["start_minute"],
        second=values["start_second"],
        millisecond=values[
            "start_millisecond"
        ],
    )

    end = timestamp_part_to_milliseconds(
        hour=values["end_hour"],
        minute=values["end_minute"],
        second=values["end_second"],
        millisecond=values[
            "end_millisecond"
        ],
    )

    if end < start:
        return None

    return start, end


def subtitle_gap_milliseconds(
    previous_block: SrtBlock,
    next_block: SrtBlock,
) -> int | None:
    """
    連続する字幕間の時間差を返す。

    タイムスタンプを解析できない場合は
    Noneを返す。

    字幕時間が重複している場合は、
    0として扱う。
    """
    previous_range = (
        parse_srt_timestamp_range(
            previous_block.timestamp
        )
    )

    next_range = parse_srt_timestamp_range(
        next_block.timestamp
    )

    if (
        previous_range is None
        or next_range is None
    ):
        return None

    previous_end = previous_range[1]
    next_start = next_range[0]

    return max(
        0,
        next_start - previous_end,
    )


def crosses_hybrid_time_boundary(
    previous_block: SrtBlock,
    next_block: SrtBlock,
    *,
    maximum_gap_milliseconds: int,
) -> bool:
    """
    2字幕間がHybridグループの
    時間境界を越えているか判定する。

    タイムスタンプを解析できない場合は、
    無関係な会話を結合しないため
    境界として扱う。
    """
    gap = subtitle_gap_milliseconds(
        previous_block,
        next_block,
    )

    if gap is None:
        return True

    return (
        gap
        > maximum_gap_milliseconds
    )


def build_hybrid_translation_group(
    target_blocks: list[SrtBlock],
    failed_ids: set[str],
    *,
    maximum_blocks: int = (
        DEFAULT_MAX_HYBRID_GROUP_BLOCKS
    ),
    maximum_gap_milliseconds: int = (
        DEFAULT_MAX_HYBRID_GAP_MILLISECONDS
    ),
) -> HybridTranslationGroup | None:
    """
    失敗字幕を含む連続文グループを構築する。

    現在のチャンク内だけを対象にする。

    複数の失敗IDがある場合は、
    それらをすべて含む1グループを作る。

    次のいずれかを満たした位置で
    グループの拡張を停止する。

    - 明確な英文末記号がある
    - 字幕間隔が上限を超えている
    - タイムスタンプを解析できない
    - 最大字幕数へ到達した

    対象IDが見つからない場合や、
    失敗ID範囲だけで最大件数を超える場合は
    Noneを返す。
    """
    if not target_blocks:
        return None

    if not failed_ids:
        return None

    if maximum_blocks < 1:
        raise ValueError(
            "maximum_blocks must be at least 1"
        )

    if maximum_gap_milliseconds < 0:
        raise ValueError(
            "maximum_gap_milliseconds "
            "must not be negative"
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

    for position in range(
        start,
        end,
    ):
        if crosses_hybrid_time_boundary(
            target_blocks[position],
            target_blocks[position + 1],
            maximum_gap_milliseconds=(
                maximum_gap_milliseconds
            ),
        ):
            return None

    while start > 0:
        previous_position = start - 1

        previous_block = target_blocks[
            previous_position
        ]

        current_block = target_blocks[
            start
        ]

        if source_text_ends_sentence(
            previous_block.text
        ):
            break

        if crosses_hybrid_time_boundary(
            previous_block,
            current_block,
            maximum_gap_milliseconds=(
                maximum_gap_milliseconds
            ),
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
        current_block = target_blocks[
            end
        ]

        next_position = end + 1

        next_block = target_blocks[
            next_position
        ]

        if source_text_ends_sentence(
            current_block.text
        ):
            break

        if crosses_hybrid_time_boundary(
            current_block,
            next_block,
            maximum_gap_milliseconds=(
                maximum_gap_milliseconds
            ),
        ):
            break

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
