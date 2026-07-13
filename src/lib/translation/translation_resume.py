from __future__ import annotations

from pathlib import Path

from lib.srt import (
    SrtBlock,
    parse_srt,
)


def validate_resume_blocks(
    source_blocks: list[SrtBlock],
    translated_blocks: list[SrtBlock],
) -> None:
    """
    途中保存されたSRTが、
    入力SRTの先頭部分と一致するか確認する。

    本文は翻訳後なので比較しない。
    字幕番号とタイムコードだけを比較する。
    """
    if len(translated_blocks) > len(source_blocks):
        raise RuntimeError(
            "Resume failed: "
            "output SRT contains more subtitles "
            "than input SRT. "
            f"input={len(source_blocks)}, "
            f"output={len(translated_blocks)}"
        )

    for index, translated_block in enumerate(
        translated_blocks
    ):
        source_block = source_blocks[index]

        if (
            translated_block.number
            != source_block.number
        ):
            raise RuntimeError(
                "Resume failed: subtitle number mismatch "
                f"at position {index + 1}. "
                f"input={source_block.number}, "
                f"output={translated_block.number}"
            )

        if (
            translated_block.timestamp
            != source_block.timestamp
        ):
            raise RuntimeError(
                "Resume failed: timestamp mismatch "
                f"at subtitle {source_block.number}. "
                f"input={source_block.timestamp!r}, "
                f"output={translated_block.timestamp!r}"
            )


def load_resume_blocks(
    source_blocks: list[SrtBlock],
    output_path: Path,
) -> list[SrtBlock]:
    """
    途中保存済みの出力SRTを読み込む。

    出力ファイルが存在しない場合は空リストを返す。

    出力ファイルが存在する場合は、
    SRTとして読み込めることと、
    入力SRTの先頭部分に対応していることを検証する。
    """
    if not output_path.exists():
        return []

    translated_blocks = parse_srt(
        output_path
    )

    if not translated_blocks:
        raise RuntimeError(
            "Resume failed: output SRT exists "
            "but contains no valid subtitle blocks: "
            f"{output_path}"
        )

    validate_resume_blocks(
        source_blocks,
        translated_blocks,
    )

    return translated_blocks
