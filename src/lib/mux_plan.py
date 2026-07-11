from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MuxStream:
    """
    Mux対象ストリーム情報。
    Phase1では将来利用する器だけ定義する。
    """

    codec_type: str
    codec_name: str
    language: str
    title: str
    default: bool
    forced: bool


@dataclass(frozen=True)
class MuxPlan:
    """
    Mux実行計画。
    """

    input_mkv: Path
    input_srt: Path
    output_mkv: Path
    temporary_output: Path

    existing_subtitle_count: int
    added_subtitle_index: int

    added_language: str
    added_title: str


def build_mux_plan(
    input_mkv: str | Path,
    input_srt: str | Path,
    output_mkv: str | Path,
    *,
    existing_subtitle_count: int,
    language: str = "jpn",
    title: str = "Japanese AI",
) -> MuxPlan:
    """
    Mux実行計画を生成する。
    """

    input_mkv = Path(input_mkv)
    input_srt = Path(input_srt)
    output_mkv = Path(output_mkv)

    temporary_output = output_mkv.with_name(
        output_mkv.name + ".part.mkv"
    )

    return MuxPlan(
        input_mkv=input_mkv,
        input_srt=input_srt,
        output_mkv=output_mkv,
        temporary_output=temporary_output,
        existing_subtitle_count=(
            existing_subtitle_count
        ),
        added_subtitle_index=(
            existing_subtitle_count
        ),
        added_language=language,
        added_title=title,
    )
