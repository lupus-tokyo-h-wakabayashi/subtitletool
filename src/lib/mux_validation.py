from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lib.ffprobe import (
    probe,
    stream_language,
    stream_title,
)

TEXT_SUBTITLE_CODECS = frozenset(
    {
        "subrip",
        "srt",
    }
)

MAX_DURATION_DIFFERENCE_SECONDS = 2.0


@dataclass(frozen=True)
class MuxValidationResult:
    """
    Mux結果の検証結果。
    """

    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def streams_by_type(
    probe_data: dict,
    codec_type: str,
) -> list[dict]:
    """
    ffprobe結果から指定種類のストリームを取得する。
    """
    return [
        stream
        for stream in probe_data.get(
            "streams",
            [],
        )
        if stream.get("codec_type") == codec_type
    ]


def format_duration(
    probe_data: dict,
) -> float | None:
    """
    ffprobe結果からコンテナの再生時間を取得する。
    """
    value = (
        probe_data.get(
            "format",
            {},
        ).get("duration")
    )

    if value in {
        None,
        "",
        "N/A",
    }:
        return None

    try:
        return float(value)
    except (
            TypeError,
            ValueError,
    ):
        return None


def validate_mux_probe_data(
    input_probe: dict,
    output_probe: dict,
    *,
    added_language: str = "jpn",
    added_title: str = "Japanese AI",
    max_duration_difference: float = (
        MAX_DURATION_DIFFERENCE_SECONDS
    ),
) -> MuxValidationResult:
    """
    Mux前後のffprobe結果を比較する。
    """
    errors: list[str] = []
    warnings: list[str] = []

    input_videos = streams_by_type(
        input_probe,
        "video",
    )
    output_videos = streams_by_type(
        output_probe,
        "video",
    )

    input_audios = streams_by_type(
        input_probe,
        "audio",
    )
    output_audios = streams_by_type(
        output_probe,
        "audio",
    )

    input_subtitles = streams_by_type(
        input_probe,
        "subtitle",
    )
    output_subtitles = streams_by_type(
        output_probe,
        "subtitle",
    )

    if len(output_videos) != len(input_videos):
        errors.append(
            "Video stream count mismatch: "
            f"input={len(input_videos)}, "
            f"output={len(output_videos)}"
        )

    if len(output_audios) != len(input_audios):
        errors.append(
            "Audio stream count mismatch: "
            f"input={len(input_audios)}, "
            f"output={len(output_audios)}"
        )

    expected_subtitle_count = (
        len(input_subtitles) + 1
    )

    if (
        len(output_subtitles)
        != expected_subtitle_count
    ):
        errors.append(
            "Subtitle stream count mismatch: "
            f"input={len(input_subtitles)}, "
            f"output={len(output_subtitles)}, "
            f"expected={expected_subtitle_count}"
        )

    if output_subtitles:
        added_subtitle = output_subtitles[-1]

        codec_name = str(
            added_subtitle.get(
                "codec_name",
                "",
            )
        )

        if codec_name not in TEXT_SUBTITLE_CODECS:
            errors.append(
                "Added subtitle codec is not text: "
                f"codec={codec_name!r}"
            )

        language = stream_language(
            added_subtitle
        )

        if language != added_language:
            errors.append(
                "Added subtitle language mismatch: "
                f"expected={added_language!r}, "
                f"actual={language!r}"
            )

        title = stream_title(
            added_subtitle
        )

        if title != added_title:
            errors.append(
                "Added subtitle title mismatch: "
                f"expected={added_title!r}, "
                f"actual={title!r}"
            )
    else:
        errors.append(
            "Output contains no subtitle streams."
        )

    input_duration = format_duration(
        input_probe
    )
    output_duration = format_duration(
        output_probe
    )

    if (
        input_duration is None
        or output_duration is None
    ):
        warnings.append(
            "Duration could not be compared: "
            f"input={input_duration!r}, "
            f"output={output_duration!r}"
        )
    else:
        duration_difference = abs(
            input_duration
            - output_duration
        )

        if (
            duration_difference
            > max_duration_difference
        ):
            errors.append(
                "Duration difference is too large: "
                f"input={input_duration:.3f}, "
                f"output={output_duration:.3f}, "
                f"difference={duration_difference:.3f}, "
                f"maximum={max_duration_difference:.3f}"
            )

    return MuxValidationResult(
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def validate_mux_output(
    input_mkv: str | Path,
    output_mkv: str | Path,
    *,
    added_language: str = "jpn",
    added_title: str = "Japanese AI",
    max_duration_difference: float = (
        MAX_DURATION_DIFFERENCE_SECONDS
    ),
) -> MuxValidationResult:
    """
    入出力ファイルをffprobeで解析してMux結果を検証する。
    """
    input_path = (
        Path(input_mkv)
        .expanduser()
        .resolve()
    )
    output_path = (
        Path(output_mkv)
        .expanduser()
        .resolve()
    )

    errors: list[str] = []

    if not input_path.is_file():
        errors.append(
            f"Input MKV not found: {input_path}"
        )

    if not output_path.is_file():
        errors.append(
            f"Output MKV not found: {output_path}"
        )
    elif output_path.stat().st_size == 0:
        errors.append(
            f"Output MKV is empty: {output_path}"
        )

    if errors:
        return MuxValidationResult(
            valid=False,
            errors=tuple(errors),
            warnings=(),
        )

    return validate_mux_probe_data(
        probe(input_path),
        probe(output_path),
        added_language=added_language,
        added_title=added_title,
        max_duration_difference=(
            max_duration_difference
        ),
    )
