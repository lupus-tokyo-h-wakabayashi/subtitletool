from pathlib import Path

from lib.mux_plan import (
    build_mux_plan,
)
from lib.mux_validation import (
    validate_mux_output,
    validate_mux_probe_data,
)


def test_build_mux_plan_creates_part_filename(
    tmp_path: Path,
) -> None:
    input_mkv = tmp_path / "movie.mkv"
    input_srt = tmp_path / "movie.ja.srt"
    output_mkv = tmp_path / "movie.ja.mkv"

    plan = build_mux_plan(
        input_mkv,
        input_srt,
        output_mkv,
        existing_subtitle_count=4,
    )

    assert plan.output_mkv == output_mkv
    assert plan.temporary_output == (
        tmp_path
        / "movie.ja.mkv.part.mkv"
    )
    assert plan.added_subtitle_index == 4


def test_validate_mux_output_missing_output(
    tmp_path: Path,
) -> None:
    input_mkv = tmp_path / "movie.mkv"
    output_mkv = tmp_path / "movie.ja.mkv.part.mkv"

    input_mkv.write_bytes(
        b"input"
    )

    result = validate_mux_output(
        input_mkv,
        output_mkv,
    )

    assert not result.valid
    assert result.warnings == ()
    assert result.errors == (
        "Output MKV not found: "
        f"{output_mkv.resolve()}",
    )


def test_validate_mux_probe_data_detects_subtitle_count_mismatch(
) -> None:
    input_probe = {
        "streams": [
            {
                "codec_type": "video",
            },
            {
                "codec_type": "audio",
            },
            {
                "codec_type": "subtitle",
                "codec_name": (
                    "hdmv_pgs_subtitle"
                ),
                "tags": {
                    "language": "eng",
                },
            },
        ],
        "format": {
            "duration": "100.0",
        },
    }

    output_probe = {
        "streams": [
            {
                "codec_type": "video",
            },
            {
                "codec_type": "audio",
            },
            {
                "codec_type": "subtitle",
                "codec_name": "subrip",
                "tags": {
                    "language": "jpn",
                    "title": "Japanese AI",
                },
            },
        ],
        "format": {
            "duration": "100.0",
        },
    }

    result = validate_mux_probe_data(
        input_probe,
        output_probe,
    )

    assert not result.valid
    assert result.warnings == ()
    assert result.errors == (
        "Subtitle stream count mismatch: "
        "input=1, output=1, expected=2",
    )
