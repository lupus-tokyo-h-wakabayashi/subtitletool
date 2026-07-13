from pathlib import Path

import lib.media.ffmpeg as ffmpeg_module
import pytest
from lib.media.mux_plan import (
    build_mux_plan,
)
from lib.media.mux_validation import (
    MuxValidationResult,
    validate_mux_output,
    validate_mux_probe_data,
)


def build_valid_mux_probe_pair(
    *,
    input_duration: str = "100.0",
    output_duration: str = "100.0",
) -> tuple[dict, dict]:
    input_probe = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "hevc",
                "tags": {},
                "disposition": {
                    "default": 1,
                    "forced": 0,
                },
            },
            {
                "codec_type": "audio",
                "codec_name": "eac3",
                "tags": {
                    "language": "eng",
                    "title": "Surround 5.1",
                },
                "disposition": {
                    "default": 1,
                    "forced": 0,
                },
            },
            {
                "codec_type": "subtitle",
                "codec_name": (
                    "hdmv_pgs_subtitle"
                ),
                "tags": {
                    "language": "eng",
                    "title": "English",
                },
                "disposition": {
                    "default": 1,
                    "forced": 0,
                },
            },
        ],
        "format": {
            "duration": input_duration,
        },
    }

    output_probe = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "hevc",
                "tags": {},
                "disposition": {
                    "default": 1,
                    "forced": 0,
                },
            },
            {
                "codec_type": "audio",
                "codec_name": "eac3",
                "tags": {
                    "language": "eng",
                    "title": "Surround 5.1",
                },
                "disposition": {
                    "default": 1,
                    "forced": 0,
                },
            },
            {
                "codec_type": "subtitle",
                "codec_name": (
                    "hdmv_pgs_subtitle"
                ),
                "tags": {
                    "language": "eng",
                    "title": "English",
                },
                "disposition": {
                    "default": 1,
                    "forced": 0,
                },
            },
            {
                "codec_type": "subtitle",
                "codec_name": "subrip",
                "tags": {
                    "language": "jpn",
                    "title": "Japanese AI",
                },
                "disposition": {
                    "default": 0,
                    "forced": 0,
                },
            },
        ],
        "format": {
            "duration": output_duration,
        },
    }

    return (
        input_probe,
        output_probe,
    )


def test_validate_mux_probe_data_detects_missing_added_subtitle_language(
) -> None:
    (
        input_probe,
        output_probe,
    ) = build_valid_mux_probe_pair()

    added_subtitle = (
        output_probe["streams"][-1]
    )

    added_subtitle["tags"].pop(
        "language"
    )

    result = validate_mux_probe_data(
        input_probe,
        output_probe,
    )

    assert not result.valid
    assert result.warnings == ()

    assert (
        "Added subtitle language mismatch: "
        "expected='jpn', actual=''"
        in result.errors
    )


def test_validate_mux_probe_data_detects_missing_added_subtitle_title(
) -> None:
    (
        input_probe,
        output_probe,
    ) = build_valid_mux_probe_pair()

    added_subtitle = (
        output_probe["streams"][-1]
    )

    added_subtitle["tags"].pop(
        "title"
    )

    result = validate_mux_probe_data(
        input_probe,
        output_probe,
    )

    assert not result.valid
    assert result.warnings == ()

    assert (
        "Added subtitle title mismatch: "
        "expected='Japanese AI', actual=''"
        in result.errors
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
    assert (
        "Subtitle stream count mismatch: "
        "input=1, output=1, expected=2"
        in result.errors
    )


def test_mux_skips_when_final_output_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_mkv = tmp_path / "movie.mkv"
    input_srt = tmp_path / "movie.ja.srt"
    output_mkv = tmp_path / "movie.ja.mkv"

    input_mkv.write_bytes(
        b"input"
    )
    input_srt.write_text(
        "subtitle",
        encoding="utf-8",
    )
    output_mkv.write_bytes(
        b"completed"
    )

    def fail_if_run(
        cmd: list[str],
    ) -> None:
        raise AssertionError(
            "FFmpeg must not run "
            "when final output exists."
        )

    monkeypatch.setattr(
        ffmpeg_module,
        "run",
        fail_if_run,
    )

    result = (
        ffmpeg_module.mux_japanese_srt(
            input_mkv,
            input_srt,
            output_mkv,
        )
    )

    assert result == output_mkv.resolve()
    assert output_mkv.read_bytes() == (
        b"completed"
    )

    temporary_output = (
        output_mkv.with_name(
            output_mkv.name
            + ".part.mkv"
        )
    )

    assert not temporary_output.exists()


def test_mux_keeps_partial_output_when_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_mkv = tmp_path / "movie.mkv"
    input_srt = tmp_path / "movie.ja.srt"
    output_mkv = tmp_path / "movie.ja.mkv"

    temporary_output = (
        output_mkv.with_name(
            output_mkv.name
            + ".part.mkv"
        )
    )

    input_mkv.write_bytes(
        b"input"
    )
    input_srt.write_text(
        "subtitle",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        ffmpeg_module,
        "subtitle_count",
        lambda path: 1,
    )

    def fake_run(
        cmd: list[str],
    ) -> None:
        output_path = Path(
            cmd[-1]
        )

        output_path.write_bytes(
            b"partial"
        )

    monkeypatch.setattr(
        ffmpeg_module,
        "run",
        fake_run,
    )

    def fake_validate_mux_output(
        input_path: str | Path,
        output_path: str | Path,
        **kwargs,
    ) -> MuxValidationResult:
        assert Path(
            input_path
        ) == input_mkv.resolve()

        assert Path(
            output_path
        ) == temporary_output.resolve()

        return MuxValidationResult(
            valid=False,
            errors=(
                "Test validation failure",
            ),
            warnings=(),
        )

    monkeypatch.setattr(
        ffmpeg_module,
        "validate_mux_output",
        fake_validate_mux_output,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Mux validation failed"
        ),
    ):
        ffmpeg_module.mux_japanese_srt(
            input_mkv,
            input_srt,
            output_mkv,
        )

    assert temporary_output.is_file()
    assert temporary_output.read_bytes() == (
        b"partial"
    )

    assert not output_mkv.exists()


def test_mux_finalizes_output_after_successful_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_mkv = tmp_path / "movie.mkv"
    input_srt = tmp_path / "movie.ja.srt"
    output_mkv = tmp_path / "movie.ja.mkv"

    temporary_output = (
        output_mkv.with_name(
            output_mkv.name
            + ".part.mkv"
        )
    )

    input_mkv.write_bytes(
        b"input"
    )
    input_srt.write_text(
        "subtitle",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        ffmpeg_module,
        "subtitle_count",
        lambda path: 1,
    )

    def fake_run(
        cmd: list[str],
    ) -> None:
        output_path = Path(
            cmd[-1]
        )

        assert output_path == (
            temporary_output.resolve()
        )

        output_path.write_bytes(
            b"completed"
        )

    monkeypatch.setattr(
        ffmpeg_module,
        "run",
        fake_run,
    )

    def fake_validate_mux_output(
        input_path: str | Path,
        output_path: str | Path,
        **kwargs,
    ) -> MuxValidationResult:
        assert Path(
            input_path
        ) == input_mkv.resolve()

        assert Path(
            output_path
        ) == temporary_output.resolve()

        return MuxValidationResult(
            valid=True,
            errors=(),
            warnings=(),
        )

    monkeypatch.setattr(
        ffmpeg_module,
        "validate_mux_output",
        fake_validate_mux_output,
    )

    result = (
        ffmpeg_module.mux_japanese_srt(
            input_mkv,
            input_srt,
            output_mkv,
        )
    )

    assert result == output_mkv.resolve()

    assert output_mkv.is_file()
    assert output_mkv.read_bytes() == (
        b"completed"
    )

    assert not temporary_output.exists()


def test_validate_mux_probe_data_detects_video_codec_change(
) -> None:
    input_probe = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "hevc",
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
                "codec_name": "h264",
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

    assert (
        "Video codec sequence mismatch: "
        "input=['hevc'], output=['h264']"
        in result.errors
    )


def test_validate_mux_probe_data_detects_audio_codec_change(
) -> None:
    input_probe = {
        "streams": [
            {
                "codec_type": "audio",
                "codec_name": "eac3",
            },
        ],
        "format": {
            "duration": "100.0",
        },
    }

    output_probe = {
        "streams": [
            {
                "codec_type": "audio",
                "codec_name": "aac",
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

    assert (
        "Audio codec sequence mismatch: "
        "input=['eac3'], output=['aac']"
        in result.errors
    )


def test_validate_mux_probe_data_detects_existing_subtitle_codec_change(
) -> None:
    input_probe = {
        "streams": [
            {
                "codec_type": "subtitle",
                "codec_name": (
                    "hdmv_pgs_subtitle"
                ),
            },
        ],
        "format": {
            "duration": "100.0",
        },
    }

    output_probe = {
        "streams": [
            {
                "codec_type": "subtitle",
                "codec_name": "subrip",
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

    assert (
        "Existing subtitle codec sequence "
        "mismatch: "
        "input=['hdmv_pgs_subtitle'], "
        "output=['subrip']"
        in result.errors
    )


def test_validate_mux_probe_data_detects_audio_language_change(
) -> None:
    input_probe = {
        "streams": [
            {
                "codec_type": "audio",
                "codec_name": "eac3",
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
                "codec_type": "audio",
                "codec_name": "eac3",
                "tags": {
                    "language": "jpn",
                },
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

    assert (
        "Audio language sequence mismatch: "
        "input=['eng'], output=['jpn']"
        in result.errors
    )


def test_validate_mux_probe_data_detects_audio_title_change(
) -> None:
    input_probe = {
        "streams": [
            {
                "codec_type": "audio",
                "codec_name": "eac3",
                "tags": {
                    "language": "eng",
                    "title": "Surround 5.1",
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
                "codec_type": "audio",
                "codec_name": "eac3",
                "tags": {
                    "language": "eng",
                    "title": "Stereo",
                },
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

    assert (
        "Audio title sequence mismatch: "
        "input=['Surround 5.1'], "
        "output=['Stereo']"
        in result.errors
    )


def test_validate_mux_probe_data_detects_existing_subtitle_disposition_change(
) -> None:
    input_probe = {
        "streams": [
            {
                "codec_type": "subtitle",
                "codec_name": (
                    "hdmv_pgs_subtitle"
                ),
                "tags": {
                    "language": "eng",
                },
                "disposition": {
                    "default": 1,
                    "forced": 0,
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
                "codec_type": "subtitle",
                "codec_name": (
                    "hdmv_pgs_subtitle"
                ),
                "tags": {
                    "language": "eng",
                },
                "disposition": {
                    "default": 0,
                    "forced": 0,
                },
            },
            {
                "codec_type": "subtitle",
                "codec_name": "subrip",
                "tags": {
                    "language": "jpn",
                    "title": "Japanese AI",
                },
                "disposition": {
                    "default": 0,
                    "forced": 0,
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

    assert (
        "Existing subtitle disposition "
        "sequence mismatch: "
        "input=[(True, False)], "
        "output=[(False, False)]"
        in result.errors
    )


@pytest.mark.parametrize(
    (
            "output_duration",
            "expected_valid",
    ),
    [
        (
                "102.000",
                True,
        ),
        (
                "102.001",
                False,
        ),
    ],
)
def test_validate_mux_probe_data_duration_boundary(
    output_duration: str,
    expected_valid: bool,
) -> None:
    (
        input_probe,
        output_probe,
    ) = build_valid_mux_probe_pair(
        input_duration="100.000",
        output_duration=output_duration,
    )

    result = validate_mux_probe_data(
        input_probe,
        output_probe,
    )

    assert result.valid is expected_valid
    assert result.warnings == ()

    duration_error = (
        "Duration difference is too large"
    )

    if expected_valid:
        assert not any(
            duration_error in error
            for error in result.errors
        )
    else:
        assert any(
            duration_error in error
            for error in result.errors
        )
