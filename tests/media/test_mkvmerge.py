from __future__ import annotations

from pathlib import Path

import pytest
from lib.media import mkvmerge
from lib.media.mkvmerge import (
    build_mkvmerge_command,
    mux_japanese_srt,
)
from lib.media.mux_plan import (
    build_mux_plan,
)
from lib.media.mux_validation import (
    MuxValidationResult,
)


def build_test_plan(
    tmp_path: Path,
):
    input_mkv = tmp_path / "movie.mkv"
    input_srt = tmp_path / "movie.ja.srt"
    output_mkv = tmp_path / "movie.ja.mkv"

    return build_mux_plan(
        input_mkv=input_mkv,
        input_srt=input_srt,
        output_mkv=output_mkv,
        existing_subtitle_count=4,
    )


def test_build_mkvmerge_command(
    tmp_path: Path,
) -> None:
    plan = build_test_plan(
        tmp_path
    )

    command = build_mkvmerge_command(
        plan
    )

    assert command == [
        "mkvmerge",
        "--output",
        str(plan.temporary_output),
        str(plan.input_mkv),
        "--language",
        "0:jpn",
        "--track-name",
        "0:Japanese AI",
        "--default-track-flag",
        "0:no",
        "--forced-display-flag",
        "0:no",
        str(plan.input_srt),
    ]


def test_mux_rejects_missing_input_mkv(
    tmp_path: Path,
) -> None:
    input_mkv = tmp_path / "missing.mkv"
    input_srt = tmp_path / "movie.ja.srt"

    input_srt.write_text(
        "1\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "日本語\n",
        encoding="utf-8",
    )

    with pytest.raises(
        FileNotFoundError,
        match="MKV not found",
    ):
        mux_japanese_srt(
            input_mkv,
            input_srt,
        )


def test_mux_rejects_missing_srt(
    tmp_path: Path,
) -> None:
    input_mkv = tmp_path / "movie.mkv"
    input_srt = tmp_path / "missing.ja.srt"

    input_mkv.write_bytes(b"mkv")

    with pytest.raises(
        FileNotFoundError,
        match="SRT not found",
    ):
        mux_japanese_srt(
            input_mkv,
            input_srt,
        )


def test_mux_rejects_missing_mkvmerge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_mkv = tmp_path / "movie.mkv"
    input_srt = tmp_path / "movie.ja.srt"

    input_mkv.write_bytes(b"mkv")

    input_srt.write_text(
        "1\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "日本語\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        mkvmerge.shutil,
        "which",
        lambda command: None,
    )

    with pytest.raises(
        RuntimeError,
        match="mkvmerge command not found",
    ):
        mux_japanese_srt(
            input_mkv,
            input_srt,
        )


def test_mux_skips_existing_output(
    tmp_path: Path,
) -> None:
    input_mkv = tmp_path / "movie.mkv"
    input_srt = tmp_path / "movie.ja.srt"
    output_mkv = tmp_path / "movie.ja.mkv"

    input_mkv.write_bytes(b"input")
    input_srt.write_text(
        "subtitle",
        encoding="utf-8",
    )
    output_mkv.write_bytes(b"existing")

    result = mux_japanese_srt(
        input_mkv,
        input_srt,
        output_mkv,
    )

    assert result == output_mkv.resolve()

    assert output_mkv.read_bytes() == (
        b"existing"
    )


def test_mux_finalizes_valid_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_mkv = tmp_path / "movie.mkv"
    input_srt = tmp_path / "movie.ja.srt"
    output_mkv = tmp_path / "movie.ja.mkv"

    input_mkv.write_bytes(b"input")

    input_srt.write_text(
        "1\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "日本語\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        mkvmerge.shutil,
        "which",
        lambda command: "/usr/bin/mkvmerge",
    )

    monkeypatch.setattr(
        mkvmerge,
        "subtitle_count",
        lambda path: 4,
    )

    def fake_run(
        command: list[str],
    ) -> None:
        output_index = (
            command.index("--output") + 1
        )

        temporary_output = Path(
            command[output_index]
        )

        temporary_output.write_bytes(
            b"muxed"
        )

    monkeypatch.setattr(
        mkvmerge,
        "run_mkvmerge",
        fake_run,
    )

    monkeypatch.setattr(
        mkvmerge,
        "validate_mux_output",
        lambda *args, **kwargs: (
            MuxValidationResult(
                valid=True,
                errors=(),
                warnings=(),
            )
        ),
    )

    result = mux_japanese_srt(
        input_mkv,
        input_srt,
        output_mkv,
    )

    assert result == output_mkv.resolve()
    assert output_mkv.read_bytes() == b"muxed"

    temporary_output = output_mkv.with_name(
        output_mkv.name + ".part.mkv"
    )

    assert not temporary_output.exists()


def test_mux_does_not_finalize_invalid_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_mkv = tmp_path / "movie.mkv"
    input_srt = tmp_path / "movie.ja.srt"
    output_mkv = tmp_path / "movie.ja.mkv"

    input_mkv.write_bytes(b"input")

    input_srt.write_text(
        "1\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "日本語\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        mkvmerge.shutil,
        "which",
        lambda command: "/usr/bin/mkvmerge",
    )

    monkeypatch.setattr(
        mkvmerge,
        "subtitle_count",
        lambda path: 4,
    )

    def fake_run(
        command: list[str],
    ) -> None:
        output_index = (
            command.index("--output") + 1
        )

        temporary_output = Path(
            command[output_index]
        )

        temporary_output.write_bytes(
            b"invalid"
        )

    monkeypatch.setattr(
        mkvmerge,
        "run_mkvmerge",
        fake_run,
    )

    monkeypatch.setattr(
        mkvmerge,
        "validate_mux_output",
        lambda *args, **kwargs: (
            MuxValidationResult(
                valid=False,
                errors=(
                    "Existing subtitle "
                    "disposition mismatch",
                ),
                warnings=(),
            )
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Mux validation failed",
    ):
        mux_japanese_srt(
            input_mkv,
            input_srt,
            output_mkv,
        )

    assert not output_mkv.exists()

    temporary_output = output_mkv.with_name(
        output_mkv.name + ".part.mkv"
    )

    assert temporary_output.is_file()
