#!/usr/bin/env python3
import subprocess
from pathlib import Path

from lib.ffprobe import subtitle_count
from lib.mux_plan import (
    build_mux_plan,
)
from lib.mux_validation import (
    validate_mux_output,
)


def run(cmd: list[str]) -> None:
    print()
    print(" ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def mux_japanese_srt(
    input_mkv: str | Path,
    ja_srt: str | Path,
    output_mkv: str | Path | None = None,
) -> Path:
    input_mkv = Path(input_mkv).expanduser().resolve()
    ja_srt = Path(ja_srt).expanduser().resolve()

    if output_mkv is None:
        output_mkv = input_mkv.with_name(f"{input_mkv.stem}.ja.mkv")
    else:
        output_mkv = Path(output_mkv).expanduser().resolve()

    if not input_mkv.exists():
        raise FileNotFoundError(f"MKV not found: {input_mkv}")

    if not ja_srt.exists():
        raise FileNotFoundError(f"SRT not found: {ja_srt}")

    if output_mkv.exists():
        print(f"Skip MUX: {output_mkv}")
        return output_mkv

    added_subtitle_index = subtitle_count(input_mkv)

    mux_plan = build_mux_plan(
        input_mkv=input_mkv,
        input_srt=ja_srt,
        output_mkv=output_mkv,
        existing_subtitle_count=(
            added_subtitle_index
        ),
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_mkv),
        "-i", str(ja_srt),

        "-map", "0",
        "-map", "1:0",

        "-c", "copy",

        f"-c:s:{mux_plan.added_subtitle_index}",
        "srt",

        f"-metadata:s:s:{mux_plan.added_subtitle_index}",
        "language=jpn",
        f"-metadata:s:s:{mux_plan.added_subtitle_index}",
        "title=Japanese AI",

        str(mux_plan.temporary_output),
    ]

    run(cmd)

    validation = validate_mux_output(
        mux_plan.input_mkv,
        mux_plan.temporary_output,
        added_language=(
            mux_plan.added_language
        ),
        added_title=(
            mux_plan.added_title
        ),
    )

    if validation.warnings:
        print()
        print("Mux Validation Warnings:")

        for warning in validation.warnings:
            print(f"  - {warning}")

    if not validation.valid:
        details = "\n".join(
            f"  - {error}"
            for error in validation.errors
        )

        raise RuntimeError(
            "Mux validation failed:\n"
            f"{details}"
        )

    print()
    print("Mux Validation: OK")

    return mux_plan.temporary_output
