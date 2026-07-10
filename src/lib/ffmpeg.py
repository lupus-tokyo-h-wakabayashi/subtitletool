#!/usr/bin/env python3
import subprocess
from pathlib import Path

from lib.ffprobe import subtitle_count


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

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_mkv),
        "-i", str(ja_srt),

        "-map", "0",
        "-map", "1:0",

        # まず全ストリームをコピー
        "-c", "copy",

        # 追加される字幕だけSRTとしてmux
        f"-c:s:{added_subtitle_index}", "srt",

        f"-metadata:s:s:{added_subtitle_index}", "language=jpn",
        f"-metadata:s:s:{added_subtitle_index}", "title=Japanese AI",

        str(output_mkv),
    ]

    run(cmd)

    return output_mkv