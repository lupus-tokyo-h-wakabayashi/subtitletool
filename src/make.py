#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

PGS_TO_SRT = str(Path.home() / "projects/PgsToSrt/src/PgsToSrt/bin/Release/net8.0/PgsToSrt.dll")
DOTNET8 = str(Path.home() / ".dotnet8/dotnet")
TRANSLATE = str(Path.home() / "projects/subtitletool/src/translate.py")

SUBTITLE_TRACK = "0:3"


def run(cmd):
    print()
    print(" ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: subtitletool make input.mkv")
        sys.exit(1)

    input_path = Path(sys.argv[1]).expanduser().resolve()

    if not input_path.exists():
        print(f"Not found: {input_path}")
        sys.exit(1)

    work_dir = input_path.parent
    stem = input_path.stem

    sup_path = work_dir / f"{stem}.eng.sup"
    eng_srt_path = work_dir / f"{stem}.eng.srt"
    ja_srt_path = work_dir / f"{stem}.ja.srt"
    output_mkv_path = work_dir / f"{stem}.ja.mkv"

    print("========================================")
    print("SubtitleTool Make")
    print("========================================")
    print(f"Input : {input_path}")
    print(f"Track : {SUBTITLE_TRACK}")
    print(f"SUP   : {sup_path}")
    print(f"ENG   : {eng_srt_path}")
    print(f"JPN   : {ja_srt_path}")
    print(f"Output: {output_mkv_path}")
    print("========================================")

    if not sup_path.exists():
        run([
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-map", SUBTITLE_TRACK,
            "-c:s", "copy",
            str(sup_path),
        ])
    else:
        print(f"Skip SUP: {sup_path}")

    if not eng_srt_path.exists():
        run([
            DOTNET8,
            PGS_TO_SRT,
            "--input", str(sup_path),
            "--output", str(eng_srt_path),
            "--tesseractlanguage", "eng",
            "--tesseractdata", "/usr/share/tesseract-ocr/4.00/tessdata",
            "--tesseractversion", "4",
        ])
    else:
        print(f"Skip ENG SRT: {eng_srt_path}")

    if not ja_srt_path.exists():
        run([
            "python3",
            TRANSLATE,
            str(eng_srt_path),
            str(ja_srt_path),
        ])
    else:
        print(f"Skip JPN SRT: {ja_srt_path}")

    if not output_mkv_path.exists():
        run([
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-i", str(ja_srt_path),
            "-map", "0",
            "-map", "1",
            "-c", "copy",
            "-c:s", "srt",
            "-metadata:s:s:0", "language=eng",
            "-metadata:s:s:1", "language=fre",
            "-metadata:s:s:2", "language=spa",
            "-metadata:s:s:3", "language=spa",
            "-metadata:s:s:4", "language=jpn",
            "-metadata:s:s:4", "title=Japanese",
            str(output_mkv_path),
        ])
    else:
        print(f"Skip MKV: {output_mkv_path}")

    print()
    print("========================================")
    print("Done")
    print(f"Output: {output_mkv_path}")
    print("========================================")


if __name__ == "__main__":
    main()
