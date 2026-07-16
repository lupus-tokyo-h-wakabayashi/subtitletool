#!/usr/bin/env python3
import sys
from pathlib import Path

from lib.media.mkvmerge import (
    mux_japanese_srt,
)
from lib.subtitle.srt import default_ja_path


def main():
    if len(sys.argv) < 2:
        print("Usage: subtitletool mux input.mkv [input.ja.srt] [output.ja.mkv]")
        sys.exit(1)

    input_mkv = Path(sys.argv[1]).expanduser().resolve()

    if len(sys.argv) >= 3:
        ja_srt = Path(sys.argv[2]).expanduser().resolve()
    else:
        ja_srt = default_ja_path(input_mkv.with_suffix(".eng.srt"))

    if len(sys.argv) >= 4:
        output_mkv = Path(sys.argv[3]).expanduser().resolve()
    else:
        output_mkv = input_mkv.with_name(f"{input_mkv.stem}.ja.mkv")

    print("========================================")
    print("SubtitleTool Mux")
    print("========================================")
    print(f"Input : {input_mkv}")
    print(f"SRT   : {ja_srt}")
    print(f"Output: {output_mkv}")
    print("========================================")

    result = mux_japanese_srt(input_mkv, ja_srt, output_mkv)

    print()
    print(f"Done: {result}")


if __name__ == "__main__":
    main()
