#!/usr/bin/env python3
import sys
from pathlib import Path

from lib.extract import extract_english_pgs
from lib.ffmpeg import mux_japanese_srt
from lib.paths import eng_srt_path, eng_sup_path, ja_mkv_path, ja_srt_path
from lib.pgstosrt import ocr_sup_to_srt
from lib.translate import translate_srt


def main():
    if len(sys.argv) < 2:
        print("Usage: subtitletool make input.mkv")
        sys.exit(1)

    input_mkv = Path(sys.argv[1]).expanduser().resolve()

    if not input_mkv.exists():
        print(f"Not found: {input_mkv}")
        sys.exit(1)

    sup_path = eng_sup_path(input_mkv)
    eng_srt = eng_srt_path(input_mkv)
    ja_srt = ja_srt_path(input_mkv)
    output_mkv = ja_mkv_path(input_mkv)

    print("========================================")
    print("SubtitleTool Make")
    print("========================================")
    print(f"Input : {input_mkv}")
    print(f"SUP   : {sup_path}")
    print(f"ENG   : {eng_srt}")
    print(f"JPN   : {ja_srt}")
    print(f"Output: {output_mkv}")
    print("========================================")

    extract_english_pgs(input_mkv, sup_path)
    ocr_sup_to_srt(sup_path, eng_srt)
    translate_srt(eng_srt, ja_srt)
    mux_japanese_srt(input_mkv, ja_srt, output_mkv)

    print()
    print("========================================")
    print("Done")
    print(f"Output: {output_mkv}")
    print("========================================")


if __name__ == "__main__":
    main()