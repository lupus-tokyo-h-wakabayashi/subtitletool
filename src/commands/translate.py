#!/usr/bin/env python3
import sys
from pathlib import Path

from lib.srt import default_ja_path
from lib.translate import MODEL, translate_srt


def main():
    if len(sys.argv) < 2:
        print("Usage: subtitletool translate input.eng.srt [output.ja.srt]")
        sys.exit(1)

    input_srt = Path(sys.argv[1]).expanduser().resolve()

    if len(sys.argv) >= 3:
        output_srt = Path(sys.argv[2]).expanduser().resolve()
    else:
        output_srt = default_ja_path(input_srt)

    if not input_srt.exists():
        print(f"Not found: {input_srt}")
        sys.exit(1)

    print("========================================")
    print("SubtitleTool Translate")
    print("========================================")
    print(f"Input : {input_srt}")
    print(f"Output: {output_srt}")
    print(f"Model : {MODEL}")
    print("========================================")

    result = translate_srt(input_srt, output_srt)

    print()
    print(f"Done: {result}")


if __name__ == "__main__":
    main()