#!/usr/bin/env python3
import sys
from pathlib import Path

from lib.media.extract import extract_english_pgs
from lib.media.ffprobe import find_best_english_pgs_subtitle
from lib.subtitle.paths import eng_sup_path


def main():
    if len(sys.argv) < 2:
        print("Usage: subtitletool extract input.mkv")
        sys.exit(1)

    input_path = Path(sys.argv[1]).expanduser().resolve()

    if not input_path.exists():
        print(f"Not found: {input_path}")
        sys.exit(1)

    subtitle = find_best_english_pgs_subtitle(input_path)

    if subtitle is None:
        print("English PGS subtitle not found.")
        sys.exit(1)

    output_path = eng_sup_path(input_path)

    print("========================================")
    print("SubtitleTool Extract")
    print("========================================")
    print(f"Input : {input_path}")
    print(f"Track : 0:{subtitle['index']}")
    print(f"Output: {output_path}")
    print("========================================")

    result = extract_english_pgs(input_path, output_path)

    print()
    print(f"Done: {result}")


if __name__ == "__main__":
    main()
