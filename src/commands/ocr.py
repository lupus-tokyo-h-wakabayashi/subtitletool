#!/usr/bin/env python3
import sys
from pathlib import Path

from lib.media.pgstosrt import ocr_sup_to_srt


def main():
    if len(sys.argv) < 2:
        print("Usage: subtitletool ocr input.eng.sup [output.eng.srt]")
        sys.exit(1)

    input_sup = Path(sys.argv[1]).expanduser().resolve()

    if len(sys.argv) >= 3:
        output_srt = Path(sys.argv[2]).expanduser().resolve()
    else:
        output_srt = input_sup.with_suffix(".srt")

    print("========================================")
    print("SubtitleTool OCR")
    print("========================================")
    print(f"Input : {input_sup}")
    print(f"Output: {output_srt}")
    print("========================================")

    ocr_sup_to_srt(input_sup, output_srt)

    print()
    print(f"Done: {output_srt}")


if __name__ == "__main__":
    main()
