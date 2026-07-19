#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

from lib.infrastructure.cleanup import (
    CleanupResult,
    cleanup_intermediate_files,
)
from lib.media.extract import extract_english_pgs
from lib.media.mkvmerge import (
    mux_japanese_srt,
)
from lib.media.pgstosrt import ocr_sup_to_srt
from lib.subtitle.paths import (
    eng_srt_path,
    eng_sup_path,
    ja_mkv_path,
    ja_srt_path,
)
from lib.translation.translate import (
    MODEL,
    translate_srt,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract English subtitles, translate them, "
            "and mux Japanese subtitles into an MKV file."
        )
    )

    parser.add_argument(
        "input_mkv",
        help="Input MKV file",
    )

    parser.add_argument(
        "--model",
        default=MODEL,
        help=(
            "Ollama model name. "
            f"Uses {MODEL} when omitted."
        ),
    )

    parser.add_argument(
        "--profile",
        default=None,
        help=(
            "Translation profile name. "
            "Uses default when omitted."
        ),
    )

    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help=(
            "Keep extracted and translated "
            "intermediate subtitle files."
        ),
    )

    return parser


def cleanup_make_intermediate_files(
    *,
    sup_path: Path,
    eng_srt: Path,
    ja_srt: Path,
    keep_intermediate: bool,
) -> CleanupResult | None:
    if keep_intermediate:
        return None

    return cleanup_intermediate_files(
        [
            sup_path,
            eng_srt,
            ja_srt,
        ]
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_mkv = (
        Path(args.input_mkv)
        .expanduser()
        .resolve()
    )

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
    print(f"Model : {args.model}")

    if args.profile is not None:
        print(f"Profile: {args.profile}")

    print("========================================")

    extract_english_pgs(input_mkv, sup_path)
    ocr_sup_to_srt(sup_path, eng_srt)
    translate_srt(
        eng_srt,
        ja_srt,
        model=args.model,
        profile_name=args.profile,
    )
    result_mkv = mux_japanese_srt(
        input_mkv,
        ja_srt,
        output_mkv,
    )

    if not result_mkv.is_file():
        raise FileNotFoundError(
            "Final MKV not found after mux: "
            f"{result_mkv}"
        )

    cleanup_result = cleanup_make_intermediate_files(
        sup_path=sup_path,
        eng_srt=eng_srt,
        ja_srt=ja_srt,
        keep_intermediate=(
            args.keep_intermediate
        ),
    )

    if cleanup_result is None:
        print()
        print("Cleanup: skipped")
    else:
        print()
        print("Cleanup:")

        for path in cleanup_result.deleted:
            print(f"  Deleted: {path}")

        for path in cleanup_result.missing:
            print(f"  Missing: {path}")

    print()
    print("========================================")
    print("Done")
    print(f"Output: {result_mkv}")
    print("========================================")


if __name__ == "__main__":
    main()
