#!/usr/bin/env python3

import argparse
from pathlib import Path

from lib.translation.translate import (
    MODEL,
    translate_srt,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Translate an English SRT subtitle file "
            "into Japanese."
        )
    )

    parser.add_argument(
        "input_srt",
        help="Input English SRT file",
    )

    parser.add_argument(
        "output_srt",
        nargs="?",
        help=(
            "Output Japanese SRT file. "
            "If omitted, .eng.srt is replaced with .ja.srt."
        ),
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
        "--style",
        default=None,
        help=(
            "Legacy profile option. "
            "Must match --glossary."
        ),
    )

    parser.add_argument(
        "--glossary",
        default=None,
        help=(
            "Legacy profile option. "
            "Must match --style."
        ),
    )

    parser.add_argument(
        "--inspect-request",
        action="store_true",
        help=(
            "Save the next Ollama translation request "
            "as JSON without sending it."
        ),
    )

    return parser


def build_output_path(
    input_srt: Path,
    output_srt: str | None,
) -> Path:
    if output_srt:
        return Path(
            output_srt
        ).expanduser().resolve()

    if input_srt.name.endswith(".eng.srt"):
        output_name = (
            input_srt.name.removesuffix(".eng.srt")
            + ".ja.srt"
        )
    else:
        output_name = (
            input_srt.stem
            + ".ja.srt"
        )

    return input_srt.with_name(output_name)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_srt = Path(
        args.input_srt
    ).expanduser().resolve()

    output_srt = build_output_path(
        input_srt,
        args.output_srt,
    )

    if not input_srt.exists():
        print(f"Not found: {input_srt}")
        raise SystemExit(1)

    print("========================================")
    print("SubtitleTool Translate")
    print("========================================")
    print(f"Input   : {input_srt}")
    print(f"Output  : {output_srt}")
    print(f"Model   : {args.model}")
    if args.profile is not None:
        print(f"Profile : {args.profile}")
    if args.style is not None:
        print(f"Style   : {args.style}")
    if args.glossary is not None:
        print(f"Glossary: {args.glossary}")
    print("========================================")

    result = translate_srt(
        input_srt,
        output_srt,
        model=args.model,
        profile_name=args.profile,
        style_name=args.style,
        glossary_name=args.glossary,
        inspect_request=args.inspect_request,
    )

    print()

    if args.inspect_request:
        print(f"Request inspection saved: {result}")
    else:
        print(f"Done: {result}")


if __name__ == "__main__":
    main()
