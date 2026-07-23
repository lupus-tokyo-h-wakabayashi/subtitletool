#!/usr/bin/env python3

import argparse
from pathlib import Path

from lib.profile.charactor import (
    charactor_path,
    extract_speaker_names,
    write_charactors,
)
from lib.profile.config import (
    resolve_profile_config,
)
from lib.subtitle.srt import parse_srt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create charactor.json from explicitly named "
            "speakers in an English SRT file."
        )
    )
    parser.add_argument(
        "input_srt",
        help="Input English SRT file",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help=(
            "Profile whose charactor.json is created. "
            "Uses default when omitted."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_srt = Path(
        args.input_srt
    ).expanduser().resolve()

    if not input_srt.is_file():
        raise FileNotFoundError(
            f"SRT not found: {input_srt}"
        )

    profile_config = resolve_profile_config(
        args.profile
    )

    if profile_config.fallback_used:
        raise FileNotFoundError(
            "Profile not found: "
            f"{args.profile!r}"
        )

    speakers = extract_speaker_names(
        parse_srt(input_srt)
    )
    output_path = write_charactors(
        charactor_path(profile_config),
        speakers,
    )

    print(f"Profile   : {profile_config.resolved_profile}")
    print(f"Speakers  : {len(speakers)}")
    print(f"Charactors: {output_path}")


if __name__ == "__main__":
    main()
