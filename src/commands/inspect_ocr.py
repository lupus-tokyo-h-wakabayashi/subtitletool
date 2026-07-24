#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from lib.profile.config import (
    resolve_profile_config,
)
from lib.profile.noise import (
    load_noise_dictionary,
)
from lib.subtitle.ocr_html import (
    write_ocr_html_report,
)
from lib.subtitle.ocr_inspection import (
    OcrInspectionReport,
    inspect_ocr_blocks,
)
from lib.subtitle.ocr_report import (
    write_ocr_json_report,
)
from lib.subtitle.srt import parse_srt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect OCR subtitle transformations "
            "without modifying the input SRT "
            "or noise dictionaries."
        )
    )

    parser.add_argument(
        "input_srt",
        help=(
            "OCR-generated English SRT file"
        ),
    )

    parser.add_argument(
        "--profile",
        default=None,
        help=(
            "Translation profile used to resolve "
            "the OCR noise dictionary. "
            "Uses default when omitted."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. "
            "Defaults to "
            "<input stem>.ocr-debug."
        ),
    )

    parser.add_argument(
        "--format",
        dest="output_format",
        choices=(
            "json",
            "html",
            "all",
        ),
        default="all",
        help=(
            "Report format. "
            "Defaults to all."
        ),
    )

    return parser


def build_default_output_dir(
    input_srt: Path,
) -> Path:
    return input_srt.with_name(
        f"{input_srt.stem}.ocr-debug"
    )


def print_report_summary(
    report: OcrInspectionReport,
) -> None:
    summary = report.summary

    print()
    print(
        "Subtitles         : "
        f"{summary.subtitle_count}"
    )
    print(
        "Speaker Detected  : "
        f"{summary.speaker_detected_count}"
    )
    print(
        "Cleanup Changed   : "
        f"{summary.cleanup_changed_count}"
    )
    print(
        "Noise Subtitles   : "
        f"{summary.noise_candidate_subtitle_count}"
    )
    print(
        "Noise Candidates  : "
        f"{summary.noise_candidate_count}"
    )
    print(
        "Noise Applied     : "
        f"{summary.noise_applied_count}"
    )
    print(
        "Suspicious        : "
        f"{summary.suspicious_subtitle_count}"
    )
    print(
        "Changed Subtitles : "
        f"{summary.changed_subtitle_count}"
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_srt = (
        Path(args.input_srt)
        .expanduser()
        .resolve()
    )

    if not input_srt.is_file():
        parser.error(
            f"SRT not found: {input_srt}"
        )

    if args.output_dir is None:
        output_dir = (
            build_default_output_dir(
                input_srt
            )
        )
    else:
        output_dir = (
            Path(args.output_dir)
            .expanduser()
            .resolve()
        )

    profile_config = (
        resolve_profile_config(
            args.profile
        )
    )

    noise_dictionary = (
        load_noise_dictionary(
            profile_config
        )
    )

    blocks = parse_srt(
        input_srt
    )

    if not blocks:
        raise RuntimeError(
            "No valid subtitle blocks: "
            f"{input_srt}"
        )

    report = inspect_ocr_blocks(
        blocks,
        source_srt=input_srt,
        profile_name=(
            profile_config.resolved_profile
        ),
        noise_dictionary=noise_dictionary,
    )

    print(
        "========================================"
    )
    print(
        "SubtitleTool OCR Inspection"
    )
    print(
        "========================================"
    )
    print(f"Input   : {input_srt}")
    print(
        "Profile : "
        f"{profile_config.resolved_profile}"
    )
    print(f"Output  : {output_dir}")

    if profile_config.fallback_used:
        print(
            "Fallback: "
            f"{profile_config.requested_profile} "
            "-> "
            f"{profile_config.resolved_profile}"
        )

    print(
        "========================================"
    )

    print_report_summary(
        report
    )

    if args.output_format in {
        "json",
        "all",
    }:
        json_path = (
            write_ocr_json_report(
                output_dir / "report.json",
                report,
            )
        )

        print()
        print(f"JSON: {json_path}")

    if args.output_format in {
        "html",
        "all",
    }:
        html_path = (
            write_ocr_html_report(
                output_dir / "report.html",
                report,
            )
        )

        print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
