#!/usr/bin/env python3
import sys
from pathlib import Path

from lib.media.ffprobe import (
    audio_streams,
    find_best_english_pgs_subtitle,
    subtitle_streams,
    video_streams,
)


def lang(stream: dict) -> str:
    return stream.get("tags", {}).get("language", "-")


def title(stream: dict) -> str:
    return stream.get("tags", {}).get("title", "-")


def main():
    if len(sys.argv) < 2:
        print("Usage: subtitletool scan input.mkv")
        sys.exit(1)

    input_path = Path(sys.argv[1]).expanduser().resolve()

    if not input_path.exists():
        print(f"Not found: {input_path}")
        sys.exit(1)

    print("========================================")
    print("SubtitleTool Scan")
    print("========================================")
    print(f"Input: {input_path}")
    print("========================================")

    print()
    print("[Video]")
    for stream in video_streams(input_path):
        print(
            f"#{stream.get('index')}: "
            f"{stream.get('codec_name')} "
            f"{stream.get('width')}x{stream.get('height')} "
            f"{stream.get('avg_frame_rate')}"
        )

    print()
    print("[Audio]")
    for stream in audio_streams(input_path):
        print(
            f"#{stream.get('index')}: "
            f"{lang(stream)} "
            f"{stream.get('codec_name')} "
            f"{stream.get('channels')}ch "
            f"title={title(stream)}"
        )

    print()
    print("[Subtitle]")
    for stream in subtitle_streams(input_path):
        print(
            f"#{stream.get('index')}: "
            f"{lang(stream)} "
            f"{stream.get('codec_name')} "
            f"frames={stream.get('nb_frames', '-')} "
            f"title={title(stream)}"
        )

    best = find_best_english_pgs_subtitle(input_path)

    print()
    print("[Recommended]")
    if best:
        print(
            f"English PGS subtitle: #{best.get('index')} "
            f"frames={best.get('nb_frames', '-')}"
        )
    else:
        print("English PGS subtitle: not found")


if __name__ == "__main__":
    main()
