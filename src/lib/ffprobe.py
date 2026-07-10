#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path


def probe(input_path: str | Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            str(input_path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    return json.loads(result.stdout)


def streams(input_path: str | Path) -> list[dict]:
    return probe(input_path).get("streams", [])


def subtitle_streams(input_path: str | Path) -> list[dict]:
    return [
        stream for stream in streams(input_path)
        if stream.get("codec_type") == "subtitle"
    ]


def audio_streams(input_path: str | Path) -> list[dict]:
    return [
        stream for stream in streams(input_path)
        if stream.get("codec_type") == "audio"
    ]


def video_streams(input_path: str | Path) -> list[dict]:
    return [
        stream for stream in streams(input_path)
        if stream.get("codec_type") == "video"
    ]


def find_best_english_pgs_subtitle(input_path: str | Path) -> dict | None:
    candidates = []

    for stream in subtitle_streams(input_path):
        tags = stream.get("tags", {})
        language = tags.get("language", "")
        codec = stream.get("codec_name", "")

        if language == "eng" and codec in ["hdmv_pgs_subtitle", "pgssub"]:
            candidates.append(stream)

    if not candidates:
        return None

    # 通常字幕を優先。強制字幕はフレーム数が少ないことが多いので避ける。
    candidates.sort(
        key=lambda s: int(s.get("nb_frames") or 0),
        reverse=True,
    )

    return candidates[0]


def subtitle_count(input_path: str | Path) -> int:
    return len(subtitle_streams(input_path))
