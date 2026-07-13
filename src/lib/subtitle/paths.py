#!/usr/bin/env python3
from pathlib import Path


def eng_sup_path(input_mkv: str | Path) -> Path:
    input_mkv = Path(input_mkv).expanduser().resolve()
    return input_mkv.with_suffix(".eng.sup")


def eng_srt_path(input_mkv: str | Path) -> Path:
    input_mkv = Path(input_mkv).expanduser().resolve()
    return input_mkv.with_suffix(".eng.srt")


def ja_srt_path(input_mkv: str | Path) -> Path:
    input_mkv = Path(input_mkv).expanduser().resolve()
    return input_mkv.with_suffix(".ja.srt")


def ja_mkv_path(input_mkv: str | Path) -> Path:
    input_mkv = Path(input_mkv).expanduser().resolve()
    return input_mkv.with_name(f"{input_mkv.stem}.ja.mkv")