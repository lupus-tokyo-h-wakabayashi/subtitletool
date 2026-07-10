#!/usr/bin/env python3
import subprocess
from pathlib import Path

from lib.ffprobe import find_best_english_pgs_subtitle
from lib.paths import eng_sup_path


def extract_english_pgs(input_mkv: str | Path, output_sup: str | Path | None = None) -> Path:
    input_mkv = Path(input_mkv).expanduser().resolve()

    if output_sup is None:
        output_sup = eng_sup_path(input_mkv)
    else:
        output_sup = Path(output_sup).expanduser().resolve()

    if not input_mkv.exists():
        raise FileNotFoundError(f"MKV not found: {input_mkv}")

    subtitle = find_best_english_pgs_subtitle(input_mkv)

    if subtitle is None:
        raise RuntimeError("English PGS subtitle not found.")

    stream_index = subtitle["index"]

    if output_sup.exists():
        print(f"Skip Extract: {output_sup}")
        return output_sup

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_mkv),
        "-map", f"0:{stream_index}",
        "-c:s", "copy",
        str(output_sup),
    ]

    print()
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    return output_sup