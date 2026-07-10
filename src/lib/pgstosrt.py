#!/usr/bin/env python3
import subprocess
from pathlib import Path

DOTNET8 = Path.home() / ".dotnet8/dotnet"
PGS_TO_SRT_DLL = Path.home() / "PgsToSrt/src/PgsToSrt/bin/Release/net8.0/PgsToSrt.dll"
TESSDATA = "/usr/share/tesseract-ocr/4.00/tessdata"


def ocr_sup_to_srt(input_sup: str | Path, output_srt: str | Path, language: str = "eng") -> Path:
    input_sup = Path(input_sup).expanduser().resolve()
    output_srt = Path(output_srt).expanduser().resolve()

    if not input_sup.exists():
        raise FileNotFoundError(f"SUP not found: {input_sup}")

    if output_srt.exists():
        print(f"Skip OCR: {output_srt}")
        return output_srt

    cmd = [
        str(DOTNET8),
        str(PGS_TO_SRT_DLL),
        "--input", str(input_sup),
        "--output", str(output_srt),
        "--tesseractlanguage", language,
        "--tesseractdata", TESSDATA,
        "--tesseractversion", "4",
    ]

    print()
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    return output_srt