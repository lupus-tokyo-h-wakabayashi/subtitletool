#!/usr/bin/env python3
from pathlib import Path

from lib.ollama import generate
from lib.srt import (
    SrtBlock,
    apply_translations,
    chunk_blocks,
    extract_text_lines,
    parse_numbered_translation,
    parse_srt,
    write_structured_srt,
)

MODEL = "qwen3:14b"
CHUNK_SIZE = 30


def build_prompt(text_lines: str, count: int) -> str:
    return f"""You are a professional subtitle translator.

Translate the following English subtitle lines into natural Japanese.

Rules:
- Output exactly {count} lines.
- Keep the same numbering format: "1. ...", "2. ...".
- Do not output timestamps.
- Do not output SRT blocks.
- Do not add explanations.
- Translate only the text after the number.
- Keep each line short and natural for subtitles.
- If a line has only symbols or no meaningful text, copy it as-is.
- Never output "blank" or "（空白）".
- Keep names consistent:
  Stargate=スターゲイト
  Destiny=デスティニー
  Colonel Young=ヤング大佐
  Eli=イーライ
  Rush=ラッシュ博士
  Chloe=クロエ
  Scott=スコット
  Lieutenant=中尉
  Senator=上院議員

Input:
{text_lines}
"""


def translate_chunk(blocks: list[SrtBlock], model: str) -> list[str]:
    text_lines = extract_text_lines(blocks)
    prompt = build_prompt(text_lines, len(blocks))

    response = generate(prompt, model=model)
    translations = parse_numbered_translation(response, len(blocks))

    # 不足がある場合は元文で補完
    fixed = []

    for block, translated in zip(blocks, translations):
        if translated.strip():
            fixed.append(translated.strip())
        else:
            fixed.append(block.text)

    return fixed


def translate_srt(
    input_srt: str | Path,
    output_srt: str | Path,
    model: str = MODEL,
    chunk_size: int = CHUNK_SIZE,
) -> Path:
    input_srt = Path(input_srt).expanduser().resolve()
    output_srt = Path(output_srt).expanduser().resolve()

    if not input_srt.exists():
        raise FileNotFoundError(f"SRT not found: {input_srt}")

    if output_srt.exists():
        print(f"Skip Translate: {output_srt}")
        return output_srt

    source_blocks = parse_srt(input_srt)

    translated_all: list[SrtBlock] = []
    total = len(source_blocks)
    done = 0

    for chunk in chunk_blocks(source_blocks, chunk_size):
        print(f"Translating {done + 1}-{done + len(chunk)} / {total} ...")

        translated_texts = translate_chunk(chunk, model)
        translated_blocks = apply_translations(chunk, translated_texts)

        translated_all.extend(translated_blocks)
        done += len(chunk)

        write_structured_srt(output_srt, translated_all)

    return output_srt