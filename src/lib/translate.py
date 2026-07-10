#!/usr/bin/env python3
import time
from pathlib import Path

from lib.ollama import generate
from lib.progress import ProgressTracker, format_duration
from lib.prompt import build_translation_prompt
from lib.srt import (
    SrtBlock,
    apply_translations,
    extract_text_lines,
    parse_numbered_translation,
    parse_srt,
    write_structured_srt,
)
from lib.text import cleanup_ocr_text

MODEL = "qwen3:14b"

# 実際に翻訳する字幕数
CHUNK_SIZE = 30

# 翻訳対象の前後に参考として渡す字幕数
CONTEXT_SIZE = 15

def cleanup_blocks(blocks: list[SrtBlock]) -> list[SrtBlock]:
    """
    字幕番号とタイムコードを維持し、
    AIへ渡す本文だけOCR前処理する。
    """
    return [
        SrtBlock(
            number=block.number,
            timestamp=block.timestamp,
            text=cleanup_ocr_text(block.text),
        )
        for block in blocks
    ]

def format_context(blocks: list[SrtBlock]) -> str:
    if not blocks:
        return "なし"

    lines = []

    for block in blocks:
        text = block.text.replace("\n", " / ").strip()
        lines.append(f"[{block.number}] {text}")

    return "\n".join(lines)

def build_prompt(
    before_context: list[SrtBlock],
    target_blocks: list[SrtBlock],
    after_context: list[SrtBlock],
    style_name: str = "stargate",
    glossary_name: str = "stargate",
) -> str:
    return build_translation_prompt(
        target_count=len(target_blocks),
        before_context=format_context(before_context),
        target_text=extract_text_lines(target_blocks),
        after_context=format_context(after_context),
        style_name=style_name,
        glossary_name=glossary_name,
    )


def translate_chunk(
    before_context: list[SrtBlock],
    target_blocks: list[SrtBlock],
    after_context: list[SrtBlock],
    model: str,
) -> list[str]:
    prompt = build_prompt(
        before_context,
        target_blocks,
        after_context,
    )

    response = generate(prompt, model=model)

    translations = parse_numbered_translation(
        response,
        len(target_blocks),
    )

    fixed = []

    for block, translated in zip(target_blocks, translations):
        text = translated.strip()

        # モデルが翻訳を返さなかった場合は原文を保持
        if not text or text in {
            "（空白）",
            "(blank)",
            "blank",
            "空白",
        }:
            text = block.text

        fixed.append(text)

    return fixed


def translate_srt(
    input_srt: str | Path,
    output_srt: str | Path,
    model: str = MODEL,
    chunk_size: int = CHUNK_SIZE,
    context_size: int = CONTEXT_SIZE,
) -> Path:
    input_srt = Path(input_srt).expanduser().resolve()
    output_srt = Path(output_srt).expanduser().resolve()

    if not input_srt.exists():
        raise FileNotFoundError(f"SRT not found: {input_srt}")

    if output_srt.exists():
        print(f"Skip Translate: {output_srt}")
        return output_srt

    source_blocks = parse_srt(input_srt)

    if not source_blocks:
        raise RuntimeError(f"No valid subtitle blocks: {input_srt}")

    translated_all: list[SrtBlock] = []

    total_blocks = len(source_blocks)
    total_chunks = (total_blocks + chunk_size - 1) // chunk_size

    translation_start = time.monotonic()
    progress = ProgressTracker(total_chunks=total_chunks)

    print()
    print("========================================")
    print("Translation Start")
    print("========================================")
    print(f"Model       : {model}")
    print(f"Subtitles   : {total_blocks}")
    print(f"Chunk Size  : {chunk_size}")
    print(f"Context     : {context_size} before / after")
    print(f"Chunks      : {total_chunks}")
    print("========================================")

    for chunk_number, start in enumerate(
        range(0, total_blocks, chunk_size),
        start=1,
    ):
        end = min(start + chunk_size, total_blocks)

        before_start = max(0, start - context_size)
        after_end = min(total_blocks, end + context_size)

        # 元字幕は、翻訳失敗時のフォールバック用に保持する
        source_target_blocks = source_blocks[start:end]

        # AIへ渡す字幕だけOCR前処理する
        before_context = cleanup_blocks(
            source_blocks[before_start:start]
        )

        target_blocks = cleanup_blocks(
            source_target_blocks
        )

        after_context = cleanup_blocks(
            source_blocks[end:after_end]
        )

        chunk_start = time.monotonic()

        print()
        print(
            f"[{chunk_number}/{total_chunks}] "
            f"Translating {start + 1}-{end} / {total_blocks} "
            f"(context: {len(before_context)} + {len(after_context)})"
        )

        translated_texts = translate_chunk(
            before_context,
            target_blocks,
            after_context,
            model,
        )

        translated_blocks = apply_translations(
            source_target_blocks,
            translated_texts,
        )

        translated_all.extend(translated_blocks)

        # 各チャンク終了時に途中保存
        write_structured_srt(output_srt, translated_all)

        chunk_elapsed = time.monotonic() - chunk_start
        progress.add(chunk_elapsed)

        elapsed = time.monotonic() - translation_start

        print(f"Progress    : {progress.progress_percent:5.1f}%")
        print(f"Chunk Time  : {format_duration(chunk_elapsed)}")
        print(f"Average     : {progress.average_seconds:.1f} sec/chunk")
        print(f"Elapsed     : {format_duration(elapsed)}")
        print(f"ETA         : {format_duration(progress.eta_seconds)}")

    total_elapsed = time.monotonic() - translation_start

    if len(translated_all) != total_blocks:
        raise RuntimeError(
            "Subtitle count mismatch: "
            f"source={total_blocks}, translated={len(translated_all)}"
        )

    print()
    print("========================================")
    print("Translation Complete")
    print("========================================")
    print(f"Subtitles   : {len(translated_all)}")
    print(f"Chunks      : {progress.completed_chunks}")
    print(f"Total Time  : {format_duration(total_elapsed)}")
    print(f"Average     : {progress.average_seconds:.1f} sec/chunk")
    print(f"Fastest     : {progress.fastest_seconds:.1f} sec")
    print(f"Slowest     : {progress.slowest_seconds:.1f} sec")
    print(f"Output      : {output_srt}")
    print("========================================")

    return output_srt