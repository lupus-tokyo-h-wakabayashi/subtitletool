#!/usr/bin/env python3
import time
from pathlib import Path

from lib.ollama import generate
from lib.progress import ProgressTracker, format_duration
from lib.srt import (
    SrtBlock,
    apply_translations,
    extract_text_lines,
    parse_numbered_translation,
    parse_srt,
    write_structured_srt,
)

MODEL = "qwen3:14b"

# 実際に翻訳する字幕数
CHUNK_SIZE = 30

# 翻訳対象の前後に参考として渡す字幕数
CONTEXT_SIZE = 15


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
) -> str:
    target_text = extract_text_lines(target_blocks)
    target_count = len(target_blocks)

    return f"""あなたは映像作品を担当するプロの日本語字幕翻訳者です。

以下の英語字幕を、前後の会話を考慮して自然な日本語字幕に翻訳してください。

目的:
- 日本人が映像を見ながら違和感なく理解できる字幕にする
- 単語の直訳ではなく、場面と会話の意図を訳す
- 登場人物の口調や関係性をできる限り維持する

重要:
- 「翻訳対象」のみ翻訳すること
- 前後の参考文脈は絶対に出力しないこと
- 出力は必ず {target_count} 行にすること
- 出力形式は必ず「1. 翻訳文」の形式にすること
- 番号を省略・追加・重複しないこと
- タイムコードや解説を出力しないこと
- Markdownコードブロックを使用しないこと
- 空欄や「（空白）」を出力しないこと
- 意味のない記号のみの字幕は原文のまま出力すること
- 1字幕内の「 / 」は原文の改行位置を表す

翻訳方針:
- 英語の語順を残さず、自然な日本語にする
- 会話では説明口調を避ける
- 主語は日本語として不要なら省略する
- 敬語、命令口調、軍人らしい口調を文脈から判断する
- 字幕として読みやすい長さを優先する
- 前後の会話と意味がつながる訳にする

用語:
- Stargate = スターゲイト
- Destiny = デスティニー
- Colonel Young = ヤング大佐
- Eli = イーライ
- Rush = ラッシュ博士
- Chloe = クロエ
- Scott = スコット
- Lieutenant = 中尉
- Senator = 上院議員
- Icarus Base = イカロス基地
- Ancient = 古代種族

【直前の参考文脈・出力禁止】
{format_context(before_context)}

【翻訳対象・この部分だけ出力】
{target_text}

【直後の参考文脈・出力禁止】
{format_context(after_context)}
"""


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

        before_context = source_blocks[before_start:start]
        target_blocks = source_blocks[start:end]
        after_context = source_blocks[end:after_end]

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
            target_blocks,
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