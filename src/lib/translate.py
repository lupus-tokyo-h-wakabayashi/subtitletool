#!/usr/bin/env python3
import time
from datetime import datetime
from pathlib import Path

from lib.ollama import generate
from lib.progress import (
    ProgressTracker,
    format_duration,
)
from lib.prompt import (
    DEFAULT_GLOSSARY_NAME,
    DEFAULT_STYLE_NAME,
    build_translation_prompt,
    load_glossary_entries,
)
from lib.srt import (
    SrtBlock,
    apply_translations,
    extract_text_lines,
    parse_srt,
    write_structured_srt,
)
from lib.text import (
    cleanup_ocr_text,
    is_suspicious_ocr_text,
)
from lib.translation_validation import (
    validate_translation_response,
)

MODEL = "qwen3:14b"

# 再試行用
MAX_TRANSLATION_ATTEMPTS = 3
TRANSLATION_DEBUG_DIR = Path(
    "/tmp/subtitletool"
)

# 実際に翻訳する字幕数
CHUNK_SIZE = 20

# 翻訳対象の前後に参考として渡す字幕数
CONTEXT_SIZE = 10


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


def build_ocr_noise_instruction(
    target_blocks: list[SrtBlock],
) -> str:
    """
    翻訳対象内のOCR破損候補を検出し、
    チャンク内番号でLLMへ通知する。
    """
    suspicious_indexes = [
        index
        for index, block in enumerate(
            target_blocks,
            start=1,
        )
        if is_suspicious_ocr_text(
            block.text
        )
    ]

    if not suspicious_indexes:
        return ""

    numbers = ", ".join(
        str(index)
        for index in suspicious_indexes
    )

    print(
        "OCR Noise Candidates: "
        f"{numbers}"
    )

    return f"""

【OCR破損の可能性がある字幕】

対象番号: {numbers}

これらの字幕にはOCRで壊れた英字列が含まれる可能性がある。

* 壊れた英字列を人名、地名、セリフとして推測しない
* 意味不明な文字列をカタカナへ音写しない
* 理解できる部分だけ翻訳する
* 判読できない部分は「（判読不能）」とする
* 原文にない意味を追加しない
"""


def build_prompt(
    before_context: list[SrtBlock],
    target_blocks: list[SrtBlock],
    after_context: list[SrtBlock],
    style_name: str = DEFAULT_STYLE_NAME,
    glossary_name: str = DEFAULT_GLOSSARY_NAME,
) -> str:
    base_prompt = build_translation_prompt(
        target_count=len(target_blocks),
        before_context=format_context(before_context),
        target_text=extract_text_lines(target_blocks),
        after_context=format_context(after_context),
        style_name=style_name,
        glossary_name=glossary_name,
    )

    ocr_noise_instruction = (
        build_ocr_noise_instruction(
            target_blocks
        )
    )

    return (
        base_prompt
        + ocr_noise_instruction
    )


def build_retry_instruction(
    errors: list[str],
) -> str:
    error_text = "\n".join(
        f"* {error}"
        for error in errors
    )

    return f"""

【前回の出力は検証に失敗したため再翻訳する】

前回は以下の問題が検出された。

{error_text}

次の規則を必ず守ること。

* 翻訳対象の字幕だけを出力する
* 出力は必ず指定件数と同じ件数にする
* 各行は「1. 翻訳文」の形式にする
* 番号は1から始まる連番にする
* 各番号は対応する原文1件だけを翻訳し、複数字幕を結合しない
* 原文の字幕を別の番号へ移動しない
* 前半の翻訳を後半で繰り返さない
* 直前・直後の参考文脈を出力しない
* 英文をそのまま出力しない
* 中国語を出力しない
* 同じ字幕を繰り返さない
* 解説、話者ラベル、補足を追加しない
* 不明なOCR文字列を勝手に補完しない
* 分からない文字列が含まれていても、レスポンス全体を反復させない
"""


def save_failed_translation_response(
    response: str,
    *,
    chunk_start: int,
    chunk_end: int,
    attempt: int,
) -> Path:
    TRANSLATION_DEBUG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d-%H%M%S-%f"
    )

    output_path = TRANSLATION_DEBUG_DIR / (
        "failed-translation-"
        f"{chunk_start}-{chunk_end}-"
        f"attempt-{attempt}-"
        f"{timestamp}.txt"
    )

    output_path.write_text(
        response,
        encoding="utf-8",
    )

    return output_path


def translate_chunk(
    before_context: list[SrtBlock],
    target_blocks: list[SrtBlock],
    after_context: list[SrtBlock],
    model: str,
    *,
    chunk_start: int,
    chunk_end: int,
    glossary_entries: dict[str, str],
    style_name: str = DEFAULT_STYLE_NAME,
    glossary_name: str = DEFAULT_GLOSSARY_NAME,
) -> list[str]:
    expected_count = len(target_blocks)
    last_errors: list[str] = []

    base_prompt = build_prompt(
        before_context,
        target_blocks,
        after_context,
        style_name=style_name,
        glossary_name=glossary_name,
    )

    for attempt in range(
        1,
        MAX_TRANSLATION_ATTEMPTS + 1,
    ):
        prompt = base_prompt

        if attempt > 1:
            prompt += build_retry_instruction(
                last_errors
            )

        response = generate(
            prompt,
            model=model,
        )

        print("=" * 80)
        print(response)
        print("=" * 80)

        validation = validate_translation_response(
            response,
            expected_count=expected_count,
            source_texts=[
                block.text
                for block in target_blocks
            ],
            glossary_entries=glossary_entries,
        )

        if validation.warnings:
            print("Validation Warnings:")

            for warning in validation.warnings:
                print(f"  - {warning}")

        if validation.valid:
            return validation.translated_texts

        failed_path = save_failed_translation_response(
            response,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            attempt=attempt,
        )

        last_errors = validation.reasons

        print(
            "Translation validation failed "
            f"(attempt {attempt}/"
            f"{MAX_TRANSLATION_ATTEMPTS})"
        )

        print("Validation Errors:")

        for reason in last_errors:
            print(f"  - {reason}")

        print(f"Saved response: {failed_path}")

        if attempt < MAX_TRANSLATION_ATTEMPTS:
            print("Retrying translation...")

    raise RuntimeError(
        "Translation failed after "
        f"{MAX_TRANSLATION_ATTEMPTS} attempts "
        f"for subtitles "
        f"{chunk_start}-{chunk_end}: "
        + "; ".join(last_errors)
    )


def translate_srt(
    input_srt: str | Path,
    output_srt: str | Path,
    model: str = MODEL,
    chunk_size: int = CHUNK_SIZE,
    context_size: int = CONTEXT_SIZE,
    style_name: str = DEFAULT_STYLE_NAME,
    glossary_name: str = DEFAULT_GLOSSARY_NAME,
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
        raise RuntimeError(
            f"No valid subtitle blocks: {input_srt}"
        )

    glossary_entries = load_glossary_entries(
        glossary_name
    )

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
    print(f"Style       : {style_name}")
    print(f"Glossary    : {glossary_name}")
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

        # タイムコードと元の字幕構造を維持するため、
        # OCR前処理前の対象ブロックを保持する
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
            chunk_start=start + 1,
            chunk_end=end,
            glossary_entries=glossary_entries,
            style_name=style_name,
            glossary_name=glossary_name,
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