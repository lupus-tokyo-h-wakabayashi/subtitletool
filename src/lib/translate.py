#!/usr/bin/env python3
import json
import re
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
    parse_speaker_from_text,
    parse_srt,
    write_structured_srt,
)
from lib.text import (
    cleanup_ocr_text,
    is_suspicious_ocr_text,
)
from lib.translation_validation import (
    source_contains_glossary_term,
    validate_translation_response,
)

MODEL = "qwen3:14b"

# 再試行用
MAX_TRANSLATION_ATTEMPTS = 3
TRANSLATION_DEBUG_DIR = Path(
    "/tmp/subtitletool"
)

# 実際に翻訳する字幕数
CHUNK_SIZE = 10

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


def build_request_item(
    block: SrtBlock,
) -> dict[str, str | None]:
    """
    SRTブロックをLLMリクエスト用JSON要素へ変換する。

    話者が明示されている場合だけspeakerへ設定し、
    本文から話者表記を除去する。
    """
    parsed = parse_speaker_from_text(
        block.text
    )

    return {
        "id": block.number,
        "speaker": parsed.speaker,
        "text": parsed.text,
    }


def build_translation_request_json(
    before_context: list[SrtBlock],
    target_blocks: list[SrtBlock],
    after_context: list[SrtBlock],
) -> str:
    """
    前後文脈と翻訳対象をJSON文字列へ変換する。
    """
    payload = {
        "context_before": [
            build_request_item(block)
            for block in before_context
        ],
        "target": [
            build_request_item(block)
            for block in target_blocks
        ],
        "context_after": [
            build_request_item(block)
            for block in after_context
        ],
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def normalize_translation_text(
    text: str,
) -> str:
    """
    翻訳文内の字幕改行記号を空白へ変換する。

    半角スラッシュは前後に空白がある場合だけ対象とし、
    24/7、km/h、URLなどは維持する。
    """
    normalized = re.sub(
        r"(?:\s+/\s+|\s*／\s*)",
        " ",
        text,
    )

    normalized = re.sub(
        r"[ \t]+",
        " ",
        normalized,
    )

    return normalized.strip()


def normalize_translation_texts(
    translated_texts: list[str],
) -> list[str]:
    """
    翻訳済み字幕をSRT保存用に一括正規化する。
    """
    return [
        normalize_translation_text(text)
        for text in translated_texts
    ]


def build_ocr_noise_instruction(
    target_blocks: list[SrtBlock],
) -> str:
    """
    翻訳対象内のOCR破損候補を検出し、
    チャンク内番号でLLMへ通知する。
    """
    suspicious_ids = [
        block.number
        for block in target_blocks
        if is_suspicious_ocr_text(
            block.text
        )
    ]

    if not suspicious_ids:
        return ""

    ids = ", ".join(
        suspicious_ids
    )

    print(
        "OCR Noise Candidates: "
        f"{ids}"
    )

    return f"""

【OCR破損の可能性がある字幕】

対象ID: {ids}

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
    request_json = build_translation_request_json(
        before_context,
        target_blocks,
        after_context,
    )

    base_prompt = build_translation_prompt(
        target_count=len(target_blocks),
        request_json=request_json,
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

* 出力はJSONオブジェクト1個だけにする
* 最上位キーはtranslationsのみにする
* translationsは入力targetと同じ件数にする
* 各要素のキーはidとtranslationだけにする
* idは入力targetのidをそのまま使用する
* idを追加、削除、変更、重複、並べ替えしない
* translationには日本語字幕だけを入れる
* 入力側のtextやspeakerを出力へコピーしない
* translationの代わりにtextを使用しない
* 複数字幕を1つのtranslationへ結合しない
* context_beforeとcontext_afterは出力しない
* JSONの前後へ説明やコードブロックを追加しない
"""


def build_required_glossary_instruction(
    target_blocks: list[SrtBlock],
    glossary_entries: dict[str, str],
) -> str:
    """
    翻訳対象チャンクに含まれる用語集項目を抽出し、
    LLMへ使用必須の訳語として通知する。
    """
    required_entries: list[tuple[str, str]] = []

    for source_term, expected_term in (
        glossary_entries.items()
    ):
        if not any(
            source_contains_glossary_term(
                block.text,
                source_term,
            )
            for block in target_blocks
        ):
            continue

        required_entries.append(
            (
                source_term,
                expected_term,
            )
        )

    if not required_entries:
        return ""

    lines = "\n".join(
        f"* {source_term} → {expected_term}"
        for source_term, expected_term
        in required_entries
    )

    return f"""

【この翻訳で必ず使用する用語】

{lines}

上記の英語表現が原文にある字幕では、
右側の日本語表記を一字一句そのまま使用すること。

* 別の日本語へ意訳しない
* 省略しない
* 一般的な訳語へ戻さない
* 長音、濁点、カタカナ表記を変更しない
* 再試行時も、すべての指定用語を維持する
"""


def build_glossary_retry_instruction(
    errors: list[str],
) -> str:
    glossary_lines = []

    pattern = re.compile(
        r"source_term=(?P<source>.+?), "
        r"expected=(?P<expected>.+?), "
        r"actual="
    )

    for error in errors:
        if not error.startswith(
            "Glossary violation:"
        ):
            continue

        match = pattern.search(error)

        if not match:
            continue

        source_term = match.group("source")
        expected_term = match.group("expected")

        glossary_lines.append(
            f"* {source_term} → {expected_term}"
        )

    if not glossary_lines:
        return ""

    terms = "\n".join(glossary_lines)

    return f"""

【今回必ず使用する用語】

{terms}

* expected に記載された表記を一字一句そのまま使用する
* 別の訳語へ言い換えない
* 長音、濁点、カタカナ表記を変更しない
* 前回正しかった他の用語を変更しない
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
    last_errors: list[str] = []

    base_prompt = build_prompt(
        before_context,
        target_blocks,
        after_context,
        style_name=style_name,
        glossary_name=glossary_name,
    )

    glossary_instruction = (
        build_required_glossary_instruction(
            target_blocks,
            glossary_entries,
        )
    )

    base_prompt += glossary_instruction

    for attempt in range(
        1,
        MAX_TRANSLATION_ATTEMPTS + 1,
    ):
        prompt = base_prompt

        if attempt > 1:
            prompt += build_retry_instruction(
                last_errors
            )

            # 再試行指示の末尾でも、対象チャンク内の
            # 全用語を改めて固定する。
            prompt += glossary_instruction

        response = generate(
            prompt,
            model=model,
        )

        display_response = "\n".join(
            normalize_translation_text(line)
            for line in response.splitlines()
        )

        print("=" * 80)
        print(display_response)
        print("=" * 80)

        validation = validate_translation_response(
            response,
            expected_ids=[
                block.number
                for block in target_blocks
            ],
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
            return normalize_translation_texts(
                validation.translated_texts
            )

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


def validate_resume_blocks(
    source_blocks: list[SrtBlock],
    translated_blocks: list[SrtBlock],
) -> None:
    """
    途中保存されたSRTが、入力SRTの先頭部分と一致するか確認する。

    本文は翻訳後なので比較しない。
    字幕番号とタイムコードだけを比較する。
    """
    if len(translated_blocks) > len(source_blocks):
        raise RuntimeError(
            "Resume failed: "
            "output SRT contains more subtitles than input SRT. "
            f"input={len(source_blocks)}, "
            f"output={len(translated_blocks)}"
        )

    for index, translated_block in enumerate(
        translated_blocks
    ):
        source_block = source_blocks[index]

        if translated_block.number != source_block.number:
            raise RuntimeError(
                "Resume failed: subtitle number mismatch "
                f"at position {index + 1}. "
                f"input={source_block.number}, "
                f"output={translated_block.number}"
            )

        if translated_block.timestamp != source_block.timestamp:
            raise RuntimeError(
                "Resume failed: timestamp mismatch "
                f"at subtitle {source_block.number}. "
                f"input={source_block.timestamp!r}, "
                f"output={translated_block.timestamp!r}"
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

    if input_srt == output_srt:
        raise ValueError(
            "Input and output SRT paths must be different: "
            f"{input_srt}"
        )

    if not input_srt.exists():
        raise FileNotFoundError(
            f"SRT not found: {input_srt}"
        )

    source_blocks = parse_srt(input_srt)

    if not source_blocks:
        raise RuntimeError(
            f"No valid subtitle blocks: {input_srt}"
        )

    translated_all: list[SrtBlock] = []

    if output_srt.exists():
        translated_all = parse_srt(output_srt)

        if not translated_all:
            raise RuntimeError(
                "Resume failed: output SRT exists "
                "but contains no valid subtitle blocks: "
                f"{output_srt}"
            )

        validate_resume_blocks(
            source_blocks,
            translated_all,
        )

    total_blocks = len(source_blocks)
    resume_start = len(translated_all)

    if resume_start == total_blocks:
        print()
        print("========================================")
        print("Translation Already Complete")
        print("========================================")
        print(f"Subtitles   : {total_blocks}")
        print(f"Output      : {output_srt}")
        print("========================================")

        return output_srt

    glossary_entries = load_glossary_entries(
        glossary_name
    )

    remaining_blocks = total_blocks - resume_start

    remaining_chunks = (
        remaining_blocks + chunk_size - 1
    ) // chunk_size

    translation_start = time.monotonic()

    progress = ProgressTracker(
        total_chunks=remaining_chunks
    )

    print()
    print("========================================")
    print("Translation Start")
    print("========================================")
    print(f"Model       : {model}")
    print(f"Style       : {style_name}")
    print(f"Glossary    : {glossary_name}")
    print(f"Subtitles   : {total_blocks}")
    print(f"Chunk Size  : {chunk_size}")
    print(
        f"Context     : "
        f"{context_size} before / after"
    )
    print(
        f"Resume      : "
        f"{'Yes' if resume_start else 'No'}"
    )
    print(f"Completed   : {resume_start}")
    print(f"Remaining   : {remaining_blocks}")
    print(f"Chunks Left : {remaining_chunks}")
    print("========================================")

    for chunk_number, start in enumerate(
        range(
            resume_start,
            total_blocks,
            chunk_size,
        ),
        start=1,
    ):
        end = min(
            start + chunk_size,
            total_blocks,
        )

        before_start = max(
            0,
            start - context_size,
        )

        after_end = min(
            total_blocks,
            end + context_size,
        )

        # タイムコードと元の字幕構造を維持するため、
        # OCR前処理前の対象ブロックを保持する
        source_target_blocks = source_blocks[
            start:end
        ]

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
            f"[{chunk_number}/{remaining_chunks}] "
            f"Translating "
            f"{start + 1}-{end} / {total_blocks} "
            f"(context: "
            f"{len(before_context)} + "
            f"{len(after_context)})"
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

        translated_all.extend(
            translated_blocks
        )

        overall_progress = (
            len(translated_all)
            / total_blocks
            * 100
        )

        # 各チャンク終了時に途中保存
        write_structured_srt(
            output_srt,
            translated_all,
        )

        chunk_elapsed = (
            time.monotonic() - chunk_start
        )

        progress.add(chunk_elapsed)

        elapsed = (
            time.monotonic()
            - translation_start
        )

        print(
            f"Session     : "
            f"{progress.progress_percent:5.1f}%"
        )
        print(
            f"Progress    : "
            f"{overall_progress:5.1f}% "
            f"({len(translated_all)}/{total_blocks})"
        )
        print(
            f"Chunk Time  : "
            f"{format_duration(chunk_elapsed)}"
        )
        print(
            f"Average     : "
            f"{progress.average_seconds:.1f} "
            "sec/chunk"
        )
        print(
            f"Elapsed     : "
            f"{format_duration(elapsed)}"
        )
        print(
            f"ETA         : "
            f"{format_duration(progress.eta_seconds)}"
        )

    total_elapsed = (
        time.monotonic()
        - translation_start
    )

    if len(translated_all) != total_blocks:
        raise RuntimeError(
            "Subtitle count mismatch: "
            f"source={total_blocks}, "
            f"translated={len(translated_all)}"
        )

    print()
    print("========================================")
    print("Translation Complete")
    print("========================================")
    print(f"Subtitles   : {len(translated_all)}")
    print(
        f"Chunks      : "
        f"{progress.completed_chunks}"
    )
    print(
        f"Total Time  : "
        f"{format_duration(total_elapsed)}"
    )
    print(
        f"Average     : "
        f"{progress.average_seconds:.1f} "
        "sec/chunk"
    )
    print(
        f"Fastest     : "
        f"{progress.fastest_seconds:.1f} sec"
    )
    print(
        f"Slowest     : "
        f"{progress.slowest_seconds:.1f} sec"
    )
    print(f"Output      : {output_srt}")
    print("========================================")

    return output_srt