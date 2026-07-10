#!/usr/bin/env python3
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SrtBlock:
    number: str
    timestamp: str
    text: str


def parse_srt(path: str | Path) -> list[SrtBlock]:
    path = Path(path).expanduser().resolve()
    raw = path.read_text(encoding="utf-8", errors="replace").strip()

    blocks = []

    for chunk in re.split(r"\n\s*\n", raw):
        lines = [line.rstrip() for line in chunk.splitlines()]

        if len(lines) < 2:
            continue

        number = lines[0].strip()
        timestamp = lines[1].strip()
        text = "\n".join(lines[2:]).strip()

        if "-->" not in timestamp:
            continue

        blocks.append(SrtBlock(number=number, timestamp=timestamp, text=text))

    return blocks


def write_structured_srt(path: str | Path, blocks: list[SrtBlock]) -> None:
    path = Path(path).expanduser().resolve()

    out = []

    for block in blocks:
        text = block.text.strip()

        if not text:
            text = block.text

        out.append(f"{block.number}\n{block.timestamp}\n{text}")

    path.write_text("\n\n".join(out).strip() + "\n", encoding="utf-8")


def chunk_blocks(blocks: list[SrtBlock], size: int):
    for i in range(0, len(blocks), size):
        yield blocks[i:i + size]


def default_ja_path(input_srt: str | Path) -> Path:
    input_srt = Path(input_srt).expanduser().resolve()
    name = input_srt.name

    if name.endswith(".eng.srt"):
        return input_srt.with_name(name.replace(".eng.srt", ".ja.srt"))

    if name.endswith(".srt"):
        return input_srt.with_name(name.replace(".srt", ".ja.srt"))

    return input_srt.with_suffix(".ja.srt")


def extract_text_lines(blocks: list[SrtBlock]) -> str:
    lines = []

    for i, block in enumerate(blocks, start=1):
        text = block.text.replace("\n", " / ").strip()
        lines.append(f"{i}. {text}")

    return "\n".join(lines)


def parse_numbered_translation(text: str, expected_count: int) -> list[str]:
    text = text.strip()
    text = re.sub(r"^```(?:text)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    result = [""] * expected_count

    pattern = re.compile(
        r"^\s*(\d+)[\.\)：:)、]\s*(.*)$"
    )

    for line in text.splitlines():
        line = line.strip()
        match = pattern.match(line)

        if not match:
            continue

        idx = int(match.group(1)) - 1
        value = match.group(2).strip()

        if 0 <= idx < expected_count:
            result[idx] = value

    return result


def apply_translations(
    source_blocks: list[SrtBlock],
    translations: list[str],
) -> list[SrtBlock]:
    output = []

    for block, translated in zip(source_blocks, translations):
        text = translated.strip()

        # 空・空白プレースホルダなら元文を残す
        if not text or text in ["（空白）", "(blank)", "blank", "空白"]:
            text = block.text

        output.append(
            SrtBlock(
                number=block.number,
                timestamp=block.timestamp,
                text=text,
            )
        )

    return output