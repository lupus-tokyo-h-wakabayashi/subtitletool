#!/usr/bin/env python3
import re


# PGS字幕に含まれる話者ラベル。
# SCOTT:, MAN 1:, WOMAN 2:, DR. RUSH: などを対象にする。
SPEAKER_LABEL_PATTERN = re.compile(
    r"""
    ^
    (?:
        [A-Z][A-Z0-9.'’_-]*
        (?:[ ]+[A-Z0-9][A-Z0-9.'’_-]*)*
    )
    :
    [ \t]*
    """,
    re.VERBOSE,
)

LATIN_TOKEN_PATTERN = re.compile(r"[A-Za-z]+")
NORMAL_ENGLISH_WORD_PATTERN = re.compile(
    r"^[A-Za-z]+(?:'[A-Za-z]+)?$"
)
KNOWN_OCR_NOISE_PATTERNS = (
    re.compile(
        r"\beRe\s+Are\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bSSeS\s+elke\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bTee\s+Ole\s+mite\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bVVNsKomCIAcM\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bHil\s+I['’]?m\b",
        re.IGNORECASE,
    ),
)

def remove_speaker_label(line: str) -> str:
    """
    行頭の話者ラベルだけを除去する。

    例:
        SCOTT: Move away -> Move away
        MAN 1: Move it back -> Move it back
        This is Scott. -> 変更なし
    """
    return SPEAKER_LABEL_PATTERN.sub("", line, count=1)


def fix_common_ocr_errors(text: str) -> str:
    """
    Tesseractで頻出する、安全性の高いOCR誤認識を補正する。
    """

    # 文頭・空白後の | を英大文字 I として補正する。
    #
    # | need a medic! -> I need a medic!
    # No. | think...  -> No. I think...
    text = re.sub(r"(?<!\S)\|(?=\s|$)", "I", text)

    # 行頭の | も補正する
    text = re.sub(r"^\|(?=\s|$)", "I", text, flags=re.MULTILINE)

    return text


def normalize_whitespace(text: str) -> str:
    """
    行単位で余分な空白を整理する。
    改行自体は維持する。
    """
    normalized_lines = []

    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        normalized_lines.append(line)

    return "\n".join(normalized_lines).strip()


def cleanup_ocr_text(text: str) -> str:
    """
    翻訳前に適用するOCR字幕の前処理。
    """
    cleaned_lines = []

    for line in text.splitlines():
        line = remove_speaker_label(line)
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = fix_common_ocr_errors(cleaned)
    cleaned = normalize_whitespace(cleaned)

    return cleaned


def is_suspicious_ocr_text(
    text: str,
) -> bool:
    """
    既知のOCR破損文字列が含まれるか判定する。

    Phase 1では誤検知を避けるため、
    実際に確認済みのパターンだけを対象にする。
    """
    normalized = text.replace(
        "\n",
        " ",
    )

    return any(
        pattern.search(normalized)
        for pattern in KNOWN_OCR_NOISE_PATTERNS
    )