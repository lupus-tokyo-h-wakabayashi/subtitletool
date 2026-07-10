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
CHINESE_SPECIFIC_PATTERN = re.compile(
    r"[这些们为发经进过还让从个里边开关车话说时对与于后会动语]+"
)
DEFAULT_ALLOWED_LATIN_TERMS = {
    "AI",
    "DNA",
    "F",
    "SG",
    "TJ",
    "T",
    "J",
    "Stargate",
    "Destiny",
    "Icarus",
    "Johansen",
    "Armstrong",
    "Wallace",
    "Rush",
    "Eli",
    "Scott",
    "Chloe",
    "Young",
}
LATIN_SEQUENCE_PATTERN = re.compile(
    r"(?<![A-Za-z])"
    r"[A-Za-z][A-Za-z'-]*"
    r"(?:[ \t]+[A-Za-z][A-Za-z'-]*)+"
    r"(?![A-Za-z])"
)


def mask_chinese_ocr_text(
    text: str,
) -> str:
    """
    OCR由来と考えられる中国語固有文字列を
    判読不能マーカーへ置換する。
    """
    return CHINESE_SPECIFIC_PATTERN.sub(
        "（OCR判読不能）",
        text,
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


def normalize_latin_token(
    token: str,
) -> str:
    """
    英字語の比較用表記へ正規化する。
    """
    return re.sub(
        r"[^A-Za-z0-9]",
        "",
        token,
    ).upper()


def is_suspicious_latin_sequence(
    sequence: str,
    *,
    allowed_terms: set[str] | None = None,
) -> bool:
    """
    複数語の英字列がOCR破損らしいか判定する。
    """
    allowed = {
        normalize_latin_token(term)
        for term in (
            allowed_terms
            or DEFAULT_ALLOWED_LATIN_TERMS
        )
    }

    words = LATIN_TOKEN_PATTERN.findall(
        sequence
    )

    if len(words) < 2:
        return False

    unknown_words = [
        word
        for word in words
        if normalize_latin_token(word)
        not in allowed
    ]

    if not unknown_words:
        return False

    short_words = sum(
        len(word) <= 2
        for word in words
    )

    unusual_case_words = sum(
        any(character.isupper() for character in word)
        and any(character.islower() for character in word)
        and not (
            word[0].isupper()
            and word[1:].islower()
        )
        for word in words
    )

    internal_upper_words = sum(
        word.isupper()
        and len(word) >= 2
        for word in words[1:]
    )

    all_unknown = (
        len(unknown_words)
        == len(words)
    )

    return (
        unusual_case_words > 0
        or internal_upper_words > 0
        or short_words >= 2
        or (
            all_unknown
            and len(words) >= 3
        )
    )


def find_suspicious_latin_sequences(
    text: str,
    *,
    allowed_terms: set[str] | None = None,
) -> list[str]:
    """
    OCR破損の可能性が高い複数語の英字列を抽出する。
    """
    normalized = text.replace(
        "\n",
        " ",
    )

    results: list[str] = []

    for match in LATIN_SEQUENCE_PATTERN.finditer(
        normalized
    ):
        sequence = match.group(0).strip()

        if not is_suspicious_latin_sequence(
            sequence,
            allowed_terms=allowed_terms,
        ):
            continue

        results.append(sequence)

    for pattern in KNOWN_OCR_NOISE_PATTERNS:
        for match in pattern.finditer(normalized):
            sequence = match.group(0).strip()

            if sequence not in results:
                results.append(sequence)

    return results


def mask_suspicious_latin_sequences(
    text: str,
    *,
    sequences: list[str] | None = None,
) -> str:
    """
    OCR破損候補の英字列を判読不能へ置換する。
    """
    targets = (
        sequences
        if sequences is not None
        else find_suspicious_latin_sequences(
            text
        )
    )

    masked = text

    for sequence in sorted(
        targets,
        key=len,
        reverse=True,
    ):
        masked = masked.replace(
            sequence,
            "（判読不能）",
        )

    return masked


def is_suspicious_ocr_text(
    text: str,
) -> bool:
    """
    OCR破損文字列が含まれる可能性を判定する。
    """
    if find_suspicious_latin_sequences(
        text
    ):
        return True

    normalized = text.replace(
        "\n",
        " ",
    )

    return any(
        pattern.search(normalized)
        for pattern in KNOWN_OCR_NOISE_PATTERNS
    )