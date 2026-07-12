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
SUSPICIOUS_MIXED_CASE_TOKEN_PATTERN = re.compile(
    r"\b[A-Za-z]{8,}\b"
)


def is_suspicious_mixed_case_token(
    token: str,
) -> bool:
    """
    長い英字列の不自然な大小文字混在を検出する。

    正常例:
        Stargate
        Johansen

    異常例:
        VVNsKomCIAcM
        MimElomIElaie
    """
    if len(token) < 8:
        return False

    upper_count = sum(
        character.isupper()
        for character in token
    )

    lower_count = sum(
        character.islower()
        for character in token
    )

    # 通常の固有名詞:
    # 先頭のみ大文字、残りは小文字。
    if (
        token[0].isupper()
        and token[1:].islower()
    ):
        return False

    return (
        upper_count >= 2
        and lower_count >= 2
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


def find_heuristic_latin_noise_sequences(
    text: str,
    *,
    allowed_terms: set[str] | None = None,
) -> list[str]:
    """
    辞書を使用せず、文字列構造だけから
    OCR破損の可能性が高い英字列を抽出する。

    対象:
        - 不自然な大小文字混在の長い英字列
    """
    normalized = text.replace(
        "\n",
        " ",
    )

    allowed = {
        normalize_latin_token(term)
        for term in (
            allowed_terms
            or DEFAULT_ALLOWED_LATIN_TERMS
        )
    }

    results: list[str] = []

    for match in (
        SUSPICIOUS_MIXED_CASE_TOKEN_PATTERN.finditer(
            normalized
        )
    ):
        token = match.group(0)

        if normalize_latin_token(token) in allowed:
            continue

        if not is_suspicious_mixed_case_token(
            token
        ):
            continue

        if token not in results:
            results.append(token)

    return results


def mask_suspicious_latin_sequences(
    text: str,
    *,
    sequences: list[str],
) -> str:
    """
    指定されたOCR破損候補を判読不能へ置換する。
    """
    masked = text

    for sequence in sorted(
        sequences,
        key=len,
        reverse=True,
    ):
        masked = masked.replace(
            sequence,
            "（判読不能）",
        )

    return masked
