from __future__ import annotations

import re

SHORT_UPPERCASE_FRAGMENT_PATTERN = re.compile(
    r"^[A-Z]{2,5} [A-Z]$"
)


def normalize_uppercase_fragment(
    text: str,
) -> str:
    return re.sub(
        r"[ \t]+",
        " ",
        text,
    ).strip()


def find_suspicious_short_uppercase_fragments(
    text: str,
    *,
    allowed_fragments: set[str] | None = None,
) -> tuple[str, ...]:
    """
    OCR誤認識の可能性がある短い大文字断片を行単位で抽出する。

    対象例:
        SST A

    対象外:
        FTL
        NASA
        Yes, sir.
        (ALARMS BLARING)

    この関数は候補を検出するだけで、
    OCR文字列の補正や置換は行わない。
    """
    allowed = {
        normalize_uppercase_fragment(
            fragment
        ).upper()
        for fragment in (
            allowed_fragments or set()
        )
    }

    candidates: list[str] = []

    for raw_line in text.splitlines():
        line = normalize_uppercase_fragment(
            raw_line
        )

        if not line:
            continue

        if line.upper() in allowed:
            continue

        if (
            SHORT_UPPERCASE_FRAGMENT_PATTERN
                .fullmatch(line)
            is None
        ):
            continue

        if line in candidates:
            continue

        candidates.append(line)

    return tuple(candidates)
