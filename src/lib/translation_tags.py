from __future__ import annotations

import re
from dataclasses import dataclass

VALID_TRANSLATION_TAG_LEVELS = {
    1,
    3,
    5,
}

TRANSLATION_TAG_PATTERN = re.compile(
    r"\[(?P<closing>/?)"
    r"(?P<level>\d+)\]"
)


@dataclass(frozen=True)
class TranslationEvaluationTag:
    """
    AIが翻訳文へ付与した自己評価タグ。

    start / endは、タグを含む元文字列上での範囲。
    value_start / value_endは、タグ内部の値の範囲。
    """

    level: int
    value: str
    start: int
    end: int
    value_start: int
    value_end: int


@dataclass(frozen=True)
class TranslationTagParseResult:
    """
    翻訳自己評価タグの解析結果。
    """

    tags: tuple[
        TranslationEvaluationTag,
        ...
    ]
    errors: tuple[str, ...]


def parse_translation_tags(
    text: str,
) -> TranslationTagParseResult:
    """
    翻訳文に含まれる自己評価タグを解析する。

    対応するタグ:
        [1]...[/1]
        [3]...[/3]
        [5]...[/5]

    次の状態はエラーにする:
        - 2、4など未対応レベル
        - 開始タグのネスト
        - 対応しない終了タグ
        - 終了タグだけ存在
        - 終了タグの欠落
        - 空のタグ
        - タグ内部の前後空白
    """
    tags: list[
        TranslationEvaluationTag
    ] = []
    errors: list[str] = []

    active_level: int | None = None
    active_tag_start: int | None = None
    value_start: int | None = None

    for match in (
        TRANSLATION_TAG_PATTERN.finditer(
            text
        )
    ):
        closing = bool(
            match.group("closing")
        )
        level = int(
            match.group("level")
        )

        if (
            level
            not in VALID_TRANSLATION_TAG_LEVELS
        ):
            errors.append(
                "Unsupported translation tag level: "
                f"level={level}, "
                f"position={match.start()}"
            )
            continue

        if not closing:
            if active_level is not None:
                errors.append(
                    "Nested translation tag: "
                    f"outer_level={active_level}, "
                    f"inner_level={level}, "
                    f"position={match.start()}"
                )
                continue

            active_level = level
            active_tag_start = match.start()
            value_start = match.end()
            continue

        if active_level is None:
            errors.append(
                "Unexpected translation closing tag: "
                f"level={level}, "
                f"position={match.start()}"
            )
            continue

        if level != active_level:
            errors.append(
                "Mismatched translation closing tag: "
                f"expected={active_level}, "
                f"actual={level}, "
                f"position={match.start()}"
            )

            active_level = None
            active_tag_start = None
            value_start = None
            continue

        if (
            active_tag_start is None
            or value_start is None
        ):
            raise RuntimeError(
                "Invalid translation tag parser state"
            )

        value_end = match.start()

        value = text[
            value_start:value_end
        ]

        if not value:
            errors.append(
                "Empty translation tag value: "
                f"level={level}, "
                f"position={active_tag_start}"
            )
        elif value != value.strip():
            errors.append(
                "Translation tag value has surrounding "
                "whitespace: "
                f"level={level}, "
                f"value={value!r}"
            )
        else:
            tags.append(
                TranslationEvaluationTag(
                    level=level,
                    value=value,
                    start=active_tag_start,
                    end=match.end(),
                    value_start=value_start,
                    value_end=value_end,
                )
            )

        active_level = None
        active_tag_start = None
        value_start = None

    if active_level is not None:
        errors.append(
            "Missing translation closing tag: "
            f"level={active_level}, "
            f"position={active_tag_start}"
        )

    return TranslationTagParseResult(
        tags=tuple(tags),
        errors=tuple(errors),
    )


def strip_translation_tags(
    text: str,
) -> str:
    """
    構造が正常な翻訳文からタグだけを除去する。

    タグ内の値はそのまま維持する。
    不正なタグ構造を含む場合は例外にする。
    """
    parse_result = parse_translation_tags(
        text
    )

    if parse_result.errors:
        raise ValueError(
            "Invalid translation tags: "
            + "; ".join(
                parse_result.errors
            )
        )

    return TRANSLATION_TAG_PATTERN.sub(
        "",
        text,
    )
