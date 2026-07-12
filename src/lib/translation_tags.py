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


@dataclass(frozen=True)
class TranslationTagProcessingResult:
    """
    自己評価タグの検証・除去結果。
    """

    translated_texts: tuple[str, ...]
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


def process_translation_tags(
    translated_texts: list[str],
    subtitle_ids: list[str],
    *,
    source_texts: list[str] | None,
) -> TranslationTagProcessingResult:
    """
    翻訳文の自己評価タグを検証し、
    正常な場合はタグだけを除去する。

    全レベルのタグ内部は、
    原文から一文字も変更せずコピーされている必要がある。

    source_textsがない状態でタグが存在する場合は、
    原文一致を確認できないためエラーにする。
    """
    if len(translated_texts) != len(
        subtitle_ids
    ):
        raise ValueError(
            "Translation tag input length mismatch: "
            f"translated={len(translated_texts)}, "
            f"ids={len(subtitle_ids)}"
        )

    if (
        source_texts is not None
        and len(source_texts)
        != len(translated_texts)
    ):
        raise ValueError(
            "Translation tag source length mismatch: "
            f"translated={len(translated_texts)}, "
            f"source={len(source_texts)}"
        )

    processed_texts: list[str] = []
    errors: list[str] = []

    for index, (
            subtitle_id,
            translated_text,
    ) in enumerate(
        zip(
            subtitle_ids,
            translated_texts,
            strict=True,
        )
    ):
        item_errors: list[str] = []

        parse_result = parse_translation_tags(
            translated_text
        )

        for error in parse_result.errors:
            item_errors.append(
                "Invalid translation evaluation tag: "
                f"subtitle_id={subtitle_id!r}, "
                f"error={error!r}, "
                f"text={translated_text!r}"
            )

        if (
            not parse_result.errors
            and parse_result.tags
            and source_texts is None
        ):
            item_errors.append(
                "Translation evaluation tags require "
                "source text: "
                f"subtitle_id={subtitle_id!r}, "
                f"text={translated_text!r}"
            )

        if (
            not parse_result.errors
            and source_texts is not None
        ):
            source_text = source_texts[
                index
            ]

            for tag in parse_result.tags:
                if tag.value in source_text:
                    continue

                item_errors.append(
                    "Translation evaluation tag value "
                    "not found in source: "
                    f"subtitle_id={subtitle_id!r}, "
                    f"level={tag.level}, "
                    f"value={tag.value!r}, "
                    f"source={source_text!r}"
                )

        if item_errors:
            errors.extend(
                item_errors
            )

            processed_texts.append(
                translated_text
            )
            continue

        processed_texts.append(
            strip_translation_tags(
                translated_text
            )
        )

    return TranslationTagProcessingResult(
        translated_texts=tuple(
            processed_texts
        ),
        errors=tuple(
            errors
        ),
    )
