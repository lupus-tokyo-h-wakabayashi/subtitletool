from __future__ import annotations

from lib.subtitle.srt import (
    SrtBlock,
    parse_speaker_from_text,
)


def build_translation_target_schema(
    block: SrtBlock,
) -> dict[str, object]:
    """
    1字幕分の翻訳レスポンスSchemaを生成する。

    sourceは入力時の値へ固定し、
    translationだけ空でない文字列として生成可能にする。
    """
    parsed = parse_speaker_from_text(
        block.text
    )

    return {
        "type": "object",
        "properties": {
            "source": {
                "type": "object",
                "properties": {
                    "speaker": {
                        "const": parsed.speaker,
                    },
                    "text": {
                        "const": parsed.text,
                    },
                },
                "required": [
                    "speaker",
                    "text",
                ],
                "additionalProperties": False,
            },
            "translation": {
                "type": "string",
                "minLength": 1,
            },
        },
        "required": [
            "source",
            "translation",
        ],
        "additionalProperties": False,
    }


def build_translation_response_schema(
    target_blocks: list[SrtBlock],
) -> dict[str, object]:
    """
    翻訳対象チャンク用のJSON Schemaを生成する。

    現行レスポンス形式を維持する。

    {
      "targets": {
        "<字幕ID>": {
          "source": {
            "speaker": null または文字列,
            "text": "原文"
          },
          "translation": "日本語字幕"
        }
      }
    }
    """
    if not target_blocks:
        raise ValueError(
            "Translation response schema requires "
            "at least one target block."
        )

    target_properties: dict[
        str,
        object,
    ] = {}

    required_target_ids: list[str] = []

    for block in target_blocks:
        subtitle_id = block.number

        if not subtitle_id.strip():
            raise ValueError(
                "Translation response schema contains "
                "an empty subtitle ID."
            )

        if subtitle_id in target_properties:
            raise ValueError(
                "Translation response schema contains "
                "a duplicate subtitle ID: "
                f"{subtitle_id!r}"
            )

        target_properties[
            subtitle_id
        ] = build_translation_target_schema(
            block
        )

        required_target_ids.append(
            subtitle_id
        )

    return {
        "type": "object",
        "properties": {
            "targets": {
                "type": "object",
                "properties": target_properties,
                "required": required_target_ids,
                "additionalProperties": False,
            },
        },
        "required": [
            "targets",
        ],
        "additionalProperties": False,
    }
