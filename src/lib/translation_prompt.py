import json

from lib.prompt import (
    build_translation_prompt,
)
from lib.srt import (
    SrtBlock,
)


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


def build_context_item(
    block: SrtBlock,
) -> dict[str, str | None]:
    """
    参考文脈をLLMリクエスト用JSONへ変換する。

    contextは出力対象ではないため、
    targetと混同されないようidを含めない。
    """
    parsed = parse_speaker_from_text(
        block.text
    )

    return {
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
            build_context_item(block)
            for block in before_context
        ],
        "target": [
            build_request_item(block)
            for block in target_blocks
        ],
        "context_after": [
            build_context_item(block)
            for block in after_context
        ],
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


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
        "OCR Noise IDs: "
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
    profile_name: str,
) -> str:
    request_json = build_translation_request_json(
        before_context,
        target_blocks,
        after_context,
    )

    base_prompt = build_translation_prompt(
        target_count=len(target_blocks),
        request_json=request_json,
        profile_name=profile_name,
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
