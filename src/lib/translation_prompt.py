from __future__ import annotations

import json

from lib.prompt import (
    build_translation_prompt,
)
from lib.srt import (
    SrtBlock,
    parse_speaker_from_text,
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
    suspicious_ids: list[str],
) -> str:
    """
    OCR破損候補がある字幕IDを
    LLMへ通知する指示文を生成する。
    """
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


def build_translation_evaluation_tag_instruction(
) -> str:
    """
    AIへ翻訳自己評価タグの仕様を伝える
    指示文を生成する。
    """
    return """

【翻訳自己評価タグ】

翻訳結果の中に、原文の英数字表記や
OCRで破損した可能性がある文字列を残す場合は、
次の3種類のタグだけを使用すること。

使用可能なタグ:

[5]原文文字列[/5]
[3]原文文字列[/3]
[1]原文文字列[/1]

[2]、[4]、その他の数字タグは使用禁止。

各タグの意味:

[5]
原文を一文字も変更せず、
そのまま保持する必要がある文字列。

対象例:
* 惑星コード
* 部隊名
* 型番
* 略語
* 固有の英数字識別子

例:
P4X-351
→ [5]P4X-351[/5]

SG-1
→ [5]SG-1[/5]

DNA
→ [5]DNA[/5]

[5]は「Glossaryに登録された語」を示すタグではない。
Glossaryによって日本語へ置換すべき用語には、
必ずしも[5]を付ける必要はない。

例:
SGCをGlossaryによって
「スターゲイト司令部」と訳す場合は、
[5]SGC[/5]とはせず、
Glossaryの指定訳を使用すること。

[5]は、翻訳後も原文表記をそのまま残す
最小単位だけを囲むこと。
英文全体や、識別子の前後の単語を含めてはいけない。

正しい例:
惑星[5]P4X-351[/5]のコア

誤った例:
[5]The planet P4X-351[/5]
[5]P4X-351 was[/5]

[3]
翻訳可能か、保持すべき表記か、
OCRノイズかを判断できない文字列。

判断できない原文部分だけを、
一文字も変更せず囲むこと。

[1]
OCRで破損した可能性が高く、
意味のある翻訳ができない文字列。

OCRノイズと思われる原文部分だけを、
一文字も変更せず囲むこと。

例:
[1]sie lexer=s-4-9 10) WV am nat (el=)[/1]

共通ルール:

* タグ内部の文字列は原文からそのままコピーする
* 大文字小文字、空白、数字、記号を変更しない
* タグ内部の先頭と末尾へ余分な空白を入れない
* タグをネストしない
* 空のタグを出力しない
* 開始タグと終了タグの数字を一致させる
* 通常の日本語訳全体をタグで囲まない
* 翻訳可能な通常の英文は日本語へ翻訳する
* JSON構造は指定形式を厳守する
"""


def build_prompt(
    before_context: list[SrtBlock],
    target_blocks: list[SrtBlock],
    after_context: list[SrtBlock],
    profile_name: str,
    ocr_noise_instruction: str = "",
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

    evaluation_tag_instruction = (
        build_translation_evaluation_tag_instruction()
    )

    return (
        base_prompt
        + evaluation_tag_instruction
        + ocr_noise_instruction
    )
