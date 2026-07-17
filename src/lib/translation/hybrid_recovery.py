from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Mapping

from lib.infrastructure.ollama import generate
from lib.profile.noise import NoiseDictionary
from lib.subtitle.srt import (
    SrtBlock,
    parse_speaker_from_text,
)
from .hybrid_group import (
    HybridTranslationGroup,
    build_hybrid_translation_group,
)
from .hybrid_inspection import (
    try_save_hybrid_attempt_report,
)
from .ocr_retry import (
    is_probable_ocr_source_line,
)
from .retry import (
    build_required_glossary_instruction,
    extract_error_subtitle_ids,
    has_structural_validation_error,
)
from .translation_validation import (
    validate_translation_response,
)

MAX_HYBRID_RECOVERY_ATTEMPTS = 3

HYBRID_OCR_PLACEHOLDER = (
    "（判読不能）"
)

JAPANESE_CHARACTER_PATTERN = re.compile(
    r"[ぁ-んァ-ヶ一-龠々ー]"
)


class HybridRecoveryError(
    RuntimeError
):
    """
    Hybrid Recoveryが対象グループを
    規定回数内に回復できなかったことを表す。
    """


@dataclass(frozen=True)
class HybridValidationResult:
    """
    Hybridレスポンスの検証結果。
    """

    valid: bool
    reasons: tuple[str, ...]
    full_translation: str | None
    segments: dict[str, str]


def build_hybrid_response_schema(
    group: HybridTranslationGroup,
) -> dict[str, object]:
    """
    Hybrid Recovery用JSON Schemaを生成する。
    """
    segment_properties = {
        subtitle_id: {
            "type": "string",
            "minLength": 1,
        }
        for subtitle_id in group.target_ids
    }

    return {
        "type": "object",
        "properties": {
            "group": {
                "type": "object",
                "properties": {
                    "full_translation": {
                        "type": "string",
                        "minLength": 1,
                    },
                    "segments": {
                        "type": "object",
                        "properties": (
                            segment_properties
                        ),
                        "required": list(
                            group.target_ids
                        ),
                        "additionalProperties": False,
                    },
                },
                "required": [
                    "full_translation",
                    "segments",
                ],
                "additionalProperties": False,
            },
        },
        "required": [
            "group",
        ],
        "additionalProperties": False,
    }


def find_group_ocr_lines(
    group: HybridTranslationGroup,
    noise_dictionary: NoiseDictionary,
) -> dict[str, list[str]]:
    """
    Hybridグループ内の高確度OCR行を抽出する。
    """
    results: dict[
        str,
        list[str],
    ] = {}

    for block in group.blocks:
        lines: list[str] = []

        for raw_line in block.text.splitlines():
            source_line = raw_line.strip()

            if not source_line:
                continue

            if not is_probable_ocr_source_line(
                source_line,
                noise_dictionary,
            ):
                continue

            lines.append(
                source_line
            )

        if lines:
            results[
                block.number
            ] = lines

    return results


def find_group_text_lines(
    group: HybridTranslationGroup,
    ocr_lines: dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    Hybridグループ内から、
    OCR行として分類されていない正常行を抽出する。

    OCR行と正常英文が同じ字幕IDに混在する場合の
    検証に使用する。
    """
    results: dict[
        str,
        list[str],
    ] = {}

    for block in group.blocks:
        block_ocr_lines = set(
            ocr_lines.get(
                block.number,
                [],
            )
        )

        text_lines: list[str] = []

        for raw_line in block.text.splitlines():
            source_line = raw_line.strip()

            if not source_line:
                continue

            if source_line in block_ocr_lines:
                continue

            text_lines.append(
                source_line
            )

        if text_lines:
            results[
                block.number
            ] = text_lines

    return results


def build_hybrid_source_payload(
    group: HybridTranslationGroup,
    ocr_lines: dict[str, list[str]],
) -> dict[str, object]:
    """
    Hybrid Promptへ渡す原文を、
    通常行とOCR行に分類して構造化する。
    """
    subtitles: list[
        dict[str, object]
    ] = []

    for block in group.blocks:
        block_ocr_lines = set(
            ocr_lines.get(
                block.number,
                [],
            )
        )

        lines = []

        for raw_line in block.text.splitlines():
            source_line = raw_line.strip()

            if not source_line:
                continue

            lines.append(
                {
                    "kind": (
                        "ocr"
                        if source_line
                           in block_ocr_lines
                        else "text"
                    ),
                    "text": source_line,
                }
            )

        subtitles.append(
            {
                "id": block.number,
                "lines": lines,
            }
        )

    return {
        "subtitles": subtitles,
    }


def build_hybrid_translation_prompt(
    group: HybridTranslationGroup,
    ocr_lines: dict[str, list[str]],
    glossary_entries: Mapping[str, str],
    *,
    retry_reasons: list[str] | None = None,
) -> str:
    """
    Hybrid Recovery用Promptを生成する。
    """
    source_payload = (
        build_hybrid_source_payload(
            group,
            ocr_lines,
        )
    )

    source_json = json.dumps(
        source_payload,
        ensure_ascii=False,
        indent=2,
    )

    target_ids_json = json.dumps(
        list(group.target_ids),
        ensure_ascii=False,
    )

    glossary_instruction = (
        build_required_glossary_instruction(
            list(group.blocks),
            glossary_entries,
        )
    )

    retry_instruction = ""

    if retry_reasons:
        details = "\n".join(
            f"* {reason}"
            for reason in retry_reasons
        )

        retry_instruction = f"""

【前回のHybrid出力の修正】

前回は次の問題で検証に失敗した。

{details}

同じ問題を繰り返さず、
full_translationとsegmentsを
両方とも修正すること。
"""

    return f"""
あなたは英語字幕を日本語字幕へ翻訳する。

今回の字幕は複数のタイムスタンプに分割されているが、
全体として1つ以上の連続した文章を構成している。

最初にグループ全体の意味を理解して、
自然な日本語の全文訳を作ること。

その後、全文訳を字幕IDごとのsegmentsへ分割すること。

【出力形式】

JSONオブジェクト1個だけを出力する。

最上位キーはgroupだけにする。

groupのキーは次の2つだけにする。

* full_translation
* segments

segmentsへ出力する字幕IDは、
次のIDだけとする。

{target_ids_json}

字幕IDを追加、削除、変更、重複、並べ替えしないこと。

【full_translation】

* グループ全体を自然な日本語へ翻訳する
* 原文の一部を省略しない
* 原文にない内容を追加しない
* 正常な英文を残さない
* OCR文字列をコピーしない
* segmentsをID順に連結した内容と一致させる

【segments】

* 各IDへ空でない日本語字幕を割り当てる
* 原文のID境界に機械的に縛られず、
  日本語として自然な位置で分割する
* 話の順序を変更しない
* 別のIDへ同じ全文訳を繰り返さない
* 英文を残さない
* 字幕区切りには必要に応じて全角スラッシュを使う
* OCR行のある字幕には必ず「（判読不能）」を配置する
* OCRの原文文字列をsegmentsへコピーしない
* OCR行以外の正常な英文は必ず日本語へ翻訳する
* OCR行と正常英文が同じIDにある場合は、
  「（判読不能）」と正常英文の日本語訳を両方含める
* OCR行と正常英文が同じIDにある場合に、
  segment全体を「（判読不能）」だけにしない
* 各segmentをID順に連結すると、
  full_translationと一致するようにする

【OCR行】

入力のkindがocrの行は、
意味を推測して翻訳しないこと。

kindがocrの行は、
対応する字幕IDで必ず「（判読不能）」と表現すること。

同じ字幕IDにkindがtextの行もある場合、
その正常な英文は必ず日本語へ翻訳すること。

例:

入力:

字幕ID 563

* kind=ocr:
  aR at-lacmanl-e
* kind=text:
  lam a good friend.

segments["563"]:

（判読不能）／私は良い友人です。

禁止例:

* （判読不能）
* ラムは良い友人です。
* aR at-lacmanl-e
* [1]aR at-lacmanl-e[/1]

【full_translation】

full_translationには、
segmentsの値を字幕ID順に連結した文字列を
一字一句そのまま入れること。

segmentsに「（判読不能）」がある場合は、
full_translationから省略しないこと。

segmentsを作った後、
その値をID順に連結して
full_translationへコピーすること。

【入力】

{source_json}

{glossary_instruction}

{retry_instruction}
""".strip()


def normalize_hybrid_join_text(
    text: str,
) -> str:
    """
    full_translationとsegmentsの連結比較用に、
    空白・改行・字幕区切りだけを除去する。
    """
    return re.sub(
        r"[\s／/]+",
        "",
        text,
    )


def validate_hybrid_response(
    response: str,
    group: HybridTranslationGroup,
    ocr_lines: dict[str, list[str]],
) -> HybridValidationResult:
    """
    Hybridレスポンスを検証する。

    検証内容:

    - JSON構造
    - 字幕IDと順序
    - segmentが空でない
    - segmentに日本語がある
    - OCR原文をコピーしていない
    - OCR行のあるIDに
      「（判読不能）」がある
    - OCR行と正常行が混在するIDに、
      判読不能以外の日本語訳がある
    - segmentsから全文を再構築する

    full_translationはLLMの文章理解用フィールドだが、
    実際の出力にはsegmentsだけを使用する。

    segmentsがすべて有効な場合は、
    segmentsをID順に連結した内容を
    正規のfull_translationとして扱う。
    """
    reasons: list[str] = []

    try:
        payload = json.loads(
            response.strip()
        )
    except json.JSONDecodeError as error:
        return HybridValidationResult(
            valid=False,
            reasons=(
                "Invalid Hybrid JSON: "
                f"line={error.lineno}, "
                f"column={error.colno}, "
                f"message={error.msg}",
            ),
            full_translation=None,
            segments={},
        )

    if not isinstance(payload, dict):
        return HybridValidationResult(
            valid=False,
            reasons=(
                "Invalid Hybrid root: "
                "expected object",
            ),
            full_translation=None,
            segments={},
        )

    if set(payload.keys()) != {
        "group",
    }:
        reasons.append(
            "Invalid Hybrid root keys: "
            f"actual={sorted(payload.keys())}"
        )

    group_payload = payload.get(
        "group"
    )

    if not isinstance(
        group_payload,
        dict,
    ):
        reasons.append(
            "Invalid Hybrid group: "
            "expected object"
        )

        return HybridValidationResult(
            valid=False,
            reasons=tuple(reasons),
            full_translation=None,
            segments={},
        )

    expected_group_keys = {
        "full_translation",
        "segments",
    }

    if (
        set(group_payload.keys())
        != expected_group_keys
    ):
        reasons.append(
            "Invalid Hybrid group keys: "
            f"expected={sorted(expected_group_keys)}, "
            f"actual={sorted(group_payload.keys())}"
        )

    raw_full_translation = group_payload.get(
        "full_translation"
    )

    if not isinstance(
        raw_full_translation,
        str,
    ):
        reasons.append(
            "Invalid Hybrid full_translation: "
            "expected string"
        )

        full_translation = None
    else:
        full_translation = (
            raw_full_translation.strip()
        )

        if not full_translation:
            reasons.append(
                "Empty Hybrid full_translation"
            )

    raw_segments = group_payload.get(
        "segments"
    )

    if not isinstance(
        raw_segments,
        dict,
    ):
        reasons.append(
            "Invalid Hybrid segments: "
            "expected object"
        )

        return HybridValidationResult(
            valid=False,
            reasons=tuple(reasons),
            full_translation=(
                full_translation
                if isinstance(
                    full_translation,
                    str,
                )
                else None
            ),
            segments={},
        )

    actual_ids = list(
        raw_segments.keys()
    )

    expected_ids = list(
        group.target_ids
    )

    if actual_ids != expected_ids:
        reasons.append(
            "Invalid Hybrid segment IDs: "
            f"expected={expected_ids}, "
            f"actual={actual_ids}"
        )

    text_lines = find_group_text_lines(
        group,
        ocr_lines,
    )

    segments: dict[str, str] = {}

    for subtitle_id in expected_ids:
        segment = raw_segments.get(
            subtitle_id
        )

        if not isinstance(
            segment,
            str,
        ):
            reasons.append(
                "Invalid Hybrid segment: "
                f"subtitle_id={subtitle_id!r}, "
                "expected string"
            )
            continue

        normalized_segment = segment.strip()

        if not normalized_segment:
            reasons.append(
                "Empty Hybrid segment: "
                f"subtitle_id={subtitle_id!r}"
            )
            continue

        if not JAPANESE_CHARACTER_PATTERN.search(
            normalized_segment
        ):
            reasons.append(
                "Hybrid segment requires Japanese: "
                f"subtitle_id={subtitle_id!r}, "
                f"text={normalized_segment!r}"
            )

        block_ocr_lines = ocr_lines.get(
            subtitle_id,
            [],
        )

        for ocr_line in block_ocr_lines:
            if ocr_line in normalized_segment:
                reasons.append(
                    "Hybrid segment contains OCR source: "
                    f"subtitle_id={subtitle_id!r}, "
                    f"text={ocr_line!r}"
                )

        if (
            block_ocr_lines
            and HYBRID_OCR_PLACEHOLDER
            not in normalized_segment
        ):
            reasons.append(
                "Hybrid OCR placeholder missing: "
                f"subtitle_id={subtitle_id!r}, "
                f"required="
                f"{HYBRID_OCR_PLACEHOLDER!r}"
            )

        block_text_lines = text_lines.get(
            subtitle_id,
            [],
        )

        if (
            block_ocr_lines
            and block_text_lines
        ):
            translation_without_placeholder = (
                normalized_segment.replace(
                    HYBRID_OCR_PLACEHOLDER,
                    "",
                )
            )

            if not JAPANESE_CHARACTER_PATTERN.search(
                translation_without_placeholder
            ):
                reasons.append(
                    "Hybrid mixed OCR segment "
                    "requires Japanese translation: "
                    f"subtitle_id={subtitle_id!r}, "
                    f"source_lines="
                    f"{block_text_lines!r}, "
                    f"text={normalized_segment!r}"
                )

        segments[
            subtitle_id
        ] = normalized_segment

    reconstructed_full_translation: (
        str | None
    ) = None

    if len(segments) == len(
        expected_ids
    ):
        reconstructed_full_translation = (
            "".join(
                segments[subtitle_id]
                for subtitle_id in expected_ids
            )
        )

    return HybridValidationResult(
        valid=not reasons,
        reasons=tuple(reasons),
        full_translation=(
            reconstructed_full_translation
            if reconstructed_full_translation
               is not None
            else full_translation
        ),
        segments=segments,
    )


def build_standard_translation_response(
    target_blocks: list[SrtBlock],
    translated_texts: list[str],
) -> str:
    """
    Hybrid適用後のチャンク全体を、
    既存Validator用JSONへ変換する。
    """
    targets: dict[
        str,
        object,
    ] = {}

    for block, translation in zip(
        target_blocks,
        translated_texts,
        strict=True,
    ):
        parsed = parse_speaker_from_text(
            block.text
        )

        targets[
            block.number
        ] = {
            "source": {
                "speaker": parsed.speaker,
                "text": parsed.text,
            },
            "translation": translation,
        }

    return json.dumps(
        {
            "targets": targets,
        },
        ensure_ascii=False,
    )


def recover_translation_with_hybrid(
    target_blocks: list[SrtBlock],
    translated_texts: list[str],
    errors: list[str],
    model: str,
    *,
    noise_dictionary: NoiseDictionary,
    glossary_entries: Mapping[str, str],
) -> list[str] | None:
    """
    通常翻訳に失敗したチャンクを、
    連続文グループ全文翻訳で回復する。

    各Hybrid試行について、
    Prompt、Schema、レスポンス、
    検証結果をtmpへ保存する。

    次の制約を設ける。

    - 現在のチャンク内だけを対象とする
    - すべての失敗IDを1グループへまとめる
    - 最大6字幕
    - 最大2回試行
    """
    if not errors:
        return None

    if has_structural_validation_error(
        errors
    ):
        return None

    if len(translated_texts) != len(
        target_blocks
    ):
        return None

    failed_ids = extract_error_subtitle_ids(
        errors
    )

    if not failed_ids:
        return None

    group = build_hybrid_translation_group(
        target_blocks,
        failed_ids,
    )

    if group is None:
        return None

    ocr_lines = find_group_ocr_lines(
        group,
        noise_dictionary,
    )

    retry_reasons: list[str] = []

    for attempt in range(
        1,
        MAX_HYBRID_RECOVERY_ATTEMPTS + 1,
    ):
        prompt = (
            build_hybrid_translation_prompt(
                group,
                ocr_lines,
                glossary_entries,
                retry_reasons=retry_reasons,
            )
        )

        response_schema = (
            build_hybrid_response_schema(
                group
            )
        )

        print()
        print(
            "Hybrid Translation Recovery:"
        )
        print(
            f"  Attempt: {attempt}/"
            f"{MAX_HYBRID_RECOVERY_ATTEMPTS}"
        )
        print(
            "  IDs: "
            + ", ".join(
                group.target_ids
            )
        )

        response = generate(
            prompt,
            model=model,
            response_format=response_schema,
        )

        print("=" * 80)
        print(response)
        print("=" * 80)

        hybrid_validation = (
            validate_hybrid_response(
                response,
                group,
                ocr_lines,
            )
        )

        if not hybrid_validation.valid:
            retry_reasons = list(
                hybrid_validation.reasons
            )

            try_save_hybrid_attempt_report(
                group=group,
                model=model,
                attempt=attempt,
                prompt=prompt,
                response_schema=response_schema,
                response=response,
                ocr_lines=ocr_lines,
                validation_stage=(
                    "hybrid_validation"
                ),
                validation_valid=False,
                validation_reasons=(
                    retry_reasons
                ),
            )

            print(
                "Hybrid validation failed:"
            )

            for reason in retry_reasons:
                print(f"  - {reason}")

            continue

        merged_texts = list(
            translated_texts
        )

        for position, block in zip(
            group.positions,
            group.blocks,
            strict=True,
        ):
            merged_texts[
                position
            ] = (
                hybrid_validation.segments[
                    block.number
                ]
            )

        standard_response = (
            build_standard_translation_response(
                target_blocks,
                merged_texts,
            )
        )

        source_speakers: list[
            str | None
            ] = []

        source_texts: list[str] = []

        for block in target_blocks:
            parsed = parse_speaker_from_text(
                block.text
            )

            source_speakers.append(
                parsed.speaker
            )

            source_texts.append(
                parsed.text
            )

        standard_validation = (
            validate_translation_response(
                standard_response,
                expected_ids=[
                    block.number
                    for block in target_blocks
                ],
                source_speakers=source_speakers,
                source_texts=source_texts,
                noise_dictionary=(
                    noise_dictionary
                ),
                glossary_entries=(
                    glossary_entries
                ),
            )
        )

        if not standard_validation.valid:
            retry_reasons = list(
                standard_validation.reasons
            )

            try_save_hybrid_attempt_report(
                group=group,
                model=model,
                attempt=attempt,
                prompt=prompt,
                response_schema=response_schema,
                response=response,
                ocr_lines=ocr_lines,
                validation_stage=(
                    "standard_validation"
                ),
                validation_valid=False,
                validation_reasons=(
                    retry_reasons
                ),
            )

            print(
                "Hybrid result failed standard "
                "validation:"
            )

            for reason in retry_reasons:
                print(f"  - {reason}")

            continue

        try_save_hybrid_attempt_report(
            group=group,
            model=model,
            attempt=attempt,
            prompt=prompt,
            response_schema=response_schema,
            response=response,
            ocr_lines=ocr_lines,
            validation_stage="complete",
            validation_valid=True,
            validation_reasons=[],
        )

        print(
            "Hybrid Translation Recovery "
            "succeeded:"
        )

        print(
            "  IDs: "
            + ", ".join(
                group.target_ids
            )
        )

        return (
            standard_validation.translated_texts
        )

    raise HybridRecoveryError(
        "Hybrid Translation Recovery failed "
        f"after "
        f"{MAX_HYBRID_RECOVERY_ATTEMPTS} "
        "attempts for subtitles "
        + ", ".join(
            group.target_ids
        )
        + ": "
        + "; ".join(
            retry_reasons
        )
    )
