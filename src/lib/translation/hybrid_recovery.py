from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Mapping

from lib.infrastructure.ollama import generate
from lib.profile.noise import NoiseDictionary
from lib.profile.ocr_scoring import (
    OcrScoringConfig,
    load_ocr_scoring_config,
)
from lib.subtitle.srt import (
    SrtBlock,
    parse_speaker_from_text,
)
from lib.subtitle.text import (
    is_sound_effect_line,
)
from .hybrid_group import (
    HybridTranslationGroup,
    build_hybrid_translation_groups,
)
from .hybrid_inspection import (
    try_save_hybrid_attempt_report,
)
from .ocr_retry import (
    find_assessed_ocr_lines_in_source,
)
from .retry import (
    build_required_glossary_instruction,
    extract_error_subtitle_ids,
    has_structural_validation_error,
)
from .translation_metrics import (
    HybridGroupMetric,
    TranslationAttemptMetric,
    TranslationChunkMetric,
    build_validation_reason_codes,
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
HYBRID_TRANSLATED_SOUND_EFFECT_PATTERN = (
    re.compile(
        r"（(?P<content>[^（）]+)）"
    )
)
HYBRID_SOUND_EFFECT_ONLY_SEGMENT_PATTERN = (
    re.compile(
        r"^"
        r"(?:（[^（）]+）)"
        r"(?:[\s／]*（[^（）]+）)*"
        r"$"
    )
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


def find_group_ocr_lines_with_assessment(
    group: HybridTranslationGroup,
    glossary_entries: Mapping[
        str,
        str,
    ],
    scoring_config: OcrScoringConfig,
) -> dict[str, list[str]]:
    """
    Hybridグループの字幕原文を
    統合OCR評価器で分類する。

    通常翻訳でValidationに失敗した字幕には
    失敗字幕用の閾値を適用する。

    グループへ文脈として追加された
    失敗していない字幕には、
    高確度OCR用の閾値を適用する。

    効果音行は評価対象から除外する。
    """
    results: dict[
        str,
        list[str],
    ] = {}

    for block in group.blocks:
        source_lines = [
            raw_line.strip()
            for raw_line in (
                block.text.splitlines()
            )
            if (
                raw_line.strip()
                and not (
                is_sound_effect_line(
                    raw_line.strip()
                )
            )
            )
        ]

        if not source_lines:
            continue

        lines = (
            find_assessed_ocr_lines_in_source(
                "\n".join(
                    source_lines
                ),
                glossary_entries,
                scoring_config,
                validation_failed=(
                    block.number
                    in group.failed_ids
                ),
            )
        )

        if not lines:
            continue

        results[
            block.number
        ] = lines

    return results


def find_group_ocr_lines(
    group: HybridTranslationGroup,
    *,
    glossary_entries: Mapping[
        str,
        str,
    ],
    scoring_config: OcrScoringConfig,
) -> dict[str, list[str]]:
    """
    Hybridグループ内のOCR行を
    統合OCR評価器で分類して返す。
    """
    return (
        find_group_ocr_lines_with_assessment(
            group,
            glossary_entries,
            scoring_config,
        )
    )


def find_group_sound_effect_lines(
    group: HybridTranslationGroup,
) -> dict[str, list[str]]:
    """
    Hybridグループ内の効果音行を
    字幕IDごとに抽出する。
    """
    results: dict[
        str,
        list[str],
    ] = {}

    for block in group.blocks:
        sound_effect_lines: list[str] = []

        for raw_line in block.text.splitlines():
            source_line = raw_line.strip()

            if not source_line:
                continue

            if not is_sound_effect_line(
                source_line
            ):
                continue

            if source_line in sound_effect_lines:
                continue

            sound_effect_lines.append(
                source_line
            )

        if sound_effect_lines:
            results[
                block.number
            ] = sound_effect_lines

    return results


def find_group_text_lines(
    group: HybridTranslationGroup,
    ocr_lines: dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    Hybridグループ内から、
    OCR行・効果音行として分類されていない
    正常行を抽出する。

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

            if is_sound_effect_line(
                source_line
            ):
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
    通常行・OCR行・効果音行に分類して
    構造化する。
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

            if is_sound_effect_line(
                source_line
            ):
                line_kind = "sound_effect"
            elif source_line in block_ocr_lines:
                line_kind = "ocr"
            else:
                line_kind = "text"

            lines.append(
                {
                    "kind": line_kind,
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


def build_hybrid_context_payload(
    blocks: list[SrtBlock] | None,
) -> list[dict[str, str]]:
    """
    Hybrid Recoveryの参考文脈を
    Prompt入力用のPayloadへ変換する。

    Contextは翻訳結果の出力対象ではないため、
    字幕IDと原文だけを保持する。
    """
    if blocks is None:
        return []

    return [
        {
            "id": block.number,
            "text": block.text,
        }
        for block in blocks
    ]


def build_hybrid_segment_requirements(
    group: HybridTranslationGroup,
    ocr_lines: dict[str, list[str]],
) -> str:
    """
    Hybrid Promptへ追加する、
    字幕IDごとの出力必須条件を生成する。

    通常文・OCR行・効果音行の分類結果を使い、
    各segmentへ必要な内容を具体的に指示する。
    """
    text_lines = find_group_text_lines(
        group,
        ocr_lines,
    )

    sound_effect_lines = (
        find_group_sound_effect_lines(
            group
        )
    )

    requirements: list[str] = []

    for block in group.blocks:
        subtitle_id = block.number

        has_text_lines = bool(
            text_lines.get(
                subtitle_id,
                [],
            )
        )

        has_ocr_lines = bool(
            ocr_lines.get(
                subtitle_id,
                [],
            )
        )

        has_sound_effect_lines = bool(
            sound_effect_lines.get(
                subtitle_id,
                [],
            )
        )

        if (
            has_sound_effect_lines
            and not has_text_lines
            and not has_ocr_lines
        ):
            requirements.append(
                f"* 字幕ID {subtitle_id}: "
                "kind=sound_effectだけなので、"
                "効果音を短い日本語へ翻訳し、"
                "segment全体を全角括弧で囲む。"
                "原文にない会話や説明を追加しない。"
            )
            continue

        instructions: list[str] = []

        if has_text_lines:
            instructions.append(
                "kind=textの正常英文を"
                "自然な日本語へ翻訳する。"
                "英文を残さない。"
            )

        if has_sound_effect_lines:
            instructions.append(
                "kind=sound_effectを"
                "短い日本語の効果音へ翻訳し、"
                "その部分を全角括弧で囲む。"
            )

        if has_ocr_lines:
            instructions.append(
                "kind=ocrの位置を"
                "「（判読不能）」で表現し、"
                "OCR原文をコピーしない。"
            )

        if (
            has_ocr_lines
            and (
            has_text_lines
            or has_sound_effect_lines
        )
        ):
            instructions.append(
                "segmentには"
                "「（判読不能）」と、"
                "それ以外の翻訳結果を"
                "両方とも含める。"
            )

        instructions.append(
            "各行の内容を原文順に配置する。"
        )

        requirements.append(
            f"* 字幕ID {subtitle_id}: "
            + "".join(
                instructions
            )
        )

    return "\n".join(
        requirements
    )


def build_hybrid_ocr_instruction(
    ocr_lines: dict[str, list[str]],
) -> str:
    """
    OCR行が存在する場合だけ、
    Hybrid PromptへOCR指示を追加する。
    """
    if not ocr_lines:
        return ""

    return """
【OCR行】

入力のkindがocrの行は、
意味を推測して翻訳しないこと。

kindがocrの行は、
対応する字幕IDで必ず
「（判読不能）」と表現すること。

OCR原文をsegmentsへコピーしないこと。

同じ字幕IDにkind=textの行もある場合、
その正常な英文は必ず日本語へ翻訳すること。

同じ字幕IDにkind=sound_effectの行もある場合、
その効果音も日本語の括弧形式へ翻訳すること。

OCR行と他の種類の行が同じIDにある場合、
segment全体を「（判読不能）」だけにしないこと。
""".strip()


def build_hybrid_sound_effect_instruction(
    group: HybridTranslationGroup,
) -> str:
    """
    効果音行が存在する場合だけ、
    Hybrid Promptへ効果音指示を追加する。
    """
    sound_effect_lines = (
        find_group_sound_effect_lines(
            group
        )
    )

    if not sound_effect_lines:
        return ""

    return """
【効果音行】

入力のkindがsound_effectの行は、
短く自然な日本語の効果音・動作説明へ翻訳すること。

効果音部分は全角括弧で囲むこと。

例:

(CHIRPING)
→
（電子音）

原文の英語効果音を残さないこと。

原文にない会話、人物、状況説明を追加しないこと。

字幕IDがkind=sound_effectだけで構成される場合は、
segment全体を1つ以上の全角括弧表現にすること。

同じ字幕IDにkind=textもある場合は、
効果音と会話文の両方を原文順に含めること。
""".strip()


def build_hybrid_translation_prompt(
    group: HybridTranslationGroup,
    ocr_lines: dict[str, list[str]],
    glossary_entries: Mapping[str, str],
    *,
    before_context: list[SrtBlock] | None = None,
    after_context: list[SrtBlock] | None = None,
    retry_reasons: list[str] | None = None,
) -> str:
    """
    Hybrid Recovery用Promptを生成する。

    OCR指示と効果音指示は、
    対象グループに該当する行がある場合だけ追加する。
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

    before_context_payload = (
        build_hybrid_context_payload(
            before_context
        )
    )

    after_context_payload = (
        build_hybrid_context_payload(
            after_context
        )
    )

    before_context_json = json.dumps(
        before_context_payload,
        ensure_ascii=False,
        indent=2,
    )

    after_context_json = json.dumps(
        after_context_payload,
        ensure_ascii=False,
        indent=2,
    )

    target_ids_json = json.dumps(
        list(group.target_ids),
        ensure_ascii=False,
    )

    segment_requirements = (
        build_hybrid_segment_requirements(
            group,
            ocr_lines,
        )
    )

    ocr_instruction = (
        build_hybrid_ocr_instruction(
            ocr_lines
        )
    )

    sound_effect_instruction = (
        build_hybrid_sound_effect_instruction(
            group
        )
    )

    glossary_instruction = (
        build_required_glossary_instruction(
            list(group.blocks),
            glossary_entries,
        )
    )

    optional_instructions = "\n\n".join(
        instruction
        for instruction in (
            ocr_instruction,
            sound_effect_instruction,
            glossary_instruction,
        )
        if instruction.strip()
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
""".strip()

    prompt_sections = [
        f"""
あなたは英語字幕を日本語字幕へ翻訳する。

今回の入力は、
複数のタイムスタンプに分割された会話、
単独の字幕、効果音、OCR破損行を含む場合がある。

最初にグループ全体の字幕内容を理解して、
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

【参考文脈】

context_beforeとcontext_afterは、
翻訳対象の前後関係を理解するための参考情報である。

context_beforeとcontext_afterの字幕は、
full_translationとsegmentsへ出力しないこと。

segmentsへ出力するのは、
指定された字幕IDだけとする。

翻訳対象が文の途中で終わっている場合や、
前後の字幕へ文章が続いている場合は、
参考文脈を使って意味を判断し、
翻訳対象部分だけを自然な日本語へ翻訳すること。

【full_translation】

* グループ全体の字幕内容を自然な日本語へ翻訳する
* 原文の一部を省略しない
* 原文にない内容を追加しない
* 正常な英文を残さない
* segmentsをID順に連結した内容と一致させる

【segments】

* 各IDへ空でない日本語字幕を割り当てる
* 原文のID境界に機械的に縛られず、
  日本語として自然な位置で分割する
* 話と字幕内容の順序を変更しない
* 別のIDへ同じ全文訳を繰り返さない
* 正常な英文を残さない
* 字幕区切りには必要に応じて全角スラッシュを使う
* 各segmentをID順に連結すると、
  full_translationと一致するようにする

【字幕IDごとの必須条件】

{segment_requirements}
""".strip(),
    ]

    if optional_instructions:
        prompt_sections.append(
            optional_instructions
        )

    prompt_sections.append(
        f"""
【full_translationの組み立て】

full_translationには、
segmentsの値を字幕ID順に連結した文字列を
一字一句そのまま入れること。

segmentsを作った後、
その値をID順に連結して
full_translationへコピーすること。

【参考文脈（前）】

{before_context_json}

【翻訳対象】

{source_json}

【参考文脈（後）】

{after_context_json}
""".strip()
    )

    if retry_instruction:
        prompt_sections.append(
            retry_instruction
        )

    return "\n\n".join(
        prompt_sections
    )


def normalize_hybrid_parentheses(
    text: str,
) -> str:
    """
    Hybrid翻訳結果の半角括弧を
    日本語字幕用の全角括弧へ統一する。
    """
    return (
        text.replace(
            "(",
            "（",
        ).replace(
            ")",
            "）",
        )
    )


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


def find_hybrid_sound_effect_segment_violations(
    subtitle_id: str,
    segment: str,
    *,
    raw_segment: str,
    source_sound_effect_lines: list[str],
    source_text_lines: list[str],
    source_ocr_lines: list[str],
) -> list[str]:
    """
    Hybrid segment内の効果音翻訳を検証する。

    効果音行が存在しない字幕では何も返さない。

    効果音だけの字幕では、
    segment全体を日本語の全角括弧表現に限定する。

    通常文・OCR行との混在字幕では、
    segment内に日本語の全角括弧表現が
    1つ以上存在することを必須とする。
    """
    if not source_sound_effect_lines:
        return []

    violations: list[str] = []

    for source_sound_effect in (
        source_sound_effect_lines
    ):
        if source_sound_effect not in raw_segment:
            continue

        violations.append(
            "Hybrid segment contains sound "
            "effect source: "
            f"subtitle_id={subtitle_id!r}, "
            f"text={source_sound_effect!r}"
        )

    translated_effect_contents = [
        match.group(
            "content"
        )
        for match in (
            HYBRID_TRANSLATED_SOUND_EFFECT_PATTERN
            .finditer(
                segment
            )
        )
    ]

    if not translated_effect_contents:
        violations.append(
            "Hybrid sound effect translation "
            "missing: "
            f"subtitle_id={subtitle_id!r}, "
            f"text={segment!r}"
        )

        return violations

    non_japanese_effects = [
        effect_content
        for effect_content
        in translated_effect_contents
        if not JAPANESE_CHARACTER_PATTERN.search(
            effect_content
        )
    ]

    if non_japanese_effects:
        violations.append(
            "Hybrid sound effect requires "
            "Japanese translation: "
            f"subtitle_id={subtitle_id!r}, "
            f"values={non_japanese_effects!r}, "
            f"text={segment!r}"
        )

    is_sound_effect_only = (
        not source_text_lines
        and not source_ocr_lines
    )

    if (
        is_sound_effect_only
        and not (
        HYBRID_SOUND_EFFECT_ONLY_SEGMENT_PATTERN
            .fullmatch(
            segment
        )
    )
    ):
        violations.append(
            "Hybrid sound-effect-only segment "
            "must contain only fullwidth "
            "parenthesized effects: "
            f"subtitle_id={subtitle_id!r}, "
            f"text={segment!r}"
        )

    return violations


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

    sound_effect_lines = (
        find_group_sound_effect_lines(
            group
        )
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

        raw_segment = segment.strip()

        if not raw_segment:
            reasons.append(
                "Empty Hybrid segment: "
                f"subtitle_id={subtitle_id!r}"
            )
            continue

        block_sound_effect_lines = (
            sound_effect_lines.get(
                subtitle_id,
                [],
            )
        )

        normalized_segment = (
            normalize_hybrid_parentheses(
                raw_segment
            )
        )

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
            if ocr_line in raw_segment:
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

        if (
            not block_ocr_lines
            and HYBRID_OCR_PLACEHOLDER
            in normalized_segment
        ):
            reasons.append(
                "Unexpected Hybrid OCR placeholder: "
                f"subtitle_id={subtitle_id!r}, "
                f"text={normalized_segment!r}"
            )

        block_text_lines = text_lines.get(
            subtitle_id,
            [],
        )

        sound_effect_violations = (
            find_hybrid_sound_effect_segment_violations(
                subtitle_id,
                normalized_segment,
                raw_segment=raw_segment,
                source_sound_effect_lines=(
                    block_sound_effect_lines
                ),
                source_text_lines=(
                    block_text_lines
                ),
                source_ocr_lines=(
                    block_ocr_lines
                ),
            )
        )

        reasons.extend(
            sound_effect_violations
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


def recover_single_hybrid_group(
    *,
    group: HybridTranslationGroup,
    target_blocks: list[SrtBlock],
    translated_texts: list[str],
    model: str,
    noise_dictionary: NoiseDictionary,
    glossary_entries: Mapping[str, str],
    before_context: list[SrtBlock] | None = None,
    after_context: list[SrtBlock] | None = None,
    group_number: int = 1,
    metrics: TranslationChunkMetric | None = None,
) -> list[str]:
    """
    1つのHybridグループを回復し、
    チャンク全体の翻訳一覧へ反映する。

    Hybrid検証後の既存Validatorは、
    他グループの未修正エラーに影響されないよう
    現在のグループだけを対象に実行する。
    """
    ocr_scoring_config = (
        load_ocr_scoring_config()
    )

    ocr_lines = find_group_ocr_lines(
        group,
        glossary_entries=(
            glossary_entries
        ),
        scoring_config=(
            ocr_scoring_config
        ),
    )

    # Phase 1-7：対象Hybridグループを取得する
    metrics_group: HybridGroupMetric | None = None

    if metrics is not None:
        try:
            metrics_group = (
                metrics.find_hybrid_group(
                    group_number
                )
            )
        except ValueError:
            metrics_group = HybridGroupMetric(
                group_number=group_number,
                target_ids=tuple(
                    group.target_ids
                ),
                failed_ids=tuple(
                    sorted(
                        group.failed_ids
                    )
                ),
            )

            metrics.add_hybrid_group(
                metrics_group
            )

    retry_reasons: list[str] = []

    for attempt in range(
        1,
        MAX_HYBRID_RECOVERY_ATTEMPTS + 1,
    ):
        attempt_started_at = (
            time.monotonic()
        )

        prompt = (
            build_hybrid_translation_prompt(
                group,
                ocr_lines,
                glossary_entries,
                before_context=before_context,
                after_context=after_context,
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

        try:
            response = generate(
                prompt,
                model=model,
                response_format=response_schema,
            )
        except Exception as error:
            if metrics_group is not None:
                metrics_group.add_attempt(
                    TranslationAttemptMetric(
                        pipeline="hybrid",
                        attempt=attempt,
                        target_ids=tuple(
                            group.target_ids
                        ),
                        elapsed_seconds=(
                            time.monotonic()
                            - attempt_started_at
                        ),
                        response_received=False,
                        validation_stage=(
                            "generation_exception"
                        ),
                        validation_valid=None,
                        exception_type=(
                            type(error).__name__
                        ),
                        exception_message=str(
                            error
                        ),
                    )
                )

                metrics_group.mark_failed()

            raise

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

            if metrics_group is not None:
                validation_reasons = tuple(
                    retry_reasons
                )

                metrics_group.add_attempt(
                    TranslationAttemptMetric(
                        pipeline="hybrid",
                        attempt=attempt,
                        target_ids=tuple(
                            group.target_ids
                        ),
                        elapsed_seconds=(
                            time.monotonic()
                            - attempt_started_at
                        ),
                        response_received=True,
                        validation_stage=(
                            "hybrid_validation"
                        ),
                        validation_valid=False,
                        validation_reasons=(
                            validation_reasons
                        ),
                        reason_codes=(
                            build_validation_reason_codes(
                                validation_reasons
                            )
                        ),
                    )
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

        group_translated_texts = [
            hybrid_validation.segments[
                block.number
            ]
            for block in group.blocks
        ]

        standard_response = (
            build_standard_translation_response(
                list(group.blocks),
                group_translated_texts,
            )
        )

        source_speakers: list[
            str | None
            ] = []

        source_texts: list[str] = []

        for block in group.blocks:
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
                    for block in group.blocks
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

            if metrics_group is not None:
                validation_reasons = tuple(
                    retry_reasons
                )

                metrics_group.add_attempt(
                    TranslationAttemptMetric(
                        pipeline="hybrid",
                        attempt=attempt,
                        target_ids=tuple(
                            group.target_ids
                        ),
                        elapsed_seconds=(
                            time.monotonic()
                            - attempt_started_at
                        ),
                        response_received=True,
                        validation_stage=(
                            "standard_validation"
                        ),
                        validation_valid=False,
                        validation_reasons=(
                            validation_reasons
                        ),
                        reason_codes=(
                            build_validation_reason_codes(
                                validation_reasons
                            )
                        ),
                    )
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

        if metrics_group is not None:
            metrics_group.add_attempt(
                TranslationAttemptMetric(
                    pipeline="hybrid",
                    attempt=attempt,
                    target_ids=tuple(
                        group.target_ids
                    ),
                    elapsed_seconds=(
                        time.monotonic()
                        - attempt_started_at
                    ),
                    response_received=True,
                    validation_stage="complete",
                    validation_valid=True,
                )
            )

            metrics_group.mark_success()

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

        merged_texts = list(
            translated_texts
        )

        for position, translation in zip(
            group.positions,
            standard_validation.translated_texts,
            strict=True,
        ):
            merged_texts[
                position
            ] = translation

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

        return merged_texts

    if metrics_group is not None:
        metrics_group.mark_failed()

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


def recover_translation_with_hybrid(
    target_blocks: list[SrtBlock],
    translated_texts: list[str],
    errors: list[str],
    model: str,
    *,
    noise_dictionary: NoiseDictionary,
    glossary_entries: Mapping[str, str],
    before_context: list[SrtBlock] | None = None,
    after_context: list[SrtBlock] | None = None,
    metrics: TranslationChunkMetric | None = None,
) -> list[str] | None:
    """
    通常翻訳に失敗したチャンクを、
    1つ以上のHybridグループへ分割して回復する。

    複数の失敗IDが異なる文章や時間範囲にある場合は、
    それぞれを独立したグループとして順番に翻訳する。

    グループごとの回復結果は
    チャンク全体の翻訳一覧へ累積して反映する。
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

    groups = build_hybrid_translation_groups(
        target_blocks,
        failed_ids,
    )

    if not groups:
        return None

    # Phase 1-7：グループ生成後にHybrid開始を記録する
    if metrics is not None:
        metrics.trigger_hybrid(
            errors
        )

        for group_number, group in enumerate(
            groups,
            start=1,
        ):
            metrics.add_hybrid_group(
                HybridGroupMetric(
                    group_number=(
                        group_number
                    ),
                    target_ids=tuple(
                        group.target_ids
                    ),
                    failed_ids=tuple(
                        sorted(
                            group.failed_ids
                        )
                    ),
                )
            )

    print()
    print(
        "Hybrid Translation Recovery Groups:"
    )
    print(
        f"  Count: {len(groups)}"
    )

    for group_number, group in enumerate(
        groups,
        start=1,
    ):
        print(
            f"  [{group_number}] "
            + ", ".join(
                group.target_ids
            )
        )

    recovered_texts = list(
        translated_texts
    )

    for group_number, group in enumerate(
        groups,
        start=1,
    ):
        print()
        print(
            "Hybrid Group Start:"
        )
        print(
            f"  Group: {group_number}/"
            f"{len(groups)}"
        )
        print(
            "  IDs: "
            + ", ".join(
                group.target_ids
            )
        )

        recovered_texts = (
            recover_single_hybrid_group(
                group=group,
                target_blocks=target_blocks,
                translated_texts=(
                    recovered_texts
                ),
                model=model,
                noise_dictionary=(
                    noise_dictionary
                ),
                glossary_entries=(
                    glossary_entries
                ),
                before_context=(
                    before_context
                ),
                after_context=(
                    after_context
                ),
                group_number=group_number,
                metrics=metrics,
            )
        )

    final_response = (
        build_standard_translation_response(
            target_blocks,
            recovered_texts,
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

    final_validation = (
        validate_translation_response(
            final_response,
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

    if not final_validation.valid:
        raise HybridRecoveryError(
            "Hybrid Translation Recovery "
            "completed all groups but final "
            "chunk validation failed: "
            + "; ".join(
                final_validation.reasons
            )
        )

    print()
    print(
        "Hybrid Translation Recovery "
        "completed all groups:"
    )
    print(
        f"  Groups: {len(groups)}"
    )

    return (
        final_validation.translated_texts
    )
