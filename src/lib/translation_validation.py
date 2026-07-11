from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from lib.text import (
    CHINESE_SPECIFIC_PATTERN,
    DEFAULT_ALLOWED_LATIN_TERMS,
    find_suspicious_latin_sequences,
)

# 英語の文として残っている可能性が高い単語。
# 固有名詞や略語だけを英語残存と誤判定しないため、
# 英単語数と組み合わせて判定する。
ENGLISH_SENTENCE_PATTERN = re.compile(
    r"\b(?:"
    r"I|you|he|she|we|they|it|"
    r"the|a|an|"
    r"is|are|am|was|were|"
    r"do|does|did|"
    r"have|has|had|"
    r"can|could|will|would|should|"
    r"what|where|when|why|how|who|"
    r"get|come|go|move|look|know|think|need|want|"
    r"this|that|these|those|"
    r"not|no|yes|"
    r"from|to|for|with|about"
    r")\b",
    re.IGNORECASE,
)

ENGLISH_WORD_PATTERN = re.compile(r"[A-Za-z]{2,}")

LATIN_TOKEN_PATTERN = re.compile(r"[A-Za-z]+")

ALLOWED_LATIN_TERMS = (
    DEFAULT_ALLOWED_LATIN_TERMS
)

DEFAULT_REPEAT_THRESHOLD = 5
DEFAULT_MAX_LINES_PER_SUBTITLE = 6
DEFAULT_MAX_CHARS_PER_SUBTITLE = 200

DEFAULT_INCOMPLETE_THRESHOLD = 3

DEFAULT_MINIMUM_SOURCE_LENGTH = 8
DEFAULT_MINIMUM_LENGTH_RATIO = 0.15
DEFAULT_MAXIMUM_LENGTH_RATIO = 4.0
DEFAULT_MAXIMUM_SEGMENTS = 4

DEFAULT_JSON_RESPONSE_OVERHEAD_LINES = 5

NUMBER_PATTERN = re.compile(r"\d+")

SOURCE_EFFECT_PATTERN = re.compile(
    r"^\s*\([A-Z0-9 ,.'’!?-]+\)\s*$"
)

TRANSLATED_EFFECT_PATTERN = re.compile(
    r"^\s*[（(].+[）)]\s*$"
)

# 字幕の末尾として不自然になりやすい助詞・接続表現。
#
# 字幕では次の字幕へ文章が続くこともあるため、
# 1件だけでは異常扱いせず、チャンク内で複数件検出された場合に
# 再翻訳対象とする。
INCOMPLETE_ENDING_PATTERN = re.compile(
    r"(?:"
    r"の|を|が|に|へ|で|"
    r"から|まで|より|"
    r"そして|または|"
    r"ので|ため|なら|けど"
    r")$"
)

# 効果音だけの字幕は未完結判定から除外する。
SOUND_EFFECT_PATTERN = re.compile(
    r"^[（(].+[）)]$"
)


@dataclass
class TranslationResponseItem:
    id: str
    translation: str


@dataclass
class ValidationResult:
    valid: bool = True
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    translated_texts: list[str] = field(default_factory=list)
    failed_ids: set[str] = field(default_factory=set)
    requires_full_retry: bool = False

    def add_error(
        self,
        reason: str,
        *,
        subtitle_id: str | None = None,
        requires_full_retry: bool = False,
    ) -> None:
        self.valid = False
        self.reasons.append(reason)

        if subtitle_id is not None:
            self.failed_ids.add(
                subtitle_id
            )

        if requires_full_retry:
            self.requires_full_retry = True

    def add_warning(self, warning: str) -> None:
        self.warnings.append(warning)


def strip_json_code_fence(
    response: str,
) -> str:
    """
    LLMが付加したJSONコードフェンスだけを除去する。

    プロンプトではコードフェンスを禁止しているが、
    正しいJSON本体を回収できる場合は許容する。
    """
    stripped = response.strip()

    stripped = re.sub(
        r"^```(?:json)?\s*",
        "",
        stripped,
        flags=re.IGNORECASE,
    )

    stripped = re.sub(
        r"\s*```$",
        "",
        stripped,
    )

    return stripped.strip()


def parse_translation_json(
    response: str,
) -> tuple[
    list[TranslationResponseItem],
    list[str],
]:
    """
    LLMレスポンスをJSON翻訳結果として解析する。

    期待形式:
        {
          "translations": [
            {
              "id": "1",
              "translation": "日本語字幕"
            }
          ]
        }
    """
    errors: list[str] = []
    items: list[TranslationResponseItem] = []

    raw_json = strip_json_code_fence(
        response
    )

    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as error:
        return [], [
            "Invalid JSON response: "
            f"line={error.lineno}, "
            f"column={error.colno}, "
            f"message={error.msg}"
        ]

    if not isinstance(payload, dict):
        return [], [
            "Invalid JSON root: expected object"
        ]

    actual_root_keys = set(
        payload.keys()
    )

    expected_root_keys = {
        "translations",
    }

    if actual_root_keys != expected_root_keys:
        errors.append(
            "Invalid JSON root keys: "
            f"expected={sorted(expected_root_keys)}, "
            f"actual={sorted(actual_root_keys)}"
        )

    translations = payload.get(
        "translations"
    )

    if not isinstance(translations, list):
        errors.append(
            "Invalid translations: expected array"
        )

        return [], errors

    for position, item in enumerate(
        translations,
        start=1,
    ):
        if not isinstance(item, dict):
            errors.append(
                "Invalid translation item: "
                f"position={position}, "
                "expected=object"
            )

            continue

        required_item_keys = {
            "id",
            "translation",
        }

        actual_item_keys = set(
            item.keys()
        )

        missing_item_keys = (
            required_item_keys
            - actual_item_keys
        )

        if missing_item_keys:
            errors.append(
                "Missing translation item keys: "
                f"position={position}, "
                f"missing={sorted(missing_item_keys)}, "
                f"actual={sorted(actual_item_keys)}"
            )

            continue

        item_id = item.get("id")
        translation = item.get(
            "translation"
        )

        if isinstance(item_id, bool):
            errors.append(
                "Invalid translation id: "
                f"position={position}, "
                "expected=string or integer"
            )

            continue

        if isinstance(item_id, int):
            normalized_id = str(item_id)
        elif isinstance(item_id, str):
            normalized_id = item_id.strip()
        else:
            errors.append(
                "Invalid translation id: "
                f"position={position}, "
                "expected=string or integer"
            )

            continue

        if not normalized_id:
            errors.append(
                "Empty translation id: "
                f"position={position}"
            )

            continue

        if not isinstance(translation, str):
            errors.append(
                "Invalid translation text: "
                f"id={normalized_id!r}, "
                "expected=string"
            )

            continue

        normalized_translation = (
            translation.strip()
        )

        if not normalized_translation:
            errors.append(
                "Empty translation: "
                f"id={normalized_id!r}"
            )

            continue

        items.append(
            TranslationResponseItem(
                id=normalized_id,
                translation=normalized_translation,
            )
        )

    return items, errors


def validate_translation_ids(
    items: list[TranslationResponseItem],
    expected_ids: list[str],
) -> list[str]:
    """
    JSONレスポンスのIDが入力targetと完全に対応するか検証する。
    """
    errors: list[str] = []

    actual_ids = [
        item.id
        for item in items
    ]

    duplicate_ids = sorted(
        item_id
        for item_id, count
        in Counter(actual_ids).items()
        if count > 1
    )

    if duplicate_ids:
        errors.append(
            "Duplicate translation IDs: "
            f"{duplicate_ids}"
        )

    actual_id_set = set(
        actual_ids
    )

    expected_id_set = set(
        expected_ids
    )

    missing_ids = [
        item_id
        for item_id in expected_ids
        if item_id not in actual_id_set
    ]

    if missing_ids:
        errors.append(
            "Missing translation IDs: "
            f"{missing_ids}"
        )

    unexpected_ids = [
        item_id
        for item_id in actual_ids
        if item_id not in expected_id_set
    ]

    if unexpected_ids:
        errors.append(
            "Unexpected translation IDs: "
            f"{unexpected_ids}"
        )

    if actual_ids != expected_ids:
        errors.append(
            "Invalid translation ID order: "
            f"expected={expected_ids}, "
            f"actual={actual_ids}"
        )

    return errors


def validate_response_size(
    response: str,
    expected_count: int,
    *,
    max_lines_per_subtitle: int = DEFAULT_MAX_LINES_PER_SUBTITLE,
    max_chars_per_subtitle: int = DEFAULT_MAX_CHARS_PER_SUBTITLE,
    response_overhead_lines: int = DEFAULT_JSON_RESPONSE_OVERHEAD_LINES,
) -> list[str]:
    """
    LLMレスポンスが異常に長くないか検証する。

    JSONのルート・配列部分には固定行数が必要なため、
    字幕件数に応じた上限へJSON構造分の余白を加える。
    """
    errors: list[str] = []

    actual_line_count = len(response.splitlines())
    actual_char_count = len(response)

    max_line_count = (
        expected_count * max_lines_per_subtitle
        + response_overhead_lines
    )
    max_char_count = (
        expected_count * max_chars_per_subtitle
    )

    if actual_line_count > max_line_count:
        errors.append(
            "Response has too many lines: "
            f"maximum={max_line_count}, "
            f"actual={actual_line_count}"
        )

    if actual_char_count > max_char_count:
        errors.append(
            "Response is too long: "
            f"maximum={max_char_count}, "
            f"actual={actual_char_count}"
        )

    return errors


def normalize_line_for_comparison(
    line: str,
) -> str:
    """
    重複判定用に空白と一部記号を正規化する。
    """
    normalized = re.sub(
        r"\s+",
        "",
        line,
    )

    normalized = normalized.strip(
        "。．.!！?？、,，"
    )

    return normalized


def normalized_text_length(
    text: str,
) -> int:
    """
    空白と字幕内改行記号を除外した文字数を返す。
    """
    normalized = re.sub(
        r"[\s／/]+",
        "",
        text,
    )

    return len(normalized)



def find_length_ratio_violations(
    source_texts: list[str],
    translated_texts: list[str],
    *,
    minimum_source_length: int = DEFAULT_MINIMUM_SOURCE_LENGTH,
    minimum_ratio: float = DEFAULT_MINIMUM_LENGTH_RATIO,
    maximum_ratio: float = DEFAULT_MAXIMUM_LENGTH_RATIO,
) -> list[str]:
    """
    原文に対して訳文が極端に短い、または長い字幕を検出する。
    """
    violations: list[str] = []

    for index, (
        source_text,
        translated_text,
    ) in enumerate(
        zip(
            source_texts,
            translated_texts,
            strict=True,
        ),
        start=1,
    ):
        source_length = normalized_text_length(
            source_text
        )
        translated_length = normalized_text_length(
            translated_text
        )

        if source_length < minimum_source_length:
            continue

        ratio = translated_length / source_length

        if minimum_ratio <= ratio <= maximum_ratio:
            continue

        violations.append(
            "Suspicious translation length: "
            f"subtitle={index}, "
            f"source_length={source_length}, "
            f"translated_length={translated_length}, "
            f"ratio={ratio:.2f}"
        )

    return violations


def find_number_mismatches(
    source_texts: list[str],
    translated_texts: list[str],
) -> list[str]:
    """
    原文と訳文に含まれる数字が一致しているか確認する。
    """
    violations: list[str] = []

    for index, (
        source_text,
        translated_text,
    ) in enumerate(
        zip(
            source_texts,
            translated_texts,
            strict=True,
        ),
        start=1,
    ):
        source_numbers = NUMBER_PATTERN.findall(
            source_text
        )
        translated_numbers = NUMBER_PATTERN.findall(
            translated_text
        )

        if source_numbers == translated_numbers:
            continue

        violations.append(
            "Number mismatch: "
            f"subtitle={index}, "
            f"source={source_numbers}, "
            f"translated={translated_numbers}"
        )

    return violations


def find_effect_format_violations(
    source_texts: list[str],
    translated_texts: list[str],
) -> list[str]:
    """
    原文が効果音のみなのに、
    訳文が括弧形式になっていない字幕を検出する。
    """
    violations: list[str] = []

    for index, (
        source_text,
        translated_text,
    ) in enumerate(
        zip(
            source_texts,
            translated_texts,
            strict=True,
        ),
        start=1,
    ):
        source = re.sub(
            r"\s+",
            " ",
            source_text,
        ).strip()

        translated = translated_text.strip()

        if not SOURCE_EFFECT_PATTERN.fullmatch(
            source
        ):
            continue

        if TRANSLATED_EFFECT_PATTERN.fullmatch(
            translated
        ):
            continue

        violations.append(
            "Sound effect format mismatch: "
            f"subtitle={index}, "
            f"source={source!r}, "
            f"translated={translated!r}"
        )

    return violations


def find_excessive_segments(
    translated_texts: list[str],
    *,
    maximum_segments: int = DEFAULT_MAXIMUM_SEGMENTS,
) -> list[str]:
    """
    1字幕内の区切りが多すぎる字幕を検出する。
    """
    violations: list[str] = []

    for index, text in enumerate(
        translated_texts,
        start=1,
    ):
        segments = [
            segment.strip()
            for segment in re.split(
                r"\s*[／/]\s*",
                text,
            )
            if segment.strip()
        ]

        if len(segments) <= maximum_segments:
            continue

        violations.append(
            "Too many subtitle segments: "
            f"subtitle={index}, "
            f"segments={len(segments)}, "
            f"text={text!r}"
        )

    return violations


def find_repeated_lines(
    lines: list[str],
    *,
    threshold: int = DEFAULT_REPEAT_THRESHOLD,
) -> list[tuple[str, int]]:
    """
    同一内容がthreshold回以上出現した字幕を返す。

    空行や極端に短い文字列は対象外にする。
    """
    original_by_normalized: dict[str, str] = {}
    normalized_lines: list[str] = []

    for line in lines:
        normalized = normalize_line_for_comparison(
            line
        )

        # 「はい」「いや」など短い字幕は偶然重複しやすい。
        if len(normalized) < 5:
            continue

        normalized_lines.append(normalized)
        original_by_normalized.setdefault(
            normalized,
            line.strip(),
        )

    counts = Counter(normalized_lines)

    repeated: list[tuple[str, int]] = []

    for normalized, count in counts.items():
        if count < threshold:
            continue

        repeated.append(
            (
                original_by_normalized[normalized],
                count,
            )
        )

    return repeated


def find_chinese_specific_characters(
    translated_texts: list[str],
    subtitle_ids: list[str],
) -> list[str]:
    """
    日本語字幕へ混入した簡体字・中国語固有文字を、
    字幕IDと本文を含めて検出する。
    """
    violations: list[str] = []

    for subtitle_id, translated_text in zip(
        subtitle_ids,
        translated_texts,
        strict=True,
    ):
        matched_characters = sorted(
            set(
                CHINESE_SPECIFIC_PATTERN.findall(
                    translated_text
                )
            )
        )

        if not matched_characters:
            continue

        violations.append(
            "Chinese-specific characters detected: "
            f"subtitle_id={subtitle_id!r}, "
            f"characters={''.join(matched_characters)!r}, "
            f"text={translated_text!r}"
        )

    return violations


def normalize_latin_term(
    token: str,
) -> str:
    return token.replace(".", "").upper()


def contains_untranslated_english(
    text: str,
) -> bool:
    """
    英文が未翻訳のまま残っている可能性を検出する。

    英字があるだけではNGにしない。
    例:
        T.J.
        SG-1
        F-302
        DNA

    英単語が複数存在し、英文で頻出する語も含まれる場合に
    未翻訳英文と判定する。
    """
    words = ENGLISH_WORD_PATTERN.findall(text)

    suspicious_words = [
        word
        for word in words
        if normalize_latin_term(word)
        not in {
            normalize_latin_term(term)
            for term in ALLOWED_LATIN_TERMS
        }
    ]

    if len(suspicious_words) < 2:
        return False

    return bool(
        ENGLISH_SENTENCE_PATTERN.search(text)
    )


def find_untranslated_english_violations(
    translated_texts: list[str],
    subtitle_ids: list[str],
) -> list[str]:
    """
    OCR破損候補を除外した上で、
    字幕単位に未翻訳英文を検出する。
    """
    violations: list[str] = []

    for subtitle_id, translated_text in zip(
        subtitle_ids,
        translated_texts,
        strict=True,
    ):
        ocr_sequences = (
            find_suspicious_latin_sequences(
                translated_text,
                allowed_terms=ALLOWED_LATIN_TERMS,
            )
        )

        text_for_check = translated_text

        for sequence in ocr_sequences:
            text_for_check = (
                text_for_check.replace(
                    sequence,
                    "",
                )
            )

        if not contains_untranslated_english(
            text_for_check
        ):
            continue

        violations.append(
            "Untranslated English sentence detected: "
            f"subtitle_id={subtitle_id!r}, "
            f"text={translated_text!r}"
        )

    return violations


def find_garbled_latin_violations(
    translated_texts: list[str],
    subtitle_ids: list[str],
) -> list[str]:
    """
    字幕ごとにOCR破損英字列を検出する。
    """
    violations: list[str] = []

    for subtitle_id, translated_text in zip(
        subtitle_ids,
        translated_texts,
        strict=True,
    ):
        sequences = (
            find_suspicious_latin_sequences(
                translated_text,
                allowed_terms=ALLOWED_LATIN_TERMS,
            )
        )

        if not sequences:
            continue

        violations.append(
            "Garbled Latin text detected: "
            f"subtitle_id={subtitle_id!r}, "
            f"sequences={sequences!r}, "
            f"text={translated_text!r}"
        )

    return violations


def validate_translation_response(
    response: str,
    *,
    expected_ids: list[str],
    source_texts: list[str] | None = None,
    glossary_entries: Mapping[str, str] | None = None,
    repeat_threshold: int = DEFAULT_REPEAT_THRESHOLD,
    incomplete_threshold: int = DEFAULT_INCOMPLETE_THRESHOLD,
) -> ValidationResult:
    """
    LLMのJSON翻訳レスポンス全体を検証する。

    期待形式:
        {
          "translations": [
            {
              "id": "入力targetのid",
              "translation": "日本語字幕"
            }
          ]
        }

    正常な場合:
        result.valid == True
        result.translated_texts にID順の翻訳結果が入る

    異常な場合:
        result.valid == False
        result.reasons に理由が入る
    """
    result = ValidationResult()

    expected_count = len(
        expected_ids
    )

    if not response.strip():
        result.add_error(
            "Translation response is empty"
        )
        return result

    for error in validate_response_size(
        response,
        expected_count,
    ):
        result.add_error(error)

    items, parse_errors = (
        parse_translation_json(
            response
        )
    )

    for error in parse_errors:
        result.add_error(error)

    # JSON構造が壊れている場合、
    # IDと原文の対応関係を保証できないため、
    # 後続の検証へ進まない。
    if parse_errors:
        return result

    id_errors = validate_translation_ids(
        items,
        expected_ids,
    )

    for error in id_errors:
        result.add_error(error)

    # IDの欠落・重複・追加・並び替えがある場合、
    # 原文と訳文の対応関係を保証できない。
    if id_errors:
        return result

    translated_texts = [
        item.translation
        for item in items
    ]

    result.translated_texts = (
        translated_texts
    )

    garbled_latin_violations = (
        find_garbled_latin_violations(
            translated_texts,
            expected_ids,
        )
    )

    for violation in garbled_latin_violations[:10]:
        result.add_error(violation)

    if len(garbled_latin_violations) > 10:
        result.add_error(
            "Additional garbled Latin violations: "
            f"{len(garbled_latin_violations) - 10}"
        )

    untranslated_english_violations = (
        find_untranslated_english_violations(
            translated_texts,
            expected_ids,
        )
    )

    for violation in (
        untranslated_english_violations[:10]
    ):
        result.add_error(violation)

    if len(
        untranslated_english_violations
    ) > 10:
        result.add_error(
            "Additional untranslated English violations: "
            f"{len(untranslated_english_violations) - 10}"
        )

    repeated_lines = find_repeated_lines(
        translated_texts,
        threshold=repeat_threshold,
    )

    for repeated_text, count in repeated_lines:
        result.add_error(
            "Repeated translation detected: "
            f"count={count}, "
            f"text={repeated_text[:80]!r}"
        )

    if (
        source_texts is not None
        and glossary_entries is not None
        and len(source_texts) == len(translated_texts)
    ):
        glossary_violations = (
            find_glossary_violations(
                source_texts,
                translated_texts,
                expected_ids,
                glossary_entries,
            )
        )

        # 再試行プロンプトが極端に長くならないよう、
        # エラーへ追加する件数を制限する。
        for violation in glossary_violations[:10]:
            result.add_error(violation)

        if len(glossary_violations) > 10:
            result.add_error(
                "Additional glossary violations: "
                f"{len(glossary_violations) - 10}"
            )

    if (
        source_texts is not None
        and len(source_texts) == len(translated_texts)
    ):
        length_violations = (
            find_length_ratio_violations(
                source_texts,
                translated_texts,
            )
        )

        for violation in length_violations[:10]:
            result.add_warning(violation)

        if len(length_violations) > 10:
            result.add_warning(
                "Additional length ratio violations: "
                f"{len(length_violations) - 10}"
            )

        number_mismatches = find_number_mismatches(
            source_texts,
            translated_texts,
        )

        for violation in number_mismatches[:10]:
            result.add_warning(violation)

        if len(number_mismatches) > 10:
            result.add_warning(
                "Additional number mismatches: "
                f"{len(number_mismatches) - 10}"
            )

        effect_violations = (
            find_effect_format_violations(
                source_texts,
                translated_texts,
            )
        )

        for violation in effect_violations[:10]:
            result.add_warning(violation)

        if len(effect_violations) > 10:
            result.add_warning(
                "Additional effect format violations: "
                f"{len(effect_violations) - 10}"
            )

    excessive_segments = find_excessive_segments(
        translated_texts
    )

    for violation in excessive_segments[:10]:
        result.add_warning(violation)

    if len(excessive_segments) > 10:
        result.add_warning(
            "Additional excessive segment warnings: "
            f"{len(excessive_segments) - 10}"
        )

    incomplete_translations = (
        find_incomplete_translations(
            translated_texts
        )
    )

    # 字幕では文章が次のブロックへ続く場合があるため、
    # 一定数以上でも警告に留め、再試行理由にはしない。
    if (
        len(incomplete_translations)
        >= incomplete_threshold
    ):
        details = ", ".join(
            f"{number}:{text!r}"
            for number, text
            in incomplete_translations[:10]
        )

        result.add_warning(
            "Possible incomplete translations: "
            f"count={len(incomplete_translations)}, "
            f"items={details}"
        )

    repeated_sequences = find_repeated_sequences(
        translated_texts,
        minimum_sequence_length=3,
    )

    if repeated_sequences:
        first_start, second_start, length = (
            repeated_sequences[0]
        )

        result.add_error(
            "Repeated translation sequence detected: "
            f"first_start={first_start}, "
            f"second_start={second_start}, "
            f"length={length}"
        )

    return result


def source_contains_glossary_term(
    source_text: str,
    source_term: str,
) -> bool:
    """
    原文に用語集の英語表現が含まれているか判定する。

    英数字の単語境界を考慮し、
    Gate が navigate 等へ誤一致しないようにする。
    """
    pattern = re.compile(
        rf"(?<![A-Za-z0-9])"
        rf"{re.escape(source_term)}"
        rf"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )

    return bool(
        pattern.search(source_text)
    )


def find_glossary_violations(
    source_texts: list[str],
    translated_texts: list[str],
    subtitle_ids: list[str],
    glossary_entries: Mapping[str, str],
) -> list[str]:
    """
    原文に用語集の英語表現が存在するのに、
    対応する指定訳が翻訳結果へ含まれていない字幕を検出する。
    """
    violations: list[str] = []

    for (
        subtitle_id,
        source_text,
        translated_text,
    ) in zip(
        subtitle_ids,
        source_texts,
        translated_texts,
        strict=True,
    ):
        for source_term, expected_term in (
            glossary_entries.items()
        ):
            if not source_contains_glossary_term(
                source_text,
                source_term,
            ):
                continue

            if expected_term in translated_text:
                continue

            violations.append(
                "Glossary violation: "
                f"subtitle_id={subtitle_id!r}, "
                f"source_term={source_term!r}, "
                f"expected={expected_term!r}, "
                f"actual={translated_text!r}"
            )

    return violations

def normalize_translation_ending(
    text: str,
) -> str:
    """
    未完結判定用に末尾の空白・句読点・引用符を除去する。
    """
    normalized = text.strip()

    normalized = normalized.rstrip(
        "。．.!！?？、,，"
        "…"
        "」』】）)"
        "\"'"
        " "
    )

    return normalized


def is_incomplete_translation(
    text: str,
) -> bool:
    """
    字幕が助詞・接続語だけで終わっている可能性を判定する。

    疑問文・感嘆文・効果音は未完結扱いしない。
    """
    raw_text = text.strip()

    if not raw_text:
        return False

    if SOUND_EFFECT_PATTERN.fullmatch(raw_text):
        return False

    # 疑問文・感嘆文は文として完結している。
    if raw_text.endswith(
        (
            "？",
            "?",
            "！",
            "!",
        )
    ):
        return False

    normalized = normalize_translation_ending(
        raw_text
    )

    if not normalized:
        return False

    # 半角・全角の字幕内改行記号へ対応する。
    last_segment = re.split(
        r"\s*[／/]\s*",
        normalized,
    )[-1].strip()

    if not last_segment:
        return True

    # 「おっと」など短い発話の誤検知を避ける。
    if len(last_segment) <= 4:
        return False

    return bool(
        INCOMPLETE_ENDING_PATTERN.search(
            last_segment
        )
    )


def find_incomplete_translations(
    translated_texts: list[str],
) -> list[tuple[int, str]]:
    """
    未完結の可能性がある字幕番号と本文を返す。
    """
    results: list[tuple[int, str]] = []

    for index, text in enumerate(
        translated_texts,
        start=1,
    ):
        if not is_incomplete_translation(text):
            continue

        results.append(
            (
                index,
                text,
            )
        )

    return results


def find_repeated_sequences(
    lines: list[str],
    *,
    minimum_sequence_length: int = 3,
) -> list[tuple[int, int, int]]:
    """
    連続する字幕列の重複を検出する。

    戻り値:
        (
            1回目の開始位置,
            2回目の開始位置,
            重複した字幕数,
        )
    """
    normalized = [
        normalize_line_for_comparison(line)
        for line in lines
    ]

    repeated: list[tuple[int, int, int]] = []
    total = len(normalized)

    for first_start in range(total):
        for second_start in range(
            first_start + minimum_sequence_length,
            total,
        ):
            length = 0

            while (
                first_start + length < second_start
                and second_start + length < total
                and normalized[first_start + length]
                == normalized[second_start + length]
            ):
                length += 1

            if length >= minimum_sequence_length:
                repeated.append(
                    (
                        first_start + 1,
                        second_start + 1,
                        length,
                    )
                )

    return repeated