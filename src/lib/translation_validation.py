from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field


# LLMへ要求している番号形式。
#
# 許可例:
#   1. 翻訳文
#   1) 翻訳文
#   [1] 翻訳文
#
# 番号の後ろに本文が必要。
NUMBERED_LINE_PATTERN = re.compile(
    r"^\s*(?:\[(\d+)\]|(\d+)[.)])\s*(.+?)\s*$"
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

# 日本語にも漢字があるため、CJK範囲全体では判定しない。
# 実際に混入が確認された簡体字・中国語固有表現を中心に検出する。
#
# 誤検知が判明した文字は、この集合から外すこと。
CHINESE_SPECIFIC_PATTERN = re.compile(
    r"[这们为发经进过还让从个里边开关车话说时对与于后会动语]"
)

LATIN_TOKEN_PATTERN = re.compile(r"[A-Za-z]+")

# 字幕内に残っても異常とは限らない英字表記。
# 必要に応じて作品別の語彙を追加する。
ALLOWED_LATIN_TERMS = {
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
}

DEFAULT_REPEAT_THRESHOLD = 5
DEFAULT_MAX_LINES_PER_SUBTITLE = 4
DEFAULT_MAX_CHARS_PER_SUBTITLE = 200

DEFAULT_INCOMPLETE_THRESHOLD = 3

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
class ValidationResult:
    valid: bool = True
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    translated_texts: list[str] = field(default_factory=list)

    def add_error(self, reason: str) -> None:
        self.valid = False
        self.reasons.append(reason)

    def add_warning(self, warning: str) -> None:
        self.warnings.append(warning)


def extract_numbered_lines(
    response: str,
) -> list[tuple[int, str]]:
    """
    LLMレスポンスから番号付き翻訳行を抽出する。

    対応形式:
        1. 翻訳文
        1) 翻訳文
        [1] 翻訳文

    番号のない行は抽出しない。
    """
    results: list[tuple[int, str]] = []

    for raw_line in response.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        match = NUMBERED_LINE_PATTERN.match(line)

        if not match:
            continue

        number_text = match.group(1) or match.group(2)
        translated_text = match.group(3).strip()

        if not number_text or not translated_text:
            continue

        results.append(
            (
                int(number_text),
                translated_text,
            )
        )

    return results


def validate_number_sequence(
    numbered_lines: list[tuple[int, str]],
    expected_count: int,
) -> str | None:
    """
    番号が1からexpected_countまで連続しているか検証する。
    """
    actual_numbers = [
        number
        for number, _ in numbered_lines
    ]

    expected_numbers = list(
        range(1, expected_count + 1)
    )

    if actual_numbers == expected_numbers:
        return None

    return (
        "Invalid translation numbering: "
        f"expected_count={expected_count}, "
        f"actual_count={len(actual_numbers)}, "
        f"actual_last={actual_numbers[-1] if actual_numbers else None}"
    )


def validate_output_count(
    translated_texts: list[str],
    expected_count: int,
) -> str | None:
    """
    翻訳件数が翻訳対象件数と一致するか検証する。
    """
    actual_count = len(translated_texts)

    if actual_count == expected_count:
        return None

    return (
        "Translation count mismatch: "
        f"expected={expected_count}, "
        f"actual={actual_count}"
    )


def validate_response_size(
    response: str,
    expected_count: int,
    *,
    max_lines_per_subtitle: int = DEFAULT_MAX_LINES_PER_SUBTITLE,
    max_chars_per_subtitle: int = DEFAULT_MAX_CHARS_PER_SUBTITLE,
) -> list[str]:
    """
    LLMレスポンスが異常に長くないか検証する。

    30字幕の場合の既定値:
        最大行数: 120
        最大文字数: 6000
    """
    errors: list[str] = []

    actual_line_count = len(response.splitlines())
    actual_char_count = len(response)

    max_line_count = (
        expected_count * max_lines_per_subtitle
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


def contains_chinese_specific_characters(
    text: str,
) -> bool:
    """
    日本語字幕へ簡体字などが混入していないか検出する。
    """
    return bool(
        CHINESE_SPECIFIC_PATTERN.search(text)
    )


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


def contains_garbled_latin(
    text: str,
) -> bool:
    """
    OCR由来と思われる不自然な英字列を検出する。

    例:
        VVNsKomCIAcM
        MimElomIElaie

    完全な判定はできないため、
    長く、かつ大文字・小文字が不自然に混ざる語を対象とする。
    """
    allowed_normalized = {
        normalize_latin_term(term)
        for term in ALLOWED_LATIN_TERMS
    }

    for token in LATIN_TOKEN_PATTERN.findall(text):
        if normalize_latin_term(token) in allowed_normalized:
            continue

        if len(token) < 8:
            continue

        upper_count = sum(
            character.isupper()
            for character in token
        )
        lower_count = sum(
            character.islower()
            for character in token
        )

        # 通常の固有名詞は先頭だけ大文字であることが多い。
        # 大文字が2文字以上かつ小文字も2文字以上なら疑わしい。
        if upper_count >= 2 and lower_count >= 2:
            return True

    return False


def validate_translation_response(
    response: str,
    *,
    expected_count: int,
    source_texts: list[str] | None = None,
    glossary_entries: Mapping[str, str] | None = None,
    repeat_threshold: int = DEFAULT_REPEAT_THRESHOLD,
    incomplete_threshold: int = DEFAULT_INCOMPLETE_THRESHOLD,
) -> ValidationResult:
    """
    LLMの翻訳レスポンス全体を検証する。

    正常な場合:
        result.valid == True
        result.translated_texts に翻訳結果が入る

    異常な場合:
        result.valid == False
        result.reasons に理由が入る
    """
    result = ValidationResult()

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

    numbered_lines = extract_numbered_lines(
        response
    )

    numbering_error = validate_number_sequence(
        numbered_lines,
        expected_count,
    )

    if numbering_error:
        result.add_error(numbering_error)

    translated_texts = [
        translated_text
        for _, translated_text in numbered_lines
    ]

    result.translated_texts = translated_texts

    count_error = validate_output_count(
        translated_texts,
        expected_count,
    )

    if count_error:
        result.add_error(count_error)

    # 件数または番号が壊れている場合、
    # 原文と訳文の対応関係を保証できないため、
    # 用語・内容の検証へ進まない。
    if numbering_error or count_error:
        return result

    joined_text = "\n".join(
        translated_texts
    )

    if contains_chinese_specific_characters(
        joined_text
    ):
        result.add_error(
            "Chinese-specific characters detected"
        )

    if contains_untranslated_english(
        joined_text
    ):
        result.add_error(
            "Untranslated English sentence detected"
        )

    if contains_garbled_latin(
        joined_text
    ):
        result.add_error(
            "Garbled Latin text detected"
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
                glossary_entries,
            )
        )

        # 再試行プロンプトが極端に長くならないよう、
        # ログへ追加する件数を制限する。
        for violation in glossary_violations[:10]:
            result.add_warning(violation)

        if len(glossary_violations) > 10:
            result.add_warning(
                "Additional glossary violations: "
                f"{len(glossary_violations) - 10}"
            )

    incomplete_translations = (
        find_incomplete_translations(
            translated_texts
        )
    )

    # 字幕では文章が次のブロックへ続く場合があるため、
    # 1件では再試行せず、一定数以上の場合のみ異常扱いする。
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
    glossary_entries: Mapping[str, str],
) -> list[str]:
    """
    原文に用語集の英語表現が存在するのに、
    対応する指定訳が翻訳結果へ含まれていない字幕を検出する。
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
                f"subtitle={index}, "
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