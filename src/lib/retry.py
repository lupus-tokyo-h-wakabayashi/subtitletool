from __future__ import annotations

import json
import re

from lib.srt import SrtBlock
from lib.translation.translation_validation import (
    source_contains_glossary_term,
)

STRUCTURAL_ERROR_PREFIXES = (
    "Invalid JSON response:",
    "Invalid JSON root:",
    "Invalid JSON root keys:",
    "Invalid translations:",
    "Invalid translation item:",
    "Invalid translation item keys:",
    "Missing translation item keys:",
    "Unexpected translation item keys:",
    "Invalid translation id:",
    "Empty translation id:",
    "Invalid translation text:",
    "Empty translation:",
    "Translation count mismatch:",
    "Duplicate translation IDs:",
    "Missing translation IDs:",
    "Unexpected translation IDs:",
    "Invalid translation ID order:",
    "Response has too many lines:",
    "Response is too long:",
)


def has_structural_validation_error(
    errors: list[str],
) -> bool:
    """
    ID対応やJSON構造を保証できないエラーがあるか判定する。
    """
    return any(
        error.startswith(
            STRUCTURAL_ERROR_PREFIXES
        )
        for error in errors
    )


def extract_error_subtitle_ids(
    errors: list[str],
    *,
    prefixes: tuple[str, ...] | None = None,
) -> set[str]:
    """
    subtitle_idを含む検証エラーから、
    修正対象のSRT字幕IDを抽出する。
    """
    subtitle_ids: set[str] = set()

    pattern = re.compile(
        r"subtitle_id=(?P<quote>['\"])"
        r"(?P<id>.+?)"
        r"(?P=quote)"
    )

    for error in errors:
        if (
            prefixes is not None
            and not error.startswith(prefixes)
        ):
            continue

        match = pattern.search(error)

        if not match:
            continue

        subtitle_ids.add(
            match.group("id")
        )

    return subtitle_ids


def extract_glossary_error_ids(
    errors: list[str],
) -> set[str]:
    """
    用語集違反がある字幕IDを抽出する。
    """
    return extract_error_subtitle_ids(
        errors,
        prefixes=(
            "Glossary violation:",
        ),
    )


def build_retry_instruction(
    errors: list[str],
) -> str:
    """
    検証失敗時の共通再試行指示を生成する。
    """
    error_text = "\n".join(
        f"* {error}"
        for error in errors
    )

    return f"""

【前回の出力は検証に失敗したため再翻訳する】

前回は以下の問題が検出された。

{error_text}

次の規則を必ず守ること。

* 出力はJSONオブジェクト1個だけにする
* 最上位キーはtranslationsのみにする
* translationsは入力targetと同じ件数にする
* 各要素のキーはidとtranslationだけにする
* idは入力targetのidをそのまま使用する
* idを追加、削除、変更、重複、並べ替えしない
* translationには日本語字幕だけを入れる
* 入力側のtextやspeakerを出力へコピーしない
* translationの代わりにtextを使用しない
* 複数字幕を1つのtranslationへ結合しない
* context_beforeとcontext_afterは出力しない
* JSONの前後へ説明やMarkdownコードブロックを追加しない
* 英文をそのまま出力しない
* 中国語を出力しない
* 同じ字幕を繰り返さない
* エラーにsubtitle_idがある場合、そのIDだけを修正する
* エラーがないIDのtranslationは前回の内容を維持する
* 用語集の指定訳を別のIDへ移動しない
* 修正対象外の字幕へ用語を追加しない
"""


def build_structural_retry_instruction(
    target_blocks: list[SrtBlock],
    errors: list[str],
) -> str:
    """
    JSON・ID構造エラー専用の再試行指示を生成する。
    """
    if not has_structural_validation_error(
        errors
    ):
        return ""

    target_ids = [
        block.number
        for block in target_blocks
    ]

    ids_json = json.dumps(
        target_ids,
        ensure_ascii=False,
    )

    return f"""

【JSON構造の修正】

今回出力してよいIDは次のIDだけである。

{ids_json}

必ず次を守ること。

* translationsは配列にする
* translationsは必ず{len(target_ids)}件にする
* 上記IDを同じ順序で1回ずつ出力する
* context_beforeとcontext_afterの内容・IDを出力しない
* 上記にないIDを追加しない
* 一部のIDだけを出力しない
* Markdownコードブロックを付けない
"""


def build_chinese_retry_instruction(
    errors: list[str],
) -> str:
    """
    中国語混入エラーから対象字幕ID・文字・本文を抽出し、
    再翻訳時に具体的な修正指示を追加する。
    """
    chinese_errors = [
        error
        for error in errors
        if error.startswith(
            "Chinese-specific characters detected:"
        )
    ]

    if not chinese_errors:
        return ""

    details = "\n".join(
        f"* {error}"
        for error in chinese_errors
    )

    return f"""

【中国語OCR混入の修正】

以下の字幕には中国語文字またはOCR破損文字列が残っている。

{details}

必ず次を守ること。

* エラーに記載されたsubtitle_idのtranslationを修正する
* charactersに記載された中国語文字を1文字も残さない
* 中国語文字列を固有名詞として保持しない
* 中国語文字列をカタカナへ音写しない
* 文脈から意味を判断し、自然な日本語へ置き換える
* 文脈から判別できない場合は「（判読不能）」へ置き換える
* 前回と同じ中国語入りtranslationを再利用しない
"""


def build_latin_ocr_retry_instruction(
    errors: list[str],
) -> str:
    """
    OCR英字破損の再試行指示を生成する。
    """
    ocr_errors = [
        error
        for error in errors
        if error.startswith(
            "Garbled Latin text detected:"
        )
    ]

    if not ocr_errors:
        return ""

    details = "\n".join(
        f"* {error}"
        for error in ocr_errors
    )

    return f"""

【英字OCR破損の修正】

以下の字幕にはOCRで壊れた英字列が残っている。

{details}

必ず次を守ること。

* sequencesに記載された文字列をtranslationへ残さない
* 壊れた文字列を人名、地名、固有名詞として推測しない
* 壊れた文字列をカタカナへ音写しない
* 文脈から意味を判断できる場合だけ自然な日本語へ置き換える
* 判断できない場合は「（判読不能）」とする
* 前回と同じOCR文字列を再利用しない
"""


def build_untranslated_english_retry_instruction(
    errors: list[str],
) -> str:
    """
    未翻訳英文またはOCR英字破損の再試行指示を生成する。
    """
    english_errors = [
        error
        for error in errors
        if error.startswith(
            "Untranslated English sentence detected:"
        )
    ]

    if not english_errors:
        return ""

    details = "\n".join(
        f"* {error}"
        for error in english_errors
    )

    return f"""

【未翻訳英文・OCR英字破損の修正】

以下の字幕には、未翻訳の英文または
OCRで壊れた英字列が残っている。

{details}

まず、残っている英字列が次のどちらかを判断すること。

1. 意味の通る正常な英文
2. OCRで壊れた意味不明な英字列

正常な英文の場合は、必ず次を守ること。

* エラーに記載されたsubtitle_idの英文をすべて日本語へ翻訳する
* translationへ英文をそのままコピーしない
* 複数行の字幕は、すべての行を日本語へ翻訳する
* 一部だけ翻訳して残りの英文を残さない
* 人名、作品固有名詞、略語以外の英文を残さない
* 前回出力した未翻訳英文を再利用しない
* 正常な英文を「（判読不能）」へ置き換えない

OCRで壊れた英字列の場合は、必ず次を守ること。

* 壊れた英字列をtranslationへそのまま残さない
* 壊れた英字列を人名、地名、専門用語として推測しない
* 壊れた英字列をカタカナへ音写しない
* 文脈から意味を判断できる場合だけ自然な日本語へ置き換える
* 文脈から判断できない場合は「（判読不能）」へ置き換える
* 前回と同じOCR破損文字列を再利用しない

出力について、必ず次を守ること。

* JSONオブジェクト1個だけを出力する
* JSONの前後へ説明を追加しない
* Markdownコードブロックを付けない
* 翻訳方針や判断理由を出力しない
"""


def build_required_glossary_instruction(
    target_blocks: list[SrtBlock],
    glossary_entries: dict[str, str],
) -> str:
    """
    翻訳対象チャンクに含まれる用語集項目を抽出し、
    LLMへ使用必須の訳語として通知する。
    """
    required_entries: list[tuple[str, str]] = []

    for source_term, expected_term in (
        glossary_entries.items()
    ):
        if not any(
            source_contains_glossary_term(
                block.text,
                source_term,
            )
            for block in target_blocks
        ):
            continue

        required_entries.append(
            (
                source_term,
                expected_term,
            )
        )

    if not required_entries:
        return ""

    lines = "\n".join(
        f"* {source_term} → {expected_term}"
        for source_term, expected_term
        in required_entries
    )

    return f"""

【この翻訳で必ず使用する用語】

{lines}

上記の英語表現が原文にある字幕では、
右側の日本語表記を一字一句そのまま使用すること。

* 別の日本語へ意訳しない
* 省略しない
* 一般的な訳語へ戻さない
* 長音、濁点、カタカナ表記を変更しない
* 再試行時も、すべての指定用語を維持する
"""


def build_glossary_retry_instruction(
    errors: list[str],
) -> str:
    """
    用語集違反の字幕ID・指定訳・前回訳を含む
    再試行指示を生成する。
    """
    glossary_errors = [
        error
        for error in errors
        if error.startswith(
            "Glossary violation:"
        )
    ]

    if not glossary_errors:
        return ""

    details = "\n".join(
        f"* {error}"
        for error in glossary_errors
    )

    target_ids = sorted(
        extract_glossary_error_ids(errors),
        key=lambda value: (
            int(value)
            if value.isdigit()
            else value
        ),
    )

    ids = ", ".join(target_ids)

    return f"""

【用語集違反の修正】

修正対象ID: {ids}

以下の用語集違反だけを修正する。

{details}

必ず次を守ること。

* 出力には入力targetの全IDを必ず含める
* 修正対象IDだけtranslationの内容を修正する
* 修正対象外IDはpreserved_translationsの訳文をそのままコピーする
* 修正対象IDだけを出力することは禁止する
* source_termが原文にある場合、expectedを一字一句そのまま使用する
* actualに記載された前回訳の問題部分を修正する
* 指定訳を別の字幕IDへ移動しない
* 指定訳を無関係な字幕へ追加しない
* 修正対象外IDの意味と内容を変更しない
* 用語を入れるために原文にない内容を追加しない
"""


def build_preserved_translations_instruction(
    target_blocks: list[SrtBlock],
    translated_texts: list[str],
    errors: list[str],
) -> str:
    """
    前回正常だった字幕を、再試行時の固定訳として通知する。
    """
    failed_ids = extract_error_subtitle_ids(
        errors
    )

    if not failed_ids:
        return ""

    preserved = [
        {
            "id": block.number,
            "translation": translation,
        }
        for block, translation in zip(
            target_blocks,
            translated_texts,
            strict=True,
        )
        if block.number not in failed_ids
    ]

    if not preserved:
        return ""

    preserved_json = json.dumps(
        {
            "preserved_translations": preserved,
        },
        ensure_ascii=False,
        indent=2,
    )

    return f"""

【変更禁止の前回正常訳】

以下は前回の検証で問題が検出されなかった字幕である。

{preserved_json}

必ず次を守ること。

* preserved_translationsのtranslationをそのまま出力する
* 表現、語尾、句読点、内容を変更しない
* 別のidへ移動しない
* 修正対象字幕の内容を混ぜない
* 出力JSONには入力targetの全IDを含める
* エラーに記載されたsubtitle_idだけ内容を修正する
* preserved_translationsのIDも省略せず出力する
"""
