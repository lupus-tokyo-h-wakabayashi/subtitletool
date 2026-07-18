from __future__ import annotations

import ast
import json
import re

from lib.subtitle.srt import SrtBlock
from .translation_validation import (
    is_glossary_term_case_sensitive,
    source_contains_glossary_term,
)

STRUCTURAL_ERROR_PREFIXES = (
    "Translation response is empty",
    "Invalid JSON response:",
    "Invalid JSON root:",
    "Invalid JSON root keys:",
    "Invalid targets:",
    "Invalid target id:",
    "Empty target id:",
    "Invalid target item:",
    "Invalid target item keys:",
    "Invalid target source:",
    "Invalid target source keys:",
    "Invalid target source speaker:",
    "Invalid target source text:",
    "Empty target source text:",
    "Invalid translation text:",
    "Empty translation:",
    "Duplicate translation IDs:",
    "Missing translation IDs:",
    "Unexpected translation IDs:",
    "Invalid translation ID order:",
    "Source speaker count mismatch:",
    "Source text count mismatch:",
    "Source speaker changed:",
    "Source text changed:",
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
    検証エラーから修正対象のSRT字幕IDを抽出する。

    次の両形式に対応する。

    単一ID:
        subtitle_id='83'

    複数ID:
        subtitle_ids=['81', '82', '83']

    同じエラーに両形式が含まれる場合は、
    すべてのIDを重複なしで返す。
    """
    subtitle_ids: set[str] = set()

    single_id_pattern = re.compile(
        r"(?<![A-Za-z_])"
        r"subtitle_id=(?P<quote>['\"])"
        r"(?P<id>.+?)"
        r"(?P=quote)"
    )

    multiple_ids_pattern = re.compile(
        r"(?<![A-Za-z_])"
        r"subtitle_ids="
        r"(?P<ids>\[[^\]]*\])"
    )

    for error in errors:
        if (
            prefixes is not None
            and not error.startswith(prefixes)
        ):
            continue

        for match in single_id_pattern.finditer(
            error
        ):
            subtitle_id = match.group(
                "id"
            ).strip()

            if not subtitle_id:
                continue

            subtitle_ids.add(
                subtitle_id
            )

        for match in multiple_ids_pattern.finditer(
            error
        ):
            raw_ids = match.group(
                "ids"
            )

            try:
                parsed_ids = ast.literal_eval(
                    raw_ids
                )
            except (
                    SyntaxError,
                    ValueError,
            ):
                continue

            if not isinstance(
                parsed_ids,
                list,
            ):
                continue

            for parsed_id in parsed_ids:
                if not isinstance(
                    parsed_id,
                    str,
                ):
                    continue

                subtitle_id = parsed_id.strip()

                if not subtitle_id:
                    continue

                subtitle_ids.add(
                    subtitle_id
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
* 最上位キーはtargetsだけにする
* targetsは配列ではなくオブジェクトにする
* targetsは入力時と同じ字幕IDを同じ順序で含める
* 各字幕オブジェクトのキーはsourceとtranslationだけにする
* sourceのキーはspeakerとtextだけにする
* source.speakerとsource.textは入力時の値をそのまま維持する
* 変更してよいのはtranslationの値だけにする
* translationには同じ字幕IDの日本語字幕だけを入れる
* 字幕IDを追加、削除、変更、重複、並べ替えしない
* sourceを追加、削除、変更しない
* 入力側のtextやspeakerをtranslationへコピーしない
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

今回出力してよい字幕IDは次のIDだけである。

{ids_json}

必ず次を守ること。

* 最上位キーはtargetsだけにする
* targetsは配列ではなくオブジェクトにする
* targetsは必ず{len(target_ids)}件にする
* 上記IDを同じ順序で1回ずつ出力する
* 各字幕IDの値はオブジェクトにする
* 各字幕オブジェクトのキーはsourceとtranslationだけにする
* sourceのキーはspeakerとtextだけにする
* source.speakerとsource.textは入力時の値をそのまま出力する
* 変更してよいのはtranslationの値だけにする
* context_beforeとcontext_afterは出力しない
* 上記にない字幕IDを追加しない
* 一部の字幕IDだけを出力しない
* sourceを省略しない
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
    probable_ocr_lines: dict[str, list[str]],
) -> str:
    """
    未翻訳英文またはOCR英字破損の再試行指示を生成する。

    probable_ocr_linesには、
    translationへそのままコピーされた原文行のうち、
    OCR破損の可能性が高い行だけを渡す。
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

    ocr_instruction = ""

    if probable_ocr_lines:
        ocr_sections: list[str] = []

        for subtitle_id, lines in (
            probable_ocr_lines.items()
        ):
            line_instructions = "\n".join(
                (
                    f"* 原文行: {line}\n"
                    f"  必須形式: [1]{line}[/1]"
                )
                for line in lines
            )

            ocr_sections.append(
                (
                    f"字幕ID: {subtitle_id}\n"
                    f"{line_instructions}"
                )
            )

        ocr_details = "\n\n".join(
            ocr_sections
        )

        ocr_instruction = f"""

【今回の高確度OCR破損行】

以下の原文行は、
未翻訳の正常英文ではなく、
OCRで破損した可能性が高い。

{ocr_details}

上記の各原文行について、
必ず次を守ること。

* translationへ文字列をそのまま裸で残さない
* 日本語の助詞や語尾をOCR文字列へ直接付けない
* 原文行全体を一文字も変更せず[1]と[/1]で囲む
* [1]タグ内の大文字小文字、空白、数字、記号を変更しない
* [1]タグは原文の完全な1行だけを囲む
* [1]タグの中へ別の原文行を含めない
* [1]タグの中へ日本語訳を含めない
* 同じ字幕内の正常な英文は日本語へ翻訳する
* 正常な英文まで[1]タグで囲まない

例:

原文:

AV Cag are T
the wrong people!

translation:

[1]AV Cag are T[/1]／間違った人たちを！
"""

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
* 複数行の字幕は、すべての正常な英文を日本語へ翻訳する
* 一部だけ翻訳して残りの正常な英文を残さない
* 人名、作品固有名詞、略語以外の英文を残さない
* 前回出力した未翻訳英文を再利用しない
* 正常な英文を「（判読不能）」へ置き換えない
* 正常な英文を[1]タグで囲まない

OCRで壊れた英字列の場合は、必ず次を守ること。

* 壊れた英字列をtranslationへ裸のまま残さない
* 壊れた英字列を人名、地名、専門用語として推測しない
* 壊れた英字列をカタカナへ音写しない
* 高確度OCR破損行として指定された原文行は[1]タグで囲む
* [1]タグ内は原文の完全な1行と一字一句一致させる
* 前回と同じOCR破損文字列を裸のまま再利用しない

出力について、必ず次を守ること。

* JSONオブジェクト1個だけを出力する
* JSONの前後へ説明を追加しない
* Markdownコードブロックを付けない
* 翻訳方針や判断理由を出力しない

{ocr_instruction}
"""


def build_required_glossary_instruction(
    target_blocks: list[SrtBlock],
    glossary_entries: dict[str, str],
) -> str:
    """
    翻訳対象チャンクに含まれる用語集項目を抽出し、
    LLMへ使用必須の訳語として通知する。

    case_sensitive対象の用語は、
    原文の大文字・小文字まで一致した場合だけ通知する。
    """
    required_entries: list[
        tuple[str, str]
    ] = []

    for source_term, expected_term in (
        glossary_entries.items()
    ):
        case_sensitive = (
            is_glossary_term_case_sensitive(
                glossary_entries,
                source_term,
            )
        )

        if not any(
            source_contains_glossary_term(
                block.text,
                source_term,
                case_sensitive=case_sensitive,
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
