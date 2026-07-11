#!/usr/bin/env python3
import ast
import json
import re
import time
from datetime import datetime
from pathlib import Path
from lib.ollama import generate
from lib.progress import (
    ProgressTracker,
    format_duration,
)
from lib.config import (
    resolve_profile_config,
)
from lib.prompt import (
    DEFAULT_GLOSSARY_NAME,
    DEFAULT_STYLE_NAME,
    build_translation_prompt,
    load_glossary_entries,
)
from lib.srt import (
    SrtBlock,
    apply_translations,
    parse_speaker_from_text,
    parse_srt,
    write_structured_srt,
)
from lib.text import (
    cleanup_ocr_text,
    find_suspicious_latin_sequences,
    is_suspicious_ocr_text,
    mask_chinese_ocr_text,
    mask_suspicious_latin_sequences,
)
from lib.translation_validation import (
    source_contains_glossary_term,
    validate_translation_response,
)
from lib.noise import (
    NoiseDictionary,
    NoiseEntry,
    append_noise_candidates,
    apply_noise_dictionary_to_text,
    load_noise_dictionary,
)


MODEL = "qwen3:14b"

# 再試行用
MAX_TRANSLATION_ATTEMPTS = 3
TRANSLATION_DEBUG_DIR = (
    Path("~/tmp/subtitletool")
    .expanduser()
)

# 実際に翻訳する字幕数
CHUNK_SIZE = 10

# 翻訳対象の前後に参考として渡す字幕数
CONTEXT_SIZE = 15


def cleanup_blocks(blocks: list[SrtBlock]) -> list[SrtBlock]:
    """
    字幕番号とタイムコードを維持し、
    AIへ渡す本文だけOCR前処理する。
    """
    return [
        SrtBlock(
            number=block.number,
            timestamp=block.timestamp,
            text=cleanup_ocr_text(block.text),
        )
        for block in blocks
    ]


def apply_noise_to_blocks(
        blocks: list[SrtBlock],
        noise_dictionary: NoiseDictionary,
) -> list[SrtBlock]:
    """
    字幕番号とタイムコードを維持し、
    本文だけへnoise辞書を適用する。
    """
    return [
        SrtBlock(
            number=block.number,
            timestamp=block.timestamp,
            text=apply_noise_dictionary_to_text(
                block.text,
                noise_dictionary,
            ),
        )
        for block in blocks
    ]


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


def normalize_translation_text(
    text: str,
) -> str:
    """
    翻訳文内の字幕改行記号を空白へ変換する。

    半角スラッシュは前後に空白がある場合だけ対象とし、
    24/7、km/h、URLなどは維持する。
    """
    normalized = re.sub(
        r"(?:\s+/\s+|\s*／\s*)",
        " ",
        text,
    )

    normalized = re.sub(
        r"[ \t]+",
        " ",
        normalized,
    )

    return normalized.strip()


def normalize_translation_texts(
    translated_texts: list[str],
) -> list[str]:
    """
    翻訳済み字幕をSRT保存用に一括正規化する。
    """
    return [
        normalize_translation_text(text)
        for text in translated_texts
    ]


def extract_noise_candidates_from_blocks(
        blocks: list[SrtBlock],
) -> list[str]:
    """
    翻訳前字幕からOCR英字破損候補を抽出する。

    同じ候補は1件にまとめ、
    字幕の出現順を維持する。
    """
    candidates: list[str] = []

    for block in blocks:
        sequences = (
            find_suspicious_latin_sequences(
                block.text
            )
        )

        for sequence in sequences:
            if sequence in candidates:
                continue

            candidates.append(
                sequence
            )

    return candidates


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
    style_name: str = DEFAULT_STYLE_NAME,
    glossary_name: str = DEFAULT_GLOSSARY_NAME,
) -> str:
    request_json = build_translation_request_json(
        before_context,
        target_blocks,
        after_context,
    )

    base_prompt = build_translation_prompt(
        target_count=len(target_blocks),
        request_json=request_json,
        style_name=style_name,
        glossary_name=glossary_name,
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


def build_retry_instruction(
    errors: list[str],
) -> str:
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


STRUCTURAL_ERROR_PREFIXES = (
    "Invalid JSON response:",
    "Invalid translations:",
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


def extract_chinese_error_ids(
    errors: list[str],
) -> set[str]:
    """
    中国語混入エラーから対象字幕IDを抽出する。
    """
    subtitle_ids: set[str] = set()

    pattern = re.compile(
        r"subtitle_id=(?P<quote>['\"])"
        r"(?P<id>.+?)"
        r"(?P=quote)"
    )

    for error in errors:
        if not error.startswith(
            "Chinese-specific characters detected:"
        ):
            continue

        match = pattern.search(error)

        if not match:
            continue

        subtitle_ids.add(
            match.group("id")
        )

    return subtitle_ids


def extract_garbled_latin_errors(
    errors: list[str],
) -> dict[str, list[str]]:
    """
    OCR英字破損エラーから字幕IDと文字列を抽出する。
    """
    results: dict[str, list[str]] = {}

    pattern = re.compile(
        r"subtitle_id=(?P<quote>['\"])"
        r"(?P<id>.+?)"
        r"(?P=quote), "
        r"sequences=(?P<sequences>\[.*?\]), "
        r"text="
    )

    for error in errors:
        if not error.startswith(
            "Garbled Latin text detected:"
        ):
            continue

        match = pattern.search(error)

        if not match:
            continue

        raw_sequences = match.group(
            "sequences"
        )

        try:
            sequences = ast.literal_eval(
                raw_sequences
            )
        except (
            SyntaxError,
            ValueError,
        ):
            continue

        if not isinstance(sequences, list):
            continue

        if not all(
            isinstance(sequence, str)
            for sequence in sequences
        ):
            continue

        results[match.group("id")] = (
            sequences
        )

    return results


def extract_garbled_latin_candidates(
        errors: list[str],
) -> list[str]:
    """
    OCR英字破損エラーから、
    noise辞書へ保存する候補文字列を抽出する。
    """
    error_details = (
        extract_garbled_latin_errors(
            errors
        )
    )

    candidates: list[str] = []

    for sequences in error_details.values():
        for sequence in sequences:
            if sequence in candidates:
                continue

            candidates.append(
                sequence
            )

    return candidates


def build_chinese_retry_blocks(
    target_blocks: list[SrtBlock],
    errors: list[str],
) -> list[SrtBlock]:
    """
    中国語混入エラーが出た字幕だけ、
    再試行用入力の中国語OCR文字列をマスクする。
    """
    error_ids = extract_chinese_error_ids(
        errors
    )

    if not error_ids:
        return target_blocks

    return [
        SrtBlock(
            number=block.number,
            timestamp=block.timestamp,
            text=(
                mask_chinese_ocr_text(block.text)
                if block.number in error_ids
                else block.text
            ),
        )
        for block in target_blocks
    ]


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
    未翻訳英文の再試行指示を生成する。
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

【未翻訳英文の修正】

以下の字幕には未翻訳の英文が残っている。

{details}

必ず次を守ること。

* エラーに記載されたsubtitle_idの英文をすべて日本語へ翻訳する
* translationへ英文をそのままコピーしない
* 複数行の字幕は、すべての行を日本語へ翻訳する
* 一部だけ翻訳して残りの英文を残さない
* 人名、作品固有名詞、略語以外の英文を残さない
* 前回出力した未翻訳英文を再利用しない
* OCR破損ではない正常な英文を「（判読不能）」へ置き換えない
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


def save_failed_translation_response(
    response: str,
    *,
    chunk_start: int,
    chunk_end: int,
    attempt: int,
) -> Path:
    TRANSLATION_DEBUG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d-%H%M%S-%f"
    )

    output_path = TRANSLATION_DEBUG_DIR / (
        "failed-translation-"
        f"{chunk_start}-{chunk_end}-"
        f"attempt-{attempt}-"
        f"{timestamp}.txt"
    )

    output_path.write_text(
        response,
        encoding="utf-8",
    )

    return output_path


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


def translate_chunk(
    before_context: list[SrtBlock],
    target_blocks: list[SrtBlock],
    after_context: list[SrtBlock],
    model: str,
    *,
    chunk_start: int,
    chunk_end: int,
    glossary_entries: dict[str, str],
    noise_dictionary: NoiseDictionary,
    style_name: str = DEFAULT_STYLE_NAME,
    glossary_name: str = DEFAULT_GLOSSARY_NAME,
) -> list[str]:
    last_errors: list[str] = []
    last_translated_texts: list[str] = []

    input_noise_candidates = (
        extract_noise_candidates_from_blocks(
            target_blocks
        )
    )

    saved_input_noise_entries = (
        append_noise_candidates(
            noise_dictionary,
            input_noise_candidates,
        )
    )

    print_saved_noise_candidates(
        saved_input_noise_entries,
        noise_dictionary,
    )

    glossary_instruction = (
        build_required_glossary_instruction(
            target_blocks,
            glossary_entries,
        )
    )

    for attempt in range(
        1,
        MAX_TRANSLATION_ATTEMPTS + 1,
    ):
        retry_target_blocks = target_blocks

        if attempt > 1:
            retry_target_blocks = (
                build_chinese_retry_blocks(
                    target_blocks,
                    last_errors,
                )
            )

            retry_target_blocks = (
                build_latin_ocr_retry_blocks(
                    retry_target_blocks,
                    last_errors,
                )
            )

        prompt = build_prompt(
            before_context,
            retry_target_blocks,
            after_context,
            style_name=style_name,
            glossary_name=glossary_name,
        )

        prompt += glossary_instruction

        if attempt > 1:
            prompt += build_retry_instruction(
                last_errors
            )

            prompt += build_structural_retry_instruction(
                target_blocks,
                last_errors,
            )

            if not has_structural_validation_error(
                last_errors
            ):
                prompt += build_chinese_retry_instruction(
                    last_errors
                )

                prompt += build_latin_ocr_retry_instruction(
                    last_errors
                )

                prompt += (
                    build_untranslated_english_retry_instruction(
                        last_errors
                    )
                )

                prompt += build_glossary_retry_instruction(
                    last_errors
                )

                prompt += (
                    build_preserved_translations_instruction(
                        target_blocks,
                        last_translated_texts,
                        last_errors,
                    )
                )

        response = generate(
            prompt,
            model=model,
        )

        display_response = "\n".join(
            normalize_translation_text(line)
            for line in response.splitlines()
        )

        print("=" * 80)
        print(display_response)
        print("=" * 80)

        validation = validate_translation_response(
            response,
            expected_ids=[
                block.number
                for block in target_blocks
            ],
            source_texts=[
                block.text
                for block in target_blocks
            ],
            glossary_entries=glossary_entries,
        )

        if validation.warnings:
            print("Validation Warnings:")

            for warning in validation.warnings:
                print(f"  - {warning}")

        if validation.valid:
            return normalize_translation_texts(
                validation.translated_texts
            )

        failed_path = save_failed_translation_response(
            response,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            attempt=attempt,
        )

        last_errors = validation.reasons

        noise_candidates = (
            extract_garbled_latin_candidates(
                last_errors
            )
        )

        added_noise_entries = append_noise_candidates(
            noise_dictionary,
            noise_candidates,
        )

        print_saved_noise_candidates(
            added_noise_entries,
            noise_dictionary,
        )

        if (
            len(validation.translated_texts)
            == len(target_blocks)
        ):
            last_translated_texts = (
                validation.translated_texts
            )
        else:
            last_translated_texts = []

        print(
            "Translation validation failed "
            f"(attempt {attempt}/"
            f"{MAX_TRANSLATION_ATTEMPTS})"
        )

        print("Validation Errors:")

        for reason in last_errors:
            print(f"  - {reason}")

        print(f"Saved response: {failed_path}")

        if attempt < MAX_TRANSLATION_ATTEMPTS:
            print("Retrying translation...")

    raise RuntimeError(
        "Translation failed after "
        f"{MAX_TRANSLATION_ATTEMPTS} attempts "
        f"for subtitles "
        f"{chunk_start}-{chunk_end}: "
        + "; ".join(last_errors)
    )


def validate_resume_blocks(
    source_blocks: list[SrtBlock],
    translated_blocks: list[SrtBlock],
) -> None:
    """
    途中保存されたSRTが、入力SRTの先頭部分と一致するか確認する。

    本文は翻訳後なので比較しない。
    字幕番号とタイムコードだけを比較する。
    """
    if len(translated_blocks) > len(source_blocks):
        raise RuntimeError(
            "Resume failed: "
            "output SRT contains more subtitles than input SRT. "
            f"input={len(source_blocks)}, "
            f"output={len(translated_blocks)}"
        )

    for index, translated_block in enumerate(
        translated_blocks
    ):
        source_block = source_blocks[index]

        if translated_block.number != source_block.number:
            raise RuntimeError(
                "Resume failed: subtitle number mismatch "
                f"at position {index + 1}. "
                f"input={source_block.number}, "
                f"output={translated_block.number}"
            )

        if translated_block.timestamp != source_block.timestamp:
            raise RuntimeError(
                "Resume failed: timestamp mismatch "
                f"at subtitle {source_block.number}. "
                f"input={source_block.timestamp!r}, "
                f"output={translated_block.timestamp!r}"
            )


def build_latin_ocr_retry_blocks(
    target_blocks: list[SrtBlock],
    errors: list[str],
) -> list[SrtBlock]:
    """
    OCR英字破損が出た字幕だけ、
    該当文字列を再試行入力でマスクする。
    """
    error_details = (
        extract_garbled_latin_errors(
            errors
        )
    )

    if not error_details:
        return target_blocks

    return [
        SrtBlock(
            number=block.number,
            timestamp=block.timestamp,
            text=(
                mask_suspicious_latin_sequences(
                    block.text,
                    sequences=error_details[
                        block.number
                    ],
                )
                if block.number in error_details
                else block.text
            ),
        )
        for block in target_blocks
    ]


def print_profile_resolution(
    requested_profile: str | None,
    resolved_profile: str,
    fallback_used: bool,
) -> None:
    """
    profile解決結果を表示する。
    """
    requested_text = (
        requested_profile
        if requested_profile is not None
        else DEFAULT_STYLE_NAME
    )

    print(
        f"Profile Req : {requested_text}"
    )
    print(
        f"Profile Use : {resolved_profile}"
    )

    if fallback_used:
        print(
            "Warning     : "
            f"Profile {requested_text!r} was not found. "
            f"Using {resolved_profile!r}."
        )


def print_saved_noise_candidates(
        entries: list[NoiseEntry],
        noise_dictionary: NoiseDictionary,
) -> None:
    """
    今回noise.local.jsonへ保存した候補を表示する。
    """
    if not entries:
        return

    print("Noise Candidates Saved:")

    for entry in entries:
        print(
            f"  - {entry.source}"
        )

    print(
        "Noise Candidate File: "
        f"{noise_dictionary.local_path}"
    )


def print_noise_dictionary_summary(
        noise_dictionary: NoiseDictionary,
) -> None:
    """
    読み込んだnoise辞書の概要を表示する。
    """
    print(
        "Noise       : "
        f"{len(noise_dictionary.entries)} entries"
    )
    print(
        "Noise Local : "
        f"{'Yes' if noise_dictionary.local_loaded else 'No'}"
    )


def translate_srt(
    input_srt: str | Path,
    output_srt: str | Path,
    model: str = MODEL,
    chunk_size: int = CHUNK_SIZE,
    context_size: int = CONTEXT_SIZE,
    style_name: str = DEFAULT_STYLE_NAME,
    glossary_name: str = DEFAULT_GLOSSARY_NAME,
) -> Path:
    input_path = (
        Path(input_srt)
        .expanduser()
        .resolve()
    )

    output_path = (
        Path(output_srt)
        .expanduser()
        .resolve()
    )

    if input_path == output_path:
        raise ValueError(
            "Input and output SRT paths must be different: "
            f"{input_path}"
        )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"SRT not found: {input_path}"
        )

    if style_name != glossary_name:
        raise ValueError(
            "Style and glossary profiles must match "
            "during profile migration: "
            f"style={style_name!r}, "
            f"glossary={glossary_name!r}"
        )

    profile_config = resolve_profile_config(
        style_name
    )

    resolved_profile = (
        profile_config.resolved_profile
    )

    noise_dictionary = load_noise_dictionary(
        profile_config
    )

    source_blocks = parse_srt(
        input_path
    )

    if not source_blocks:
        raise RuntimeError(
            "No valid subtitle blocks: "
            f"{input_path}"
        )

    translated_blocks_all: list[SrtBlock] = []

    if output_path.exists():
        translated_blocks_all = parse_srt(
            output_path
        )

        if not translated_blocks_all:
            raise RuntimeError(
                "Resume failed: output SRT exists "
                "but contains no valid subtitle blocks: "
                f"{output_path}"
            )

        validate_resume_blocks(
            source_blocks,
            translated_blocks_all,
        )

    total_blocks = len(source_blocks)
    resume_start = len(
        translated_blocks_all
    )

    if resume_start == total_blocks:
        print()
        print("========================================")
        print("Translation Already Complete")
        print("========================================")

        print_profile_resolution(
            profile_config.requested_profile,
            resolved_profile,
            profile_config.fallback_used,
        )

        print_noise_dictionary_summary(
            noise_dictionary
        )

        print(f"Subtitles   : {total_blocks}")
        print(f"Output      : {output_path}")
        print("========================================")

        return output_path

    glossary_entries = load_glossary_entries(
        resolved_profile
    )

    remaining_blocks = (
        total_blocks - resume_start
    )

    remaining_chunks = (
        remaining_blocks
        + chunk_size
        - 1
    ) // chunk_size

    translation_started_at = (
        time.monotonic()
    )

    progress = ProgressTracker(
        total_chunks=remaining_chunks
    )

    print()
    print("========================================")
    print("Translation Start")
    print("========================================")
    print(f"Model       : {model}")

    print_profile_resolution(
        profile_config.requested_profile,
        resolved_profile,
        profile_config.fallback_used,
    )

    print_noise_dictionary_summary(
        noise_dictionary
    )

    print(f"Style       : {resolved_profile}")
    print(f"Glossary    : {resolved_profile}")
    print(f"Subtitles   : {total_blocks}")
    print(f"Chunk Size  : {chunk_size}")
    print(
        "Context     : "
        f"{context_size} before / after"
    )
    print(
        "Resume      : "
        f"{'Yes' if resume_start else 'No'}"
    )
    print(f"Completed   : {resume_start}")
    print(f"Remaining   : {remaining_blocks}")
    print(f"Chunks Left : {remaining_chunks}")
    print("========================================")

    chunk_starts = range(
        resume_start,
        total_blocks,
        chunk_size,
    )

    for chunk_number, start in enumerate(
        chunk_starts,
        start=1,
    ):
        end = min(
            start + chunk_size,
            total_blocks,
        )

        before_start = max(
            0,
            start - context_size,
        )

        after_end = min(
            total_blocks,
            end + context_size,
        )

        # タイムコードと元の字幕構造を維持するため、
        # OCR前処理前の翻訳対象を保持する。
        source_target_blocks = (
            source_blocks[start:end]
        )

        # AIへ渡す字幕本文だけOCR前処理する。
        before_context = cleanup_blocks(
            source_blocks[
                before_start:start
            ]
        )

        target_blocks = apply_noise_to_blocks(
            cleanup_blocks(
                source_target_blocks
            ),
            noise_dictionary,
        )

        after_context = cleanup_blocks(
            source_blocks[
                end:after_end
            ]
        )

        chunk_started_at = time.monotonic()

        print()
        print(
            f"[{chunk_number}/{remaining_chunks}] "
            f"Translating "
            f"{start + 1}-{end} / {total_blocks} "
            f"(context: "
            f"{len(before_context)} + "
            f"{len(after_context)})"
        )

        translated_texts = translate_chunk(
            before_context,
            target_blocks,
            after_context,
            model,
            chunk_start=start + 1,
            chunk_end=end,
            glossary_entries=glossary_entries,
            noise_dictionary=noise_dictionary,
            style_name=resolved_profile,
            glossary_name=resolved_profile,
        )

        translated_chunk_blocks = (
            apply_translations(
                source_target_blocks,
                translated_texts,
            )
        )

        translated_blocks_all.extend(
            translated_chunk_blocks
        )

        # 各チャンク終了時に途中保存する。
        write_structured_srt(
            output_path,
            translated_blocks_all,
        )

        chunk_elapsed = (
            time.monotonic()
            - chunk_started_at
        )

        progress.add(
            chunk_elapsed
        )

        elapsed = (
            time.monotonic()
            - translation_started_at
        )

        translated_count = len(
            translated_blocks_all
        )

        overall_progress = (
            translated_count
            / total_blocks
            * 100
        )

        print(
            "Session     : "
            f"{progress.progress_percent:5.1f}%"
        )
        print(
            "Progress    : "
            f"{overall_progress:5.1f}% "
            f"({translated_count}/{total_blocks})"
        )
        print(
            "Chunk Time  : "
            f"{format_duration(chunk_elapsed)}"
        )
        print(
            "Average     : "
            f"{progress.average_seconds:.1f} "
            "sec/chunk"
        )
        print(
            "Elapsed     : "
            f"{format_duration(elapsed)}"
        )
        print(
            "ETA         : "
            f"{format_duration(progress.eta_seconds)}"
        )

    total_elapsed = (
        time.monotonic()
        - translation_started_at
    )

    translated_count = len(
        translated_blocks_all
    )

    if translated_count != total_blocks:
        raise RuntimeError(
            "Subtitle count mismatch: "
            f"source={total_blocks}, "
            f"translated={translated_count}"
        )

    print()
    print("========================================")
    print("Translation Complete")
    print("========================================")
    print(f"Subtitles   : {translated_count}")
    print(
        "Chunks      : "
        f"{progress.completed_chunks}"
    )
    print(
        "Total Time  : "
        f"{format_duration(total_elapsed)}"
    )
    print(
        "Average     : "
        f"{progress.average_seconds:.1f} "
        "sec/chunk"
    )
    print(
        "Fastest     : "
        f"{progress.fastest_seconds:.1f} sec"
    )
    print(
        "Slowest     : "
        f"{progress.slowest_seconds:.1f} sec"
    )
    print(f"Output      : {output_path}")
    print("========================================")

    return output_path
