from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lib.config import ProfileConfig
from lib.text import (
    DEFAULT_ALLOWED_LATIN_TERMS,
    find_heuristic_latin_noise_sequences,
    normalize_latin_token,
)

SUPPORTED_NOISE_VERSION = 1

VALID_NOISE_ACTIONS = {
    "mask",
    "replace",
    "ignore",
    "warn",
}

VALID_NOISE_STATUSES = {
    "confirmed",
    "candidate",
    "ignored",
}

MIN_NOISE_CANDIDATE_LENGTH = 4
MAX_NOISE_CANDIDATE_LENGTH = 120
MIN_NOISE_CANDIDATE_ASCII_LETTERS = 4
MAX_NOISE_CANDIDATES_PER_APPEND = 20


@dataclass(frozen=True)
class NoiseEntry:
    """
    OCRノイズ辞書の1エントリ。
    """

    source: str
    replacement: str
    action: str
    status: str


@dataclass(frozen=True)
class NoiseDictionary:
    """
    profile内の正式辞書とローカル辞書を
    マージした読み込み結果。
    """

    profile_name: str
    entries: dict[str, NoiseEntry]
    official_path: Path
    local_path: Path
    local_loaded: bool


def read_noise_json(
    path: Path,
    *,
    required: bool,
) -> dict[str, Any] | None:
    """
    noise辞書JSONを読み込む。

    required=Falseかつファイルが存在しない場合はNoneを返す。
    """
    if not path.is_file():
        if required:
            raise FileNotFoundError(
                f"Noise dictionary not found: {path}"
            )

        return None

    try:
        raw_text = path.read_text(
            encoding="utf-8",
            errors="strict",
        )
    except OSError as error:
        raise RuntimeError(
            f"Failed to read noise dictionary: {path}"
        ) from error

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Invalid noise dictionary JSON: "
            f"path={path}, "
            f"line={error.lineno}, "
            f"column={error.colno}, "
            f"message={error.msg}"
        ) from error

    if not isinstance(data, dict):
        raise RuntimeError(
            "Noise dictionary root must be an object: "
            f"path={path}"
        )

    return data


def validate_noise_document(
    data: dict[str, Any],
    *,
    path: Path,
) -> list[NoiseEntry]:
    """
    noise辞書JSONの構造と各エントリを検証する。
    """
    version = data.get("version")

    if version != SUPPORTED_NOISE_VERSION:
        raise RuntimeError(
            "Unsupported noise dictionary version: "
            f"path={path}, "
            f"expected={SUPPORTED_NOISE_VERSION}, "
            f"actual={version!r}"
        )

    raw_entries = data.get("entries")

    if not isinstance(raw_entries, list):
        raise RuntimeError(
            "Noise dictionary entries must be a list: "
            f"path={path}"
        )

    entries: list[NoiseEntry] = []
    seen_sources: set[str] = set()

    for index, raw_entry in enumerate(
        raw_entries,
        start=1,
    ):
        if not isinstance(raw_entry, dict):
            raise RuntimeError(
                "Noise dictionary entry must be an object: "
                f"path={path}, "
                f"entry={index}"
            )

        source = raw_entry.get("source")
        replacement = raw_entry.get(
            "replacement"
        )
        action = raw_entry.get("action")
        status = raw_entry.get("status")

        if not isinstance(source, str):
            raise RuntimeError(
                "Noise entry source must be a string: "
                f"path={path}, "
                f"entry={index}"
            )

        source = source.strip()

        if not source:
            raise RuntimeError(
                "Noise entry source must not be empty: "
                f"path={path}, "
                f"entry={index}"
            )

        if not isinstance(replacement, str):
            raise RuntimeError(
                "Noise entry replacement must be a string: "
                f"path={path}, "
                f"entry={index}, "
                f"source={source!r}"
            )

        if not isinstance(action, str):
            raise RuntimeError(
                "Noise entry action must be a string: "
                f"path={path}, "
                f"entry={index}, "
                f"source={source!r}"
            )

        if action not in VALID_NOISE_ACTIONS:
            raise RuntimeError(
                "Invalid noise entry action: "
                f"path={path}, "
                f"entry={index}, "
                f"source={source!r}, "
                f"action={action!r}"
            )

        if not isinstance(status, str):
            raise RuntimeError(
                "Noise entry status must be a string: "
                f"path={path}, "
                f"entry={index}, "
                f"source={source!r}"
            )

        if status not in VALID_NOISE_STATUSES:
            raise RuntimeError(
                "Invalid noise entry status: "
                f"path={path}, "
                f"entry={index}, "
                f"source={source!r}, "
                f"status={status!r}"
            )

        if source in seen_sources:
            raise RuntimeError(
                "Duplicate noise entry source: "
                f"path={path}, "
                f"source={source!r}"
            )

        seen_sources.add(source)

        entries.append(
            NoiseEntry(
                source=source,
                replacement=replacement,
                action=action,
                status=status,
            )
        )

    return entries


def load_noise_entries(
    path: Path,
    *,
    required: bool,
) -> list[NoiseEntry]:
    """
    noise辞書を読み込み、検証済みエントリを返す。
    """
    data = read_noise_json(
        path,
        required=required,
    )

    if data is None:
        return []

    return validate_noise_document(
        data,
        path=path,
    )


def merge_noise_entries(
    official_entries: list[NoiseEntry],
    local_entries: list[NoiseEntry],
) -> dict[str, NoiseEntry]:
    """
    正式辞書とローカル辞書をsource単位でマージする。

    同じsourceが存在する場合はlocal側を優先する。
    """
    merged = {
        entry.source: entry
        for entry in official_entries
    }

    for entry in local_entries:
        merged[entry.source] = entry

    return merged


def apply_noise_entry(
    text: str,
    entry: NoiseEntry,
) -> str:
    """
    確認済みのnoise辞書エントリを文字列へ適用する。

    mask:
        sourceをreplacementへ置換する

    replace:
        sourceをreplacementへ置換する

    ignore / warn:
        この段階では文字列を変更しない
    """
    if entry.action in {
        "ignore",
        "warn",
    }:
        return text

    if entry.action in {
        "mask",
        "replace",
    }:
        return text.replace(
            entry.source,
            entry.replacement,
        )

    raise RuntimeError(
        "Unsupported noise action: "
        f"source={entry.source!r}, "
        f"action={entry.action!r}"
    )


def build_noise_source_pattern(
    source: str,
) -> re.Pattern[str]:
    """
    noise辞書のsourceを検索用パターンへ変換する。

    source自体は正規表現として扱わず、
    空白数と大文字小文字の差だけを吸収する。
    """
    parts = re.split(
        r"\s+",
        source.strip(),
    )

    expression = r"\s+".join(
        re.escape(part)
        for part in parts
        if part
    )

    return re.compile(
        expression,
        re.IGNORECASE,
    )


def find_confirmed_noise_sequences(
    text: str,
    noise_dictionary: NoiseDictionary,
) -> list[str]:
    """
    confirmedのnoise辞書と一致する文字列を、
    本文中の出現順で返す。
    """
    matches: list[
        tuple[int, str]
    ] = []

    for entry in (
        noise_dictionary.entries.values()
    ):
        if entry.status != "confirmed":
            continue

        if entry.action not in {
            "mask",
            "replace",
            "warn",
        }:
            continue

        pattern = build_noise_source_pattern(
            entry.source
        )

        for match in pattern.finditer(text):
            matches.append(
                (
                    match.start(),
                    match.group(0),
                )
            )

    matches.sort(
        key=lambda item: item[0]
    )

    results: list[str] = []

    for _, sequence in matches:
        if sequence in results:
            continue

        results.append(sequence)

    return results


def find_suspicious_latin_sequences(
    text: str,
    noise_dictionary: NoiseDictionary,
    *,
    allowed_terms: set[str] | None = None,
) -> list[str]:
    """
    confirmed辞書と汎用ヒューリスティックを使って、
    OCR破損候補を抽出する。
    """
    dictionary_sequences = (
        find_confirmed_noise_sequences(
            text,
            noise_dictionary,
        )
    )

    heuristic_sequences = (
        find_heuristic_latin_noise_sequences(
            text,
            allowed_terms=allowed_terms,
        )
    )

    candidates: list[str] = []

    for sequence in [
        *dictionary_sequences,
        *heuristic_sequences,
    ]:
        if sequence in candidates:
            continue

        candidates.append(sequence)

    candidates.sort(
        key=lambda sequence: text.find(sequence)
    )

    return candidates


def apply_noise_dictionary_to_text(
    text: str,
    noise_dictionary: NoiseDictionary,
) -> str:
    """
    noise辞書を文字列へ適用する。

    confirmedかつmask/replaceのエントリだけを適用する。
    長いsourceから処理し、部分一致による誤置換を避ける。
    """
    applicable_entries = [
        entry
        for entry in noise_dictionary.entries.values()
        if (
            entry.status == "confirmed"
            and entry.action in {
                "mask",
                "replace",
            }
            and entry.source in text
        )
    ]

    applicable_entries.sort(
        key=lambda entry: len(entry.source),
        reverse=True,
    )

    result = text

    for entry in applicable_entries:
        result = apply_noise_entry(
            result,
            entry,
        )

    return result


def load_noise_dictionary(
    profile_config: ProfileConfig,
) -> NoiseDictionary:
    """
    解決済みprofileのnoise辞書を読み込む。

    読み込み順:
        noise.json
        noise.local.json

    local側を後勝ちでマージする。
    """
    official_entries = load_noise_entries(
        profile_config.noise_path,
        required=True,
    )

    local_entries = load_noise_entries(
        profile_config.noise_local_path,
        required=False,
    )

    merged_entries = merge_noise_entries(
        official_entries,
        local_entries,
    )

    return NoiseDictionary(
        profile_name=(
            profile_config.resolved_profile
        ),
        entries=merged_entries,
        official_path=profile_config.noise_path,
        local_path=(
            profile_config.noise_local_path
        ),
        local_loaded=(
            profile_config.noise_local_path.is_file()
        ),
    )


def normalize_noise_candidate(
    source: str,
) -> str:
    """
    OCRノイズ候補を保存用の表記へ正規化する。
    """
    return re.sub(
        r"\s+",
        " ",
        source,
    ).strip()


def is_valid_noise_candidate(
    source: str,
) -> bool:
    """
    OCRノイズ候補として保存可能か判定する。
    """
    normalized = normalize_noise_candidate(
        source
    )

    if not normalized:
        return False

    if (
        len(normalized)
        < MIN_NOISE_CANDIDATE_LENGTH
    ):
        return False

    if (
        len(normalized)
        > MAX_NOISE_CANDIDATE_LENGTH
    ):
        return False

    ascii_letter_count = sum(
        character.isascii()
        and character.isalpha()
        for character in normalized
    )

    if (
        ascii_letter_count
        < MIN_NOISE_CANDIDATE_ASCII_LETTERS
    ):
        return False

    allowed_terms = {
        normalize_latin_token(term)
        for term in DEFAULT_ALLOWED_LATIN_TERMS
    }

    normalized_term = normalize_latin_token(
        normalized
    )

    if normalized_term in allowed_terms:
        return False

    return True


def serialize_noise_entry(
    entry: NoiseEntry,
) -> dict[str, str]:
    """
    NoiseEntryをJSON保存用辞書へ変換する。
    """
    return {
        "source": entry.source,
        "replacement": entry.replacement,
        "action": entry.action,
        "status": entry.status,
    }


def write_noise_entries(
    path: Path,
    entries: list[NoiseEntry],
) -> None:
    """
    noise辞書をJSONとして安全に保存する。
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "version": SUPPORTED_NOISE_VERSION,
        "entries": [
            serialize_noise_entry(entry)
            for entry in entries
        ],
    }

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(path)


def append_noise_candidates(
    noise_dictionary: NoiseDictionary,
    sources: list[str],
) -> list[NoiseEntry]:
    """
    未登録のOCRノイズ候補をnoise.local.jsonへ追記する。

    既にofficialまたはlocalに登録済みのsourceは追加しない。
    戻り値は今回新たに追加したエントリ。
    """
    normalized_sources: list[str] = []

    for source in sources:
        normalized = normalize_noise_candidate(
            source
        )

        if not is_valid_noise_candidate(
            normalized
        ):
            continue

        if normalized in normalized_sources:
            continue

        normalized_sources.append(
            normalized
        )

        if (
            len(normalized_sources)
            >= MAX_NOISE_CANDIDATES_PER_APPEND
        ):
            break

    existing_sources = set(
        noise_dictionary.entries.keys()
    )

    new_entries = [
        NoiseEntry(
            source=source,
            replacement="（判読不能）",
            action="mask",
            status="candidate",
        )
        for source in normalized_sources
        if source not in existing_sources
    ]

    if not new_entries:
        return []

    local_entries = load_noise_entries(
        noise_dictionary.local_path,
        required=False,
    )

    local_sources = {
        entry.source
        for entry in local_entries
    }

    entries_to_append = [
        entry
        for entry in new_entries
        if entry.source not in local_sources
    ]

    if not entries_to_append:
        return []

    updated_local_entries = [
        *local_entries,
        *entries_to_append,
    ]

    write_noise_entries(
        noise_dictionary.local_path,
        updated_local_entries,
    )

    return entries_to_append
