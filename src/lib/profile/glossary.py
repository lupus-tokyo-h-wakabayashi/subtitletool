from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_PROFILE_NAME,
    resolve_profile_config,
)

SUPPORTED_GLOSSARY_VERSION = 1

GLOSSARY_ROOT_REQUIRED_KEYS = {
    "version",
    "entries",
}

GLOSSARY_ROOT_ALLOWED_KEYS = {
    "version",
    "description",
    "entries",
}

GLOSSARY_ENTRY_REQUIRED_KEYS = {
    "source",
    "target",
}

GLOSSARY_ENTRY_ALLOWED_KEYS = {
    "source",
    "target",
}


@dataclass(frozen=True)
class GlossaryEntry:
    """
    Glossaryの1エントリ。
    """

    source: str
    target: str


@dataclass(frozen=True)
class GlossaryDocument:
    """
    検証済みGlossary JSON。
    """

    version: int
    description: str | None
    entries: tuple[GlossaryEntry, ...]
    path: Path


def read_glossary_json(
    path: Path,
) -> dict[str, Any]:
    """
    Glossary JSONを読み込む。
    """
    try:
        raw_text = path.read_text(
            encoding="utf-8",
            errors="strict",
        )
    except OSError as error:
        raise RuntimeError(
            "Failed to read glossary JSON: "
            f"path={path}, error={error}"
        ) from error

    try:
        payload = json.loads(
            raw_text
        )
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Invalid glossary JSON: "
            f"path={path}, "
            f"line={error.lineno}, "
            f"column={error.colno}, "
            f"message={error.msg}"
        ) from error

    if not isinstance(payload, dict):
        raise RuntimeError(
            "Invalid glossary root: "
            f"path={path}, expected=object"
        )

    return payload


def validate_root_keys(
    payload: dict[str, Any],
    path: Path,
) -> None:
    """
    ルートの必須キーと不明キーを検証する。
    """
    actual_keys = set(
        payload.keys()
    )

    missing_keys = (
        GLOSSARY_ROOT_REQUIRED_KEYS
        - actual_keys
    )

    if missing_keys:
        raise RuntimeError(
            "Missing glossary root keys: "
            f"path={path}, "
            f"keys={sorted(missing_keys)}"
        )

    unknown_keys = (
        actual_keys
        - GLOSSARY_ROOT_ALLOWED_KEYS
    )

    if unknown_keys:
        raise RuntimeError(
            "Unknown glossary root keys: "
            f"path={path}, "
            f"keys={sorted(unknown_keys)}"
        )


def validate_version(
    value: Any,
    path: Path,
) -> int:
    """
    versionが整数の1であることを検証する。
    """
    if type(value) is not int:
        raise RuntimeError(
            "Invalid glossary version type: "
            f"path={path}, "
            f"value={value!r}, "
            "expected=integer"
        )

    if value != SUPPORTED_GLOSSARY_VERSION:
        raise RuntimeError(
            "Unsupported glossary version: "
            f"path={path}, "
            f"value={value}, "
            "supported="
            f"{SUPPORTED_GLOSSARY_VERSION}"
        )

    return value


def validate_description(
    payload: dict[str, Any],
    path: Path,
) -> str | None:
    """
    任意descriptionを検証する。

    キーが存在しない場合だけNoneを返す。
    指定されている場合は、
    空でない文字列である必要がある。
    """
    if "description" not in payload:
        return None

    value = payload["description"]

    if not isinstance(value, str):
        raise RuntimeError(
            "Invalid glossary description: "
            f"path={path}, "
            "expected=string"
        )

    description = value.strip()

    if not description:
        raise RuntimeError(
            "Empty glossary description: "
            f"path={path}"
        )

    return description


def parse_glossary_entry(
    value: Any,
    *,
    index: int,
    path: Path,
) -> GlossaryEntry:
    """
    entries配列内の1要素を検証する。
    """
    if not isinstance(value, dict):
        raise RuntimeError(
            "Invalid glossary entry: "
            f"path={path}, "
            f"index={index}, "
            "expected=object"
        )

    actual_keys = set(
        value.keys()
    )

    missing_keys = (
        GLOSSARY_ENTRY_REQUIRED_KEYS
        - actual_keys
    )

    if missing_keys:
        raise RuntimeError(
            "Missing glossary entry keys: "
            f"path={path}, "
            f"index={index}, "
            f"keys={sorted(missing_keys)}"
        )

    unknown_keys = (
        actual_keys
        - GLOSSARY_ENTRY_ALLOWED_KEYS
    )

    if unknown_keys:
        raise RuntimeError(
            "Unknown glossary entry keys: "
            f"path={path}, "
            f"index={index}, "
            f"keys={sorted(unknown_keys)}"
        )

    source_value = value["source"]
    target_value = value["target"]

    if not isinstance(source_value, str):
        raise RuntimeError(
            "Invalid glossary source: "
            f"path={path}, "
            f"index={index}, "
            "expected=string"
        )

    if not isinstance(target_value, str):
        raise RuntimeError(
            "Invalid glossary target: "
            f"path={path}, "
            f"index={index}, "
            "expected=string"
        )

    source = source_value.strip()
    target = target_value.strip()

    if not source:
        raise RuntimeError(
            "Empty glossary source: "
            f"path={path}, "
            f"index={index}"
        )

    if not target:
        raise RuntimeError(
            "Empty glossary target: "
            f"path={path}, "
            f"index={index}"
        )

    return GlossaryEntry(
        source=source,
        target=target,
    )


def parse_glossary_document(
    payload: dict[str, Any],
    *,
    path: Path,
    allow_empty: bool,
) -> GlossaryDocument:
    """
    Glossary JSON全体を検証する。
    """
    validate_root_keys(
        payload,
        path,
    )

    version = validate_version(
        payload["version"],
        path,
    )

    description = validate_description(
        payload,
        path,
    )

    raw_entries = payload["entries"]

    if not isinstance(raw_entries, list):
        raise RuntimeError(
            "Invalid glossary entries: "
            f"path={path}, expected=array"
        )

    entries: list[GlossaryEntry] = []
    seen_sources: dict[str, int] = {}

    for index, raw_entry in enumerate(
        raw_entries,
        start=1,
    ):
        entry = parse_glossary_entry(
            raw_entry,
            index=index,
            path=path,
        )

        normalized_source = (
            entry.source.casefold()
        )

        previous_index = seen_sources.get(
            normalized_source
        )

        if previous_index is not None:
            raise RuntimeError(
                "Duplicate glossary source: "
                f"path={path}, "
                f"source={entry.source!r}, "
                f"first_index={previous_index}, "
                f"duplicate_index={index}"
            )

        seen_sources[
            normalized_source
        ] = index

        entries.append(
            entry
        )

    if not entries and not allow_empty:
        raise RuntimeError(
            "No valid glossary entries: "
            f"path={path}"
        )

    return GlossaryDocument(
        version=version,
        description=description,
        entries=tuple(entries),
        path=path,
    )


def read_glossary_document(
    path: Path,
    *,
    allow_empty: bool,
) -> GlossaryDocument:
    """
    指定パスのGlossary JSONを読み込み、
    スキーマ検証済みDocumentを返す。
    """
    payload = read_glossary_json(
        path
    )

    return parse_glossary_document(
        payload,
        path=path,
        allow_empty=allow_empty,
    )


def load_glossary_document(
    profile_name: str | None = None,
) -> GlossaryDocument:
    """
    profileを解決してGlossary JSONを読み込む。

    defaultのみ空entriesを許可する。
    """
    config = resolve_profile_config(
        profile_name
    )

    return read_glossary_document(
        config.glossary_path,
        allow_empty=(
            config.resolved_profile
            == DEFAULT_PROFILE_NAME
        ),
    )


def load_glossary_entries(
    profile_name: str | None = None,
) -> dict[str, str]:
    """
    Validation用の辞書を配列順のまま返す。
    """
    document = load_glossary_document(
        profile_name
    )

    return {
        entry.source: entry.target
        for entry in document.entries
    }


def build_glossary_prompt(
    profile_name: str | None = None,
) -> str:
    """
    LLM Prompt用の従来形式へ整形する。
    """
    document = load_glossary_document(
        profile_name
    )

    return "\n".join(
        (
            f"{entry.source} = "
            f"{entry.target}"
        )
        for entry in document.entries
    )
