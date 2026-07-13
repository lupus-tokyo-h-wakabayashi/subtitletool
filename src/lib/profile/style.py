from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import (
    resolve_profile_config,
)

SUPPORTED_STYLE_VERSION = 1

STYLE_ROOT_REQUIRED_KEYS = {
    "version",
    "sections",
}

STYLE_ROOT_ALLOWED_KEYS = {
    "version",
    "description",
    "sections",
}

STYLE_SECTION_REQUIRED_KEYS = {
    "name",
    "rules",
}

STYLE_SECTION_ALLOWED_KEYS = {
    "name",
    "rules",
}


@dataclass(frozen=True)
class StyleSection:
    """
    Style JSONの1セクション。
    """

    name: str
    rules: tuple[str, ...]


@dataclass(frozen=True)
class StyleDocument:
    """
    検証済みStyle JSON。
    """

    version: int
    description: str | None
    sections: tuple[StyleSection, ...]
    path: Path


def read_style_json(
    path: Path,
) -> dict[str, Any]:
    """
    Style JSONを読み込む。
    """
    try:
        raw_text = path.read_text(
            encoding="utf-8",
            errors="strict",
        )
    except OSError as error:
        raise RuntimeError(
            "Failed to read style JSON: "
            f"path={path}, error={error}"
        ) from error

    try:
        payload = json.loads(
            raw_text
        )
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Invalid style JSON: "
            f"path={path}, "
            f"line={error.lineno}, "
            f"column={error.colno}, "
            f"message={error.msg}"
        ) from error

    if not isinstance(payload, dict):
        raise RuntimeError(
            "Invalid style root: "
            f"path={path}, expected=object"
        )

    return payload


def validate_style_root_keys(
    payload: dict[str, Any],
    path: Path,
) -> None:
    """
    Style JSONルートのキーを検証する。
    """
    actual_keys = set(
        payload.keys()
    )

    missing_keys = (
        STYLE_ROOT_REQUIRED_KEYS
        - actual_keys
    )

    if missing_keys:
        raise RuntimeError(
            "Missing style root keys: "
            f"path={path}, "
            f"keys={sorted(missing_keys)}"
        )

    unknown_keys = (
        actual_keys
        - STYLE_ROOT_ALLOWED_KEYS
    )

    if unknown_keys:
        raise RuntimeError(
            "Unknown style root keys: "
            f"path={path}, "
            f"keys={sorted(unknown_keys)}"
        )


def validate_style_version(
    value: Any,
    path: Path,
) -> int:
    """
    versionが整数の1であることを検証する。
    """
    if type(value) is not int:
        raise RuntimeError(
            "Invalid style version type: "
            f"path={path}, "
            f"value={value!r}, "
            "expected=integer"
        )

    if value != SUPPORTED_STYLE_VERSION:
        raise RuntimeError(
            "Unsupported style version: "
            f"path={path}, "
            f"value={value}, "
            "supported="
            f"{SUPPORTED_STYLE_VERSION}"
        )

    return value


def validate_style_description(
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
            "Invalid style description: "
            f"path={path}, "
            "expected=string"
        )

    description = value.strip()

    if not description:
        raise RuntimeError(
            "Empty style description: "
            f"path={path}"
        )

    return description


def parse_style_section(
    value: Any,
    *,
    index: int,
    path: Path,
) -> StyleSection:
    """
    sections配列内の1要素を検証する。
    """
    if not isinstance(value, dict):
        raise RuntimeError(
            "Invalid style section: "
            f"path={path}, "
            f"index={index}, "
            "expected=object"
        )

    actual_keys = set(
        value.keys()
    )

    missing_keys = (
        STYLE_SECTION_REQUIRED_KEYS
        - actual_keys
    )

    if missing_keys:
        raise RuntimeError(
            "Missing style section keys: "
            f"path={path}, "
            f"index={index}, "
            f"keys={sorted(missing_keys)}"
        )

    unknown_keys = (
        actual_keys
        - STYLE_SECTION_ALLOWED_KEYS
    )

    if unknown_keys:
        raise RuntimeError(
            "Unknown style section keys: "
            f"path={path}, "
            f"index={index}, "
            f"keys={sorted(unknown_keys)}"
        )

    name_value = value["name"]
    rules_value = value["rules"]

    if not isinstance(name_value, str):
        raise RuntimeError(
            "Invalid style section name: "
            f"path={path}, "
            f"index={index}, "
            "expected=string"
        )

    name = name_value.strip()

    if not name:
        raise RuntimeError(
            "Empty style section name: "
            f"path={path}, "
            f"index={index}"
        )

    if not isinstance(rules_value, list):
        raise RuntimeError(
            "Invalid style rules: "
            f"path={path}, "
            f"index={index}, "
            "expected=array"
        )

    if not rules_value:
        raise RuntimeError(
            "Empty style rules: "
            f"path={path}, "
            f"index={index}"
        )

    rules: list[str] = []

    for rule_index, rule_value in enumerate(
        rules_value,
        start=1,
    ):
        if not isinstance(rule_value, str):
            raise RuntimeError(
                "Invalid style rule: "
                f"path={path}, "
                f"section_index={index}, "
                f"rule_index={rule_index}, "
                "expected=string"
            )

        rule = rule_value.strip()

        if not rule:
            raise RuntimeError(
                "Empty style rule: "
                f"path={path}, "
                f"section_index={index}, "
                f"rule_index={rule_index}"
            )

        rules.append(
            rule
        )

    return StyleSection(
        name=name,
        rules=tuple(rules),
    )


def parse_style_document(
    payload: dict[str, Any],
    *,
    path: Path,
) -> StyleDocument:
    """
    Style JSON全体を検証する。
    """
    validate_style_root_keys(
        payload,
        path,
    )

    version = validate_style_version(
        payload["version"],
        path,
    )

    description = validate_style_description(
        payload,
        path,
    )

    raw_sections = payload["sections"]

    if not isinstance(raw_sections, list):
        raise RuntimeError(
            "Invalid style sections: "
            f"path={path}, expected=array"
        )

    if not raw_sections:
        raise RuntimeError(
            "No style sections: "
            f"path={path}"
        )

    sections: list[StyleSection] = []
    seen_names: dict[str, int] = {}

    for index, raw_section in enumerate(
        raw_sections,
        start=1,
    ):
        section = parse_style_section(
            raw_section,
            index=index,
            path=path,
        )

        normalized_name = (
            section.name.casefold()
        )

        previous_index = seen_names.get(
            normalized_name
        )

        if previous_index is not None:
            raise RuntimeError(
                "Duplicate style section name: "
                f"path={path}, "
                f"name={section.name!r}, "
                f"first_index={previous_index}, "
                f"duplicate_index={index}"
            )

        seen_names[
            normalized_name
        ] = index

        sections.append(
            section
        )

    return StyleDocument(
        version=version,
        description=description,
        sections=tuple(sections),
        path=path,
    )


def read_style_document(
    path: Path,
) -> StyleDocument:
    """
    指定パスのStyle JSONを読み込み、
    スキーマ検証済みDocumentを返す。
    """
    payload = read_style_json(
        path
    )

    return parse_style_document(
        payload,
        path=path,
    )


def load_style_document(
    profile_name: str | None = None,
) -> StyleDocument:
    """
    profileを解決してStyle JSONを読み込み、
    スキーマ検証済みDocumentを返す。
    """
    config = resolve_profile_config(
        profile_name
    )

    return read_style_document(
        config.style_path
    )


def build_style_prompt(
    profile_name: str | None = None,
) -> str:
    """
    Style JSONを、
    翻訳Promptへ埋め込む文字列へ変換する。
    """
    document = load_style_document(
        profile_name
    )

    section_texts: list[str] = []

    for section in document.sections:
        lines = [
            f"【{section.name}】",
            "",
        ]

        lines.extend(
            f"* {rule}"
            for rule in section.rules
        )

        section_texts.append(
            "\n".join(
                lines
            )
        )

    return "\n\n".join(
        section_texts
    )
