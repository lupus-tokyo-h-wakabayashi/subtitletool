#!/usr/bin/env python3

import json
from pathlib import Path

from .charactor import (
    build_charactor_prompt,
)
from .config import (
    resolve_profile_config,
)
from .glossary import (
    build_glossary_prompt,
    load_glossary_entries as load_json_glossary_entries,
)
from .style import (
    build_style_prompt,
)

DEFAULT_PROMPT_NAME = "translate"


def extract_request_speaker_names(
    request_json: str,
) -> list[str]:
    """
    翻訳リクエストに明示された話者名だけを、
    前文脈・対象・後文脈の順で抽出する。
    """
    try:
        payload = json.loads(
            request_json
        )
    except json.JSONDecodeError:
        return []

    if not isinstance(payload, dict):
        return []

    speakers: list[str] = []
    seen: set[str] = set()

    def add_speaker(value: object) -> None:
        if not isinstance(value, str):
            return

        speaker = value.strip()

        if not speaker:
            return

        identity = speaker.casefold()

        if identity in seen:
            return

        seen.add(identity)
        speakers.append(speaker)

    def add_context_speakers(
        context_key: str,
    ) -> None:
        context = payload.get(
            context_key
        )

        if not isinstance(context, list):
            return

        for item in context:
            if isinstance(item, dict):
                add_speaker(
                    item.get("speaker")
                )

    add_context_speakers(
        "context_before"
    )

    targets = payload.get(
        "targets"
    )

    if isinstance(targets, dict):
        for item in targets.values():
            if not isinstance(item, dict):
                continue

            source = item.get(
                "source"
            )

            if isinstance(source, dict):
                add_speaker(
                    source.get("speaker")
                )

    add_context_speakers(
        "context_after"
    )

    return speakers


def read_config_file(path: Path) -> str:
    text = path.read_text(
        encoding="utf-8",
        errors="strict",
    ).strip()

    if not text:
        raise RuntimeError(
            f"Config file is empty: {path}"
        )

    return text


def load_prompt_template(
    prompt_name: str = DEFAULT_PROMPT_NAME,
) -> str:
    """
    全profile共通の翻訳プロンプトを読み込む。

    prompt_nameは既存呼び出しとの互換性のため残す。
    現在使用するプロンプトはconfig/prompt.txtのみ。
    """
    if prompt_name != DEFAULT_PROMPT_NAME:
        raise ValueError(
            "Unsupported prompt name: "
            f"{prompt_name!r}. "
            f"Expected {DEFAULT_PROMPT_NAME!r}."
        )

    config = resolve_profile_config(
        None
    )

    template = read_config_file(
        config.prompt_path
    )

    required_placeholders = {
        "{target_count}",
        "{glossary}",
        "{style}",
        "{request_json}",
    }

    missing = [
        placeholder
        for placeholder in required_placeholders
        if placeholder not in template
    ]

    if missing:
        raise RuntimeError(
            "Translation prompt is missing placeholders: "
            + ", ".join(sorted(missing))
        )

    return template


def load_glossary(
    profile_name: str | None = None,
) -> str:
    """
    Prompt用Glossary文字列を返す。

    呼出元との互換性のため、
    API名を維持する。
    """
    return build_glossary_prompt(
        profile_name
    )


def load_glossary_entries(
    profile_name: str | None = None,
) -> dict[str, str]:
    """
    Validation用Glossary辞書を返す。

    呼出元との互換性のため、
    API名を維持する。
    """
    return load_json_glossary_entries(
        profile_name
    )


def load_style(
    profile_name: str | None = None,
) -> str:
    """
    Prompt用Style文字列を返す。

    呼出元との互換性のため、
    API名を維持する。
    """
    return build_style_prompt(
        profile_name
    )


def build_translation_prompt(
    *,
    target_count: int,
    request_json: str,
    profile_name: str | None = None,
    prompt_name: str = DEFAULT_PROMPT_NAME,
) -> str:
    template = load_prompt_template(
        prompt_name
    )

    glossary = load_glossary(
        profile_name
    )

    style = load_style(
        profile_name
    )

    profile_config = resolve_profile_config(
        profile_name
    )

    charactor = build_charactor_prompt(
        profile_config,
        speaker_names=(
            extract_request_speaker_names(
                request_json
            )
        ),
    )

    return template.format(
        target_count=target_count,
        glossary=glossary,
        style=style,
        request_json=request_json,
    ) + charactor
