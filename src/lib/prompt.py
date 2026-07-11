#!/usr/bin/env python3

from pathlib import Path
from lib.config import (
    DEFAULT_PROFILE_NAME,
    resolve_profile_config,
)


DEFAULT_PROMPT_NAME = "translate"
DEFAULT_STYLE_NAME = "default"
DEFAULT_GLOSSARY_NAME = "default"


def read_config_file(path: Path) -> str:
    text = path.read_text(
        encoding="utf-8",
        errors="strict",
    ).strip()

    if not text:
        raise RuntimeError(f"Config file is empty: {path}")

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
    glossary_name: str = DEFAULT_GLOSSARY_NAME,
) -> str:
    """
    指定profileの用語集を読み込む。

    profileが存在しない場合はdefaultへフォールバックする。
    """
    config = resolve_profile_config(
        glossary_name
    )

    return read_config_file(
        config.glossary_path
    )


def parse_glossary_entries(
    glossary_text: str,
) -> dict[str, str]:
    """
    次の形式の用語集を辞書へ変換する。

    English term = 日本語訳

    空行と # から始まるコメントは無視する。
    """
    entries: dict[str, str] = {}

    for raw_line in glossary_text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        if "=" not in line:
            continue

        source_term, translated_term = line.split(
            "=",
            maxsplit=1,
        )

        source_term = source_term.strip()
        translated_term = translated_term.strip()

        if not source_term or not translated_term:
            continue

        entries[source_term] = translated_term

    return entries


def load_glossary_entries(
    glossary_name: str = DEFAULT_GLOSSARY_NAME,
) -> dict[str, str]:
    """
    指定profileの用語集を検証用辞書として読み込む。

    default profileの用語集は空でも許可する。
    それ以外のprofileは有効な用語が1件以上必要。
    """
    config = resolve_profile_config(
        glossary_name
    )

    glossary_text = read_config_file(
        config.glossary_path
    )

    entries = parse_glossary_entries(
        glossary_text
    )

    if (
        not entries
        and config.resolved_profile
        != DEFAULT_PROFILE_NAME
    ):
        raise RuntimeError(
            "No valid glossary entries: "
            f"profile={config.resolved_profile!r}, "
            f"path={config.glossary_path}"
        )

    return entries


def load_style(
    style_name: str = DEFAULT_STYLE_NAME,
) -> str:
    """
    指定profileの字幕スタイルを読み込む。

    defaultと指定profileの連結は行わない。
    """
    config = resolve_profile_config(
        style_name
    )

    return read_config_file(
        config.style_path
    )


def build_translation_prompt(
    *,
    target_count: int,
    request_json: str,
    prompt_name: str = DEFAULT_PROMPT_NAME,
    glossary_name: str = DEFAULT_GLOSSARY_NAME,
    style_name: str = DEFAULT_STYLE_NAME,
) -> str:
    template = load_prompt_template(
        prompt_name
    )

    glossary = load_glossary(
        glossary_name
    )

    style = load_style(
        style_name
    )

    return template.format(
        target_count=target_count,
        glossary=glossary,
        style=style,
        request_json=request_json,
    )
