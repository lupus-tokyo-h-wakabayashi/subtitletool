#!/usr/bin/env python3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROMPT_DIR = PROJECT_ROOT / "config" / "prompts"
GLOSSARY_DIR = PROJECT_ROOT / "config" / "glossary"
STYLE_DIR = PROJECT_ROOT / "config" / "styles"

DEFAULT_PROMPT_NAME = "translate"
DEFAULT_STYLE_NAME = "default"
DEFAULT_GLOSSARY_NAME = "default"


def resolve_config_file(
    directory: Path,
    name: str,
) -> Path:
    """
    ローカル設定ファイルを優先し、
    存在しなければ example ファイルへフォールバックする。

    例:
        stargate.txt
        stargate.example.txt
    """
    local_path = directory / f"{name}.txt"

    if local_path.exists():
        return local_path

    example_path = directory / f"{name}.example.txt"

    if example_path.exists():
        return example_path

    raise FileNotFoundError(
        f"Config file not found: "
        f"{local_path} or {example_path}"
    )


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
    path = resolve_config_file(
        PROMPT_DIR,
        prompt_name,
    )

    template = read_config_file(path)

    required_placeholders = {
        "{target_count}",
        "{glossary}",
        "{style}",
        "{before_context}",
        "{target_text}",
        "{after_context}",
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
    path = resolve_config_file(
        GLOSSARY_DIR,
        glossary_name,
    )

    return read_config_file(path)


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
    指定した用語集を検証用の辞書として読み込む。
    """
    glossary_text = load_glossary(
        glossary_name
    )

    entries = parse_glossary_entries(
        glossary_text
    )

    if not entries and glossary_name != DEFAULT_GLOSSARY_NAME:
        raise RuntimeError(
            f"No valid glossary entries: "
            f"{glossary_name}"
        )

    return entries

def load_style(
    style_name: str = DEFAULT_STYLE_NAME,
) -> str:
    common_path = resolve_config_file(
        STYLE_DIR,
        "common",
    )

    style_path = resolve_config_file(
        STYLE_DIR,
        style_name,
    )

    common_style = read_config_file(common_path)
    specific_style = read_config_file(style_path)

    return (
        f"{common_style}\n\n"
        f"{specific_style}"
    )


def build_translation_prompt(
    *,
    target_count: int,
    before_context: str,
    target_text: str,
    after_context: str,
    prompt_name: str = DEFAULT_PROMPT_NAME,
    glossary_name: str = DEFAULT_GLOSSARY_NAME,
    style_name: str = DEFAULT_STYLE_NAME,
) -> str:
    template = load_prompt_template(prompt_name)
    glossary = load_glossary(glossary_name)
    style = load_style(style_name)

    return template.format(
        target_count=target_count,
        glossary=glossary,
        style=style,
        before_context=before_context,
        target_text=target_text,
        after_context=after_context,
    )