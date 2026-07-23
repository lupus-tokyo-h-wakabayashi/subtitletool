from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lib.subtitle.srt import (
    SrtBlock,
    parse_speaker_from_text,
)

from .config import ProfileConfig

CHARACTOR_FILE_NAME = "charactor.json"


@dataclass(frozen=True)
class Charactor:
    charactor: str
    description: str


def charactor_path(
    profile_config: ProfileConfig,
) -> Path:
    return (
        profile_config.profile_dir
        / CHARACTOR_FILE_NAME
    )


def _parse_charactor(
    value: Any,
    *,
    index: int,
    path: Path,
) -> Charactor:
    if not isinstance(value, dict):
        raise RuntimeError(
            "Invalid charactor entry: "
            f"path={path}, index={index}, "
            "expected=object"
        )

    expected_keys = {
        "charactor",
        "description",
    }

    if set(value) != expected_keys:
        raise RuntimeError(
            "Invalid charactor entry keys: "
            f"path={path}, index={index}, "
            f"expected={sorted(expected_keys)}, "
            f"actual={sorted(value)}"
        )

    name = value["charactor"]
    description = value["description"]

    if not isinstance(name, str) or not name.strip():
        raise RuntimeError(
            "Invalid charactor name: "
            f"path={path}, index={index}, "
            "expected=non-empty string"
        )

    if not isinstance(description, str):
        raise RuntimeError(
            "Invalid charactor description: "
            f"path={path}, index={index}, "
            "expected=string"
        )

    return Charactor(
        charactor=name.strip(),
        description=description.strip(),
    )


def load_charactors(
    path: Path,
) -> tuple[Charactor, ...]:
    try:
        raw_text = path.read_text(
            encoding="utf-8",
            errors="strict",
        )

        if not raw_text.strip():
            return ()

        payload = json.loads(
            raw_text
        )
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Invalid charactor JSON: "
            f"path={path}, line={error.lineno}, "
            f"column={error.colno}, "
            f"message={error.msg}"
        ) from error
    except OSError as error:
        raise RuntimeError(
            "Failed to read charactor JSON: "
            f"path={path}, error={error}"
        ) from error

    if not isinstance(payload, list):
        raise RuntimeError(
            "Invalid charactor JSON root: "
            f"path={path}, expected=array"
        )

    result: list[Charactor] = []
    seen: set[str] = set()

    for index, value in enumerate(payload):
        item = _parse_charactor(
            value,
            index=index,
            path=path,
        )
        identity = item.charactor.casefold()

        if identity in seen:
            raise RuntimeError(
                "Duplicate charactor name: "
                f"path={path}, "
                f"charactor={item.charactor!r}"
            )

        seen.add(identity)
        result.append(item)

    return tuple(result)


def extract_speaker_names(
    blocks: list[SrtBlock],
) -> list[str]:
    speakers: list[str] = []
    seen: set[str] = set()

    for block in blocks:
        speaker = parse_speaker_from_text(
            block.text
        ).speaker

        if speaker is None:
            continue

        identity = speaker.casefold()

        if identity in seen:
            continue

        seen.add(identity)
        speakers.append(speaker)

    return speakers


def write_charactors(
    path: Path,
    speaker_names: list[str],
) -> Path:
    existing_descriptions: dict[str, str] = {}

    if path.is_file():
        existing_descriptions = {
            item.charactor.casefold(): (
                item.description
            )
            for item in load_charactors(path)
        }

    payload = [
        {
            "charactor": speaker,
            "description": (
                existing_descriptions.get(
                    speaker.casefold(),
                    "",
                )
            ),
        }
        for speaker in speaker_names
    ]

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    return path


def build_charactor_prompt(
    profile_config: ProfileConfig,
    speaker_names: list[str] | None = None,
) -> str:
    path = charactor_path(
        profile_config
    )

    if not path.is_file():
        return ""

    charactors = load_charactors(path)

    if speaker_names is not None:
        requested_speakers = {
            speaker.casefold()
            for speaker in speaker_names
        }
        charactors = tuple(
            item
            for item in charactors
            if item.charactor.casefold()
            in requested_speakers
        )

    if not charactors:
        return ""

    payload = [
        {
            "charactor": item.charactor,
            "description": item.description,
        }
        for item in charactors
    ]

    return (
        "\n\n【話者の人物設定】\n\n"
        "source.speakerとcharactorが大文字・小文字を除いて一致する場合は、"
        "descriptionの人物像を口調へ反映すること。\n"
        "原文の意味や感情より人物設定を優先してはならない。\n\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )
