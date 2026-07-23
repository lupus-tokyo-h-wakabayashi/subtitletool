import json
from pathlib import Path

from lib.profile.charactor import (
    build_charactor_prompt,
    extract_speaker_names,
    load_charactors,
    write_charactors,
)
from lib.profile.config import ProfileConfig
from lib.subtitle.srt import SrtBlock


def make_block(number: str, text: str) -> SrtBlock:
    return SrtBlock(
        number=number,
        timestamp="00:00:01,000 --> 00:00:02,000",
        text=text,
    )


def make_profile_config(
    profile_dir: Path,
) -> ProfileConfig:
    return ProfileConfig(
        requested_profile="test",
        resolved_profile="test",
        profile_dir=profile_dir,
        prompt_path=profile_dir / "prompt.txt",
        glossary_path=profile_dir / "glossary.json",
        style_path=profile_dir / "style.json",
        noise_path=profile_dir / "noise.json",
        noise_local_path=(
            profile_dir / "noise.local.json"
        ),
        fallback_used=False,
    )


def test_extract_speaker_names_keeps_first_spelling(
) -> None:
    blocks = [
        make_block("1", "Daniel: Open the gate."),
        make_block("2", "DANIEL: Now."),
        make_block("3", "[O'NEILL] Move."),
        make_block("4", "No explicit speaker."),
    ]

    assert extract_speaker_names(blocks) == [
        "Daniel",
        "O'NEILL",
    ]


def test_write_charactors_creates_empty_descriptions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "charactor.json"

    write_charactors(
        path,
        ["Daniel", "O'NEILL"],
    )

    assert json.loads(
        path.read_text(encoding="utf-8")
    ) == [
        {
            "charactor": "Daniel",
            "description": "",
        },
        {
            "charactor": "O'NEILL",
            "description": "",
        },
    ]


def test_write_charactors_preserves_existing_description(
    tmp_path: Path,
) -> None:
    path = tmp_path / "charactor.json"
    path.write_text(
        json.dumps(
            [
                {
                    "charactor": "DANIEL",
                    "description": "Archaeologist",
                }
            ]
        ),
        encoding="utf-8",
    )

    write_charactors(
        path,
        ["Daniel", "SCOTT"],
    )

    result = load_charactors(path)

    assert [
        (item.charactor, item.description)
        for item in result
    ] == [
        ("Daniel", "Archaeologist"),
        ("SCOTT", ""),
    ]


def test_build_charactor_prompt_is_optional(
    tmp_path: Path,
) -> None:
    config = make_profile_config(tmp_path)

    assert build_charactor_prompt(config) == ""


def test_build_charactor_prompt_includes_description(
    tmp_path: Path,
) -> None:
    path = tmp_path / "charactor.json"
    path.write_text(
        json.dumps(
            [
                {
                    "charactor": "Daniel",
                    "description": "男性の考古学者。",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    prompt = build_charactor_prompt(
        make_profile_config(tmp_path)
    )

    assert "【話者の人物設定】" in prompt
    assert '\"charactor\": \"Daniel\"' in prompt
    assert '\"description\": \"男性の考古学者。\"' in prompt
