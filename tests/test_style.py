import json
from pathlib import Path

import lib.profile.config as config_module
import pytest
from lib.profile.style import (
    build_style_prompt,
    load_style_document,
    parse_style_document,
    read_style_document,
)


def write_style(
    path: Path,
    payload,
) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def valid_payload() -> dict:
    return {
        "version": 1,
        "description": "Test style",
        "sections": [
            {
                "name": "基本方針",
                "rules": [
                    "自然な日本語にする",
                    "簡潔にする",
                ],
            },
            {
                "name": "口調",
                "rules": [
                    "人物関係を反映する",
                ],
            },
        ],
    }


def test_read_style_document_preserves_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "style.json"

    write_style(
        path,
        valid_payload(),
    )

    document = read_style_document(
        path
    )

    assert document.version == 1
    assert document.description == (
        "Test style"
    )

    assert [
               section.name
               for section in document.sections
           ] == [
               "基本方針",
               "口調",
           ]

    assert document.sections[0].rules == (
        "自然な日本語にする",
        "簡潔にする",
    )


def test_parse_style_document_allows_missing_description(
) -> None:
    payload = valid_payload()
    del payload["description"]

    document = parse_style_document(
        payload,
        path=Path("style.json"),
    )

    assert document.description is None


@pytest.mark.parametrize(
    "description",
    [
        None,
        123,
        True,
        [],
        {},
    ],
)
def test_parse_style_document_rejects_invalid_description(
    description,
) -> None:
    payload = valid_payload()
    payload["description"] = description

    with pytest.raises(
        RuntimeError,
        match="Invalid style description",
    ):
        parse_style_document(
            payload,
            path=Path("style.json"),
        )


@pytest.mark.parametrize(
    "description",
    [
        "",
        "   ",
    ],
)
def test_parse_style_document_rejects_empty_description(
    description: str,
) -> None:
    payload = valid_payload()
    payload["description"] = description

    with pytest.raises(
        RuntimeError,
        match="Empty style description",
    ):
        parse_style_document(
            payload,
            path=Path("style.json"),
        )


def test_parse_style_document_trims_values(
) -> None:
    payload = {
        "version": 1,
        "sections": [
            {
                "name": "  基本方針  ",
                "rules": [
                    "  自然な日本語にする  ",
                ],
            },
        ],
    }

    document = parse_style_document(
        payload,
        path=Path("style.json"),
    )

    assert document.sections[0].name == (
        "基本方針"
    )

    assert document.sections[0].rules == (
        "自然な日本語にする",
    )


@pytest.mark.parametrize(
    "version",
    [
        "1",
        1.0,
        True,
        None,
    ],
)
def test_parse_style_document_rejects_non_integer_version(
    version,
) -> None:
    payload = valid_payload()
    payload["version"] = version

    with pytest.raises(
        RuntimeError,
        match="Invalid style version type",
    ):
        parse_style_document(
            payload,
            path=Path("style.json"),
        )


def test_parse_style_document_rejects_unsupported_version(
) -> None:
    payload = valid_payload()
    payload["version"] = 2

    with pytest.raises(
        RuntimeError,
        match="Unsupported style version",
    ):
        parse_style_document(
            payload,
            path=Path("style.json"),
        )


@pytest.mark.parametrize(
    "missing_key",
    [
        "version",
        "sections",
    ],
)
def test_parse_style_document_rejects_missing_root_key(
    missing_key: str,
) -> None:
    payload = valid_payload()
    del payload[missing_key]

    with pytest.raises(
        RuntimeError,
        match="Missing style root keys",
    ):
        parse_style_document(
            payload,
            path=Path("style.json"),
        )


def test_parse_style_document_rejects_unknown_root_key(
) -> None:
    payload = valid_payload()
    payload["unknown"] = True

    with pytest.raises(
        RuntimeError,
        match="Unknown style root keys",
    ):
        parse_style_document(
            payload,
            path=Path("style.json"),
        )


@pytest.mark.parametrize(
    "sections",
    [
        {},
        "invalid",
        1,
        None,
    ],
)
def test_parse_style_document_rejects_non_array_sections(
    sections,
) -> None:
    payload = valid_payload()
    payload["sections"] = sections

    with pytest.raises(
        RuntimeError,
        match="Invalid style sections",
    ):
        parse_style_document(
            payload,
            path=Path("style.json"),
        )


def test_parse_style_document_rejects_empty_sections(
) -> None:
    payload = valid_payload()
    payload["sections"] = []

    with pytest.raises(
        RuntimeError,
        match="No style sections",
    ):
        parse_style_document(
            payload,
            path=Path("style.json"),
        )


@pytest.mark.parametrize(
    "section",
    [
        [],
        "invalid",
        1,
        None,
    ],
)
def test_parse_style_document_rejects_non_object_section(
    section,
) -> None:
    payload = {
        "version": 1,
        "sections": [
            section,
        ],
    }

    with pytest.raises(
        RuntimeError,
        match="Invalid style section",
    ):
        parse_style_document(
            payload,
            path=Path("style.json"),
        )


@pytest.mark.parametrize(
    "missing_key",
    [
        "name",
        "rules",
    ],
)
def test_parse_style_document_rejects_missing_section_key(
    missing_key: str,
) -> None:
    payload = valid_payload()
    del payload["sections"][0][missing_key]

    with pytest.raises(
        RuntimeError,
        match="Missing style section keys",
    ):
        parse_style_document(
            payload,
            path=Path("style.json"),
        )


def test_parse_style_document_rejects_unknown_section_key(
) -> None:
    payload = valid_payload()
    payload["sections"][0]["unknown"] = True

    with pytest.raises(
        RuntimeError,
        match="Unknown style section keys",
    ):
        parse_style_document(
            payload,
            path=Path("style.json"),
        )


@pytest.mark.parametrize(
    "name",
    [
        123,
        True,
        [],
        {},
        None,
    ],
)
def test_parse_style_document_rejects_invalid_section_name(
    name,
) -> None:
    payload = valid_payload()
    payload["sections"][0]["name"] = name

    with pytest.raises(
        RuntimeError,
        match="Invalid style section name",
    ):
        parse_style_document(
            payload,
            path=Path("style.json"),
        )


@pytest.mark.parametrize(
    "name",
    [
        "",
        "   ",
    ],
)
def test_parse_style_document_rejects_empty_section_name(
    name: str,
) -> None:
    payload = valid_payload()
    payload["sections"][0]["name"] = name

    with pytest.raises(
        RuntimeError,
        match="Empty style section name",
    ):
        parse_style_document(
            payload,
            path=Path("style.json"),
        )


@pytest.mark.parametrize(
    "rules",
    [
        {},
        "invalid",
        1,
        None,
    ],
)
def test_parse_style_document_rejects_non_array_rules(
    rules,
) -> None:
    payload = valid_payload()
    payload["sections"][0]["rules"] = rules

    with pytest.raises(
        RuntimeError,
        match="Invalid style rules",
    ):
        parse_style_document(
            payload,
            path=Path("style.json"),
        )


def test_parse_style_document_rejects_empty_rules(
) -> None:
    payload = valid_payload()
    payload["sections"][0]["rules"] = []

    with pytest.raises(
        RuntimeError,
        match="Empty style rules",
    ):
        parse_style_document(
            payload,
            path=Path("style.json"),
        )


@pytest.mark.parametrize(
    "rule",
    [
        123,
        True,
        [],
        {},
        None,
    ],
)
def test_parse_style_document_rejects_invalid_rule(
    rule,
) -> None:
    payload = valid_payload()
    payload["sections"][0]["rules"] = [
        rule,
    ]

    with pytest.raises(
        RuntimeError,
        match="Invalid style rule",
    ):
        parse_style_document(
            payload,
            path=Path("style.json"),
        )


@pytest.mark.parametrize(
    "rule",
    [
        "",
        "   ",
    ],
)
def test_parse_style_document_rejects_empty_rule(
    rule: str,
) -> None:
    payload = valid_payload()
    payload["sections"][0]["rules"] = [
        rule,
    ]

    with pytest.raises(
        RuntimeError,
        match="Empty style rule",
    ):
        parse_style_document(
            payload,
            path=Path("style.json"),
        )


def test_parse_style_document_rejects_case_insensitive_duplicate_section(
) -> None:
    payload = {
        "version": 1,
        "sections": [
            {
                "name": "Basic",
                "rules": [
                    "Rule 1",
                ],
            },
            {
                "name": "basic",
                "rules": [
                    "Rule 2",
                ],
            },
        ],
    }

    with pytest.raises(
        RuntimeError,
        match="Duplicate style section name",
    ):
        parse_style_document(
            payload,
            path=Path("style.json"),
        )


def test_read_style_document_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "style.json"

    path.write_text(
        "{invalid",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="Invalid style JSON",
    ):
        read_style_document(
            path
        )


@pytest.mark.parametrize(
    "payload",
    [
        [],
        "invalid",
        1,
        None,
    ],
)
def test_read_style_document_rejects_non_object_root(
    tmp_path: Path,
    payload,
) -> None:
    path = tmp_path / "style.json"

    write_style(
        path,
        payload,
    )

    with pytest.raises(
        RuntimeError,
        match="Invalid style root",
    ):
        read_style_document(
            path
        )


def test_default_style_json_integration(
) -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "default"
        / "style.json"
    )

    document = read_style_document(
        path
    )

    assert document.version == 1

    assert [
               section.name
               for section in document.sections
           ] == [
               "共通字幕スタイル",
               "長さ",
               "会話",
               "効果音",
               "話者ラベル",
           ]


def test_profile_style_prompt_integration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = (
        tmp_path
        / "config"
    )

    profile_dir = (
        config_dir
        / "test-profile"
    )

    profile_dir.mkdir(
        parents=True,
    )

    (
        config_dir
        / "prompt.txt"
    ).write_text(
        (
            "{target_count}\n"
            "{glossary}\n"
            "{style}\n"
            "{request_json}\n"
        ),
        encoding="utf-8",
    )

    (
        profile_dir
        / "glossary.json"
    ).write_text(
        (
            '{\n'
            '  "version": 1,\n'
            '  "entries": [\n'
            '    {\n'
            '      "source": "Gate",\n'
            '      "target": "ゲート"\n'
            '    }\n'
            '  ]\n'
            '}\n'
        ),
        encoding="utf-8",
    )

    (
        profile_dir
        / "style.json"
    ).write_text(
        (
            '{\n'
            '  "version": 1,\n'
            '  "description": "Test profile",\n'
            '  "sections": [\n'
            '    {\n'
            '      "name": "基本方針",\n'
            '      "rules": [\n'
            '        "自然な日本語にする",\n'
            '        "簡潔にする"\n'
            '      ]\n'
            '    },\n'
            '    {\n'
            '      "name": "口調",\n'
            '      "rules": [\n'
            '        "人物関係を反映する"\n'
            '      ]\n'
            '    }\n'
            '  ]\n'
            '}\n'
        ),
        encoding="utf-8",
    )

    (
        profile_dir
        / "noise.json"
    ).write_text(
        (
            '{\n'
            '  "version": 1,\n'
            '  "entries": []\n'
            '}\n'
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        config_module,
        "CONFIG_DIR",
        config_dir,
    )

    document = load_style_document(
        "test-profile"
    )

    assert [
               section.name
               for section in document.sections
           ] == [
               "基本方針",
               "口調",
           ]

    assert build_style_prompt(
        "test-profile"
    ) == (
               "【基本方針】\n"
               "\n"
               "* 自然な日本語にする\n"
               "* 簡潔にする\n"
               "\n"
               "【口調】\n"
               "\n"
               "* 人物関係を反映する"
           )
