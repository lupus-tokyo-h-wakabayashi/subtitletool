from pathlib import Path

import lib.config as config_module
import pytest


def create_required_profile_files(
    profile_dir: Path,
    *,
    create_glossary_json: bool,
    create_glossary_txt: bool = False,
    create_style_json: bool = True,
    create_style_txt: bool = False,
) -> None:
    profile_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if create_glossary_json:
        (
            profile_dir
            / "glossary.json"
        ).write_text(
            (
                '{\n'
                '  "version": 1,\n'
                '  "entries": []\n'
                '}\n'
            ),
            encoding="utf-8",
        )

    if create_glossary_txt:
        (
            profile_dir
            / "glossary.txt"
        ).write_text(
            "Chevron = シェブロン\n",
            encoding="utf-8",
        )

    if create_style_json:
        (
            profile_dir
            / "style.json"
        ).write_text(
            (
                '{\n'
                '  "version": 1,\n'
                '  "sections": [\n'
                '    {\n'
                '      "name": "Test",\n'
                '      "rules": [\n'
                '        "Test style"\n'
                '      ]\n'
                '    }\n'
                '  ]\n'
                '}\n'
            ),
            encoding="utf-8",
        )

    if create_style_txt:
        (
            profile_dir
            / "style.txt"
        ).write_text(
            "Test style\n",
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


def create_prompt_file(
    config_dir: Path,
) -> None:
    config_dir.mkdir(
        parents=True,
        exist_ok=True,
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


def test_resolve_profile_config_reports_legacy_glossary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = (
        tmp_path
        / "config"
    )

    create_prompt_file(
        config_dir
    )

    profile_dir = (
        config_dir
        / "legacy"
    )

    create_required_profile_files(
        profile_dir,
        create_glossary_json=False,
        create_glossary_txt=True,
    )

    monkeypatch.setattr(
        config_module,
        "CONFIG_DIR",
        config_dir,
    )

    with pytest.raises(
        FileNotFoundError,
        match=(
            "Glossary JSON is missing"
        ),
    ) as error:
        config_module.resolve_profile_config(
            "legacy"
        )

    message = str(
        error.value
    )

    assert (
        "glossary.json"
        in message
    )

    assert (
        "glossary.txt"
        in message
    )

    assert (
        "Convert glossary.txt "
        "to glossary.json"
        in message
    )


def test_resolve_profile_config_reports_missing_glossary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = (
        tmp_path
        / "config"
    )

    create_prompt_file(
        config_dir
    )

    profile_dir = (
        config_dir
        / "missing"
    )

    create_required_profile_files(
        profile_dir,
        create_glossary_json=False,
    )

    monkeypatch.setattr(
        config_module,
        "CONFIG_DIR",
        config_dir,
    )

    with pytest.raises(
        FileNotFoundError,
        match=(
            "Profile configuration is incomplete"
        ),
    ) as error:
        config_module.resolve_profile_config(
            "missing"
        )

    assert (
        "glossary="
        in str(error.value)
    )


def test_resolve_profile_config_uses_glossary_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = (
        tmp_path
        / "config"
    )

    create_prompt_file(
        config_dir
    )

    profile_dir = (
        config_dir
        / "valid"
    )

    create_required_profile_files(
        profile_dir,
        create_glossary_json=True,
        create_glossary_txt=True,
    )

    monkeypatch.setattr(
        config_module,
        "CONFIG_DIR",
        config_dir,
    )

    result = config_module.resolve_profile_config(
        "valid"
    )

    assert result.glossary_path == (
        profile_dir
        / "glossary.json"
    )


def test_resolve_profile_config_reports_legacy_style(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = (
        tmp_path
        / "config"
    )

    create_prompt_file(
        config_dir
    )

    profile_dir = (
        config_dir
        / "legacy-style"
    )

    create_required_profile_files(
        profile_dir,
        create_glossary_json=True,
        create_style_json=False,
        create_style_txt=True,
    )

    monkeypatch.setattr(
        config_module,
        "CONFIG_DIR",
        config_dir,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Style JSON is missing",
    ) as error:
        config_module.resolve_profile_config(
            "legacy-style"
        )

    message = str(
        error.value
    )

    assert (
        "style.json"
        in message
    )

    assert (
        "style.txt"
        in message
    )

    assert (
        "Convert style.txt "
        "to style.json"
        in message
    )


def test_resolve_profile_config_uses_style_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = (
        tmp_path
        / "config"
    )

    create_prompt_file(
        config_dir
    )

    profile_dir = (
        config_dir
        / "valid-style"
    )

    create_required_profile_files(
        profile_dir,
        create_glossary_json=True,
        create_style_json=True,
        create_style_txt=True,
    )

    monkeypatch.setattr(
        config_module,
        "CONFIG_DIR",
        config_dir,
    )

    result = (
        config_module.resolve_profile_config(
            "valid-style"
        )
    )

    assert result.style_path == (
        profile_dir
        / "style.json"
    )
