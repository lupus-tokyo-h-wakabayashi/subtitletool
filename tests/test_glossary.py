import json
from pathlib import Path

import pytest
from lib.glossary import (
    build_glossary_prompt,
    load_glossary_entries,
    parse_glossary_document,
    read_glossary_document,
)


def write_glossary(
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
        "description": "Test glossary",
        "entries": [
            {
                "source": "Chevron",
                "target": "シェブロン",
            },
            {
                "source": "P4X351",
                "target": "P4X-351",
            },
        ],
    }


def test_read_glossary_document_preserves_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "glossary.json"

    write_glossary(
        path,
        valid_payload(),
    )

    document = read_glossary_document(
        path,
        allow_empty=False,
    )

    assert document.version == 1
    assert document.description == (
        "Test glossary"
    )

    assert [
               entry.source
               for entry in document.entries
           ] == [
               "Chevron",
               "P4X351",
           ]


def test_parse_glossary_document_allows_missing_description(
) -> None:
    payload = valid_payload()
    del payload["description"]

    document = parse_glossary_document(
        payload,
        path=Path("glossary.json"),
        allow_empty=False,
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
def test_parse_glossary_document_rejects_invalid_description(
    description,
) -> None:
    payload = valid_payload()
    payload["description"] = description

    with pytest.raises(
        RuntimeError,
        match="Invalid glossary description",
    ):
        parse_glossary_document(
            payload,
            path=Path("glossary.json"),
            allow_empty=False,
        )


@pytest.mark.parametrize(
    "description",
    [
        "",
        "   ",
    ],
)
def test_parse_glossary_document_rejects_empty_description(
    description: str,
) -> None:
    payload = valid_payload()
    payload["description"] = description

    with pytest.raises(
        RuntimeError,
        match="Empty glossary description",
    ):
        parse_glossary_document(
            payload,
            path=Path("glossary.json"),
            allow_empty=False,
        )


def test_parse_glossary_document_trims_values(
) -> None:
    payload = {
        "version": 1,
        "entries": [
            {
                "source": "  Chevron  ",
                "target": "  シェブロン  ",
            },
        ],
    }

    document = parse_glossary_document(
        payload,
        path=Path("glossary.json"),
        allow_empty=False,
    )

    assert document.entries[0].source == (
        "Chevron"
    )
    assert document.entries[0].target == (
        "シェブロン"
    )


def test_parse_glossary_document_allows_empty_default(
) -> None:
    document = parse_glossary_document(
        {
            "version": 1,
            "entries": [],
        },
        path=Path("glossary.json"),
        allow_empty=True,
    )

    assert document.entries == ()


def test_parse_glossary_document_rejects_empty_non_default(
) -> None:
    with pytest.raises(
        RuntimeError,
        match="No valid glossary entries",
    ):
        parse_glossary_document(
            {
                "version": 1,
                "entries": [],
            },
            path=Path("glossary.json"),
            allow_empty=False,
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
def test_parse_glossary_document_rejects_non_integer_version(
    version,
) -> None:
    payload = valid_payload()
    payload["version"] = version

    with pytest.raises(
        RuntimeError,
        match="Invalid glossary version type",
    ):
        parse_glossary_document(
            payload,
            path=Path("glossary.json"),
            allow_empty=False,
        )


def test_parse_glossary_document_rejects_unsupported_version(
) -> None:
    payload = valid_payload()
    payload["version"] = 2

    with pytest.raises(
        RuntimeError,
        match="Unsupported glossary version",
    ):
        parse_glossary_document(
            payload,
            path=Path("glossary.json"),
            allow_empty=False,
        )


def test_parse_glossary_document_rejects_unknown_root_key(
) -> None:
    payload = valid_payload()
    payload["unknown"] = True

    with pytest.raises(
        RuntimeError,
        match="Unknown glossary root keys",
    ):
        parse_glossary_document(
            payload,
            path=Path("glossary.json"),
            allow_empty=False,
        )


def test_parse_glossary_document_rejects_unknown_entry_key(
) -> None:
    payload = valid_payload()
    payload["entries"][0]["unknown"] = True

    with pytest.raises(
        RuntimeError,
        match="Unknown glossary entry keys",
    ):
        parse_glossary_document(
            payload,
            path=Path("glossary.json"),
            allow_empty=False,
        )


@pytest.mark.parametrize(
    "missing_key",
    [
        "version",
        "entries",
    ],
)
def test_parse_glossary_document_rejects_missing_root_key(
    missing_key: str,
) -> None:
    payload = valid_payload()
    del payload[missing_key]

    with pytest.raises(
        RuntimeError,
        match="Missing glossary root keys",
    ):
        parse_glossary_document(
            payload,
            path=Path("glossary.json"),
            allow_empty=False,
        )


@pytest.mark.parametrize(
    "missing_key",
    [
        "source",
        "target",
    ],
)
def test_parse_glossary_document_rejects_missing_entry_key(
    missing_key: str,
) -> None:
    payload = valid_payload()
    del payload["entries"][0][missing_key]

    with pytest.raises(
        RuntimeError,
        match="Missing glossary entry keys",
    ):
        parse_glossary_document(
            payload,
            path=Path("glossary.json"),
            allow_empty=False,
        )


@pytest.mark.parametrize(
    (
            "field_name",
            "field_value",
            "error_message",
    ),
    [
        (
                "source",
                123,
                "Invalid glossary source",
        ),
        (
                "target",
                123,
                "Invalid glossary target",
        ),
        (
                "source",
                "   ",
                "Empty glossary source",
        ),
        (
                "target",
                "   ",
                "Empty glossary target",
        ),
    ],
)
def test_parse_glossary_document_rejects_invalid_entry_value(
    field_name: str,
    field_value,
    error_message: str,
) -> None:
    payload = valid_payload()
    payload["entries"][0][
        field_name
    ] = field_value

    with pytest.raises(
        RuntimeError,
        match=error_message,
    ):
        parse_glossary_document(
            payload,
            path=Path("glossary.json"),
            allow_empty=False,
        )


def test_parse_glossary_document_rejects_case_insensitive_duplicate(
) -> None:
    payload = {
        "version": 1,
        "entries": [
            {
                "source": "Chevron",
                "target": "シェブロン",
            },
            {
                "source": "chevron",
                "target": "別訳",
            },
        ],
    }

    with pytest.raises(
        RuntimeError,
        match="Duplicate glossary source",
    ):
        parse_glossary_document(
            payload,
            path=Path("glossary.json"),
            allow_empty=False,
        )


def test_read_glossary_document_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "glossary.json"

    path.write_text(
        "{invalid",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="Invalid glossary JSON",
    ):
        read_glossary_document(
            path,
            allow_empty=False,
        )


def test_default_glossary_integration(
) -> None:
    assert load_glossary_entries(
        None
    ) == {}

    assert build_glossary_prompt(
        None
    ) == ""


@pytest.mark.parametrize(
    "payload",
    [
        [],
        "invalid",
        1,
        None,
    ],
)
def test_read_glossary_document_rejects_non_object_root(
    tmp_path: Path,
    payload,
) -> None:
    path = tmp_path / "glossary.json"

    write_glossary(
        path,
        payload,
    )

    with pytest.raises(
        RuntimeError,
        match="Invalid glossary root",
    ):
        read_glossary_document(
            path,
            allow_empty=False,
        )


@pytest.mark.parametrize(
    "entries",
    [
        {},
        "invalid",
        1,
        None,
    ],
)
def test_parse_glossary_document_rejects_non_array_entries(
    entries,
) -> None:
    payload = valid_payload()
    payload["entries"] = entries

    with pytest.raises(
        RuntimeError,
        match="Invalid glossary entries",
    ):
        parse_glossary_document(
            payload,
            path=Path("glossary.json"),
            allow_empty=False,
        )


@pytest.mark.parametrize(
    "entry",
    [
        [],
        "invalid",
        1,
        None,
    ],
)
def test_parse_glossary_document_rejects_non_object_entry(
    entry,
) -> None:
    payload = {
        "version": 1,
        "entries": [
            entry,
        ],
    }

    with pytest.raises(
        RuntimeError,
        match="Invalid glossary entry",
    ):
        parse_glossary_document(
            payload,
            path=Path("glossary.json"),
            allow_empty=False,
        )
