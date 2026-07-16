import pytest

from lib.subtitle.srt import SrtBlock
from lib.translation.translation_schema import (
    build_translation_response_schema,
    build_translation_target_schema,
)


def test_build_translation_target_schema_with_speaker(
) -> None:
    block = SrtBlock(
        number="11",
        timestamp=(
            "00:00:01,000 --> "
            "00:00:03,000"
        ),
        text=(
            "DANIEL: "
            "This is the Stargate."
        ),
    )

    schema = build_translation_target_schema(
        block
    )

    assert schema == {
        "type": "object",
        "properties": {
            "source": {
                "type": "object",
                "properties": {
                    "speaker": {
                        "const": "DANIEL",
                    },
                    "text": {
                        "const": (
                            "This is the Stargate."
                        ),
                    },
                },
                "required": [
                    "speaker",
                    "text",
                ],
                "additionalProperties": False,
            },
            "translation": {
                "type": "string",
                "minLength": 1,
            },
        },
        "required": [
            "source",
            "translation",
        ],
        "additionalProperties": False,
    }


def test_build_translation_response_schema_preserves_target_order(
) -> None:
    target_blocks = [
        SrtBlock(
            number="11",
            timestamp=(
                "00:00:01,000 --> "
                "00:00:03,000"
            ),
            text=(
                "DANIEL: "
                "This is the Stargate."
            ),
        ),
        SrtBlock(
            number="12",
            timestamp=(
                "00:00:04,000 --> "
                "00:00:06,000"
            ),
            text="That's a pity.",
        ),
    ]

    schema = build_translation_response_schema(
        target_blocks
    )

    assert schema["type"] == "object"

    assert schema["required"] == [
        "targets",
    ]

    assert (
        schema["additionalProperties"]
        is False
    )

    root_properties = schema[
        "properties"
    ]

    assert isinstance(
        root_properties,
        dict,
    )

    targets_schema = root_properties[
        "targets"
    ]

    assert isinstance(
        targets_schema,
        dict,
    )

    assert targets_schema[
               "type"
           ] == "object"

    assert targets_schema[
               "required"
           ] == [
               "11",
               "12",
           ]

    assert (
        targets_schema[
            "additionalProperties"
        ]
        is False
    )

    target_properties = targets_schema[
        "properties"
    ]

    assert isinstance(
        target_properties,
        dict,
    )

    assert list(
        target_properties.keys()
    ) == [
               "11",
               "12",
           ]


def test_build_translation_response_schema_uses_null_speaker_const(
) -> None:
    target_blocks = [
        SrtBlock(
            number="1",
            timestamp=(
                "00:00:01,000 --> "
                "00:00:03,000"
            ),
            text="This is a pen.",
        ),
    ]

    schema = build_translation_response_schema(
        target_blocks
    )

    root_properties = schema[
        "properties"
    ]

    assert isinstance(
        root_properties,
        dict,
    )

    targets_schema = root_properties[
        "targets"
    ]

    assert isinstance(
        targets_schema,
        dict,
    )

    target_properties = targets_schema[
        "properties"
    ]

    assert isinstance(
        target_properties,
        dict,
    )

    target_schema = target_properties[
        "1"
    ]

    assert isinstance(
        target_schema,
        dict,
    )

    item_properties = target_schema[
        "properties"
    ]

    assert isinstance(
        item_properties,
        dict,
    )

    source_schema = item_properties[
        "source"
    ]

    assert isinstance(
        source_schema,
        dict,
    )

    source_properties = source_schema[
        "properties"
    ]

    assert isinstance(
        source_properties,
        dict,
    )

    assert source_properties[
               "speaker"
           ] == {
               "const": None,
           }

    assert source_properties[
               "text"
           ] == {
               "const": "This is a pen.",
           }


def test_build_translation_response_schema_rejects_empty_blocks(
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "requires at least one "
            "target block"
        ),
    ):
        build_translation_response_schema(
            []
        )


def test_build_translation_response_schema_rejects_empty_id(
) -> None:
    target_blocks = [
        SrtBlock(
            number=" ",
            timestamp=(
                "00:00:01,000 --> "
                "00:00:03,000"
            ),
            text="This is a pen.",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="empty subtitle ID",
    ):
        build_translation_response_schema(
            target_blocks
        )


def test_build_translation_response_schema_rejects_duplicate_id(
) -> None:
    target_blocks = [
        SrtBlock(
            number="1",
            timestamp=(
                "00:00:01,000 --> "
                "00:00:03,000"
            ),
            text="First subtitle.",
        ),
        SrtBlock(
            number="1",
            timestamp=(
                "00:00:04,000 --> "
                "00:00:06,000"
            ),
            text="Second subtitle.",
        ),
    ]

    with pytest.raises(
        ValueError,
        match=(
            "duplicate subtitle ID: '1'"
        ),
    ):
        build_translation_response_schema(
            target_blocks
        )
