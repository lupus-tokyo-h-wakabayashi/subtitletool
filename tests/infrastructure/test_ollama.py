from copy import deepcopy

from lib.infrastructure.ollama import (
    build_generate_payload,
)


def test_build_generate_payload_without_response_format(
) -> None:
    payload = build_generate_payload(
        prompt="Translate this subtitle.",
        model="qwen3:14b",
        temperature=0.2,
        top_p=0.9,
        response_format=None,
    )

    assert payload == {
        "model": "qwen3:14b",
        "prompt": "Translate this subtitle.",
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
        },
    }

    assert "format" not in payload


def test_build_generate_payload_with_response_format(
) -> None:
    response_format = {
        "type": "object",
        "properties": {
            "targets": {
                "type": "object",
            },
        },
        "required": [
            "targets",
        ],
        "additionalProperties": False,
    }

    payload = build_generate_payload(
        prompt="Translate this subtitle.",
        model="qwen3:14b",
        temperature=0.2,
        top_p=0.9,
        response_format=response_format,
    )

    assert payload == {
        "model": "qwen3:14b",
        "prompt": "Translate this subtitle.",
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
        },
        "format": response_format,
    }

    assert payload["format"] is response_format


def test_build_generate_payload_does_not_modify_response_format(
) -> None:
    response_format = {
        "type": "object",
        "properties": {
            "targets": {
                "type": "object",
                "properties": {
                    "1": {
                        "type": "object",
                    },
                },
                "required": [
                    "1",
                ],
                "additionalProperties": False,
            },
        },
        "required": [
            "targets",
        ],
        "additionalProperties": False,
    }

    original_response_format = deepcopy(
        response_format
    )

    build_generate_payload(
        prompt="Translate this subtitle.",
        model="qwen3:14b",
        temperature=0.2,
        top_p=0.9,
        response_format=response_format,
    )

    assert (
        response_format
        == original_response_format
    )
