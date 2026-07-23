import json
from copy import deepcopy

import lib.infrastructure.ollama as ollama_module
from lib.infrastructure.ollama import (
    OllamaGenerateExchange,
    build_generate_payload,
    generate,
    generate_exchange,
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


class FakeOllamaResponse:
    def __init__(
        self,
        body: str,
    ) -> None:
        self.body = body

    def __enter__(
        self,
    ) -> "FakeOllamaResponse":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        return None

    def read(
        self,
    ) -> bytes:
        return self.body.encode("utf-8")


def test_generate_exchange_returns_request_and_raw_response(
    monkeypatch,
) -> None:
    raw_response_body = (
        '{"model":"qwen3:14b",'
        '"response":"  翻訳結果  ",'
        '"done":true}'
    )
    captured_requests: list[object] = []

    monkeypatch.setattr(
        ollama_module,
        "ensure_available",
        lambda: None,
    )
    monkeypatch.setattr(
        ollama_module,
        "get_ollama_base_url",
        lambda: "http://127.0.0.1:11434",
    )

    def fake_urlopen(
        request: object,
        timeout: int,
    ) -> FakeOllamaResponse:
        captured_requests.append(
            request
        )

        assert timeout == 123

        return FakeOllamaResponse(
            raw_response_body
        )

    monkeypatch.setattr(
        ollama_module.urllib.request,
        "urlopen",
        fake_urlopen,
    )

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

    exchange = generate_exchange(
        prompt="Translate target 459.",
        model="qwen3:14b",
        temperature=0.1,
        top_p=0.8,
        timeout=123,
        response_format=response_format,
    )

    assert exchange.request_payload == {
        "model": "qwen3:14b",
        "prompt": "Translate target 459.",
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.8,
        },
        "format": response_format,
    }
    assert (
        exchange.raw_response_body
        == raw_response_body
    )
    assert exchange.response_body == {
        "model": "qwen3:14b",
        "response": "  翻訳結果  ",
        "done": True,
    }
    assert exchange.generated_text == "翻訳結果"
    assert len(captured_requests) == 1

    request = captured_requests[0]

    assert (
        request.full_url
        == "http://127.0.0.1:11434/api/generate"
    )
    assert request.method == "POST"
    assert json.loads(
        request.data.decode("utf-8")
    ) == exchange.request_payload


def test_generate_keeps_string_return_value(
    monkeypatch,
) -> None:
    expected_exchange = OllamaGenerateExchange(
        request_payload={
            "model": "qwen3:14b",
        },
        raw_response_body=(
            '{"response":"翻訳結果"}'
        ),
        response_body={
            "response": "翻訳結果",
        },
        generated_text="翻訳結果",
    )
    captured_arguments: dict[str, object] = {}

    def fake_generate_exchange(
        prompt: str,
        model: str,
        temperature: float,
        top_p: float,
        timeout: int,
        response_format: dict[str, object] | None,
    ) -> OllamaGenerateExchange:
        captured_arguments.update(
            {
                "prompt": prompt,
                "model": model,
                "temperature": temperature,
                "top_p": top_p,
                "timeout": timeout,
                "response_format": response_format,
            }
        )

        return expected_exchange

    monkeypatch.setattr(
        ollama_module,
        "generate_exchange",
        fake_generate_exchange,
    )

    result = generate(
        prompt="Translate target 459.",
        model="qwen3.5:9b",
        temperature=0.3,
        top_p=0.7,
        timeout=456,
        response_format={
            "type": "object",
        },
    )

    assert result == "翻訳結果"
    assert isinstance(
        result,
        str,
    )
    assert captured_arguments == {
        "prompt": "Translate target 459.",
        "model": "qwen3.5:9b",
        "temperature": 0.3,
        "top_p": 0.7,
        "timeout": 456,
        "response_format": {
            "type": "object",
        },
    }
