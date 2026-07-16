from lib.subtitle.srt import (
    SrtBlock,
)
from lib.translation.translation_chunk import (
    generate_translation_response,
    normalize_translation_text,
)


def test_normalize_translation_text() -> None:
    result = normalize_translation_text(
        "  スターゲイトです。  "
    )

    assert isinstance(
        result,
        str,
    )
    assert result
    assert (
        result.strip()
        == result
    )


def test_generate_translation_response_passes_schema(
    monkeypatch,
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

    response_schema = {
        "type": "object",
        "properties": {
            "targets": {
                "type": "object",
            },
        },
    }

    captured: dict[str, object] = {}

    def fake_build_translation_response_schema(
        blocks: list[SrtBlock],
    ) -> dict[str, object]:
        assert blocks is target_blocks

        return response_schema

    def fake_generate(
        prompt: str,
        *,
        model: str,
        response_format: dict[str, object],
    ) -> str:
        captured["prompt"] = prompt
        captured["model"] = model
        captured[
            "response_format"
        ] = response_format

        return '{"targets": {}}'

    monkeypatch.setattr(
        (
            "lib.translation.translation_chunk."
            "build_translation_response_schema"
        ),
        fake_build_translation_response_schema,
    )

    monkeypatch.setattr(
        (
            "lib.translation.translation_chunk."
            "generate"
        ),
        fake_generate,
    )

    result = generate_translation_response(
        "TRANSLATION PROMPT",
        "qwen3:14b",
        target_blocks,
    )

    assert result == '{"targets": {}}'

    assert captured == {
        "prompt": "TRANSLATION PROMPT",
        "model": "qwen3:14b",
        "response_format": response_schema,
    }
