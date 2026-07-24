from pathlib import Path

from lib.subtitle.srt import (
    SrtBlock,
)
from lib.translation import translation_chunk
from lib.translation.translation_artifacts import (
    TranslationArtifactRegistry,
)
from lib.translation.translation_chunk import (
    generate_translation_response,
    normalize_translation_text,
    save_failed_translation_response,
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


def test_save_failed_translation_response_registers_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    debug_directory = (
        tmp_path
        / "tmp"
    )

    monkeypatch.setattr(
        translation_chunk,
        "TRANSLATION_DEBUG_DIR",
        debug_directory,
    )

    registry = TranslationArtifactRegistry(
        root_directory=debug_directory
    )

    saved_path = save_failed_translation_response(
        '{"targets": {}}',
        chunk_start=1,
        chunk_end=10,
        attempt=2,
        artifact_registry=registry,
    )

    assert saved_path.exists()

    assert saved_path.read_text(
        encoding="utf-8"
    ) == '{"targets": {}}'

    assert saved_path.parent == (
        debug_directory
    )

    assert saved_path.name.startswith(
        "failed-translation-"
        "1-10-attempt-2-"
    )

    assert registry.files == (
        saved_path.resolve(),
    )


def test_save_failed_translation_response_without_registry(
    monkeypatch,
    tmp_path: Path,
) -> None:
    debug_directory = (
        tmp_path
        / "tmp"
    )

    monkeypatch.setattr(
        translation_chunk,
        "TRANSLATION_DEBUG_DIR",
        debug_directory,
    )

    saved_path = save_failed_translation_response(
        "invalid response",
        chunk_start=11,
        chunk_end=20,
        attempt=1,
    )

    assert saved_path.exists()

    assert saved_path.read_text(
        encoding="utf-8"
    ) == "invalid response"
