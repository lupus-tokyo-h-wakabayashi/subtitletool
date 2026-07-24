from pathlib import Path

import pytest
from lib.profile.config import ProfileConfig
from lib.subtitle.srt import SrtBlock
from lib.translation import translate
from lib.translation.translate import (
    filter_empty_source_blocks,
    resolve_requested_profile,
    translate_srt,
)
from lib.translation.translation_artifacts import (
    TranslationArtifactRegistry,
)
from .helpers import (
    build_test_noise_dictionary,
)


def test_resolve_requested_profile_with_profile() -> None:
    result = resolve_requested_profile(
        profile_name="stargate",
        style_name=None,
        glossary_name=None,
    )

    assert result == "stargate"


def test_resolve_requested_profile_with_legacy_options() -> None:
    result = resolve_requested_profile(
        profile_name=None,
        style_name="stargate",
        glossary_name="stargate",
    )

    assert result == "stargate"


def test_resolve_requested_profile_rejects_legacy_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Style and glossary profiles "
            "must match"
        ),
    ):
        resolve_requested_profile(
            profile_name=None,
            style_name="stargate",
            glossary_name="default",
        )


def test_resolve_requested_profile_rejects_conflict() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Profile conflicts with "
            "legacy options"
        ),
    ):
        resolve_requested_profile(
            profile_name="default",
            style_name="stargate",
            glossary_name="stargate",
        )


def test_filter_empty_source_blocks_removes_empty_text() -> None:
    blocks = [
        SrtBlock(
            number="216",
            timestamp=(
                "00:10:00,000 --> "
                "00:10:02,000"
            ),
            text="I need to talk to you.",
        ),
        SrtBlock(
            number="217",
            timestamp=(
                "00:10:02,000 --> "
                "00:10:04,000"
            ),
            text="",
        ),
        SrtBlock(
            number="218",
            timestamp=(
                "00:10:04,000 --> "
                "00:10:06,000"
            ),
            text="Alone. It's important.",
        ),
    ]

    (
        translation_blocks,
        skipped_blocks,
    ) = filter_empty_source_blocks(
        blocks
    )

    assert [
               block.number
               for block in translation_blocks
           ] == [
               "216",
               "218",
           ]

    assert [
               block.number
               for block in skipped_blocks
           ] == [
               "217",
           ]


def test_filter_empty_source_blocks_removes_whitespace_only_text() -> None:
    blocks = [
        SrtBlock(
            number="1",
            timestamp=(
                "00:00:01,000 --> "
                "00:00:02,000"
            ),
            text=" \n\t ",
        ),
    ]

    (
        translation_blocks,
        skipped_blocks,
    ) = filter_empty_source_blocks(
        blocks
    )

    assert translation_blocks == []

    assert [
               block.number
               for block in skipped_blocks
           ] == [
               "1",
           ]


def test_filter_empty_source_blocks_preserves_order() -> None:
    blocks = [
        SrtBlock(
            number="10",
            timestamp=(
                "00:00:10,000 --> "
                "00:00:11,000"
            ),
            text="First",
        ),
        SrtBlock(
            number="11",
            timestamp=(
                "00:00:11,000 --> "
                "00:00:12,000"
            ),
            text="",
        ),
        SrtBlock(
            number="12",
            timestamp=(
                "00:00:12,000 --> "
                "00:00:13,000"
            ),
            text="Second",
        ),
    ]

    (
        translation_blocks,
        skipped_blocks,
    ) = filter_empty_source_blocks(
        blocks
    )

    assert [
               block.number
               for block in translation_blocks
           ] == [
               "10",
               "12",
           ]

    assert [
               block.number
               for block in skipped_blocks
           ] == [
               "11",
           ]


def test_filter_empty_source_blocks_removes_speaker_only_text() -> None:
    blocks = [
        SrtBlock(
            number="15",
            timestamp=(
                "00:00:15,000 --> "
                "00:00:16,000"
            ),
            text="Previous subtitle.",
        ),
        SrtBlock(
            number="16",
            timestamp=(
                "00:00:16,000 --> "
                "00:00:17,000"
            ),
            text="VARRO:",
        ),
        SrtBlock(
            number="17",
            timestamp=(
                "00:00:17,000 --> "
                "00:00:18,000"
            ),
            text="VARRO: Actual dialogue.",
        ),
    ]

    (
        translation_blocks,
        skipped_blocks,
    ) = filter_empty_source_blocks(
        blocks
    )

    assert [
               block.number
               for block in translation_blocks
           ] == [
               "15",
               "17",
           ]

    assert [
               block.number
               for block in skipped_blocks
           ] == [
               "16",
           ]


def test_translate_srt_passes_artifact_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = (
        tmp_path
        / "input.eng.srt"
    )

    output_path = (
        tmp_path
        / "output.ja.srt"
    )

    input_path.write_text(
        (
            "1\n"
            "00:00:01,000 --> "
            "00:00:02,000\n"
            "First subtitle.\n"
        ),
        encoding="utf-8",
    )

    profile_directory = (
        tmp_path
        / "profile"
    )

    profile_config = ProfileConfig(
        requested_profile="test",
        resolved_profile="test",
        profile_dir=profile_directory,
        prompt_path=(
            profile_directory
            / "prompt.txt"
        ),
        glossary_path=(
            profile_directory
            / "glossary.json"
        ),
        style_path=(
            profile_directory
            / "style.json"
        ),
        noise_path=(
            profile_directory
            / "noise.json"
        ),
        noise_local_path=(
            profile_directory
            / "noise.local.json"
        ),
        fallback_used=False,
    )

    noise_dictionary = (
        build_test_noise_dictionary(
            []
        )
    )

    registry = TranslationArtifactRegistry(
        root_directory=tmp_path
    )

    received_registries: list[
        TranslationArtifactRegistry | None
        ] = []

    monkeypatch.setattr(
        translate,
        "resolve_profile_config",
        lambda requested_profile: (
            profile_config
        ),
    )

    monkeypatch.setattr(
        translate,
        "load_noise_dictionary",
        lambda config: noise_dictionary,
    )

    monkeypatch.setattr(
        translate,
        "load_resume_blocks",
        lambda source_blocks, path: [],
    )

    monkeypatch.setattr(
        translate,
        "build_translation_artifact_registry",
        lambda: registry,
    )

    def fake_run_translation_session(
        **kwargs: object,
    ) -> None:
        received_registries.append(
            kwargs.get(
                "artifact_registry"
            )
        )

        return None

    monkeypatch.setattr(
        translate,
        "run_translation_session",
        fake_run_translation_session,
    )

    result = translate_srt(
        input_path,
        output_path,
        profile_name="test",
    )

    assert result == output_path.resolve()

    assert received_registries == [
        registry,
    ]
