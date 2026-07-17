from __future__ import annotations

import json
from pathlib import Path

import pytest
from lib.profile.noise import (
    NoiseDictionary,
)
from lib.subtitle.srt import SrtBlock
from lib.translation import hybrid_recovery
from lib.translation.hybrid_group import (
    build_hybrid_translation_group,
)
from lib.translation.hybrid_recovery import (
    build_hybrid_response_schema,
    recover_translation_with_hybrid,
    validate_hybrid_response,
)

OCR_LINE = (
    "=) EWA eam CO ma = Ae lan"
)


@pytest.fixture
def noise_dictionary(
    tmp_path: Path,
) -> NoiseDictionary:
    return NoiseDictionary(
        profile_name="test",
        entries={},
        official_path=(
            tmp_path
            / "noise.json"
        ),
        local_path=(
            tmp_path
            / "noise.local.json"
        ),
        local_loaded=False,
    )


@pytest.fixture
def target_blocks(
) -> list[SrtBlock]:
    return [
        SrtBlock(
            number="281",
            timestamp=(
                "00:17:10,988 --> "
                "00:17:12,865"
            ),
            text=(
                "It's under control,\n"
                "but as a precaution,"
            ),
        ),
        SrtBlock(
            number="282",
            timestamp=(
                "00:17:12,949 --> "
                "00:17:15,201"
            ),
            text=(
                f"{OCR_LINE}\n"
                "to their quarters"
            ),
        ),
        SrtBlock(
            number="283",
            timestamp=(
                "00:17:15,284 --> "
                "00:17:18,079"
            ),
            text=(
                "and remain there\n"
                "until further notice.\n"
                "Thank you."
            ),
        ),
    ]


def valid_hybrid_payload(
) -> dict[str, object]:
    segments = {
        "281": (
            "事態は制御下です。"
            "しかし念のため、"
        ),
        "282": (
            "（判読不能）"
            "各自の居住区へ戻り、"
        ),
        "283": (
            "追って通知があるまで"
            "そこに留まってください。"
            "以上です。"
        ),
    }

    return {
        "group": {
            "full_translation": (
                segments["281"]
                + segments["282"]
                + segments["283"]
            ),
            "segments": segments,
        },
    }


def test_hybrid_schema_requires_exact_ids(
    target_blocks: list[SrtBlock],
) -> None:
    group = build_hybrid_translation_group(
        target_blocks,
        {
            "282",
        },
    )

    assert group is not None

    schema = build_hybrid_response_schema(
        group
    )

    segments_schema = (
        schema[
            "properties"
        ][
            "group"
        ][
            "properties"
        ][
            "segments"
        ]
    )

    assert segments_schema[
               "required"
           ] == [
               "281",
               "282",
               "283",
           ]

    assert (
        segments_schema[
            "additionalProperties"
        ]
        is False
    )


def test_validate_hybrid_response_accepts_valid_result(
    target_blocks: list[SrtBlock],
) -> None:
    group = build_hybrid_translation_group(
        target_blocks,
        {
            "282",
        },
    )

    assert group is not None

    response = json.dumps(
        valid_hybrid_payload(),
        ensure_ascii=False,
    )

    validation = validate_hybrid_response(
        response,
        group,
        {
            "282": [
                OCR_LINE,
            ],
        },
    )

    assert validation.valid is True
    assert validation.reasons == ()

    assert validation.segments[
               "282"
           ] == (
               "（判読不能）"
               "各自の居住区へ戻り、"
           )


def test_validate_hybrid_response_rejects_id_mismatch(
    target_blocks: list[SrtBlock],
) -> None:
    group = build_hybrid_translation_group(
        target_blocks,
        {
            "282",
        },
    )

    assert group is not None

    payload = valid_hybrid_payload()

    del payload[
        "group"
    ][
        "segments"
    ][
        "282"
    ]

    response = json.dumps(
        payload,
        ensure_ascii=False,
    )

    validation = validate_hybrid_response(
        response,
        group,
        {
            "282": [
                OCR_LINE,
            ],
        },
    )

    assert validation.valid is False

    assert any(
        reason.startswith(
            "Invalid Hybrid segment IDs:"
        )
        for reason in validation.reasons
    )


def test_validate_hybrid_response_rejects_full_text_mismatch(
    target_blocks: list[SrtBlock],
) -> None:
    group = build_hybrid_translation_group(
        target_blocks,
        {
            "282",
        },
    )

    assert group is not None

    payload = valid_hybrid_payload()

    payload[
        "group"
    ][
        "full_translation"
    ] = "別の全文訳"

    response = json.dumps(
        payload,
        ensure_ascii=False,
    )

    validation = validate_hybrid_response(
        response,
        group,
        {
            "282": [
                OCR_LINE,
            ],
        },
    )

    assert validation.valid is False

    assert (
        "Hybrid full translation mismatch: "
        "segments do not reconstruct "
        "full_translation"
        in validation.reasons
    )


def test_validate_hybrid_response_rejects_ocr_source_copy(
    target_blocks: list[SrtBlock],
) -> None:
    group = build_hybrid_translation_group(
        target_blocks,
        {
            "282",
        },
    )

    assert group is not None

    payload = valid_hybrid_payload()

    payload[
        "group"
    ][
        "segments"
    ][
        "282"
    ] = (
        f"{OCR_LINE}／"
        "各自の居住区へ戻り"
    )

    payload[
        "group"
    ][
        "full_translation"
    ] = "".join(
        payload[
            "group"
        ][
            "segments"
        ].values()
    )

    response = json.dumps(
        payload,
        ensure_ascii=False,
    )

    validation = validate_hybrid_response(
        response,
        group,
        {
            "282": [
                OCR_LINE,
            ],
        },
    )

    assert validation.valid is False

    assert any(
        reason.startswith(
            "Hybrid segment contains OCR source:"
        )
        for reason in validation.reasons
    )


def test_hybrid_recovery_replaces_group_only(
    monkeypatch: pytest.MonkeyPatch,
    target_blocks: list[SrtBlock],
    noise_dictionary: NoiseDictionary,
) -> None:
    response = json.dumps(
        valid_hybrid_payload(),
        ensure_ascii=False,
    )

    monkeypatch.setattr(
        hybrid_recovery,
        "generate",
        lambda *args, **kwargs: response,
    )

    previous_texts = [
        "制御されていますが、",
        (
            f"[1]{OCR_LINE}[/1]／"
            "to their quarters"
        ),
        "追って通知があるまで。",
    ]

    result = recover_translation_with_hybrid(
        target_blocks,
        previous_texts,
        [
            (
                "Level 1 translation tag requires "
                "Japanese translation in the same "
                "subtitle: subtitle_id='282'"
            ),
            (
                "Untranslated English sentence "
                "detected: subtitle_id='282', "
                "text='to their quarters'"
            ),
        ],
        "test-model",
        noise_dictionary=noise_dictionary,
        glossary_entries={},
    )

    assert result is not None

    assert result == [
        (
            "事態は制御下です。"
            "しかし念のため、"
        ),
        (
            "（判読不能）"
            "各自の居住区へ戻り、"
        ),
        (
            "追って通知があるまで"
            "そこに留まってください。"
            "以上です。"
        ),
    ]
