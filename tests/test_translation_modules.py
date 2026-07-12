import pytest

from lib.noise import (
    is_valid_noise_candidate,
    normalize_noise_candidate,
)
from lib.srt import SrtBlock
from lib.translate import (
    resolve_requested_profile,
)
from lib.translation_chunk import (
    normalize_translation_text,
)
from lib.translation_prompt import (
    build_ocr_noise_instruction,
    build_request_item,
)
from lib.translation_resume import (
    load_resume_blocks,
)


def test_build_request_item_parses_speaker() -> None:
    block = SrtBlock(
        number="1",
        timestamp=(
            "00:00:01,000 --> "
            "00:00:03,000"
        ),
        text=(
            "DANIEL: "
            "This is the Stargate."
        ),
    )

    result = build_request_item(
        block
    )

    assert result == {
        "id": "1",
        "speaker": "DANIEL",
        "text": "This is the Stargate.",
    }


def test_build_ocr_noise_instruction() -> None:
    blocks = [
        SrtBlock(
            number="1",
            timestamp=(
                "00:00:01,000 --> "
                "00:00:03,000"
            ),
            text=(
                "Our guests have arrived. "
                "CTL EA rare"
            ),
        ),
    ]

    result = (
        build_ocr_noise_instruction(
            blocks
        )
    )

    assert "対象ID: 1" in result
    assert "OCR破損" in result
    assert "（判読不能）" in result


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


def test_load_resume_blocks_without_output(
    tmp_path,
) -> None:
    source_blocks = [
        SrtBlock(
            number="1",
            timestamp=(
                "00:00:01,000 --> "
                "00:00:03,000"
            ),
            text="Test",
        ),
    ]

    output_path = (
        tmp_path
        / "not-found.ja.srt"
    )

    result = load_resume_blocks(
        source_blocks,
        output_path,
    )

    assert result == []


def test_load_resume_blocks_with_valid_output(
    tmp_path,
) -> None:
    source_blocks = [
        SrtBlock(
            number="1",
            timestamp=(
                "00:00:01,000 --> "
                "00:00:03,000"
            ),
            text="Test",
        ),
        SrtBlock(
            number="2",
            timestamp=(
                "00:00:04,000 --> "
                "00:00:06,000"
            ),
            text="Next",
        ),
    ]

    output_path = (
        tmp_path
        / "resume.ja.srt"
    )

    output_path.write_text(
        "\n".join(
            [
                "1",
                (
                    "00:00:01,000 --> "
                    "00:00:03,000"
                ),
                "テスト",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = load_resume_blocks(
        source_blocks,
        output_path,
    )

    assert len(result) == 1
    assert result[0].number == "1"
    assert result[0].timestamp == (
        "00:00:01,000 --> "
        "00:00:03,000"
    )


def test_load_resume_blocks_rejects_invalid_number(
    tmp_path,
) -> None:
    source_blocks = [
        SrtBlock(
            number="1",
            timestamp=(
                "00:00:01,000 --> "
                "00:00:03,000"
            ),
            text="Test",
        ),
    ]

    output_path = (
        tmp_path
        / "invalid.ja.srt"
    )

    output_path.write_text(
        "\n".join(
            [
                "9",
                (
                    "00:00:01,000 --> "
                    "00:00:03,000"
                ),
                "テスト",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "subtitle number mismatch"
        ),
    ):
        load_resume_blocks(
            source_blocks,
            output_path,
        )


def test_normalize_noise_candidate() -> None:
    source = (
        "  VViat=\n"
        "lancom   Rom  "
    )

    assert normalize_noise_candidate(
        source
    ) == "VViat= lancom Rom"


def test_is_valid_noise_candidate_accepts_ocr_noise() -> None:
    assert is_valid_noise_candidate(
        "VViat= lancom Rom (ele) .qi ale nce]"
    )


def test_is_valid_noise_candidate_rejects_invalid_values() -> None:
    cases = [
        "",
        "  ",
        "mm",
        "[0)",
        "12345",
        "Stargate",
        "FTL",
    ]

    for source in cases:
        assert not is_valid_noise_candidate(
            source
        )
