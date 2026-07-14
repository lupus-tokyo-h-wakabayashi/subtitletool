import pytest
from lib.profile.noise import (
    NoiseEntry,
)
from lib.profile.noise import (
    find_confirmed_noise_sequences,
    find_suspicious_latin_sequences,
    is_valid_noise_candidate,
    normalize_noise_candidate,
)
from lib.subtitle.srt import SrtBlock
from lib.subtitle.text import (
    detect_simplified_chinese,
    mask_chinese_ocr_text,
)
from lib.translation.translation_chunk import (
    find_noise_candidate_ids,
)
from lib.translation.translation_validation import (
    validate_translation_response,
)
from .helpers import build_test_noise_dictionary


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


def test_find_confirmed_noise_sequences() -> None:
    noise_dictionary = (
        build_test_noise_dictionary(
            [
                NoiseEntry(
                    source="eRe Are",
                    replacement="（判読不能）",
                    action="mask",
                    status="confirmed",
                ),
            ]
        )
    )

    assert find_confirmed_noise_sequences(
        "Before eRe   Are after",
        noise_dictionary,
    ) == [
               "eRe   Are",
           ]


def test_find_confirmed_noise_sequences_ignores_candidate() -> None:
    noise_dictionary = (
        build_test_noise_dictionary(
            [
                NoiseEntry(
                    source="eRe Are",
                    replacement="（判読不能）",
                    action="mask",
                    status="candidate",
                ),
            ]
        )
    )

    assert find_confirmed_noise_sequences(
        "Before eRe Are after",
        noise_dictionary,
    ) == []


def test_find_suspicious_latin_sequences_combines_dictionary_and_heuristic() -> None:
    noise_dictionary = (
        build_test_noise_dictionary(
            [
                NoiseEntry(
                    source="eRe Are",
                    replacement="（判読不能）",
                    action="mask",
                    status="confirmed",
                ),
            ]
        )
    )

    assert find_suspicious_latin_sequences(
        "eRe Are and AbCdEfGhIj",
        noise_dictionary,
    ) == [
               "eRe Are",
               "AbCdEfGhIj",
           ]


def test_noise_dictionary_replaces_legacy_pattern_detection() -> None:
    noise_dictionary = build_test_noise_dictionary(
        [
            NoiseEntry(
                source="CTL EA rare",
                replacement="（判読不能）",
                action="mask",
                status="confirmed",
            ),
        ]
    )

    assert find_suspicious_latin_sequences(
        "Before ctl   ea   RARE after",
        noise_dictionary,
    ) == [
               "ctl   ea   RARE",
           ]


def test_find_noise_candidate_ids_uses_noise_dictionary() -> None:
    noise_dictionary = (
        build_test_noise_dictionary(
            [
                NoiseEntry(
                    source="eRe Are",
                    replacement="（判読不能）",
                    action="mask",
                    status="confirmed",
                ),
            ]
        )
    )

    blocks = [
        SrtBlock(
            number="1",
            timestamp=(
                "00:00:00,000 --> "
                "00:00:01,000"
            ),
            text="Normal subtitle",
        ),
        SrtBlock(
            number="2",
            timestamp=(
                "00:00:01,000 --> "
                "00:00:02,000"
            ),
            text="Before eRe   Are after",
        ),
    ]

    assert find_noise_candidate_ids(
        blocks,
        noise_dictionary,
    ) == [
               "2",
           ]


@pytest.mark.parametrize(
    "text",
    [
        "最後の機会だと思いました。",
        "ラッシュ博士に会いたいです。",
        "国際評議会代表として。",
        "私にチャンスを与えてください。",
        "この件に関与させます。",
        "（判読不能）",
        "第九のシェブロンアドレスへの接続",
        "接続を継続します。",
    ],
)
def test_detect_simplified_chinese_accepts_japanese_text(
    text: str,
) -> None:
    result = detect_simplified_chinese(
        text
    )

    assert not result.detected
    assert result.characters == ()


@pytest.mark.parametrize(
    (
            "source_character",
            "expected_detected",
    ),
    [
        (
                "这",
                True,
        ),
        (
                "们",
                True,
        ),
        (
                "会",
                False,
        ),
        (
                "与",
                False,
        ),
        (
                "関",
                False,
        ),
        (
                "読",
                False,
        ),
        (
                "続",
                False,
        ),
    ],
)
def test_detect_simplified_chinese_character_boundary(
    source_character: str,
    expected_detected: bool,
) -> None:
    result = detect_simplified_chinese(
        source_character
    )

    assert (
        result.detected
        is expected_detected
    )


def test_detect_simplified_chinese_finds_mixed_text(
) -> None:
    result = detect_simplified_chinese(
        "兵曹、这些人を落ち着かせてくれ。"
    )

    assert result.detected
    assert "这" in result.characters


def test_detect_simplified_chinese_finds_chinese_text(
) -> None:
    result = detect_simplified_chinese(
        "我们已经准备好了。"
    )

    assert result.detected
    assert "们" in result.characters


def test_validation_accepts_japanese_opencc_variants(
) -> None:
    noise_dictionary = (
        build_test_noise_dictionary(
            []
        )
    )

    response = """
{
  "translations": [
    {
      "id": "1",
      "translation": "最後の機会だと思いました。"
    },
    {
      "id": "2",
      "translation": "私にチャンスを与えてください。"
    }
  ]
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "1",
            "2",
        ],
        noise_dictionary=noise_dictionary,
    )

    assert result.valid
    assert result.reasons == []


def test_mask_chinese_ocr_text_keeps_japanese(
) -> None:
    source = (
        "最後の機会です。"
        "接続を継続します。"
    )

    result = mask_chinese_ocr_text(
        source
    )

    assert result == source


def test_mask_chinese_ocr_text_masks_simplified_characters(
) -> None:
    result = mask_chinese_ocr_text(
        "兵曹、这些人を落ち着かせてくれ。"
    )

    assert result == (
        "兵曹、（OCR判読不能）"
        "を落ち着かせてくれ。"
    )


def test_validation_rejects_simplified_chinese_with_opencc(
) -> None:
    noise_dictionary = (
        build_test_noise_dictionary(
            []
        )
    )

    response = """
{
  "translations": [
    {
      "id": "1",
      "translation": "兵曹、这些人を落ち着かせてくれ。"
    }
  ]
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "1",
        ],
        noise_dictionary=noise_dictionary,
    )

    assert not result.valid

    assert any(
        (
            "Chinese-specific characters detected:"
            in reason
        )
        for reason in result.reasons
    )
