from lib.profile.noise import (
    NoiseEntry,
)
from lib.translation.translation_validation import (
    normalize_source_text_with_glossary,
    validate_translation_response,
)

from .helpers import build_test_noise_dictionary


def test_validation_uses_confirmed_noise_dictionary() -> None:
    noise_dictionary = build_test_noise_dictionary(
        [
            NoiseEntry(
                source="eRe Are",
                replacement="（判読不能）",
                action="mask",
                status="confirmed",
            ),
        ]
    )

    response = """
{
  "translations": [
    {
      "id": "1",
      "translation": "これは eRe   Are です。"
    }
  ]
}
"""

    result = validate_translation_response(
        response,
        expected_ids=["1"],
        noise_dictionary=noise_dictionary,
    )

    assert not result.valid

    assert any(
        "Garbled Latin text detected"
        in reason
        for reason in result.reasons
    )


def test_validation_accepts_level_5_source_copy(
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
      "translation": "惑星[5]P4X-351[/5]は不安定です。"
    }
  ]
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "1",
        ],
        source_texts=[
            (
                "The planet P4X-351 "
                "is unstable."
            ),
        ],
        noise_dictionary=noise_dictionary,
    )

    assert result.valid
    assert result.reasons == []

    assert result.translated_texts == [
        "惑星P4X-351は不安定です。",
    ]


def test_validation_rejects_modified_level_5_value(
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
      "translation": "惑星[5]P4X351[/5]は不安定です。"
    }
  ]
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "1",
        ],
        source_texts=[
            (
                "The planet P4X-351 "
                "is unstable."
            ),
        ],
        noise_dictionary=noise_dictionary,
    )

    assert not result.valid

    assert any(
        (
            "Translation evaluation tag value "
            "not found in source"
        )
        in reason
        for reason in result.reasons
    )


def test_validation_rejects_invalid_evaluation_tag_structure(
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
      "translation": "惑星[5]P4X-351[/3]は不安定です。"
    }
  ]
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "1",
        ],
        source_texts=[
            (
                "The planet P4X-351 "
                "is unstable."
            ),
        ],
        noise_dictionary=noise_dictionary,
    )

    assert not result.valid

    assert any(
        "Invalid translation evaluation tag"
        in reason
        for reason in result.reasons
    )


def test_validation_rejects_evaluation_tag_without_source_text(
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
      "translation": "惑星[5]P4X-351[/5]です。"
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
            "Translation evaluation tags require "
            "source text"
        )
        in reason
        for reason in result.reasons
    )


def test_validation_processes_level_3_with_normal_validation(
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
      "translation": "船名は[3]Destiny[/3]です。"
    }
  ]
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "1",
        ],
        source_texts=[
            "The ship is Destiny.",
        ],
        noise_dictionary=noise_dictionary,
    )

    assert result.valid

    assert result.translated_texts == [
        "船名はDestinyです。",
    ]


def test_validation_rejects_untranslated_level_3_sentence(
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
      "translation": "[3]the connection to the gate[/3]"
    }
  ]
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "1",
        ],
        source_texts=[
            "the connection to the gate",
        ],
        noise_dictionary=noise_dictionary,
    )

    assert not result.valid

    assert any(
        "Untranslated English sentence detected"
        in reason
        for reason in result.reasons
    )


def test_validation_applies_glossary_after_level_5_tag_removal(
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
      "translation": "[5]SGC[/5]からの命令です。"
    }
  ]
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "1",
        ],
        source_texts=[
            "This is an order from SGC.",
        ],
        glossary_entries={
            "SGC": "スターゲイト司令部",
        },
        noise_dictionary=noise_dictionary,
    )

    assert not result.valid

    assert any(
        "Glossary violation"
        in reason
        for reason in result.reasons
    )


def test_validation_masks_level_1_ocr_noise(
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
      "id": "773",
      "translation": "[1]sie lexer=s-4-9 10) WV am nat (el=)[/1]\\n第九のシェブロンアドレスへの接続"
    }
  ]
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "773",
        ],
        source_texts=[
            (
                "sie lexer=s-4-9 10) "
                "WV am nat (el=)\n"
                "the connection to the\n"
                "ninth chevron address."
            ),
        ],
        noise_dictionary=noise_dictionary,
    )

    assert result.valid
    assert result.reasons == []

    assert result.translated_texts == [
        (
            "（判読不能）\n"
            "第九のシェブロンアドレスへの接続"
        ),
    ]

    assert result.noise_candidates == [
        "sie lexer=s-4-9 10) WV am nat (el=)",
    ]


def test_validation_rejects_level_1_without_japanese_translation(
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
      "translation": "[1]garbled OCR text[/1]"
    }
  ]
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "1",
        ],
        source_texts=[
            "garbled OCR text",
        ],
        noise_dictionary=noise_dictionary,
    )

    assert not result.valid
    assert result.noise_candidates == []

    assert any(
        "requires Japanese translation"
        in reason
        for reason in result.reasons
    )


def test_validation_rejects_partial_level_1_source_line(
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
      "translation": "[1]sie lexer=s-4-9[/1]\\n接続を確立します。"
    }
  ]
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "1",
        ],
        source_texts=[
            (
                "sie lexer=s-4-9 10) "
                "WV am nat (el=)\n"
                "establishing connection"
            ),
        ],
        noise_dictionary=noise_dictionary,
    )

    assert not result.valid
    assert result.noise_candidates == []

    assert any(
        "must match a complete source line"
        in reason
        for reason in result.reasons
    )


def test_validation_records_failed_id_for_invalid_tag(
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
      "id": "7",
      "translation": "[5]P4X-351[/3]"
    }
  ]
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "7",
        ],
        source_texts=[
            "P4X-351",
        ],
        noise_dictionary=noise_dictionary,
    )

    assert not result.valid

    assert result.failed_ids == {
        "7",
    }


def test_normalize_source_text_with_glossary(
) -> None:
    source = (
        "The core\n"
        "of the planet P4X351\n"
        "had become unstable,"
    )

    result = (
        normalize_source_text_with_glossary(
            source,
            {
                "P4X351": "P4X-351",
            },
        )
    )

    assert result == (
        "The core\n"
        "of the planet P4X-351\n"
        "had become unstable,"
    )


def test_validation_accepts_level_5_glossary_normalized_value(
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
      "id": "779",
      "translation": "惑星[5]P4X-351[/5]のコアが不安定になっていた。"
    }
  ]
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "779",
        ],
        source_texts=[
            (
                "The core\n"
                "of the planet P4X351\n"
                "had become unstable,"
            ),
        ],
        glossary_entries={
            "P4X351": "P4X-351",
        },
        noise_dictionary=noise_dictionary,
    )

    assert result.valid
    assert result.reasons == []

    assert result.translated_texts == [
        (
            "惑星P4X-351のコアが"
            "不安定になっていた。"
        ),
    ]


def test_validation_rejects_level_5_value_without_glossary_normalization(
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
      "id": "779",
      "translation": "惑星[5]P4X-351[/5]のコアが不安定になっていた。"
    }
  ]
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "779",
        ],
        source_texts=[
            (
                "The core\n"
                "of the planet P4X351\n"
                "had become unstable,"
            ),
        ],
        noise_dictionary=noise_dictionary,
    )

    assert not result.valid

    assert any(
        (
            "Translation evaluation tag value "
            "not found in source"
        )
        in reason
        for reason in result.reasons
    )


def test_validation_does_not_normalize_level_1_with_glossary(
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
      "translation": "[1]P4X-351[/1]\\n識別子を確認しました。"
    }
  ]
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "1",
        ],
        source_texts=[
            (
                "P4X351\n"
                "identifier confirmed"
            ),
        ],
        glossary_entries={
            "P4X351": "P4X-351",
        },
        noise_dictionary=noise_dictionary,
    )

    assert not result.valid

    assert result.noise_candidates == []

    assert any(
        (
            "Translation evaluation tag value "
            "not found in source"
        )
        in reason
        for reason in result.reasons
    )
