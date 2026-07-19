from __future__ import annotations

import json

import pytest
from lib.profile.glossary import (
    GlossaryEntries,
    GlossaryEntry,
)
from lib.profile.noise import (
    NoiseEntry,
)
from lib.translation.translation_tags import (
    process_translation_tags,
)
from lib.translation.translation_validation import (
    find_chinese_specific_characters,
    find_glossary_violations,
    find_repeated_sequence_subtitle_ids,
    find_repeated_translation_subtitle_ids,
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
  "targets": {
    "1": {
      "source": {
        "speaker": null,
        "text": "This is eRe Are."
      },
      "translation": "これは eRe   Are です。"
    }
  }
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "1",
        ],
        source_speakers=[
            None,
        ],
        source_texts=[
            "This is eRe Are.",
        ],
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
  "targets": {
    "1": {
      "source": {
        "speaker": null,
        "text": "The planet P4X-351 is unstable."
      },
      "translation": "惑星[5]P4X-351[/5]は不安定です。"
    }
  }
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "1",
        ],
        source_speakers=[
            None,
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


def test_validation_accepts_level_5_context_correction(
) -> None:
    noise_dictionary = (
        build_test_noise_dictionary(
            []
        )
    )

    response = """
{
  "targets": {
    "1021": {
      "source": {
        "speaker": null,
        "text": "it's indicating malfunction.\\nOthers are failing."
      },
      "translation": "スコット、これは[5]SG-1[/5]です。"
    }
  }
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "1021",
        ],
        source_speakers=[
            None,
        ],
        source_texts=[
            (
                "it's indicating malfunction.\n"
                "Others are failing."
            ),
        ],
        noise_dictionary=noise_dictionary,
    )

    assert result.valid
    assert result.reasons == []

    assert result.translated_texts == [
        "スコット、これはSG-1です。",
    ]


def test_validation_rejects_invalid_evaluation_tag_structure(
) -> None:
    noise_dictionary = (
        build_test_noise_dictionary(
            []
        )
    )

    response = """
{
  "targets": {
    "1": {
      "source": {
        "speaker": null,
        "text": "The planet P4X-351 is unstable."
      },
      "translation": "惑星[5]P4X-351[/3]は不安定です。"
    }
  }
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "1",
        ],
        source_speakers=[
            None,
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


def test_validation_accepts_level_5_without_source_text(
) -> None:
    noise_dictionary = (
        build_test_noise_dictionary(
            []
        )
    )

    response = """
{
  "targets": {
    "1": {
      "source": {
        "speaker": null,
        "text": "The planet is P4X-351."
      },
      "translation": "惑星[5]P4X-351[/5]です。"
    }
  }
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "1",
        ],
        noise_dictionary=noise_dictionary,
    )

    assert result.valid
    assert result.reasons == []

    assert result.translated_texts == [
        "惑星P4X-351です。",
    ]


def test_validation_processes_level_3_with_normal_validation(
) -> None:
    noise_dictionary = (
        build_test_noise_dictionary(
            []
        )
    )

    response = """
{
  "targets": {
    "1": {
      "source": {
        "speaker": null,
        "text": "The ship is Destiny."
      },
      "translation": "船名は[3]Destiny[/3]です。"
    }
  }
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "1",
        ],
        source_speakers=[
            None,
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


def test_level_3_source_match_normalizes_whitespace(
) -> None:
    result = process_translation_tags(
        translated_texts=[
            (
                "[3]CIN Maal- male Ws "
                "of the land.[/3]"
            ),
        ],
        subtitle_ids=[
            "316",
        ],
        source_texts=[
            (
                "CIN Maal- male Ws\n"
                "of the land."
            ),
        ],
    )

    assert result.errors == ()

    assert result.translated_texts == (
        "CIN Maal- male Ws of the land.",
    )


def test_level_3_source_match_still_rejects_text_difference(
) -> None:
    result = process_translation_tags(
        translated_texts=[
            (
                "[3]CIN Maal- female Ws "
                "of the land.[/3]"
            ),
        ],
        subtitle_ids=[
            "316",
        ],
        source_texts=[
            (
                "CIN Maal- male Ws\n"
                "of the land."
            ),
        ],
    )

    assert len(
        result.errors
    ) == 1

    assert (
        "Translation evaluation tag value "
        "not found in source"
        in result.errors[0].message
    )

    assert (
        "level=3"
        in result.errors[0].message
    )


def test_validation_rejects_untranslated_level_3_sentence(
) -> None:
    noise_dictionary = (
        build_test_noise_dictionary(
            []
        )
    )

    response = """
{
  "targets": {
    "1": {
      "source": {
        "speaker": null,
        "text": "the connection to the gate"
      },
      "translation": "[3]the connection to the gate[/3]"
    }
  }
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "1",
        ],
        source_speakers=[
            None,
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
  "targets": {
    "1": {
      "source": {
        "speaker": null,
        "text": "This is an order from SGC."
      },
      "translation": "[5]SGC[/5]からの命令です。"
    }
  }
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "1",
        ],
        source_speakers=[
            None,
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
  "targets": {
    "773": {
      "source": {
        "speaker": null,
        "text": "sie lexer=s-4-9 10) WV am nat (el=)\\nthe connection to the\\nninth chevron address."
      },
      "translation": "[1]sie lexer=s-4-9 10) WV am nat (el=)[/1]\\n第九のシェブロンアドレスへの接続"
    }
  }
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "773",
        ],
        source_speakers=[
            None,
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


def test_validation_accepts_level_1_only_ocr_noise(
) -> None:
    noise_dictionary = (
        build_test_noise_dictionary(
            []
        )
    )

    response = """
{
  "targets": {
    "1037": {
      "source": {
        "speaker": null,
        "text": "0) WV"
      },
      "translation": "[1]0) WV[/1]"
    }
  }
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "1037",
        ],
        source_speakers=[
            None,
        ],
        source_texts=[
            "0) WV",
        ],
        noise_dictionary=noise_dictionary,
    )

    assert result.valid
    assert result.reasons == []

    assert result.translated_texts == [
        "（判読不能）",
    ]

    assert result.noise_candidates == [
        "0) WV",
    ]


def test_validation_accepts_level_1_with_short_ocr_residue(
) -> None:
    noise_dictionary = (
        build_test_noise_dictionary(
            []
        )
    )

    response = """
{
  "targets": {
    "1052": {
      "source": {
        "speaker": null,
        "text": "Nec\\n\\" 0) WV"
      },
      "translation": "Nec\\n\\" [1]0) WV[/1]"
    }
  }
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "1052",
        ],
        source_speakers=[
            None,
        ],
        source_texts=[
            (
                "Nec\n"
                "\" 0) WV"
            ),
        ],
        noise_dictionary=noise_dictionary,
    )

    assert result.valid
    assert result.reasons == []

    assert result.translated_texts == [
        "（判読不能）",
    ]

    assert result.noise_candidates == [
        "0) WV",
    ]


def test_validation_rejects_level_1_with_non_japanese_text(
) -> None:
    noise_dictionary = (
        build_test_noise_dictionary(
            []
        )
    )

    response = """
{
  "targets": {
    "1": {
      "source": {
        "speaker": null,
        "text": "OCR noise\\nconnection failed"
      },
      "translation": "[1]OCR noise[/1]\\nconnection failed"
    }
  }
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "1",
        ],
        source_speakers=[
            None,
        ],
        source_texts=[
            (
                "OCR noise\n"
                "connection failed"
            ),
        ],
        noise_dictionary=noise_dictionary,
    )

    assert not result.valid

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
  "targets": {
    "1": {
      "source": {
        "speaker": null,
        "text": "sie lexer=s-4-9 10) WV am nat (el=)\\nestablishing connection"
      },
      "translation": "[1]sie lexer=s-4-9[/1]\\n接続を確立します。"
    }
  }
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "1",
        ],
        source_speakers=[
            None,
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
  "targets": {
    "7": {
      "source": {
        "speaker": null,
        "text": "P4X-351"
      },
      "translation": "[5]P4X-351[/3]"
    }
  }
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "7",
        ],
        source_speakers=[
            None,
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


def test_validation_accepts_level_5_value_without_source_match(
) -> None:
    noise_dictionary = (
        build_test_noise_dictionary(
            []
        )
    )

    response = """
{
  "targets": {
    "779": {
      "source": {
        "speaker": null,
        "text": "The core\\nof the planet P4X351\\nhad become unstable,"
      },
      "translation": "惑星[5]P4X-351[/5]のコアが不安定になっていた。"
    }
  }
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "779",
        ],
        source_speakers=[
            None,
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
        "惑星P4X-351のコアが不安定になっていた。",
    ]


def test_validation_accepts_level_5_value_without_glossary_normalization(
) -> None:
    noise_dictionary = (
        build_test_noise_dictionary(
            []
        )
    )

    response = """
{
  "targets": {
    "779": {
      "source": {
        "speaker": null,
        "text": "The core\\nof the planet P4X351\\nhad become unstable,"
      },
      "translation": "惑星[5]P4X-351[/5]のコアが不安定になっていた。"
    }
  }
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "779",
        ],
        source_speakers=[
            None,
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

    assert result.valid
    assert result.reasons == []

    assert result.translated_texts == [
        (
            "惑星P4X-351のコアが"
            "不安定になっていた。"
        ),
    ]


def test_validation_does_not_normalize_level_1_with_glossary(
) -> None:
    noise_dictionary = (
        build_test_noise_dictionary(
            []
        )
    )

    response = """
{
  "targets": {
    "1": {
      "source": {
        "speaker": null,
        "text": "P4X351\\nidentifier confirmed"
      },
      "translation": "[1]P4X-351[/1]\\n識別子を確認しました。"
    }
  }
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "1",
        ],
        source_speakers=[
            None,
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


def test_validation_uses_retry_source_for_response_validation(
) -> None:
    noise_dictionary = (
        build_test_noise_dictionary(
            []
        )
    )

    response = """
{
  "targets": {
    "1": {
      "source": {
        "speaker": null,
        "text": "（OCR判読不能） establishing connection"
      },
      "translation": "接続を確立します。"
    }
  }
}
"""

    result = validate_translation_response(
        response,
        expected_ids=[
            "1",
        ],
        source_speakers=[
            None,
        ],
        source_texts=[
            (
                "这些人 "
                "establishing connection"
            ),
        ],
        response_source_speakers=[
            None,
        ],
        response_source_texts=[
            (
                "（OCR判読不能） "
                "establishing connection"
            ),
        ],
        noise_dictionary=noise_dictionary,
    )

    assert result.valid
    assert result.reasons == []

    assert result.translated_texts == [
        "接続を確立します。",
    ]


def test_case_sensitive_glossary_matches_exact_case(
) -> None:
    glossary_entries = GlossaryEntries(
        (
            GlossaryEntry(
                source="Destiny",
                target="デスティニー",
                case_sensitive=True,
            ),
        )
    )

    violations = find_glossary_violations(
        source_texts=[
            "We have returned to Destiny.",
        ],
        translated_texts=[
            "デスティニーへ戻った。",
        ],
        subtitle_ids=[
            "1",
        ],
        glossary_entries=glossary_entries,
    )

    assert violations == []


def test_case_sensitive_glossary_rejects_wrong_translation(
) -> None:
    glossary_entries = GlossaryEntries(
        (
            GlossaryEntry(
                source="Destiny",
                target="デスティニー",
                case_sensitive=True,
            ),
        )
    )

    violations = find_glossary_violations(
        source_texts=[
            "We have returned to Destiny.",
        ],
        translated_texts=[
            "船へ戻った。",
        ],
        subtitle_ids=[
            "1",
        ],
        glossary_entries=glossary_entries,
    )

    assert len(violations) == 1

    assert violations[0].startswith(
        "Glossary violation:"
    )


def test_case_sensitive_glossary_ignores_lowercase_common_noun(
) -> None:
    glossary_entries = GlossaryEntries(
        (
            GlossaryEntry(
                source="Destiny",
                target="デスティニー",
                case_sensitive=True,
            ),
        )
    )

    violations = find_glossary_violations(
        source_texts=[
            "Coming here was my destiny.",
        ],
        translated_texts=[
            (
                "ここに来ることこそが、"
                "私の運命でした。"
            ),
        ],
        subtitle_ids=[
            "190",
        ],
        glossary_entries=glossary_entries,
    )

    assert violations == []


@pytest.mark.parametrize(
    "source_text",
    [
        "destinies",
        "predestiny",
        "DESTINY",
    ],
)
def test_case_sensitive_glossary_does_not_partially_match(
    source_text: str,
) -> None:
    glossary_entries = GlossaryEntries(
        (
            GlossaryEntry(
                source="Destiny",
                target="デスティニー",
                case_sensitive=True,
            ),
        )
    )

    violations = find_glossary_violations(
        source_texts=[
            source_text,
        ],
        translated_texts=[
            "任意の日本語",
        ],
        subtitle_ids=[
            "1",
        ],
        glossary_entries=glossary_entries,
    )

    assert violations == []


def test_plain_dictionary_keeps_case_insensitive_behavior(
) -> None:
    violations = find_glossary_violations(
        source_texts=[
            "return to chevron",
        ],
        translated_texts=[
            "帰還する",
        ],
        subtitle_ids=[
            "1",
        ],
        glossary_entries={
            "Chevron": "シェブロン",
        },
    )

    assert len(violations) == 1


def test_find_repeated_translation_subtitle_ids(
) -> None:
    subtitle_ids = [
        str(number)
        for number in range(
            81,
            91,
        )
    ]

    translated_texts = [
        "では、容疑者はいますか？"
        for _ in subtitle_ids
    ]

    actual = (
        find_repeated_translation_subtitle_ids(
            translated_texts,
            subtitle_ids,
            "では、容疑者はいますか？",
        )
    )

    assert actual == subtitle_ids


def test_find_repeated_sequence_subtitle_ids(
) -> None:
    subtitle_ids = [
        str(number)
        for number in range(
            81,
            91,
        )
    ]

    actual = (
        find_repeated_sequence_subtitle_ids(
            subtitle_ids,
            first_start=1,
            second_start=4,
            length=3,
        )
    )

    assert actual == [
        "81",
        "82",
        "83",
        "84",
        "85",
        "86",
    ]


@pytest.mark.parametrize(
    (
            "first_start",
            "second_start",
            "length",
    ),
    [
        (
                0,
                4,
                3,
        ),
        (
                1,
                0,
                3,
        ),
        (
                1,
                4,
                0,
        ),
        (
                9,
                10,
                3,
        ),
    ],
)
def test_find_repeated_sequence_subtitle_ids_rejects_invalid_range(
    first_start: int,
    second_start: int,
    length: int,
) -> None:
    subtitle_ids = [
        str(number)
        for number in range(
            81,
            91,
        )
    ]

    actual = (
        find_repeated_sequence_subtitle_ids(
            subtitle_ids,
            first_start=first_start,
            second_start=second_start,
            length=length,
        )
    )

    assert actual == []


def test_validation_reports_repeated_translation_subtitle_ids(
) -> None:
    subtitle_ids = [
        str(number)
        for number in range(
            81,
            91,
        )
    ]

    source_texts = [
        f"Source subtitle {subtitle_id}."
        for subtitle_id in subtitle_ids
    ]

    repeated_translation = (
        "では、容疑者はいますか？"
    )

    response = json.dumps(
        {
            "targets": {
                subtitle_id: {
                    "source": {
                        "speaker": None,
                        "text": source_text,
                    },
                    "translation": (
                        repeated_translation
                    ),
                }
                for (
                    subtitle_id,
                    source_text,
                ) in zip(
                    subtitle_ids,
                    source_texts,
                    strict=True,
                )
            },
        },
        ensure_ascii=False,
    )

    result = validate_translation_response(
        response,
        expected_ids=subtitle_ids,
        source_speakers=[
            None
            for _ in subtitle_ids
        ],
        source_texts=source_texts,
        noise_dictionary=(
            build_test_noise_dictionary(
                []
            )
        ),
    )

    assert not result.valid

    assert (
        "Repeated translation detected: "
        "count=10, "
        "text='では、容疑者はいますか？', "
        "subtitle_ids="
        "['81', '82', '83', '84', '85', "
        "'86', '87', '88', '89', '90']"
        in result.reasons
    )

    assert (
        "Repeated translation sequence detected: "
        "first_start=1, "
        "second_start=4, "
        "length=3, "
        "subtitle_ids="
        "['81', '82', '83', '84', '85', "
        "'86']"
        in result.reasons
    )


def test_chinese_validation_accepts_ambiguous_japanese_character(
) -> None:
    translated_text = (
        "この円は、デスティニーの範囲内にある"
        "ゲートを表しています。"
    )

    violations = (
        find_chinese_specific_characters(
            translated_texts=[
                translated_text,
            ],
            subtitle_ids=[
                "136",
            ],
        )
    )

    assert violations == []


# E19-2-3：翻訳Validationの回帰テストを追加する
def test_chinese_validation_accepts_japanese_occupation_character(
) -> None:
    translated_text = (
        "敵が施設を占領している。"
    )

    violations = (
        find_chinese_specific_characters(
            translated_texts=[
                translated_text,
            ],
            subtitle_ids=[
                "316",
            ],
        )
    )

    assert violations == []


def test_chinese_validation_accepts_e15_japanese_phrases(
) -> None:
    translated_texts = [
        (
            "仮に、我々がいる惑星の範囲内にある"
            "ゲートについて考えましょう。"
        ),
        (
            "この円は、デスティニーの範囲内にある"
            "ゲートを表しています。"
        ),
        (
            "現在、幸いにも、それぞれの範囲内にある"
            "ゲートが存在することを願っています。"
        ),
    ]

    violations = (
        find_chinese_specific_characters(
            translated_texts=translated_texts,
            subtitle_ids=[
                "134",
                "136",
                "140",
            ],
        )
    )

    assert violations == []


def test_chinese_validation_rejects_high_confidence_chinese(
) -> None:
    translated_text = (
        "これらは这些人です。"
    )

    violations = (
        find_chinese_specific_characters(
            translated_texts=[
                translated_text,
            ],
            subtitle_ids=[
                "140",
            ],
        )
    )

    assert len(
        violations
    ) == 1

    assert violations[0].startswith(
        "Chinese-specific characters detected: "
        "subtitle_id='140'"
    )

    assert (
        "这"
        in violations[0]
    )


def test_chinese_validation_reports_only_high_confidence_characters(
) -> None:
    translated_text = (
        "範囲内に这些人がいる。"
    )

    violations = (
        find_chinese_specific_characters(
            translated_texts=[
                translated_text,
            ],
            subtitle_ids=[
                "140",
            ],
        )
    )

    assert len(
        violations
    ) == 1

    assert (
        "这"
        in violations[0]
    )

    assert (
        "characters='内"
        not in violations[0]
    )


def test_e15_standard_validation_accepts_ambiguous_japanese_text(
) -> None:
    noise_dictionary = (
        build_test_noise_dictionary(
            []
        )
    )

    ocr_line = (
        "oX=¥AN(o1) 0 MUA L= S310] KO (otoe"
    )

    source_texts = [
        (
            "Let's consider the gates\n"
            "within range of the planet\n"
            "we are on."
        ),
        (
            "This circle represents\n"
            "the gates within range\n"
            "of Destiny."
        ),
        (
            "Now, hopefully, there is a gate\n"
            "within range of each one\n"
            f"{ocr_line}"
        ),
    ]

    translated_texts = [
        (
            "仮に、我々がいる惑星の範囲内にある"
            "ゲートを考えてみよう。"
        ),
        (
            "この円は、デスティニーの範囲内にある"
            "ゲートを表している。"
        ),
        (
            "現在、幸いにも、それぞれの範囲内にある"
            "ゲートが存在することを願っている。"
            f"[1]{ocr_line}[/1]"
        ),
    ]

    response = json.dumps(
        {
            "targets": {
                subtitle_id: {
                    "source": {
                        "speaker": None,
                        "text": source_text,
                    },
                    "translation": translation,
                }
                for (
                    subtitle_id,
                    source_text,
                    translation,
                ) in zip(
                    [
                        "134",
                        "136",
                        "140",
                    ],
                    source_texts,
                    translated_texts,
                    strict=True,
                )
            },
        },
        ensure_ascii=False,
    )

    result = validate_translation_response(
        response,
        expected_ids=[
            "134",
            "136",
            "140",
        ],
        source_speakers=[
            None,
            None,
            None,
        ],
        source_texts=source_texts,
        noise_dictionary=noise_dictionary,
        glossary_entries={},
    )

    assert result.valid is True

    assert result.reasons == []

    assert not any(
        reason.startswith(
            "Chinese-specific characters detected:"
        )
        for reason in result.reasons
    )

    assert result.failed_ids == set()

    assert result.translated_texts == [
        (
            "仮に、我々がいる惑星の範囲内にある"
            "ゲートを考えてみよう。"
        ),
        (
            "この円は、デスティニーの範囲内にある"
            "ゲートを表している。"
        ),
        (
            "現在、幸いにも、それぞれの範囲内にある"
            "ゲートが存在することを願っている。"
            "（判読不能）"
        ),
    ]

    assert result.noise_candidates == [
        ocr_line,
    ]


def test_e15_standard_validation_still_rejects_real_chinese_text(
) -> None:
    noise_dictionary = (
        build_test_noise_dictionary(
            []
        )
    )

    source_text = (
        "The gates are within range."
    )

    translated_text = (
        "ゲートは範囲内にありますが、"
        "这些人が近くにいます。"
    )

    response = json.dumps(
        {
            "targets": {
                "136": {
                    "source": {
                        "speaker": None,
                        "text": source_text,
                    },
                    "translation": translated_text,
                },
            },
        },
        ensure_ascii=False,
    )

    result = validate_translation_response(
        response,
        expected_ids=[
            "136",
        ],
        source_speakers=[
            None,
        ],
        source_texts=[
            source_text,
        ],
        noise_dictionary=noise_dictionary,
        glossary_entries={},
    )

    assert result.valid is False

    assert any(
        (
            reason.startswith(
                "Chinese-specific characters detected: "
                "subtitle_id='136'"
            )
            and "这" in reason
        )
        for reason in result.reasons
    )

    assert not any(
        "characters='内"
        in reason
        for reason in result.reasons
    )
