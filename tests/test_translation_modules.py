from pathlib import Path

import pytest
from lib.noise import (
    NoiseDictionary,
    NoiseEntry,
    find_confirmed_noise_sequences,
    find_suspicious_latin_sequences,
    is_valid_noise_candidate,
    normalize_noise_candidate,
)
from lib.srt import SrtBlock
from lib.translate import (
    resolve_requested_profile,
)
from lib.translation_chunk import (
    find_noise_candidate_ids,
    normalize_translation_text,
)
from lib.translation_prompt import (
    build_ocr_noise_instruction,
    build_prompt,
    build_request_item,
    build_translation_evaluation_tag_instruction,
)
from lib.translation_resume import (
    load_resume_blocks,
)
from lib.translation_tags import (
    parse_translation_tags,
    render_translation_tags,
    strip_translation_tags,
)
from lib.translation_validation import (
    validate_translation_response,
)


def build_test_noise_dictionary(
    entries: list[NoiseEntry],
) -> NoiseDictionary:
    return NoiseDictionary(
        profile_name="test",
        entries={
            entry.source: entry
            for entry in entries
        },
        official_path=Path("noise.json"),
        local_path=Path("noise.local.json"),
        local_loaded=False,
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
    instruction = (
        build_ocr_noise_instruction(
            [
                "10",
                "12",
            ]
        )
    )

    assert "対象ID: 10, 12" in instruction
    assert "（判読不能）" in instruction


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


def test_build_ocr_noise_instruction_without_ids() -> None:
    assert build_ocr_noise_instruction(
        []
    ) == ""


def test_build_translation_evaluation_tag_instruction(
) -> None:
    instruction = (
        build_translation_evaluation_tag_instruction()
    )

    assert "[1]原文文字列[/1]" in instruction
    assert "[3]原文文字列[/3]" in instruction
    assert "[5]原文文字列[/5]" in instruction

    assert (
        "[2]、[4]、その他の数字タグは使用禁止"
        in instruction
    )

    assert (
        "[5]は「Glossaryに登録された語」"
        in instruction
    )

    assert (
        "[5]SGC[/5]とはせず"
        in instruction
    )

    assert (
        "[5]P4X-351[/5]"
        in instruction
    )

    assert (
        "[1]sie lexer=s-4-9 10) "
        "WV am nat (el=)[/1]"
        in instruction
    )

    assert (
        "タグ内部の文字列は原文から"
        "そのままコピーする"
        in instruction
    )

    assert (
        "タグをネストしない"
        in instruction
    )


def test_build_prompt_includes_translation_evaluation_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_build_translation_prompt(
        *,
        target_count: int,
        request_json: str,
        profile_name: str,
    ) -> str:
        assert target_count == 1
        assert '"id": "1"' in request_json
        assert profile_name == "stargate"

        return "BASE PROMPT"

    monkeypatch.setattr(
        (
            "lib.translation_prompt."
            "build_translation_prompt"
        ),
        fake_build_translation_prompt,
    )

    target_blocks = [
        SrtBlock(
            number="1",
            timestamp=(
                "00:00:01,000 --> "
                "00:00:03,000"
            ),
            text=(
                "The planet P4X-351 "
                "is unstable."
            ),
        ),
    ]

    prompt = build_prompt(
        before_context=[],
        target_blocks=target_blocks,
        after_context=[],
        profile_name="stargate",
        ocr_noise_instruction=(
            "\nOCR INSTRUCTION\n"
        ),
    )

    assert prompt.startswith(
        "BASE PROMPT"
    )

    assert (
        "[5]P4X-351[/5]"
        in prompt
    )

    assert (
        "[1]sie lexer=s-4-9 10) "
        "WV am nat (el=)[/1]"
        in prompt
    )

    assert (
        "Glossaryに登録された語"
        in prompt
    )

    assert prompt.endswith(
        "\nOCR INSTRUCTION\n"
    )


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


def test_parse_translation_tags_returns_values(
) -> None:
    result = parse_translation_tags(
        (
            "惑星"
            "[5]P4X-351[/5]"
            "と"
            "[3]Destiny[/3]"
            "、"
            "[1]garbled text[/1]"
        )
    )

    assert result.errors == ()

    assert [
               (
                   tag.level,
                   tag.value,
               )
               for tag in result.tags
           ] == [
               (
                   5,
                   "P4X-351",
               ),
               (
                   3,
                   "Destiny",
               ),
               (
                   1,
                   "garbled text",
               ),
           ]


def test_strip_translation_tags_keeps_values(
) -> None:
    result = strip_translation_tags(
        (
            "惑星"
            "[5]P4X-351[/5]"
            "の"
            "[3]Destiny[/3]"
        )
    )

    assert result == (
        "惑星P4X-351のDestiny"
    )


@pytest.mark.parametrize(
    "level",
    [
        2,
        4,
        6,
    ],
)
def test_parse_translation_tags_rejects_unsupported_levels(
    level: int,
) -> None:
    result = parse_translation_tags(
        (
            f"[{level}]"
            "P4X-351"
            f"[/{level}]"
        )
    )

    assert result.tags == ()

    assert any(
        (
            "Unsupported translation tag level"
            in error
        )
        for error in result.errors
    )


def test_parse_translation_tags_detects_missing_closing_tag(
) -> None:
    result = parse_translation_tags(
        "[5]P4X-351"
    )

    assert result.tags == ()

    assert result.errors == (
        (
            "Missing translation closing tag: "
            "level=5, position=0"
        ),
    )


def test_parse_translation_tags_detects_mismatched_closing_tag(
) -> None:
    result = parse_translation_tags(
        "[5]P4X-351[/3]"
    )

    assert result.tags == ()

    assert any(
        (
            "Mismatched translation closing tag"
            in error
        )
        for error in result.errors
    )


def test_parse_translation_tags_detects_unexpected_closing_tag(
) -> None:
    result = parse_translation_tags(
        "P4X-351[/5]"
    )

    assert result.tags == ()

    assert any(
        (
            "Unexpected translation closing tag"
            in error
        )
        for error in result.errors
    )


def test_parse_translation_tags_detects_nested_tags(
) -> None:
    result = parse_translation_tags(
        (
            "[5]"
            "P4X-"
            "[3]351[/3]"
            "[/5]"
        )
    )

    assert any(
        (
            "Nested translation tag"
            in error
        )
        for error in result.errors
    )


@pytest.mark.parametrize(
    (
            "source",
            "expected_error",
    ),
    [
        (
                "[1][/1]",
                "Empty translation tag value",
        ),
        (
                "[5] P4X-351[/5]",
                (
                    "Translation tag value has "
                    "surrounding whitespace"
                ),
        ),
        (
                "[5]P4X-351 [/5]",
                (
                    "Translation tag value has "
                    "surrounding whitespace"
                ),
        ),
    ],
)
def test_parse_translation_tags_rejects_invalid_values(
    source: str,
    expected_error: str,
) -> None:
    result = parse_translation_tags(
        source
    )

    assert result.tags == ()

    assert any(
        expected_error in error
        for error in result.errors
    )


def test_strip_translation_tags_rejects_invalid_structure(
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Invalid translation tags"
        ),
    ):
        strip_translation_tags(
            "[5]P4X-351[/3]"
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


def test_render_translation_tags_replaces_only_tagged_level_1_value(
) -> None:
    result = render_translation_tags(
        (
            "[1]noise[/1] "
            "noise"
        ),
        level_1_replacement=(
            "（判読不能）"
        ),
    )

    assert result == (
        "（判読不能） noise"
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
