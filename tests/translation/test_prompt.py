import pytest
from lib.profile.prompt import (
    build_translation_prompt as build_profile_translation_prompt,
)
from lib.subtitle.srt import SrtBlock
from lib.translation.translation_prompt import (
    build_ocr_noise_instruction,
    build_prompt,
    build_request_item,
    build_translation_evaluation_tag_instruction,
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
            "lib.translation.translation_prompt."
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


def test_build_profile_translation_prompt_includes_target_context_isolation(
) -> None:
    request_json = """
{
  "context_before": [
    {
      "speaker": null,
      "text": "Context before text."
    }
  ],
  "target": [
    {
      "id": "1",
      "speaker": null,
      "text": "Target text."
    }
  ],
  "context_after": [
    {
      "speaker": null,
      "text": "Context after text."
    }
  ]
}
""".strip()

    prompt = build_profile_translation_prompt(
        target_count=1,
        request_json=request_json,
        profile_name=None,
    )

    assert (
        "【targetとcontextの境界】"
        in prompt
    )

    assert (
        "翻訳対象はtargetだけである。"
        in prompt
    )

    assert (
        "各translationは、同じidのtarget.textだけを翻訳する"
        in prompt
    )

    assert (
        "別のtarget、context_before、context_afterにある文字列を"
        in prompt
    )

    assert (
        "評価タグへ使用してはいけない。"
        in prompt
    )

    assert request_json in prompt
