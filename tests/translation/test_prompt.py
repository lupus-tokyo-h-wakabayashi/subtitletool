import json

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
    build_translation_request_json,
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
        "source": {
            "speaker": "DANIEL",
            "text": "This is the Stargate.",
        },
        "translation": "",
    }


def test_build_translation_request_json_uses_id_keyed_targets(
) -> None:
    before_context = [
        SrtBlock(
            number="10",
            timestamp=(
                "00:00:01,000 --> "
                "00:00:03,000"
            ),
            text="Before context.",
        ),
    ]

    target_blocks = [
        SrtBlock(
            number="11",
            timestamp=(
                "00:00:04,000 --> "
                "00:00:06,000"
            ),
            text=(
                "DANIEL: "
                "This is the Stargate."
            ),
        ),
        SrtBlock(
            number="12",
            timestamp=(
                "00:00:07,000 --> "
                "00:00:09,000"
            ),
            text="That's a pity.",
        ),
    ]

    after_context = [
        SrtBlock(
            number="13",
            timestamp=(
                "00:00:10,000 --> "
                "00:00:12,000"
            ),
            text="After context.",
        ),
    ]

    request_json = (
        build_translation_request_json(
            before_context,
            target_blocks,
            after_context,
        )
    )

    payload = json.loads(
        request_json
    )

    assert payload == {
        "context_before": [
            {
                "speaker": None,
                "text": "Before context.",
            },
        ],
        "targets": {
            "11": {
                "source": {
                    "speaker": "DANIEL",
                    "text": "This is the Stargate.",
                },
                "translation": "",
            },
            "12": {
                "source": {
                    "speaker": None,
                    "text": "That's a pity.",
                },
                "translation": "",
            },
        },
        "context_after": [
            {
                "speaker": None,
                "text": "After context.",
            },
        ],
    }

    assert "target" not in payload
    assert "translations" not in payload


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
        payload = json.loads(
            request_json
        )

        assert target_count == 1
        assert profile_name == "stargate"

        assert payload == {
            "context_before": [],
            "targets": {
                "1": {
                    "source": {
                        "speaker": None,
                        "text": (
                            "The planet P4X-351 "
                            "is unstable."
                        ),
                    },
                    "translation": "",
                },
            },
            "context_after": [],
        }

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


def test_build_profile_translation_prompt_includes_editable_targets_schema(
) -> None:
    request_json = """
{
  "context_before": [
    {
      "speaker": null,
      "text": "Context before text."
    }
  ],
  "targets": {
    "1": {
      "source": {
        "speaker": null,
        "text": "Target text."
      },
      "translation": ""
    }
  },
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
        "【JSON編集タスク】"
        in prompt
    )

    assert (
        "新しいJSONを生成するタスクではない"
        in prompt
    )

    assert (
        "返却JSONのtargetsへ複製すること"
        in prompt
    )

    assert (
        "targets配下の各字幕オブジェクトにあるtranslationだけ"
        in prompt
    )

    assert (
        "context_beforeとcontext_afterは返却しない"
        in prompt
    )

    assert (
        "最上位キーはtargetsだけ"
        in prompt
    )

    assert (
        "source.speakerとsource.textを変更してはいけません"
        in prompt
    )

    assert (
        "translation以外は、"
        in prompt
    )

    assert request_json in prompt
