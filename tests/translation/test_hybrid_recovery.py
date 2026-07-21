from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from lib.profile.glossary import (
    GlossaryEntries,
    GlossaryEntry,
)
from lib.profile.noise import (
    NoiseDictionary,
)
from lib.profile.ocr_scoring import (
    OcrScoringConfig,
    load_ocr_scoring_config,
)
from lib.subtitle.srt import SrtBlock
from lib.translation import hybrid_recovery
from lib.translation.hybrid_group import (
    HybridTranslationGroup,
    build_hybrid_translation_group,
)
from lib.translation.hybrid_recovery import (
    HYBRID_OCR_PLACEHOLDER,
    HybridRecoveryError,
    build_hybrid_context_payload,
    build_hybrid_response_schema,
    build_hybrid_source_payload,
    build_hybrid_translation_prompt,
    find_group_ocr_lines,
    find_group_sound_effect_lines,
    normalize_hybrid_parentheses,
    validate_hybrid_response,
    recover_translation_with_hybrid,
)
from lib.translation.translation_metrics import (
    TranslationChunkMetric,
)
from lib.translation.translation_validation import (
    ValidationResult,
)

OCR_LINE = (
    "=) EWA eam CO ma = Ae lan"
)

E08_OCR_LINE = (
    "Ui maar i mele aah ml iaa"
)
E08_NORMAL_LINE = (
    "How can I not"
)
E08_FOLLOWING_TEXT = (
    'The "us" on that recording\n'
    "dropped out of FTL\n"
    "and went to the planet."
)

E13_NORMAL_LINE = (
    "Okay, what about"
)
E13_SHORT_OCR_LINE = (
    "dam IAN el ESie"
)
E13_SOURCE_TEXT = (
    f"{E13_NORMAL_LINE}\n"
    f"{E13_SHORT_OCR_LINE}"
)

E09_SOURCE_TEXTS = (
    "So, any suspects?",
    (
        "Well, excuse me\n"
        "for being blunt,"
    ),
    (
        "el mal em)\n"
        "a killer onboard the ship."
    ),
    (
        "Do we have any\n"
        "idea who did this?"
    ),
    (
        "I'm still trying to\n"
        "wrap my head around it."
    ),
    "It's unbelievable.",
    (
        "You put ordinary people\n"
        "under enough stress,"
    ),
    (
        "I think you'll find\n"
        "they're capable of\n"
        "just about anything."
    ),
    (
        "Add to that the fact\n"
        "he was hoarding\n"
        "water and food,"
    ),
    (
        "involved in\n"
        "several confrontations."
    ),
)
E09_REPEATED_TRANSLATION = (
    "では、容疑者はいますか？"
)
E09_RECOVERED_TRANSLATIONS = (
    "容疑者に心当たりは？",
    "率直に言って悪いが、",
    "船に殺人犯がいる。",
    "誰がやったか分かるのか？",
    "まだ理解しようとしている。",
    "信じられない。",
    (
        "普通の人でも強い圧力を"
        "受ければ、"
    ),
    (
        "ほとんど何でも"
        "できてしまう。"
    ),
    (
        "彼は水と食料を"
        "ため込んでいたうえ、"
    ),
    (
        "何度も衝突を"
        "起こしていた。"
    ),
)


@pytest.fixture
def e09_target_blocks(
) -> list[SrtBlock]:
    timestamps = (
        (
            "00:05:00,000 --> "
            "00:05:01,000"
        ),
        (
            "00:05:01,100 --> "
            "00:05:02,000"
        ),
        (
            "00:05:02,100 --> "
            "00:05:03,000"
        ),
        (
            "00:05:03,100 --> "
            "00:05:04,000"
        ),
        (
            "00:05:04,100 --> "
            "00:05:05,000"
        ),
        (
            "00:05:05,100 --> "
            "00:05:06,000"
        ),
        (
            "00:05:06,100 --> "
            "00:05:07,000"
        ),
        (
            "00:05:07,100 --> "
            "00:05:08,000"
        ),
        (
            "00:05:08,100 --> "
            "00:05:09,000"
        ),
        (
            "00:05:09,100 --> "
            "00:05:10,000"
        ),
    )

    return [
        SrtBlock(
            number=str(number),
            timestamp=timestamp,
            text=source_text,
        )
        for (
            number,
            timestamp,
            source_text,
        ) in zip(
            range(
                81,
                91,
            ),
            timestamps,
            E09_SOURCE_TEXTS,
            strict=True,
        )
    ]


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
def ocr_scoring_config(
) -> OcrScoringConfig:
    return load_ocr_scoring_config()


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


@pytest.fixture
def e08_target_blocks(
) -> list[SrtBlock]:
    return [
        SrtBlock(
            number="496",
            timestamp=(
                "00:31:59,418 --> "
                "00:32:01,044"
            ),
            text=(
                f"{E08_NORMAL_LINE}\n"
                f"{E08_OCR_LINE}"
            ),
        ),
        SrtBlock(
            number="497",
            timestamp=(
                "00:32:01,128 --> "
                "00:32:04,506"
            ),
            text=E08_FOLLOWING_TEXT,
        ),
    ]


def test_e08_failed_subtitle_detects_only_word_salad_as_ocr(
    e08_target_blocks: list[SrtBlock],
    ocr_scoring_config: OcrScoringConfig,
) -> None:
    group = build_hybrid_translation_group(
        e08_target_blocks,
        {
            "496",
        },
    )

    assert group is not None

    assert group.target_ids == (
        "496",
        "497",
    )

    assert group.failed_ids == frozenset(
        {
            "496",
        }
    )

    actual = find_group_ocr_lines(
        group,
        glossary_entries=(
            GlossaryEntries(())
        ),
        scoring_config=(
            ocr_scoring_config
        ),
    )

    assert actual == {
        "496": [
            E08_OCR_LINE,
        ],
    }


def test_e08_hybrid_source_payload_classifies_lines(
    e08_target_blocks: list[SrtBlock],
    ocr_scoring_config: OcrScoringConfig,
) -> None:
    group = build_hybrid_translation_group(
        e08_target_blocks,
        {
            "496",
        },
    )

    assert group is not None

    ocr_lines = find_group_ocr_lines(
        group,
        glossary_entries=(
            GlossaryEntries(())
        ),
        scoring_config=(
            ocr_scoring_config
        ),
    )

    actual = build_hybrid_source_payload(
        group,
        ocr_lines,
    )

    assert actual == {
        "subtitles": [
            {
                "id": "496",
                "lines": [
                    {
                        "kind": "text",
                        "text": E08_NORMAL_LINE,
                    },
                    {
                        "kind": "ocr",
                        "text": E08_OCR_LINE,
                    },
                ],
            },
            {
                "id": "497",
                "lines": [
                    {
                        "kind": "text",
                        "text": (
                            'The "us" on that recording'
                        ),
                    },
                    {
                        "kind": "text",
                        "text": (
                            "dropped out of FTL"
                        ),
                    },
                    {
                        "kind": "text",
                        "text": (
                            "and went to the planet."
                        ),
                    },
                ],
            },
        ],
    }


def test_hybrid_source_payload_separates_speaker_sound_effect_and_ocr(
    ocr_scoring_config: OcrScoringConfig,
) -> None:
    damaged_dialogue = (
        "Ui maar i mele aah ml iaa"
    )

    block = SrtBlock(
        number="627",
        timestamp=(
            "00:31:10,000 --> "
            "00:31:12,000"
        ),
        text=(
            "[RUSH] "
            "(CHUCKLING) "
            f"{damaged_dialogue}"
        ),
    )

    group = HybridTranslationGroup(
        positions=(
            0,
        ),
        blocks=(
            block,
        ),
        failed_ids=frozenset(
            {
                "627",
            }
        ),
    )

    ocr_lines = find_group_ocr_lines(
        group,
        glossary_entries={},
        scoring_config=(
            ocr_scoring_config
        ),
    )

    actual = build_hybrid_source_payload(
        group,
        ocr_lines,
    )

    assert ocr_lines == {
        "627": [
            damaged_dialogue,
        ],
    }

    assert actual == {
        "subtitles": [
            {
                "id": "627",
                "speaker": "RUSH",
                "lines": [
                    {
                        "kind": (
                            "sound_effect"
                        ),
                        "text": (
                            "(CHUCKLING)"
                        ),
                    },
                    {
                        "kind": "ocr",
                        "text": (
                            damaged_dialogue
                        ),
                    },
                ],
            },
        ],
    }


def test_hybrid_source_payload_separates_leading_sound_effect_from_text(
) -> None:
    block = SrtBlock(
        number="628",
        timestamp=(
            "00:31:12,000 --> "
            "00:31:14,000"
        ),
        text=(
            "[YOUNG] "
            "(GASPS) Get out!"
        ),
    )

    group = HybridTranslationGroup(
        positions=(
            0,
        ),
        blocks=(
            block,
        ),
        failed_ids=frozenset(
            {
                "628",
            }
        ),
    )

    actual = build_hybrid_source_payload(
        group,
        {},
    )

    assert actual == {
        "subtitles": [
            {
                "id": "628",
                "speaker": "YOUNG",
                "lines": [
                    {
                        "kind": (
                            "sound_effect"
                        ),
                        "text": "(GASPS)",
                    },
                    {
                        "kind": "text",
                        "text": "Get out!",
                    },
                ],
            },
        ],
    }


def test_find_group_sound_effect_lines_extracts_leading_effect_from_mixed_line(
) -> None:
    block = SrtBlock(
        number="629",
        timestamp=(
            "00:31:14,000 --> "
            "00:31:16,000"
        ),
        text=(
            "[SCOTT] "
            "(PANTING) "
            "We have to move."
        ),
    )

    group = HybridTranslationGroup(
        positions=(
            0,
        ),
        blocks=(
            block,
        ),
        failed_ids=frozenset(
            {
                "629",
            }
        ),
    )

    actual = (
        find_group_sound_effect_lines(
            group
        )
    )

    assert actual == {
        "629": [
            "(PANTING)",
        ],
    }


def test_build_hybrid_context_payload_separates_speaker_from_text(
) -> None:
    blocks = [
        SrtBlock(
            number="626",
            timestamp=(
                "00:31:08,000 --> "
                "00:31:10,000"
            ),
            text=(
                "[RUSH] "
                "(SIGHS) "
                "We need more time."
            ),
        ),
    ]

    actual = (
        build_hybrid_context_payload(
            blocks
        )
    )

    assert actual == [
        {
            "id": "626",
            "speaker": "RUSH",
            "text": (
                "(SIGHS) "
                "We need more time."
            ),
        },
    ]


def test_build_hybrid_context_payload_preserves_id_and_text(
) -> None:
    blocks = [
        SrtBlock(
            number="103",
            timestamp=(
                "00:04:43,575 --> "
                "00:04:44,284"
            ),
            text="Don't let her.",
        ),
        SrtBlock(
            number="105",
            timestamp=(
                "00:04:45,911 --> "
                "00:04:47,120"
            ),
            text=(
                "Don't release her.\n"
                "Keep her here."
            ),
        ),
    ]

    actual = build_hybrid_context_payload(
        blocks
    )

    assert actual == [
        {
            "id": "103",
            "text": "Don't let her.",
        },
        {
            "id": "105",
            "text": (
                "Don't release her.\n"
                "Keep her here."
            ),
        },
    ]


def test_build_hybrid_context_payload_returns_empty_for_none(
) -> None:
    actual = build_hybrid_context_payload(
        None
    )

    assert actual == []


def test_build_hybrid_translation_prompt_includes_surrounding_context(
) -> None:
    target_block = SrtBlock(
        number="104",
        timestamp=(
            "00:04:44,576 --> "
            "00:04:45,619"
        ),
        text=(
            "Okay, when\n"
            "she returns home..."
        ),
    )

    group = HybridTranslationGroup(
        positions=(
            0,
        ),
        blocks=(
            target_block,
        ),
        failed_ids=frozenset(
            {
                "104",
            }
        ),
    )

    before_context = [
        SrtBlock(
            number="103",
            timestamp=(
                "00:04:43,575 --> "
                "00:04:44,284"
            ),
            text="Don't let her.",
        ),
    ]

    after_context = [
        SrtBlock(
            number="105",
            timestamp=(
                "00:04:45,911 --> "
                "00:04:47,120"
            ),
            text=(
                "Don't release her.\n"
                "Keep her here."
            ),
        ),
    ]

    prompt = build_hybrid_translation_prompt(
        group,
        {},
        {},
        before_context=before_context,
        after_context=after_context,
    )

    assert (
        "【参考文脈（前）】"
        in prompt
    )

    assert (
        '"id": "103"'
        in prompt
    )

    assert (
        '"text": "Don\'t let her."'
        in prompt
    )

    assert (
        "【翻訳対象】"
        in prompt
    )

    assert (
        '"id": "104"'
        in prompt
    )

    assert (
        '"text": "Okay, when"'
        in prompt
    )

    assert (
        '"text": "she returns home..."'
        in prompt
    )

    assert (
        "【参考文脈（後）】"
        in prompt
    )

    assert (
        '"id": "105"'
        in prompt
    )

    assert (
        "Don\'t release her.\\nKeep her here."
        in prompt
    )

    assert (
        "context_beforeとcontext_afterの字幕は、"
        in prompt
    )

    assert (
        "full_translationとsegmentsへ出力しないこと。"
        in prompt
    )


def test_low_symbol_word_salad_is_limited_to_failed_ids(
    e08_target_blocks: list[SrtBlock],
    ocr_scoring_config: OcrScoringConfig,
) -> None:
    group = HybridTranslationGroup(
        positions=(
            0,
            1,
        ),
        blocks=tuple(
            e08_target_blocks
        ),
        failed_ids=frozenset(
            {
                "497",
            }
        ),
    )

    actual = find_group_ocr_lines(
        group,
        glossary_entries=(
            GlossaryEntries(())
        ),
        scoring_config=(
            ocr_scoring_config
        ),
    )

    assert actual == {}


def test_assessed_group_ocr_lines_detects_structural_damage(
    ocr_scoring_config: OcrScoringConfig,
) -> None:
    damaged_line = '"(CLR (=r 108'

    block = SrtBlock(
        number="497",
        timestamp=(
            "00:25:22,563 --> "
            "00:25:24,773"
        ),
        text=damaged_line,
    )

    group = HybridTranslationGroup(
        positions=(
            0,
        ),
        blocks=(
            block,
        ),
        failed_ids=frozenset(
            {
                "497",
            }
        ),
    )

    actual = find_group_ocr_lines(
        group,
        glossary_entries=(
            GlossaryEntries(())
        ),
        scoring_config=(
            ocr_scoring_config
        ),
    )

    assert actual == {
        "497": [
            damaged_line,
        ],
    }


def test_assessed_group_ocr_lines_preserves_normal_mixed_line(
    ocr_scoring_config: OcrScoringConfig,
) -> None:
    damaged_line = (
        "Ui maar i mele aah ml iaa"
    )

    normal_line = (
        "seeing the old homestead again."
    )

    block = SrtBlock(
        number="98",
        timestamp=(
            "00:05:00,000 --> "
            "00:05:02,000"
        ),
        text=(
            f"{damaged_line}\n"
            f"{normal_line}"
        ),
    )

    group = HybridTranslationGroup(
        positions=(
            0,
        ),
        blocks=(
            block,
        ),
        failed_ids=frozenset(
            {
                "98",
            }
        ),
    )

    actual = find_group_ocr_lines(
        group,
        glossary_entries=(
            GlossaryEntries(())
        ),
        scoring_config=(
            ocr_scoring_config
        ),
    )

    assert actual == {
        "98": [
            damaged_line,
        ],
    }


def test_assessed_group_ocr_lines_uses_high_threshold_for_context(
    ocr_scoring_config: OcrScoringConfig,
) -> None:
    context_line = (
        "dam IAN el ESie"
    )

    failed_line = (
        '"(CLR (=r 108'
    )

    blocks = (
        SrtBlock(
            number="10",
            timestamp=(
                "00:00:10,000 --> "
                "00:00:11,000"
            ),
            text=context_line,
        ),
        SrtBlock(
            number="11",
            timestamp=(
                "00:00:11,000 --> "
                "00:00:12,000"
            ),
            text=failed_line,
        ),
    )

    group = HybridTranslationGroup(
        positions=(
            0,
            1,
        ),
        blocks=blocks,
        failed_ids=frozenset(
            {
                "11",
            }
        ),
    )

    actual = find_group_ocr_lines(
        group,
        glossary_entries=(
            GlossaryEntries(())
        ),
        scoring_config=(
            ocr_scoring_config
        ),
    )

    assert actual == {
        "11": [
            failed_line,
        ],
    }


def test_assessed_group_ocr_lines_ignores_sound_effect(
    ocr_scoring_config: OcrScoringConfig,
) -> None:
    block = SrtBlock(
        number="43",
        timestamp=(
            "00:02:17,429 --> "
            "00:02:17,804"
        ),
        text="(SCOTT GRUNTING)",
    )

    group = HybridTranslationGroup(
        positions=(
            0,
        ),
        blocks=(
            block,
        ),
        failed_ids=frozenset(
            {
                "43",
            }
        ),
    )

    actual = find_group_ocr_lines(
        group,
        glossary_entries=(
            GlossaryEntries(())
        ),
        scoring_config=(
            ocr_scoring_config
        ),
    )

    assert actual == {}


def test_assessed_group_ocr_lines_protects_glossary_identifier(
    ocr_scoring_config: OcrScoringConfig,
) -> None:
    block = SrtBlock(
        number="280",
        timestamp=(
            "00:14:41,506 --> "
            "00:14:42,799"
        ),
        text="SG-1",
    )

    group = HybridTranslationGroup(
        positions=(
            0,
        ),
        blocks=(
            block,
        ),
        failed_ids=frozenset(
            {
                "280",
            }
        ),
    )

    glossary_entries = GlossaryEntries(
        (
            GlossaryEntry(
                source="SG-1",
                target="SG-1",
                case_sensitive=True,
            ),
        )
    )

    actual = find_group_ocr_lines(
        group,
        glossary_entries=(
            GlossaryEntries(())
        ),
        scoring_config=(
            ocr_scoring_config
        ),
    )

    assert actual == {}


@pytest.mark.parametrize(
    "source_text",
    [
        (
            "Hopefully, we've proven\n"
            "that's not our goal."
        ),
        (
            "I couldn't deal with it,\n"
            "the thought of you\n"
            "being trapped on that ship."
        ),
    ],
)
def test_assessed_group_ocr_lines_preserves_contractions(
    source_text: str,
    ocr_scoring_config: OcrScoringConfig,
) -> None:
    block = SrtBlock(
        number="602",
        timestamp=(
            "00:00:01,100 --> "
            "00:00:03,000"
        ),
        text=source_text,
    )

    group = HybridTranslationGroup(
        positions=(
            0,
        ),
        blocks=(
            block,
        ),
        failed_ids=frozenset(
            {
                "602",
            }
        ),
    )

    actual = find_group_ocr_lines(
        group,
        glossary_entries=(
            GlossaryEntries(())
        ),
        scoring_config=(
            ocr_scoring_config
        ),
    )

    assert actual == {}


def valid_e08_hybrid_payload(
) -> dict[str, object]:
    segments = {
        "496": (
            "無視できるわけがない。"
            "（判読不能）"
        ),
        "497": (
            "録画に映っていた私たちは"
            "超光速航行を終了し、"
            "惑星へ向かった。"
        ),
    }

    return {
        "group": {
            "full_translation": (
                segments["496"]
                + segments["497"]
            ),
            "segments": segments,
        },
    }


def test_e08_prompt_contains_segment_requirements(
    e08_target_blocks: list[SrtBlock],
    ocr_scoring_config: OcrScoringConfig,
) -> None:
    group = build_hybrid_translation_group(
        e08_target_blocks,
        {
            "496",
        },
    )

    assert group is not None

    ocr_lines = find_group_ocr_lines(
        group,
        glossary_entries=(
            GlossaryEntries(())
        ),
        scoring_config=(
            ocr_scoring_config
        ),
    )

    prompt = build_hybrid_translation_prompt(
        group,
        ocr_lines,
        {},
    )

    assert (
        "* 字幕ID 496: "
        "kind=textの正常英文を"
        "自然な日本語へ翻訳する。"
        "英文を残さない。"
        "kind=ocrの位置を"
        "「（判読不能）」で表現し、"
        "OCR原文をコピーしない。"
        "segmentには"
        "「（判読不能）」と、"
        "それ以外の翻訳結果を"
        "両方とも含める。"
        "各行の内容を原文順に配置する。"
        in prompt
    )

    assert (
        "* 字幕ID 497: "
        "kind=textの正常英文を"
        "自然な日本語へ翻訳する。"
        "英文を残さない。"
        "各行の内容を原文順に配置する。"
        in prompt
    )

    assert "【OCR行】" in prompt

    assert E08_OCR_LINE in prompt

    assert HYBRID_OCR_PLACEHOLDER in prompt

    assert (
        "aR at-lacmanl-e"
        not in prompt
    )

    assert (
        "私は良い友人です"
        not in prompt
    )


def test_e08_validation_accepts_mixed_ocr_result(
    e08_target_blocks: list[SrtBlock],
    ocr_scoring_config: OcrScoringConfig,
) -> None:
    group = build_hybrid_translation_group(
        e08_target_blocks,
        {
            "496",
        },
    )

    assert group is not None

    ocr_lines = find_group_ocr_lines(
        group,
        glossary_entries=(
            GlossaryEntries(())
        ),
        scoring_config=(
            ocr_scoring_config
        ),
    )

    validation = validate_hybrid_response(
        json.dumps(
            valid_e08_hybrid_payload(),
            ensure_ascii=False,
        ),
        group,
        ocr_lines,
    )

    assert validation.valid is True
    assert validation.reasons == ()

    assert validation.segments == {
        "496": (
            "無視できるわけがない。"
            "（判読不能）"
        ),
        "497": (
            "録画に映っていた私たちは"
            "超光速航行を終了し、"
            "惑星へ向かった。"
        ),
    }


def test_e08_validation_rejects_placeholder_only_for_496(
    e08_target_blocks: list[SrtBlock],
    ocr_scoring_config: OcrScoringConfig,
) -> None:
    group = build_hybrid_translation_group(
        e08_target_blocks,
        {
            "496",
        },
    )

    assert group is not None

    ocr_lines = find_group_ocr_lines(
        group,
        glossary_entries=(
            GlossaryEntries(())
        ),
        scoring_config=(
            ocr_scoring_config
        ),
    )

    payload = valid_e08_hybrid_payload()

    payload[
        "group"
    ][
        "segments"
    ][
        "496"
    ] = HYBRID_OCR_PLACEHOLDER

    validation = validate_hybrid_response(
        json.dumps(
            payload,
            ensure_ascii=False,
        ),
        group,
        ocr_lines,
    )

    assert validation.valid is False

    assert any(
        reason.startswith(
            "Hybrid mixed OCR segment "
            "requires Japanese translation:"
        )
        for reason in validation.reasons
    )


def test_e08_validation_rejects_missing_placeholder_for_496(
    e08_target_blocks: list[SrtBlock],
    ocr_scoring_config: OcrScoringConfig,
) -> None:
    group = build_hybrid_translation_group(
        e08_target_blocks,
        {
            "496",
        },
    )

    assert group is not None

    ocr_lines = find_group_ocr_lines(
        group,
        glossary_entries=(
            GlossaryEntries(())
        ),
        scoring_config=(
            ocr_scoring_config
        ),
    )

    payload = valid_e08_hybrid_payload()

    payload[
        "group"
    ][
        "segments"
    ][
        "496"
    ] = "無視できるわけがない。"

    validation = validate_hybrid_response(
        json.dumps(
            payload,
            ensure_ascii=False,
        ),
        group,
        ocr_lines,
    )

    assert validation.valid is False

    assert (
        "Hybrid OCR placeholder missing: "
        "subtitle_id='496', "
        f"required={HYBRID_OCR_PLACEHOLDER!r}"
        in validation.reasons
    )


def test_e08_validation_rejects_english_for_497(
    e08_target_blocks: list[SrtBlock],
    ocr_scoring_config: OcrScoringConfig,
) -> None:
    group = build_hybrid_translation_group(
        e08_target_blocks,
        {
            "496",
        },
    )

    assert group is not None

    ocr_lines = find_group_ocr_lines(
        group,
        glossary_entries=(
            GlossaryEntries(())
        ),
        scoring_config=(
            ocr_scoring_config
        ),
    )

    payload = valid_e08_hybrid_payload()

    payload[
        "group"
    ][
        "segments"
    ][
        "497"
    ] = E08_FOLLOWING_TEXT

    validation = validate_hybrid_response(
        json.dumps(
            payload,
            ensure_ascii=False,
        ),
        group,
        ocr_lines,
    )

    assert validation.valid is False

    assert (
        "Hybrid segment requires Japanese: "
        "subtitle_id='497', "
        f"text={E08_FOLLOWING_TEXT!r}"
        in validation.reasons
    )


def test_e08_validation_rejects_placeholder_for_497(
    e08_target_blocks: list[SrtBlock],
    ocr_scoring_config: OcrScoringConfig,
) -> None:
    group = build_hybrid_translation_group(
        e08_target_blocks,
        {
            "496",
        },
    )

    assert group is not None

    ocr_lines = find_group_ocr_lines(
        group,
        glossary_entries=(
            GlossaryEntries(())
        ),
        scoring_config=(
            ocr_scoring_config
        ),
    )

    payload = valid_e08_hybrid_payload()

    payload[
        "group"
    ][
        "segments"
    ][
        "497"
    ] = (
        "（判読不能）"
        "録画に映っていた私たちは"
        "惑星へ向かった。"
    )

    validation = validate_hybrid_response(
        json.dumps(
            payload,
            ensure_ascii=False,
        ),
        group,
        ocr_lines,
    )

    assert validation.valid is False

    assert any(
        reason.startswith(
            "Unexpected Hybrid OCR placeholder: "
            "subtitle_id='497'"
        )
        for reason in validation.reasons
    )


def test_e08_hybrid_recovery_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    e08_target_blocks: list[SrtBlock],
    noise_dictionary: NoiseDictionary,
) -> None:
    generated_requests: list[
        dict[str, object]
    ] = []

    saved_reports: list[
        dict[str, object]
    ] = []

    response = json.dumps(
        valid_e08_hybrid_payload(),
        ensure_ascii=False,
    )

    def fake_generate(
        prompt: str,
        *,
        model: str,
        response_format: dict[str, object],
    ) -> str:
        generated_requests.append(
            {
                "prompt": prompt,
                "model": model,
                "response_format": response_format,
            }
        )

        return response

    def fake_save_report(
        **kwargs: object,
    ) -> None:
        saved_reports.append(
            dict(
                kwargs
            )
        )

    monkeypatch.setattr(
        hybrid_recovery,
        "generate",
        fake_generate,
    )

    monkeypatch.setattr(
        hybrid_recovery,
        "try_save_hybrid_attempt_report",
        fake_save_report,
    )

    previous_texts = [
        (
            "How can I not\n"
            f"{E08_OCR_LINE}"
        ),
        E08_FOLLOWING_TEXT,
    ]

    original_previous_texts = list(
        previous_texts
    )

    errors = [
        (
            "Untranslated English sentence detected: "
            "subtitle_id='496', "
            "text='How can I not\\n"
            f"{E08_OCR_LINE}'"
        ),
    ]

    result = recover_translation_with_hybrid(
        e08_target_blocks,
        previous_texts,
        errors,
        "test-model",
        noise_dictionary=noise_dictionary,
        glossary_entries={},
    )

    assert result == [
        (
            "無視できるわけがない。"
            "（判読不能）"
        ),
        (
            "録画に映っていた私たちは"
            "超光速航行を終了し、"
            "惑星へ向かった。"
        ),
    ]

    assert previous_texts == (
        original_previous_texts
    )

    assert len(
        generated_requests
    ) == 1

    generated_request = (
        generated_requests[0]
    )

    assert generated_request[
               "model"
           ] == "test-model"

    prompt = generated_request[
        "prompt"
    ]

    assert isinstance(
        prompt,
        str,
    )

    assert (
        '"kind": "text",\n'
        f'          "text": "{E08_NORMAL_LINE}"'
        in prompt
    )

    assert (
        '"kind": "ocr",\n'
        f'          "text": "{E08_OCR_LINE}"'
        in prompt
    )

    assert (
        "* 字幕ID 496: "
        "kind=textの正常英文を"
        "自然な日本語へ翻訳する。"
        "英文を残さない。"
        "kind=ocrの位置を"
        "「（判読不能）」で表現し、"
        "OCR原文をコピーしない。"
        "segmentには"
        "「（判読不能）」と、"
        "それ以外の翻訳結果を"
        "両方とも含める。"
        "各行の内容を原文順に配置する。"
        in prompt
    )

    assert (
        "* 字幕ID 497: "
        "kind=textの正常英文を"
        "自然な日本語へ翻訳する。"
        "英文を残さない。"
        "各行の内容を原文順に配置する。"
        in prompt
    )

    response_format = generated_request[
        "response_format"
    ]

    assert isinstance(
        response_format,
        dict,
    )

    required_ids = (
        response_format[
            "properties"
        ][
            "group"
        ][
            "properties"
        ][
            "segments"
        ][
            "required"
        ]
    )

    assert required_ids == [
        "496",
        "497",
    ]

    assert len(
        saved_reports
    ) == 1

    saved_report = saved_reports[0]

    assert saved_report[
               "validation_stage"
           ] == "complete"

    assert (
        saved_report[
            "validation_valid"
        ]
        is True
    )

    assert saved_report[
               "validation_reasons"
           ] == []

    assert saved_report[
               "ocr_lines"
           ] == {
               "496": [
                   E08_OCR_LINE,
               ],
           }


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


def test_validate_hybrid_response_rebuilds_full_text_from_segments(
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

    assert validation.valid is True
    assert validation.reasons == ()

    assert validation.full_translation == (
        "".join(
            payload[
                "group"
            ][
                "segments"
            ].values()
        )
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

    monkeypatch.setattr(
        hybrid_recovery,
        "try_save_hybrid_attempt_report",
        lambda **kwargs: None,
    )

    metrics = build_chunk_metrics()

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
        metrics=metrics,
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

    assert metrics.hybrid_triggered is True

    assert len(
        metrics.hybrid_groups
    ) == 1

    group_metric = metrics.hybrid_groups[0]

    assert group_metric.group_number == 1

    assert group_metric.target_ids == (
        "281",
        "282",
        "283",
    )

    assert group_metric.failed_ids == (
        "282",
    )

    assert group_metric.result == "success"

    assert len(
        group_metric.attempts
    ) == 1

    attempt_metric = group_metric.attempts[0]

    assert attempt_metric.pipeline == "hybrid"
    assert attempt_metric.attempt == 1

    assert attempt_metric.target_ids == (
        "281",
        "282",
        "283",
    )

    assert attempt_metric.response_received is True

    assert (
        attempt_metric.validation_stage
        == "complete"
    )

    assert attempt_metric.validation_valid is True
    assert attempt_metric.validation_reasons == ()
    assert attempt_metric.reason_codes == ()
    assert attempt_metric.elapsed_seconds >= 0


# Hybrid Validation失敗後の再試行
def test_hybrid_validation_failure_records_each_attempt(
    monkeypatch: pytest.MonkeyPatch,
    target_blocks: list[SrtBlock],
    noise_dictionary: NoiseDictionary,
) -> None:
    invalid_payload = valid_hybrid_payload()

    invalid_payload[
        "group"
    ][
        "segments"
    ][
        "282"
    ] = "各自の居住区へ戻り、"

    responses = iter(
        [
            json.dumps(
                invalid_payload,
                ensure_ascii=False,
            ),
            json.dumps(
                valid_hybrid_payload(),
                ensure_ascii=False,
            ),
        ]
    )

    monkeypatch.setattr(
        hybrid_recovery,
        "generate",
        lambda *args, **kwargs: next(
            responses
        ),
    )

    monkeypatch.setattr(
        hybrid_recovery,
        "try_save_hybrid_attempt_report",
        lambda **kwargs: None,
    )

    metrics = build_chunk_metrics()

    result = recover_translation_with_hybrid(
        target_blocks,
        [
            "制御されていますが、",
            "to their quarters",
            "追って通知があるまで。",
        ],
        [
            (
                "Untranslated English sentence "
                "detected: subtitle_id='282', "
                "text='to their quarters'"
            ),
        ],
        "test-model",
        noise_dictionary=noise_dictionary,
        glossary_entries={},
        metrics=metrics,
    )

    assert result is not None

    assert len(
        metrics.hybrid_groups
    ) == 1

    group_metric = metrics.hybrid_groups[0]

    assert group_metric.result == "success"

    assert len(
        group_metric.attempts
    ) == 2

    first_attempt = group_metric.attempts[0]
    second_attempt = group_metric.attempts[1]

    assert first_attempt.attempt == 1

    assert (
        first_attempt.validation_stage
        == "hybrid_validation"
    )

    assert first_attempt.validation_valid is False

    assert (
        "hybrid_ocr_placeholder_missing"
        in first_attempt.reason_codes
    )

    assert second_attempt.attempt == 2

    assert (
        second_attempt.validation_stage
        == "complete"
    )

    assert second_attempt.validation_valid is True


# Hybrid結果の既存Validator失敗
def test_standard_validation_failure_is_recorded(
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

    validations = iter(
        [
            # Hybrid試行1回目のグループ検証
            ValidationResult(
                valid=False,
                reasons=[
                    (
                        "Glossary violation: "
                        "subtitle_id='282'"
                    ),
                ],
                translated_texts=[],
            ),
            # Hybrid試行2回目のグループ検証
            ValidationResult(
                valid=True,
                translated_texts=[
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
                ],
            ),
            # Hybrid反映後のチャンク全体検証
            ValidationResult(
                valid=True,
                translated_texts=[
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
                ],
            ),
        ]
    )

    monkeypatch.setattr(
        hybrid_recovery,
        "validate_translation_response",
        lambda *args, **kwargs: next(
            validations
        ),
    )

    monkeypatch.setattr(
        hybrid_recovery,
        "try_save_hybrid_attempt_report",
        lambda **kwargs: None,
    )

    metrics = build_chunk_metrics()

    result = recover_translation_with_hybrid(
        target_blocks,
        [
            "制御されていますが、",
            "to their quarters",
            "追って通知があるまで。",
        ],
        [
            (
                "Untranslated English sentence "
                "detected: subtitle_id='282', "
                "text='to their quarters'"
            ),
        ],
        "test-model",
        noise_dictionary=noise_dictionary,
        glossary_entries={},
        metrics=metrics,
    )

    assert result is not None

    group_metric = metrics.hybrid_groups[0]

    assert group_metric.result == "success"

    assert len(
        group_metric.attempts
    ) == 2

    first_attempt = group_metric.attempts[0]
    second_attempt = group_metric.attempts[1]

    assert (
        first_attempt.validation_stage
        == "standard_validation"
    )

    assert first_attempt.validation_valid is False

    assert first_attempt.reason_codes == (
        "glossary_violation",
    )

    assert second_attempt.validation_stage == "complete"
    assert second_attempt.validation_valid is True


def test_validate_hybrid_response_requires_ocr_placeholder(
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
    ] = "各自の居住区へ戻り、"

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
        "Hybrid OCR placeholder missing: "
        "subtitle_id='282', "
        f"required={HYBRID_OCR_PLACEHOLDER!r}"
        in validation.reasons
    )


def test_e07_mixed_ocr_subtitle_accepts_placeholder_and_translation(
) -> None:
    blocks = [
        SrtBlock(
            number="562",
            timestamp=(
                "00:00:00,000 --> "
                "00:00:01,000"
            ),
            text="sa",
        ),
        SrtBlock(
            number="563",
            timestamp=(
                "00:00:01,100 --> "
                "00:00:02,000"
            ),
            text=(
                "aR at-lacmanl-e\n"
                "lam a good friend."
            ),
        ),
    ]

    group = build_hybrid_translation_group(
        blocks,
        {
            "563",
        },
    )

    assert group is not None

    payload = {
        "group": {
            "full_translation": (
                "サ"
                "（判読不能）"
                "私は良い友人です。"
            ),
            "segments": {
                "562": "サ",
                "563": (
                    "（判読不能）"
                    "私は良い友人です。"
                ),
            },
        },
    }

    validation = validate_hybrid_response(
        json.dumps(
            payload,
            ensure_ascii=False,
        ),
        group,
        {
            "563": [
                "aR at-lacmanl-e",
            ],
        },
    )

    assert validation.valid is True
    assert validation.reasons == ()

    assert validation.segments[
               "563"
           ] == (
               "（判読不能）"
               "私は良い友人です。"
           )


# Hybrid LLM生成例外
def test_hybrid_generation_exception_is_recorded_and_raised(
    monkeypatch: pytest.MonkeyPatch,
    target_blocks: list[SrtBlock],
    noise_dictionary: NoiseDictionary,
) -> None:
    def raise_generation_error(
        *args,
        **kwargs,
    ) -> str:
        raise RuntimeError(
            "Hybrid generation failed"
        )

    monkeypatch.setattr(
        hybrid_recovery,
        "generate",
        raise_generation_error,
    )

    metrics = build_chunk_metrics()

    with pytest.raises(
        RuntimeError,
        match="Hybrid generation failed",
    ):
        recover_translation_with_hybrid(
            target_blocks,
            [
                "状況は制御下です。",
                "to their quarters",
                "そこに留まってください。",
            ],
            [
                (
                    "Untranslated English sentence "
                    "detected: subtitle_id='282', "
                    "text='to their quarters'"
                ),
            ],
            "test-model",
            noise_dictionary=noise_dictionary,
            glossary_entries={},
            metrics=metrics,
        )

    assert len(
        metrics.hybrid_groups
    ) == 1

    group_metric = metrics.hybrid_groups[0]

    assert group_metric.result == "failed"

    assert len(
        group_metric.attempts
    ) == 1

    attempt_metric = group_metric.attempts[0]

    assert attempt_metric.pipeline == "hybrid"
    assert attempt_metric.attempt == 1
    assert attempt_metric.response_received is False

    assert (
        attempt_metric.validation_stage
        == "generation_exception"
    )

    assert attempt_metric.validation_valid is None

    assert (
        attempt_metric.exception_type
        == "RuntimeError"
    )

    assert (
        attempt_metric.exception_message
        == "Hybrid generation failed"
    )

    assert attempt_metric.elapsed_seconds >= 0


def test_hybrid_recovery_raises_final_hybrid_error(
    monkeypatch: pytest.MonkeyPatch,
    target_blocks: list[SrtBlock],
    noise_dictionary: NoiseDictionary,
) -> None:
    invalid_response = json.dumps(
        {
            "group": {
                "full_translation": "判読不能",
                "segments": {
                    "281": "状況は制御下です。",
                    "282": "居住区へ戻り、",
                    "283": "そこに留まってください。",
                },
            },
        },
        ensure_ascii=False,
    )

    monkeypatch.setattr(
        hybrid_recovery,
        "generate",
        lambda *args, **kwargs: (
            invalid_response
        ),
    )

    monkeypatch.setattr(
        hybrid_recovery,
        "try_save_hybrid_attempt_report",
        lambda **kwargs: None,
    )

    metrics = build_chunk_metrics()

    with pytest.raises(
        HybridRecoveryError,
        match=(
            "Hybrid Translation Recovery failed "
            "after 3 attempts"
        ),
    ) as captured:
        recover_translation_with_hybrid(
            target_blocks,
            [
                "状況は制御下です。",
                "to their quarters",
                "そこに留まってください。",
            ],
            [
                (
                    "Untranslated English sentence "
                    "detected: subtitle_id='282', "
                    "text='to their quarters'"
                ),
            ],
            "test-model",
            noise_dictionary=noise_dictionary,
            glossary_entries={},
            metrics=metrics,
        )

    assert (
        "Hybrid OCR placeholder missing:"
        in str(captured.value)
    )

    assert len(
        metrics.hybrid_groups
    ) == 1

    group_metric = metrics.hybrid_groups[0]

    assert group_metric.result == "failed"

    assert len(
        group_metric.attempts
    ) == 3

    assert [
               attempt.attempt
               for attempt in group_metric.attempts
           ] == [
               1,
               2,
               3,
           ]

    assert all(
        attempt.pipeline == "hybrid"
        for attempt in group_metric.attempts
    )

    assert all(
        attempt.response_received is True
        for attempt in group_metric.attempts
    )

    assert all(
        (
            attempt.validation_stage
            == "hybrid_validation"
        )
        for attempt in group_metric.attempts
    )

    assert all(
        attempt.validation_valid is False
        for attempt in group_metric.attempts
    )

    assert all(
        attempt.elapsed_seconds >= 0
        for attempt in group_metric.attempts
    )


def test_validate_hybrid_response_requires_translation_for_mixed_ocr_text(
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
    ] = HYBRID_OCR_PLACEHOLDER

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
            "Hybrid mixed OCR segment "
            "requires Japanese translation:"
        )
        for reason in validation.reasons
    )


def test_hybrid_recovery_handles_multiple_independent_groups(
    monkeypatch: pytest.MonkeyPatch,
    noise_dictionary: NoiseDictionary,
) -> None:
    target_blocks = [
        SrtBlock(
            number="601",
            timestamp=(
                "00:00:00,000 --> "
                "00:00:01,000"
            ),
            text="Good.",
        ),
        SrtBlock(
            number="602",
            timestamp=(
                "00:00:01,100 --> "
                "00:00:03,000"
            ),
            text=(
                "Hopefully, we've proven\n"
                "that's not our goal."
            ),
        ),
        SrtBlock(
            number="603",
            timestamp=(
                "00:00:03,100 --> "
                "00:00:04,000"
            ),
            text="I'm sorry.",
        ),
        SrtBlock(
            number="604",
            timestamp=(
                "00:00:06,000 --> "
                "00:00:07,000"
            ),
            text="I couldn't deal with it,",
        ),
        SrtBlock(
            number="605",
            timestamp=(
                "00:00:07,100 --> "
                "00:00:09,000"
            ),
            text=(
                "the thought of you\n"
                "being trapped on that ship."
            ),
        ),
    ]

    responses = iter(
        [
            json.dumps(
                {
                    "group": {
                        "full_translation": (
                            "それが目的ではないと"
                            "証明できたはずです。"
                        ),
                        "segments": {
                            "602": (
                                "それが目的ではないと"
                                "証明できたはずです。"
                            ),
                        },
                    },
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "group": {
                        "full_translation": (
                            "あなたが船に閉じ込められる"
                            "と思うと耐えられませんでした。"
                        ),
                        "segments": {
                            "604": (
                                "あなたが船に"
                                "閉じ込められると思うと"
                            ),
                            "605": (
                                "耐えられませんでした。"
                            ),
                        },
                    },
                },
                ensure_ascii=False,
            ),
        ]
    )

    monkeypatch.setattr(
        hybrid_recovery,
        "generate",
        lambda *args, **kwargs: next(
            responses
        ),
    )

    monkeypatch.setattr(
        hybrid_recovery,
        "try_save_hybrid_attempt_report",
        lambda **kwargs: None,
    )

    metrics = build_chunk_metrics(
        target_ids=(
            "601",
            "602",
            "603",
            "604",
            "605",
        ),
    )

    result = recover_translation_with_hybrid(
        target_blocks,
        [
            "いいでしょう。",
            (
                "Hopefully, we've proven\n"
                "that's not our goal."
            ),
            "ごめんなさい。",
            "I couldn't deal with it,",
            (
                "あなたが船に"
                "閉じ込められることを。"
            ),
        ],
        [
            (
                "Untranslated English sentence "
                "detected: subtitle_id='602', "
                "text=\"Hopefully, we've proven "
                "that's not our goal.\""
            ),
            (
                "Untranslated English sentence "
                "detected: subtitle_id='604', "
                "text=\"I couldn't deal with it,\""
            ),
        ],
        "test-model",
        noise_dictionary=noise_dictionary,
        glossary_entries={},
        metrics=metrics,
    )

    assert result == [
        "いいでしょう。",
        (
            "それが目的ではないと"
            "証明できたはずです。"
        ),
        "ごめんなさい。",
        (
            "あなたが船に"
            "閉じ込められると思うと"
        ),
        "耐えられませんでした。",
    ]

    assert metrics.hybrid_triggered is True

    assert len(
        metrics.hybrid_groups
    ) == 2

    first_group = metrics.hybrid_groups[0]
    second_group = metrics.hybrid_groups[1]

    assert first_group.group_number == 1
    assert first_group.target_ids == (
        "602",
    )
    assert first_group.failed_ids == (
        "602",
    )
    assert first_group.result == "success"

    assert second_group.group_number == 2
    assert second_group.target_ids == (
        "604",
        "605",
    )
    assert second_group.failed_ids == (
        "604",
    )
    assert second_group.result == "success"

    assert len(first_group.attempts) == 1
    assert len(second_group.attempts) == 1

    assert (
        first_group.attempts[0]
        .validation_stage
        == "complete"
    )

    assert (
        second_group.attempts[0]
        .validation_stage
        == "complete"
    )


def test_e09_repeated_translation_errors_trigger_hybrid_recovery(
    monkeypatch: pytest.MonkeyPatch,
    e09_target_blocks: list[SrtBlock],
    noise_dictionary: NoiseDictionary,
) -> None:
    captured_failed_ids: set[str] = set()
    recovered_translations = list(
        E09_RECOVERED_TRANSLATIONS
    )

    def fake_recover_single_hybrid_group(
        *,
        group: HybridTranslationGroup,
        target_blocks: list[SrtBlock],
        translated_texts: list[str],
        model: str,
        noise_dictionary: NoiseDictionary,
        glossary_entries: object,
        before_context: list[SrtBlock] | None = None,
        after_context: list[SrtBlock] | None = None,
        group_number: int = 1,
        metrics: TranslationChunkMetric | None = None,
    ) -> list[str]:
        del before_context
        del after_context
        del group_number
        del metrics

        captured_failed_ids.update(
            group.failed_ids
        )

        merged_texts = list(
            translated_texts
        )

        for position in group.positions:
            merged_texts[position] = (
                recovered_translations[
                    position
                ]
            )

        return merged_texts

    monkeypatch.setattr(
        hybrid_recovery,
        "recover_single_hybrid_group",
        fake_recover_single_hybrid_group,
    )

    repeated_translation_error = (
        "Repeated translation detected: "
        "count=10, "
        "text='では、容疑者はいますか？', "
        "subtitle_ids="
        "['81', '82', '83', '84', '85', "
        "'86', '87', '88', '89', '90']"
    )

    repeated_sequence_error = (
        "Repeated translation sequence detected: "
        "first_start=1, "
        "second_start=4, "
        "length=3, "
        "subtitle_ids="
        "['81', '82', '83', '84', '85', "
        "'86']"
    )

    result = recover_translation_with_hybrid(
        e09_target_blocks,
        [
            E09_REPEATED_TRANSLATION
            for _ in e09_target_blocks
        ],
        [
            repeated_translation_error,
            repeated_sequence_error,
        ],
        "test-model",
        noise_dictionary=noise_dictionary,
        glossary_entries={},
    )

    assert captured_failed_ids == {
        str(number)
        for number in range(
            81,
            91,
        )
    }

    assert result == (
        recovered_translations
    )


def test_e11_sound_effect_is_not_classified_as_ocr(
    ocr_scoring_config: OcrScoringConfig,
) -> None:
    blocks = [
        SrtBlock(
            number="321",
            timestamp=(
                "00:24:14,119 --> "
                "00:24:15,204"
            ),
            text="(CHIRPING)",
        ),
    ]

    group = build_hybrid_translation_group(
        blocks,
        {
            "321",
        },
    )

    assert group is not None

    ocr_lines = find_group_ocr_lines(
        group,
        glossary_entries=(
            GlossaryEntries(())
        ),
        scoring_config=(
            ocr_scoring_config
        ),
    )

    assert ocr_lines == {}


def test_e11_sound_effect_source_payload(
    ocr_scoring_config: OcrScoringConfig,
) -> None:
    blocks = [
        SrtBlock(
            number="321",
            timestamp=(
                "00:24:14,119 --> "
                "00:24:15,204"
            ),
            text="(CHIRPING)",
        ),
    ]

    group = build_hybrid_translation_group(
        blocks,
        {
            "321",
        },
    )

    assert group is not None

    ocr_lines = find_group_ocr_lines(
        group,
        glossary_entries=(
            GlossaryEntries(())
        ),
        scoring_config=(
            ocr_scoring_config
        ),
    )

    payload = build_hybrid_source_payload(
        group,
        ocr_lines,
    )

    assert payload == {
        "subtitles": [
            {
                "id": "321",
                "lines": [
                    {
                        "kind": "sound_effect",
                        "text": "(CHIRPING)",
                    },
                ],
            },
        ],
    }


def test_e11_sound_effect_prompt_excludes_ocr_example(
    ocr_scoring_config: OcrScoringConfig,
) -> None:
    blocks = [
        SrtBlock(
            number="321",
            timestamp=(
                "00:24:14,119 --> "
                "00:24:15,204"
            ),
            text="(CHIRPING)",
        ),
    ]

    group = build_hybrid_translation_group(
        blocks,
        {
            "321",
        },
    )

    assert group is not None

    ocr_lines = find_group_ocr_lines(
        group,
        glossary_entries=(
            GlossaryEntries(())
        ),
        scoring_config=(
            ocr_scoring_config
        ),
    )

    assert ocr_lines == {}

    prompt = build_hybrid_translation_prompt(
        group,
        ocr_lines,
        {},
    )

    assert (
        '"kind": "sound_effect"'
        in prompt
    )

    assert "(CHIRPING)" in prompt
    assert "【効果音行】" in prompt
    assert "（電子音）" in prompt

    assert "【OCR行】" not in prompt

    assert (
        HYBRID_OCR_PLACEHOLDER
        not in prompt
    )

    assert (
        "aR at-lacmanl-e"
        not in prompt
    )

    assert (
        "私は良い友人です"
        not in prompt
    )


def test_mixed_sound_effect_and_text_prompt_requirements(
    ocr_scoring_config: OcrScoringConfig,
) -> None:
    blocks = [
        SrtBlock(
            number="160",
            timestamp=(
                "00:10:00,000 --> "
                "00:10:02,000"
            ),
            text=(
                "(ON RADIO)\n"
                "Colonel Young, come in."
            ),
        ),
    ]

    group = build_hybrid_translation_group(
        blocks,
        {
            "160",
        },
    )

    assert group is not None

    ocr_lines = find_group_ocr_lines(
        group,
        glossary_entries=(
            GlossaryEntries(())
        ),
        scoring_config=(
            ocr_scoring_config
        ),
    )

    prompt = build_hybrid_translation_prompt(
        group,
        ocr_lines,
        {},
    )

    assert (
        "* 字幕ID 160: "
        "kind=textの正常英文を"
        "自然な日本語へ翻訳する。"
        "英文を残さない。"
        "kind=sound_effectを"
        "短い日本語の効果音へ翻訳し、"
        "その部分を全角括弧で囲む。"
        "各行の内容を原文順に配置する。"
        in prompt
    )

    assert (
        '"kind": "sound_effect"'
        in prompt
    )

    assert '"kind": "text"' in prompt
    assert "【効果音行】" in prompt
    assert "【OCR行】" not in prompt


def test_find_group_sound_effect_lines(
) -> None:
    blocks = [
        SrtBlock(
            number="160",
            timestamp=(
                "00:10:00,000 --> "
                "00:10:02,000"
            ),
            text=(
                "(ON RADIO)\n"
                "Colonel Young, come in."
            ),
        ),
        SrtBlock(
            number="161",
            timestamp=(
                "00:10:02,100 --> "
                "00:10:03,000"
            ),
            text="(CHIRPING)",
        ),
    ]

    group = HybridTranslationGroup(
        positions=(
            0,
            1,
        ),
        blocks=tuple(
            blocks
        ),
        failed_ids=frozenset(
            {
                "160",
                "161",
            }
        ),
    )

    actual = find_group_sound_effect_lines(
        group
    )

    assert actual == {
        "160": [
            "(ON RADIO)",
        ],
        "161": [
            "(CHIRPING)",
        ],
    }


def build_e11_sound_effect_group(
) -> HybridTranslationGroup:
    block = SrtBlock(
        number="321",
        timestamp=(
            "00:24:14,119 --> "
            "00:24:15,204"
        ),
        text="(CHIRPING)",
    )

    return HybridTranslationGroup(
        positions=(
            0,
        ),
        blocks=(
            block,
        ),
        failed_ids=frozenset(
            {
                "321",
            }
        ),
    )


def test_e11_validation_accepts_japanese_sound_effect(
) -> None:
    group = build_e11_sound_effect_group()

    response = json.dumps(
        {
            "group": {
                "full_translation": (
                    "（電子音）"
                ),
                "segments": {
                    "321": "（電子音）",
                },
            },
        },
        ensure_ascii=False,
    )

    validation = validate_hybrid_response(
        response,
        group,
        {},
    )

    assert validation.valid is True
    assert validation.reasons == ()

    assert validation.segments == {
        "321": "（電子音）",
    }


def test_e11_validation_rejects_ocr_placeholder_and_hallucinated_dialogue(
) -> None:
    group = build_e11_sound_effect_group()

    invalid_translation = (
        "（判読不能）／私は良い友人です。"
    )

    response = json.dumps(
        {
            "group": {
                "full_translation": (
                    invalid_translation
                ),
                "segments": {
                    "321": invalid_translation,
                },
            },
        },
        ensure_ascii=False,
    )

    validation = validate_hybrid_response(
        response,
        group,
        {},
    )

    assert validation.valid is False

    assert any(
        reason.startswith(
            "Unexpected Hybrid OCR placeholder: "
            "subtitle_id='321'"
        )
        for reason in validation.reasons
    )

    assert any(
        reason.startswith(
            "Hybrid sound-effect-only segment "
            "must contain only fullwidth "
            "parenthesized effects: "
            "subtitle_id='321'"
        )
        for reason in validation.reasons
    )


def test_e11_validation_rejects_unparenthesized_sound_effect(
) -> None:
    group = build_e11_sound_effect_group()

    response = json.dumps(
        {
            "group": {
                "full_translation": "電子音",
                "segments": {
                    "321": "電子音",
                },
            },
        },
        ensure_ascii=False,
    )

    validation = validate_hybrid_response(
        response,
        group,
        {},
    )

    assert validation.valid is False

    assert (
        "Hybrid sound effect translation missing: "
        "subtitle_id='321', "
        "text='電子音'"
        in validation.reasons
    )


def test_e11_validation_rejects_sound_effect_source_copy(
) -> None:
    group = build_e11_sound_effect_group()

    response = json.dumps(
        {
            "group": {
                "full_translation": (
                    "(CHIRPING)"
                ),
                "segments": {
                    "321": "(CHIRPING)",
                },
            },
        },
        ensure_ascii=False,
    )

    validation = validate_hybrid_response(
        response,
        group,
        {},
    )

    assert validation.valid is False

    assert (
        "Hybrid segment contains sound "
        "effect source: "
        "subtitle_id='321', "
        "text='(CHIRPING)'"
        in validation.reasons
    )
    assert (
        "Hybrid sound effect requires "
        "Japanese translation: "
        "subtitle_id='321', "
        "values=['CHIRPING'], "
        "text='（CHIRPING）'"
        in validation.reasons
    )


def test_e11_validation_rejects_english_in_fullwidth_parentheses(
) -> None:
    group = build_e11_sound_effect_group()

    response = json.dumps(
        {
            "group": {
                "full_translation": (
                    "（CHIRPING）"
                ),
                "segments": {
                    "321": "（CHIRPING）",
                },
            },
        },
        ensure_ascii=False,
    )

    validation = validate_hybrid_response(
        response,
        group,
        {},
    )

    assert validation.valid is False

    assert (
        "Hybrid sound effect requires "
        "Japanese translation: "
        "subtitle_id='321', "
        "values=['CHIRPING'], "
        "text='（CHIRPING）'"
        in validation.reasons
    )


def test_validation_accepts_mixed_sound_effect_and_text(
) -> None:
    block = SrtBlock(
        number="160",
        timestamp=(
            "00:10:00,000 --> "
            "00:10:02,000"
        ),
        text=(
            "(ON RADIO)\n"
            "Colonel Young, come in."
        ),
    )

    group = HybridTranslationGroup(
        positions=(
            0,
        ),
        blocks=(
            block,
        ),
        failed_ids=frozenset(
            {
                "160",
            }
        ),
    )

    translated_text = (
        "（無線）ヤング大佐、"
        "応答してください。"
    )

    response = json.dumps(
        {
            "group": {
                "full_translation": (
                    translated_text
                ),
                "segments": {
                    "160": translated_text,
                },
            },
        },
        ensure_ascii=False,
    )

    validation = validate_hybrid_response(
        response,
        group,
        {},
    )

    assert validation.valid is True
    assert validation.reasons == ()


def test_validation_rejects_missing_effect_in_mixed_subtitle(
) -> None:
    block = SrtBlock(
        number="160",
        timestamp=(
            "00:10:00,000 --> "
            "00:10:02,000"
        ),
        text=(
            "(ON RADIO)\n"
            "Colonel Young, come in."
        ),
    )

    group = HybridTranslationGroup(
        positions=(
            0,
        ),
        blocks=(
            block,
        ),
        failed_ids=frozenset(
            {
                "160",
            }
        ),
    )

    translated_text = (
        "ヤング大佐、応答してください。"
    )

    response = json.dumps(
        {
            "group": {
                "full_translation": (
                    translated_text
                ),
                "segments": {
                    "160": translated_text,
                },
            },
        },
        ensure_ascii=False,
    )

    validation = validate_hybrid_response(
        response,
        group,
        {},
    )

    assert validation.valid is False

    assert (
        "Hybrid sound effect translation "
        "missing: "
        "subtitle_id='160', "
        f"text={translated_text!r}"
        in validation.reasons
    )


def test_e11_hybrid_recovery_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    noise_dictionary: NoiseDictionary,
) -> None:
    target_blocks = [
        SrtBlock(
            number="321",
            timestamp=(
                "00:24:14,119 --> "
                "00:24:15,204"
            ),
            text="(CHIRPING)",
        ),
    ]

    original_translated_texts = [
        "(CHIRPING)",
    ]

    errors = [
        (
            "Untranslated English sentence "
            "detected: subtitle_id='321', "
            "text='(CHIRPING)'"
        ),
    ]

    generated_requests: list[
        dict[str, object]
    ] = []

    saved_reports: list[
        dict[str, object]
    ] = []

    response = json.dumps(
        {
            "group": {
                "full_translation": (
                    "（電子音）"
                ),
                "segments": {
                    "321": "（電子音）",
                },
            },
        },
        ensure_ascii=False,
    )

    def fake_generate(
        prompt: str,
        *,
        model: str,
        response_format: dict[
            str,
            object,
        ],
    ) -> str:
        generated_requests.append(
            {
                "prompt": prompt,
                "model": model,
                "response_format": (
                    response_format
                ),
            }
        )

        return response

    def fake_save_report(
        **kwargs: object,
    ) -> None:
        saved_reports.append(
            dict(
                kwargs
            )
        )

    monkeypatch.setattr(
        hybrid_recovery,
        "generate",
        fake_generate,
    )

    monkeypatch.setattr(
        hybrid_recovery,
        "try_save_hybrid_attempt_report",
        fake_save_report,
    )

    result = recover_translation_with_hybrid(
        target_blocks,
        original_translated_texts,
        errors,
        "test-model",
        noise_dictionary=noise_dictionary,
        glossary_entries={},
    )

    assert result == [
        "（電子音）",
    ]

    assert original_translated_texts == [
        "(CHIRPING)",
    ]

    assert len(
        generated_requests
    ) == 1

    generated_request = (
        generated_requests[0]
    )

    assert (
        generated_request[
            "model"
        ]
        == "test-model"
    )

    prompt = generated_request[
        "prompt"
    ]

    assert isinstance(
        prompt,
        str,
    )

    assert (
        '"kind": "sound_effect"'
        in prompt
    )

    assert "(CHIRPING)" in prompt
    assert "【効果音行】" in prompt
    assert "（電子音）" in prompt

    assert "【OCR行】" not in prompt

    assert (
        HYBRID_OCR_PLACEHOLDER
        not in prompt
    )

    assert (
        "aR at-lacmanl-e"
        not in prompt
    )

    assert (
        "私は良い友人です"
        not in prompt
    )

    assert len(
        saved_reports
    ) == 1

    saved_report = (
        saved_reports[0]
    )

    assert (
        saved_report[
            "validation_stage"
        ]
        == "complete"
    )

    assert (
        saved_report[
            "validation_valid"
        ]
        is True
    )

    assert (
        saved_report[
            "validation_reasons"
        ]
        == []
    )


def build_e13_mixed_ocr_group(
) -> HybridTranslationGroup:
    block = SrtBlock(
        number="490",
        timestamp=(
            "00:35:00,000 --> "
            "00:35:02,000"
        ),
        text=E13_SOURCE_TEXT,
    )

    return HybridTranslationGroup(
        positions=(
            0,
        ),
        blocks=(
            block,
        ),
        failed_ids=frozenset(
            {
                "490",
            }
        ),
    )


def test_e13_short_mixed_case_line_is_classified_as_ocr(
    ocr_scoring_config: OcrScoringConfig,
) -> None:
    group = build_e13_mixed_ocr_group()

    actual = find_group_ocr_lines(
        group,
        glossary_entries=(
            GlossaryEntries(())
        ),
        scoring_config=(
            ocr_scoring_config
        ),
    )

    assert actual == {
        "490": [
            E13_SHORT_OCR_LINE,
        ],
    }


def test_e13_normal_line_is_not_classified_as_ocr(
    ocr_scoring_config: OcrScoringConfig,
) -> None:
    group = build_e13_mixed_ocr_group()

    ocr_lines = find_group_ocr_lines(
        group,
        glossary_entries=(
            GlossaryEntries(())
        ),
        scoring_config=(
            ocr_scoring_config
        ),
    )

    assert (
        E13_NORMAL_LINE
        not in ocr_lines.get(
        "490",
        [],
    )
    )


def test_short_mixed_case_line_without_normal_sibling_is_not_hybrid_ocr(
    ocr_scoring_config: OcrScoringConfig,
) -> None:
    block = SrtBlock(
        number="490",
        timestamp=(
            "00:35:00,000 --> "
            "00:35:02,000"
        ),
        text=E13_SHORT_OCR_LINE,
    )

    group = HybridTranslationGroup(
        positions=(
            0,
        ),
        blocks=(
            block,
        ),
        failed_ids=frozenset(
            {
                "490",
            }
        ),
    )

    actual = find_group_ocr_lines(
        group,
        glossary_entries=(
            GlossaryEntries(())
        ),
        scoring_config=(
            ocr_scoring_config
        ),
    )

    assert actual == {}


def test_short_mixed_case_line_in_non_failed_subtitle_is_not_hybrid_ocr(
    ocr_scoring_config: OcrScoringConfig,
) -> None:
    block = SrtBlock(
        number="490",
        timestamp=(
            "00:35:00,000 --> "
            "00:35:02,000"
        ),
        text=E13_SOURCE_TEXT,
    )

    group = HybridTranslationGroup(
        positions=(
            0,
        ),
        blocks=(
            block,
        ),
        failed_ids=frozenset(),
    )

    actual = find_group_ocr_lines(
        group,
        glossary_entries=(
            GlossaryEntries(())
        ),
        scoring_config=(
            ocr_scoring_config
        ),
    )

    assert actual == {}


def test_e13_hybrid_source_payload_separates_text_and_ocr(
    ocr_scoring_config: OcrScoringConfig,
) -> None:
    group = build_e13_mixed_ocr_group()

    ocr_lines = find_group_ocr_lines(
        group,
        glossary_entries=(
            GlossaryEntries(())
        ),
        scoring_config=(
            ocr_scoring_config
        ),
    )

    payload = build_hybrid_source_payload(
        group,
        ocr_lines,
    )

    assert payload == {
        "subtitles": [
            {
                "id": "490",
                "lines": [
                    {
                        "kind": "text",
                        "text": E13_NORMAL_LINE,
                    },
                    {
                        "kind": "ocr",
                        "text": E13_SHORT_OCR_LINE,
                    },
                ],
            },
        ],
    }


def test_e13_hybrid_prompt_requires_translation_and_ocr_placeholder(
    ocr_scoring_config: OcrScoringConfig,
) -> None:
    group = build_e13_mixed_ocr_group()

    ocr_lines = find_group_ocr_lines(
        group,
        glossary_entries=(
            GlossaryEntries(())
        ),
        scoring_config=(
            ocr_scoring_config
        ),
    )

    prompt = build_hybrid_translation_prompt(
        group,
        ocr_lines,
        {},
    )

    assert (
        '"kind": "text"'
        in prompt
    )

    assert (
        f'"text": "{E13_NORMAL_LINE}"'
        in prompt
    )

    assert (
        '"kind": "ocr"'
        in prompt
    )

    assert (
        f'"text": "{E13_SHORT_OCR_LINE}"'
        in prompt
    )

    assert (
        "* 字幕ID 490: "
        "kind=textの正常英文を"
        "自然な日本語へ翻訳する。"
        "英文を残さない。"
        "kind=ocrの位置を"
        "「（判読不能）」で表現し、"
        "OCR原文をコピーしない。"
        "segmentには"
        "「（判読不能）」と、"
        "それ以外の翻訳結果を"
        "両方とも含める。"
        "各行の内容を原文順に配置する。"
        in prompt
    )


def valid_e13_hybrid_payload(
) -> dict[str, object]:
    segments = {
        "490": (
            "では、何について？"
            "（判読不能）"
        ),
    }

    return {
        "group": {
            "full_translation": (
                segments["490"]
            ),
            "segments": segments,
        },
    }


def test_e13_hybrid_recovery_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    noise_dictionary: NoiseDictionary,
) -> None:
    generated_requests: list[
        dict[str, object]
    ] = []

    saved_reports: list[
        dict[str, object]
    ] = []

    response = json.dumps(
        valid_e13_hybrid_payload(),
        ensure_ascii=False,
    )

    def fake_generate(
        prompt: str,
        *,
        model: str,
        response_format: dict[str, object],
    ) -> str:
        generated_requests.append(
            {
                "prompt": prompt,
                "model": model,
                "response_format": response_format,
            }
        )

        return response

    def fake_save_report(
        **kwargs: object,
    ) -> None:
        saved_reports.append(
            dict(
                kwargs
            )
        )

    monkeypatch.setattr(
        hybrid_recovery,
        "generate",
        fake_generate,
    )

    monkeypatch.setattr(
        hybrid_recovery,
        "try_save_hybrid_attempt_report",
        fake_save_report,
    )

    target_blocks = [
        SrtBlock(
            number="490",
            timestamp=(
                "00:35:00,000 --> "
                "00:35:02,000"
            ),
            text=E13_SOURCE_TEXT,
        ),
    ]

    previous_texts = [
        E13_SOURCE_TEXT,
    ]

    original_previous_texts = list(
        previous_texts
    )

    errors = [
        (
            "Untranslated English sentence detected: "
            "subtitle_id='490', "
            "text='Okay, what about\\n"
            f"{E13_SHORT_OCR_LINE}'"
        ),
    ]

    result = recover_translation_with_hybrid(
        target_blocks,
        previous_texts,
        errors,
        "test-model",
        noise_dictionary=noise_dictionary,
        glossary_entries={},
    )

    assert result == [
        (
            "では、何について？"
            "（判読不能）"
        ),
    ]

    assert previous_texts == (
        original_previous_texts
    )

    assert len(
        generated_requests
    ) == 1

    generated_request = (
        generated_requests[0]
    )

    assert (
        generated_request["model"]
        == "test-model"
    )

    prompt = generated_request[
        "prompt"
    ]

    assert isinstance(
        prompt,
        str,
    )

    assert (
        '"kind": "text",\n'
        f'          "text": "{E13_NORMAL_LINE}"'
        in prompt
    )

    assert (
        '"kind": "ocr",\n'
        f'          "text": "{E13_SHORT_OCR_LINE}"'
        in prompt
    )

    assert (
        "* 字幕ID 490: "
        "kind=textの正常英文を"
        "自然な日本語へ翻訳する。"
        "英文を残さない。"
        "kind=ocrの位置を"
        "「（判読不能）」で表現し、"
        "OCR原文をコピーしない。"
        "segmentには"
        "「（判読不能）」と、"
        "それ以外の翻訳結果を"
        "両方とも含める。"
        "各行の内容を原文順に配置する。"
        in prompt
    )

    response_format = generated_request[
        "response_format"
    ]

    assert isinstance(
        response_format,
        dict,
    )

    required_ids = (
        response_format[
            "properties"
        ][
            "group"
        ][
            "properties"
        ][
            "segments"
        ][
            "required"
        ]
    )

    assert required_ids == [
        "490",
    ]

    assert len(
        saved_reports
    ) == 1

    saved_report = (
        saved_reports[0]
    )

    assert (
        saved_report[
            "validation_stage"
        ]
        == "complete"
    )

    assert (
        saved_report[
            "validation_valid"
        ]
        is True
    )

    assert (
        saved_report[
            "validation_reasons"
        ]
        == []
    )

    assert (
        saved_report[
            "ocr_lines"
        ]
        == {
            "490": [
                E13_SHORT_OCR_LINE,
            ],
        }
    )


def test_e15_hybrid_recovery_accepts_ambiguous_japanese_text(
    monkeypatch: pytest.MonkeyPatch,
    noise_dictionary: NoiseDictionary,
) -> None:
    generated_requests: list[
        dict[str, object]
    ] = []

    saved_reports: list[
        dict[str, object]
    ] = []

    target_blocks = [
        SrtBlock(
            number="136",
            timestamp=(
                "00:08:00,000 --> "
                "00:08:03,000"
            ),
            text=(
                "This circle represents\n"
                "the gates within range\n"
                "of Destiny"
            ),
        ),
        SrtBlock(
            number="137",
            timestamp=(
                "00:08:03,100 --> "
                "00:08:05,000"
            ),
            text=(
                "Next time we drop\n"
                "out of FTL,"
            ),
        ),
    ]

    segments = {
        "136": (
            "この円は、デスティニーの範囲内にある"
            "ゲートを表しています。"
        ),
        "137": (
            "次にFTLから離脱するとき、"
        ),
    }

    response = json.dumps(
        {
            "group": {
                "full_translation": (
                    segments["136"]
                    + segments["137"]
                ),
                "segments": segments,
            },
        },
        ensure_ascii=False,
    )

    def fake_generate(
        prompt: str,
        *,
        model: str,
        response_format: dict[str, object],
    ) -> str:
        generated_requests.append(
            {
                "prompt": prompt,
                "model": model,
                "response_format": response_format,
            }
        )

        return response

    def fake_save_report(
        **kwargs: object,
    ) -> None:
        saved_reports.append(
            dict(
                kwargs
            )
        )

    monkeypatch.setattr(
        hybrid_recovery,
        "generate",
        fake_generate,
    )

    monkeypatch.setattr(
        hybrid_recovery,
        "try_save_hybrid_attempt_report",
        fake_save_report,
    )

    previous_texts = [
        (
            "この円は、デスティニーの範囲内にある"
            "ゲートを表しています。"
        ),
        (
            "次にFTLから離脱するとき、"
        ),
    ]

    original_previous_texts = list(
        previous_texts
    )

    errors = [
        (
            "Chinese-specific characters detected: "
            "subtitle_id='136', "
            "characters='内', "
            "text='この円は、デスティニーの範囲内にある"
            "ゲートを表しています。'"
        ),
    ]

    result = recover_translation_with_hybrid(
        target_blocks,
        previous_texts,
        errors,
        "test-model",
        noise_dictionary=noise_dictionary,
        glossary_entries={},
    )

    assert result == [
        segments["136"],
        segments["137"],
    ]

    assert previous_texts == (
        original_previous_texts
    )

    assert len(
        generated_requests
    ) == 1

    generated_request = (
        generated_requests[0]
    )

    assert (
        generated_request["model"]
        == "test-model"
    )

    prompt = generated_request[
        "prompt"
    ]

    assert isinstance(
        prompt,
        str,
    )

    assert (
        '"id": "136"'
        in prompt
    )

    assert (
        '"id": "137"'
        in prompt
    )

    assert (
        '"kind": "ocr"'
        not in prompt
    )

    response_format = generated_request[
        "response_format"
    ]

    assert isinstance(
        response_format,
        dict,
    )

    required_ids = (
        response_format[
            "properties"
        ][
            "group"
        ][
            "properties"
        ][
            "segments"
        ][
            "required"
        ]
    )

    assert required_ids == [
        "136",
        "137",
    ]

    assert len(
        saved_reports
    ) == 1

    saved_report = (
        saved_reports[0]
    )

    assert (
        saved_report[
            "validation_stage"
        ]
        == "complete"
    )

    assert (
        saved_report[
            "validation_valid"
        ]
        is True
    )

    assert (
        saved_report[
            "validation_reasons"
        ]
        == []
    )

    assert (
        saved_report[
            "ocr_lines"
        ]
        == {}
    )


def test_e15_hybrid_recovery_handles_ocr_and_ambiguous_text(
    monkeypatch: pytest.MonkeyPatch,
    noise_dictionary: NoiseDictionary,
) -> None:
    generated_requests: list[
        dict[str, object]
    ] = []

    saved_reports: list[
        dict[str, object]
    ] = []

    ocr_line = (
        "oX=¥AN(o1) 0 MUA L= S310] KO (otoe"
    )

    source_text = (
        "Now, hopefully, there is a gate\n"
        "within range of each one\n"
        f"{ocr_line}"
    )


def build_chunk_metrics(
    *,
    target_ids: tuple[str, ...] = (
            "281",
            "282",
            "283",
    ),
) -> TranslationChunkMetric:
    return TranslationChunkMetric(
        chunk_number=1,
        chunk_start=1,
        chunk_end=len(
            target_ids
        ),
        target_ids=target_ids,
        started_at=datetime(
            2026,
            7,
            19,
            12,
            0,
            0,
        ),
    )

    target_blocks = [
        SrtBlock(
            number="140",
            timestamp=(
                "00:08:09,000 --> "
                "00:08:12,000"
            ),
            text=source_text,
        ),
    ]

    recovered_translation = (
        "現在、幸いにも、それぞれの範囲内にある"
        "ゲートが存在することを願っています。"
        "（判読不能）"
    )

    response = json.dumps(
        {
            "group": {
                "full_translation": (
                    recovered_translation
                ),
                "segments": {
                    "140": (
                        recovered_translation
                    ),
                },
            },
        },
        ensure_ascii=False,
    )

    def fake_generate(
        prompt: str,
        *,
        model: str,
        response_format: dict[str, object],
    ) -> str:
        generated_requests.append(
            {
                "prompt": prompt,
                "model": model,
                "response_format": response_format,
            }
        )

        return response

    def fake_save_report(
        **kwargs: object,
    ) -> None:
        saved_reports.append(
            dict(
                kwargs
            )
        )

    monkeypatch.setattr(
        hybrid_recovery,
        "generate",
        fake_generate,
    )

    monkeypatch.setattr(
        hybrid_recovery,
        "try_save_hybrid_attempt_report",
        fake_save_report,
    )

    previous_texts = [
        (
            "現在、幸いにも、それぞれの範囲内にある"
            "ゲートが存在することを願っている"
            f"{ocr_line}"
        ),
    ]

    original_previous_texts = list(
        previous_texts
    )

    errors = [
        (
            "Untranslated English sentence detected: "
            "subtitle_id='140', "
            "text='現在、幸いにも、それぞれの範囲内にある"
            "ゲートが存在することを願っている"
            f"{ocr_line}'"
        ),
    ]

    result = recover_translation_with_hybrid(
        target_blocks,
        previous_texts,
        errors,
        "test-model",
        noise_dictionary=noise_dictionary,
        glossary_entries={},
    )

    assert result == [
        recovered_translation,
    ]

    assert previous_texts == (
        original_previous_texts
    )

    assert len(
        generated_requests
    ) == 1

    generated_request = (
        generated_requests[0]
    )

    assert (
        generated_request["model"]
        == "test-model"
    )

    prompt = generated_request[
        "prompt"
    ]

    assert isinstance(
        prompt,
        str,
    )

    assert (
        '"id": "140"'
        in prompt
    )

    assert (
        '"kind": "text",\n'
        '          "text": "Now, hopefully, there is a gate"'
        in prompt
    )

    assert (
        '"kind": "text",\n'
        '          "text": "within range of each one"'
        in prompt
    )

    assert (
        '"kind": "ocr",\n'
        f'          "text": "{ocr_line}"'
        in prompt
    )

    assert (
        "* 字幕ID 140: "
        "kind=textの正常英文を"
        "自然な日本語へ翻訳する。"
        "英文を残さない。"
        "kind=ocrの位置を"
        "「（判読不能）」で表現し、"
        "OCR原文をコピーしない。"
        "segmentには"
        "「（判読不能）」と、"
        "それ以外の翻訳結果を"
        "両方とも含める。"
        "各行の内容を原文順に配置する。"
        in prompt
    )

    assert (
        "範囲内"
        in result[0]
    )

    assert (
        "（判読不能）"
        in result[0]
    )

    assert (
        ocr_line
        not in result[0]
    )

    assert len(
        saved_reports
    ) == 1

    saved_report = (
        saved_reports[0]
    )

    assert (
        saved_report[
            "validation_stage"
        ]
        == "complete"
    )

    assert (
        saved_report[
            "validation_valid"
        ]
        is True
    )

    assert (
        saved_report[
            "validation_reasons"
        ]
        == []
    )

    assert (
        saved_report[
            "ocr_lines"
        ]
        == {
            "140": [
                ocr_line,
            ],
        }
    )


def test_normalize_hybrid_parentheses(
) -> None:
    actual = (
        normalize_hybrid_parentheses(
            "(スコットがうめく音)"
        )
    )

    assert actual == (
        "（スコットがうめく音）"
    )


def test_normalize_hybrid_parentheses_in_mixed_text(
) -> None:
    actual = (
        normalize_hybrid_parentheses(
            (
                "(スコットがうめく音)"
                "大丈夫か？"
            )
        )
    )

    assert actual == (
        "（スコットがうめく音）"
        "大丈夫か？"
    )


def test_normalize_hybrid_parentheses_converts_english_content(
) -> None:
    actual = (
        normalize_hybrid_parentheses(
            "(SG-1)"
        )
    )

    assert actual == "（SG-1）"


def test_normalize_hybrid_parentheses_preserves_fullwidth_parentheses(
) -> None:
    actual = (
        normalize_hybrid_parentheses(
            "（既に全角）"
        )
    )

    assert actual == "（既に全角）"


def test_normalize_hybrid_parentheses_preserves_plain_text(
) -> None:
    actual = (
        normalize_hybrid_parentheses(
            "括弧を含まない字幕です。"
        )
    )

    assert actual == (
        "括弧を含まない字幕です。"
    )


def test_validate_hybrid_response_normalizes_sound_effect_parentheses(
) -> None:
    block = SrtBlock(
        number="43",
        timestamp=(
            "00:02:17,429 --> "
            "00:02:17,804"
        ),
        text="(SCOTT GRUNTING)",
    )

    group = HybridTranslationGroup(
        positions=(
            0,
        ),
        blocks=(
            block,
        ),
        failed_ids=frozenset(
            {
                "43",
            }
        ),
    )

    response = json.dumps(
        {
            "group": {
                "full_translation": (
                    "(スコットがうめく音)"
                ),
                "segments": {
                    "43": (
                        "(スコットがうめく音)"
                    ),
                },
            },
        },
        ensure_ascii=False,
    )

    validation = validate_hybrid_response(
        response,
        group,
        {},
    )

    assert validation.valid is True
    assert validation.reasons == ()

    assert validation.segments == {
        "43": (
            "（スコットがうめく音）"
        ),
    }

    assert validation.full_translation == (
        "（スコットがうめく音）"
    )


def test_e05_validation_normalizes_halfwidth_ocr_placeholder(
) -> None:
    ocr_line = (
        "= PV Fel oael "
        "(ct V a bY ate)’ <16|"
    )

    block = SrtBlock(
        number="98",
        timestamp=(
            "00:04:50,999 --> "
            "00:04:54,961"
        ),
        text=(
            f"{ocr_line}\n"
            "seeing the old homestead\n"
            "again."
        ),
    )

    group = HybridTranslationGroup(
        positions=(
            0,
        ),
        blocks=(
            block,
        ),
        failed_ids=frozenset(
            {
                "98",
            }
        ),
    )

    response = json.dumps(
        {
            "group": {
                "full_translation": (
                    "(判読不能)"
                    "懐かしい我が家を"
                    "再び見られた。"
                ),
                "segments": {
                    "98": (
                        "(判読不能)"
                        "懐かしい我が家を"
                        "再び見られた。"
                    ),
                },
            },
        },
        ensure_ascii=False,
    )

    validation = validate_hybrid_response(
        response,
        group,
        {
            "98": [
                ocr_line,
            ],
        },
    )

    assert validation.valid is True
    assert validation.reasons == ()

    assert validation.segments == {
        "98": (
            "（判読不能）"
            "懐かしい我が家を"
            "再び見られた。"
        ),
    }

    assert validation.full_translation == (
        "（判読不能）"
        "懐かしい我が家を"
        "再び見られた。"
    )


def test_e05_validation_detects_ocr_source_before_parentheses_normalization(
) -> None:
    ocr_line = (
        "= PV Fel oael "
        "(ct V a bY ate)’ <16|"
    )

    block = SrtBlock(
        number="98",
        timestamp=(
            "00:04:50,999 --> "
            "00:04:54,961"
        ),
        text=(
            f"{ocr_line}\n"
            "seeing the old homestead\n"
            "again."
        ),
    )

    group = HybridTranslationGroup(
        positions=(
            0,
        ),
        blocks=(
            block,
        ),
        failed_ids=frozenset(
            {
                "98",
            }
        ),
    )

    response = json.dumps(
        {
            "group": {
                "full_translation": (
                    f"{ocr_line}\n"
                    "(判読不能)"
                    "懐かしい我が家を"
                    "再び見られた。"
                ),
                "segments": {
                    "98": (
                        f"{ocr_line}\n"
                        "(判読不能)"
                        "懐かしい我が家を"
                        "再び見られた。"
                    ),
                },
            },
        },
        ensure_ascii=False,
    )

    validation = validate_hybrid_response(
        response,
        group,
        {
            "98": [
                ocr_line,
            ],
        },
    )

    assert validation.valid is False

    assert (
        "Hybrid segment contains OCR source: "
        "subtitle_id='98', "
        f"text={ocr_line!r}"
        in validation.reasons
    )
