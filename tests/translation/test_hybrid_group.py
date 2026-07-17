from lib.subtitle.srt import SrtBlock
from lib.translation.hybrid_group import (
    build_hybrid_translation_group,
    source_text_ends_sentence,
)


def make_block(
    number: str,
    text: str,
) -> SrtBlock:
    return SrtBlock(
        number=number,
        timestamp=(
            "00:00:00,000 --> "
            "00:00:01,000"
        ),
        text=text,
    )


def test_sentence_end_detection(
) -> None:
    assert source_text_ends_sentence(
        "Thank you."
    ) is True

    assert source_text_ends_sentence(
        "Why?"
    ) is True

    assert source_text_ends_sentence(
        "but as a precaution,"
    ) is False

    assert source_text_ends_sentence(
        "to their quarters"
    ) is False


def test_build_hybrid_group_expands_split_sentence(
) -> None:
    blocks = [
        make_block(
            "281",
            (
                "It's under control,\n"
                "but as a precaution,"
            ),
        ),
        make_block(
            "282",
            (
                "=) EWA eam CO ma = Ae lan\n"
                "to their quarters"
            ),
        ),
        make_block(
            "283",
            (
                "and remain there\n"
                "until further notice.\n"
                "Thank you."
            ),
        ),
        make_block(
            "284",
            "What happened?",
        ),
    ]

    group = build_hybrid_translation_group(
        blocks,
        {
            "282",
        },
    )

    assert group is not None

    assert group.target_ids == (
        "281",
        "282",
        "283",
    )

    assert group.positions == (
        0,
        1,
        2,
    )


def test_hybrid_group_does_not_cross_completed_sentence(
) -> None:
    blocks = [
        make_block(
            "280",
            "Everyone is safe.",
        ),
        make_block(
            "281",
            "This continues,",
        ),
        make_block(
            "282",
            "to their quarters",
        ),
        make_block(
            "283",
            "and remain there.",
        ),
        make_block(
            "284",
            "What happened?",
        ),
    ]

    group = build_hybrid_translation_group(
        blocks,
        {
            "282",
        },
    )

    assert group is not None

    assert group.target_ids == (
        "281",
        "282",
        "283",
    )


def test_hybrid_group_rejects_unknown_failed_id(
) -> None:
    blocks = [
        make_block(
            "1",
            "Hello.",
        ),
    ]

    group = build_hybrid_translation_group(
        blocks,
        {
            "999",
        },
    )

    assert group is None


def test_hybrid_group_respects_maximum_size(
) -> None:
    blocks = [
        make_block(
            str(index),
            "unfinished,",
        )
        for index in range(
            1,
            8,
        )
    ]

    group = build_hybrid_translation_group(
        blocks,
        {
            "4",
        },
        maximum_blocks=3,
    )

    assert group is not None
    assert len(group.blocks) == 3
