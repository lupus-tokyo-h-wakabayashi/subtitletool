from lib.subtitle.srt import SrtBlock
from lib.translation.hybrid_group import (
    build_hybrid_translation_group,
    build_hybrid_translation_groups,
    crosses_hybrid_time_boundary,
    parse_srt_timestamp_range,
    source_text_ends_sentence,
    subtitle_gap_milliseconds,
)


def make_block(
    number: str,
    text: str,
    timestamp: str = (
        "00:00:00,000 --> "
        "00:00:01,000"
    ),
) -> SrtBlock:
    return SrtBlock(
        number=number,
        timestamp=timestamp,
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


def test_parse_srt_timestamp_range(
) -> None:
    result = parse_srt_timestamp_range(
        "01:02:03,456 --> 01:02:05,789"
    )

    assert result == (
        3_723_456,
        3_725_789,
    )


def test_parse_srt_timestamp_range_rejects_invalid_value(
) -> None:
    assert parse_srt_timestamp_range(
        "invalid timestamp"
    ) is None

    assert parse_srt_timestamp_range(
        "00:00:02,000 --> 00:00:01,000"
    ) is None


def test_subtitle_gap_milliseconds(
) -> None:
    previous_block = make_block(
        "1",
        "First,",
        timestamp=(
            "00:00:00,000 --> "
            "00:00:01,000"
        ),
    )

    next_block = make_block(
        "2",
        "continues.",
        timestamp=(
            "00:00:01,250 --> "
            "00:00:02,000"
        ),
    )

    assert subtitle_gap_milliseconds(
        previous_block,
        next_block,
    ) == 250


def test_overlapping_subtitles_have_zero_gap(
) -> None:
    previous_block = make_block(
        "1",
        "First,",
        timestamp=(
            "00:00:00,000 --> "
            "00:00:02,000"
        ),
    )

    next_block = make_block(
        "2",
        "continues.",
        timestamp=(
            "00:00:01,500 --> "
            "00:00:03,000"
        ),
    )

    assert subtitle_gap_milliseconds(
        previous_block,
        next_block,
    ) == 0


def test_invalid_timestamp_is_hybrid_boundary(
) -> None:
    previous_block = make_block(
        "1",
        "First,",
        timestamp="invalid",
    )

    next_block = make_block(
        "2",
        "continues.",
    )

    assert crosses_hybrid_time_boundary(
        previous_block,
        next_block,
        maximum_gap_milliseconds=1_500,
    ) is True


def test_hybrid_group_does_not_cross_large_gap_backward(
) -> None:
    blocks = [
        make_block(
            "280",
            "Earlier dialogue,",
            timestamp=(
                "00:00:00,000 --> "
                "00:00:01,000"
            ),
        ),
        make_block(
            "281",
            "It's under control,",
            timestamp=(
                "00:00:05,000 --> "
                "00:00:06,000"
            ),
        ),
        make_block(
            "282",
            "to their quarters",
            timestamp=(
                "00:00:06,100 --> "
                "00:00:07,000"
            ),
        ),
        make_block(
            "283",
            "and remain there.",
            timestamp=(
                "00:00:07,100 --> "
                "00:00:08,000"
            ),
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


def test_hybrid_group_does_not_cross_large_gap_forward(
) -> None:
    blocks = [
        make_block(
            "281",
            "It's under control,",
            timestamp=(
                "00:00:00,000 --> "
                "00:00:01,000"
            ),
        ),
        make_block(
            "282",
            "to their quarters",
            timestamp=(
                "00:00:01,100 --> "
                "00:00:02,000"
            ),
        ),
        make_block(
            "283",
            "unrelated dialogue.",
            timestamp=(
                "00:00:05,000 --> "
                "00:00:06,000"
            ),
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
    )


def test_hybrid_group_rejects_failed_ids_across_large_gap(
) -> None:
    blocks = [
        make_block(
            "281",
            "First failure,",
            timestamp=(
                "00:00:00,000 --> "
                "00:00:01,000"
            ),
        ),
        make_block(
            "282",
            "second failure",
            timestamp=(
                "00:00:05,000 --> "
                "00:00:06,000"
            ),
        ),
    ]

    group = build_hybrid_translation_group(
        blocks,
        {
            "281",
            "282",
        },
    )

    assert group is None


def test_build_hybrid_translation_groups_splits_independent_failures(
) -> None:
    blocks = [
        make_block(
            "601",
            "Good.",
            timestamp=(
                "00:00:00,000 --> "
                "00:00:01,000"
            ),
        ),
        make_block(
            "602",
            (
                "Hopefully, we've proven\n"
                "that's not our goal."
            ),
            timestamp=(
                "00:00:01,100 --> "
                "00:00:03,000"
            ),
        ),
        make_block(
            "603",
            "I'm sorry.",
            timestamp=(
                "00:00:03,100 --> "
                "00:00:04,000"
            ),
        ),
        make_block(
            "604",
            "I couldn't deal with it,",
            timestamp=(
                "00:00:06,000 --> "
                "00:00:07,000"
            ),
        ),
        make_block(
            "605",
            (
                "the thought of you\n"
                "being trapped on that ship."
            ),
            timestamp=(
                "00:00:07,100 --> "
                "00:00:09,000"
            ),
        ),
    ]

    groups = build_hybrid_translation_groups(
        blocks,
        {
            "602",
            "604",
        },
    )

    assert len(groups) == 2

    assert groups[0].target_ids == (
        "602",
    )

    assert groups[0].failed_ids == (
        frozenset(
            {
                "602",
            }
        )
    )

    assert groups[1].target_ids == (
        "604",
        "605",
    )

    assert groups[1].failed_ids == (
        frozenset(
            {
                "604",
            }
        )
    )


def test_build_hybrid_translation_groups_merges_overlapping_groups(
) -> None:
    blocks = [
        make_block(
            "281",
            "As a precaution,",
            timestamp=(
                "00:00:00,000 --> "
                "00:00:01,000"
            ),
        ),
        make_block(
            "282",
            "return to your quarters",
            timestamp=(
                "00:00:01,100 --> "
                "00:00:02,000"
            ),
        ),
        make_block(
            "283",
            "and remain there.",
            timestamp=(
                "00:00:02,100 --> "
                "00:00:03,000"
            ),
        ),
    ]

    groups = build_hybrid_translation_groups(
        blocks,
        {
            "282",
            "283",
        },
    )

    assert len(groups) == 1

    assert groups[0].target_ids == (
        "281",
        "282",
        "283",
    )

    assert groups[0].failed_ids == (
        frozenset(
            {
                "282",
                "283",
            }
        )
    )
