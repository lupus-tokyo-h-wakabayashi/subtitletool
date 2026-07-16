from __future__ import annotations

from pathlib import Path

from lib.profile.noise import (
    NoiseDictionary,
    NoiseEntry,
)
from lib.subtitle.ocr_inspection import (
    STEP_NOISE_DETECTED,
    STEP_NOISE_DICTIONARY,
    STEP_OCR_CLEANUP,
    STEP_SPEAKER_PARSE,
    inspect_ocr_block,
    inspect_ocr_blocks,
)
from lib.subtitle.srt import SrtBlock


def build_noise_dictionary(
    tmp_path: Path,
) -> NoiseDictionary:
    return NoiseDictionary(
        profile_name="test",
        entries={
            "VVNsKomCIAcM": NoiseEntry(
                source="VVNsKomCIAcM",
                replacement="（判読不能）",
                action="mask",
                status="confirmed",
            ),
            "CandidateNoise": NoiseEntry(
                source="CandidateNoise",
                replacement="（判読不能）",
                action="mask",
                status="candidate",
            ),
        },
        official_path=(
            tmp_path / "noise.json"
        ),
        local_path=(
            tmp_path / "noise.local.json"
        ),
        local_loaded=False,
    )


def build_block(
    text: str,
    *,
    number: str = "1",
) -> SrtBlock:
    return SrtBlock(
        number=number,
        timestamp=(
            "00:00:01,000 --> "
            "00:00:03,000"
        ),
        text=text,
    )


def test_normal_text_is_not_changed(
    tmp_path: Path,
) -> None:
    dictionary = build_noise_dictionary(
        tmp_path
    )

    entry = inspect_ocr_block(
        build_block(
            "This is normal text."
        ),
        dictionary,
    )

    assert entry.raw_text == (
        "This is normal text."
    )

    assert entry.cleaned_text == (
        "This is normal text."
    )

    assert entry.noise_applied_text == (
        "This is normal text."
    )

    assert entry.changed_steps == ()


def test_common_ocr_error_is_recorded(
    tmp_path: Path,
) -> None:
    dictionary = build_noise_dictionary(
        tmp_path
    )

    entry = inspect_ocr_block(
        build_block(
            "| think this is correct."
        ),
        dictionary,
    )

    assert entry.cleaned_text == (
        "I think this is correct."
    )

    assert STEP_OCR_CLEANUP in (
        entry.changed_steps
    )


def test_bracket_speaker_is_detected(
    tmp_path: Path,
) -> None:
    dictionary = build_noise_dictionary(
        tmp_path
    )

    entry = inspect_ocr_block(
        build_block(
            "[DANIEL] This is the Stargate."
        ),
        dictionary,
    )

    assert entry.speaker == "DANIEL"

    assert entry.parsed_text == (
        "This is the Stargate."
    )

    assert entry.cleaned_text == (
        "This is the Stargate."
    )

    assert STEP_SPEAKER_PARSE in (
        entry.changed_steps
    )


def test_colon_speaker_is_detected(
    tmp_path: Path,
) -> None:
    dictionary = build_noise_dictionary(
        tmp_path
    )

    entry = inspect_ocr_block(
        build_block(
            "DANIEL: Move away."
        ),
        dictionary,
    )

    assert entry.speaker == "DANIEL"
    assert entry.parsed_text == "Move away."
    assert entry.cleaned_text == "Move away."


def test_cleanup_is_applied_only_to_body(
    tmp_path: Path,
) -> None:
    dictionary = build_noise_dictionary(
        tmp_path
    )

    entry = inspect_ocr_block(
        build_block(
            "[DANIEL] | think this works."
        ),
        dictionary,
    )

    assert entry.speaker == "DANIEL"

    assert entry.cleaned_text == (
        "I think this works."
    )


def test_multiline_text_is_preserved(
    tmp_path: Path,
) -> None:
    dictionary = build_noise_dictionary(
        tmp_path
    )

    entry = inspect_ocr_block(
        build_block(
            "First   line\nSecond   line"
        ),
        dictionary,
    )

    assert entry.cleaned_text == (
        "First line\nSecond line"
    )


def test_confirmed_noise_is_detected_and_applied(
    tmp_path: Path,
) -> None:
    dictionary = build_noise_dictionary(
        tmp_path
    )

    entry = inspect_ocr_block(
        build_block(
            "Move VVNsKomCIAcM away."
        ),
        dictionary,
    )

    assert entry.noise_candidates == (
        "VVNsKomCIAcM",
    )

    assert entry.noise_applied_text == (
        "Move （判読不能） away."
    )

    assert STEP_NOISE_DETECTED in (
        entry.changed_steps
    )

    assert STEP_NOISE_DICTIONARY in (
        entry.changed_steps
    )


def test_candidate_noise_is_not_applied(
    tmp_path: Path,
) -> None:
    dictionary = build_noise_dictionary(
        tmp_path
    )

    entry = inspect_ocr_block(
        build_block(
            "CandidateNoise"
        ),
        dictionary,
    )

    assert entry.noise_applied_text == (
        "CandidateNoise"
    )

    assert STEP_NOISE_DICTIONARY not in (
        entry.changed_steps
    )


def test_inspection_does_not_write_noise_file(
    tmp_path: Path,
) -> None:
    dictionary = build_noise_dictionary(
        tmp_path
    )

    inspect_ocr_block(
        build_block(
            "Move VVNsKomCIAcM away."
        ),
        dictionary,
    )

    assert not (
        tmp_path / "noise.local.json"
    ).exists()


def test_changed_steps_have_stable_order(
    tmp_path: Path,
) -> None:
    dictionary = build_noise_dictionary(
        tmp_path
    )

    entry = inspect_ocr_block(
        build_block(
            "[DANIEL] | think VVNsKomCIAcM"
        ),
        dictionary,
    )

    assert entry.changed_steps == (
        STEP_SPEAKER_PARSE,
        STEP_OCR_CLEANUP,
        STEP_NOISE_DETECTED,
        STEP_NOISE_DICTIONARY,
    )


def test_report_summary_is_calculated(
    tmp_path: Path,
) -> None:
    dictionary = build_noise_dictionary(
        tmp_path
    )

    report = inspect_ocr_blocks(
        [
            build_block(
                "Normal text.",
                number="1",
            ),
            build_block(
                "[DANIEL] | think "
                "VVNsKomCIAcM",
                number="2",
            ),
        ],
        source_srt=(
            tmp_path / "input.eng.srt"
        ),
        profile_name="test",
        noise_dictionary=dictionary,
    )

    assert report.summary.subtitle_count == 2

    assert (
        report.summary
        .speaker_detected_count
        == 1
    )

    assert (
        report.summary
        .cleanup_changed_count
        == 1
    )

    assert (
        report.summary
        .noise_candidate_subtitle_count
        == 1
    )

    assert (
        report.summary
        .noise_candidate_count
        == 1
    )

    assert (
        report.summary
        .noise_applied_count
        == 1
    )

    assert (
        report.summary
        .changed_subtitle_count
        == 1
    )


def test_empty_blocks_create_empty_report(
    tmp_path: Path,
) -> None:
    dictionary = build_noise_dictionary(
        tmp_path
    )

    report = inspect_ocr_blocks(
        [],
        source_srt=(
            tmp_path / "empty.eng.srt"
        ),
        profile_name="test",
        noise_dictionary=dictionary,
    )

    assert report.entries == ()
    assert report.summary.subtitle_count == 0
