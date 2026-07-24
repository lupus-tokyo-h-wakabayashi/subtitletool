from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from lib.profile.noise import (
    NoiseDictionary,
    apply_noise_dictionary_to_text,
    find_suspicious_latin_sequences,
)
from lib.subtitle.ocr_quality import (
    find_suspicious_short_uppercase_fragments,
)
from lib.subtitle.srt import (
    SrtBlock,
    parse_speaker_from_text,
)
from lib.subtitle.text import cleanup_ocr_text

STEP_SPEAKER_PARSE = "speaker_parse"
STEP_OCR_CLEANUP = "ocr_cleanup"
STEP_NOISE_DETECTED = "noise_detected"
STEP_SHORT_UPPERCASE_FRAGMENT_DETECTED = (
    "short_uppercase_fragment_detected"
)
STEP_NOISE_DICTIONARY = "noise_dictionary"


class OcrInspectionStatus(StrEnum):
    ACCEPTED = "accepted"
    SUSPICIOUS = "suspicious"
    CONFIRMED_NOISE = "confirmed_noise"
    UNRESOLVED = "unresolved"


class OcrInspectionReason(StrEnum):
    SUSPICIOUS_LATIN_SEQUENCE = (
        "suspicious_latin_sequence"
    )
    SHORT_UPPERCASE_FRAGMENT = (
        "short_uppercase_fragment"
    )
    NOISE_DICTIONARY_APPLIED = (
        "noise_dictionary_applied"
    )


@dataclass(frozen=True)
class OcrInspectionEntry:
    subtitle_id: str
    timestamp: str
    raw_text: str
    speaker: str | None
    parsed_text: str
    cleaned_text: str
    noise_candidates: tuple[str, ...]
    short_uppercase_fragment_candidates: (
        tuple[str, ...]
    )
    noise_applied_text: str
    status: OcrInspectionStatus
    reasons: tuple[OcrInspectionReason, ...]
    resolved_text: str | None
    changed_steps: tuple[str, ...]

    @property
    def observed(self) -> bool:
        """
        話者検出やNoise候補検出を含め、
        何らかの観測結果が存在するかを返す。
        """
        return bool(self.changed_steps)

    @property
    def changed(self) -> bool:
        """
        OCR処理によって字幕本文が実際に変更されたかを返す。

        話者抽出とNoise候補検出だけでは、
        本文変更として扱わない。
        """
        return any(
            step in {
                STEP_OCR_CLEANUP,
                STEP_NOISE_DICTIONARY,
            }
            for step in self.changed_steps
        )


@dataclass(frozen=True)
class OcrInspectionSummary:
    subtitle_count: int
    speaker_detected_count: int
    cleanup_changed_count: int
    noise_candidate_subtitle_count: int
    noise_candidate_count: int
    noise_applied_count: int
    suspicious_subtitle_count: int
    changed_subtitle_count: int


@dataclass(frozen=True)
class OcrInspectionReport:
    source_srt: Path
    profile_name: str
    summary: OcrInspectionSummary
    entries: tuple[OcrInspectionEntry, ...]


def build_ocr_inspection_quality(
    *,
    cleaned_text: str,
    noise_candidates: tuple[str, ...],
    short_uppercase_fragment_candidates: (
        tuple[str, ...]
    ),
    noise_applied_text: str,
) -> tuple[
    OcrInspectionStatus,
    tuple[OcrInspectionReason, ...],
    str | None,
]:
    if noise_applied_text != cleaned_text:
        return (
            OcrInspectionStatus.CONFIRMED_NOISE,
            (
                OcrInspectionReason
                .NOISE_DICTIONARY_APPLIED,
            ),
            noise_applied_text,
        )

    reasons: list[OcrInspectionReason] = []

    if noise_candidates:
        reasons.append(
            OcrInspectionReason
            .SUSPICIOUS_LATIN_SEQUENCE
        )

    if short_uppercase_fragment_candidates:
        reasons.append(
            OcrInspectionReason
            .SHORT_UPPERCASE_FRAGMENT
        )

    if reasons:
        return (
            OcrInspectionStatus.SUSPICIOUS,
            tuple(reasons),
            None,
        )

    return (
        OcrInspectionStatus.ACCEPTED,
        (),
        noise_applied_text,
    )


def inspect_ocr_block(
    block: SrtBlock,
    noise_dictionary: NoiseDictionary,
) -> OcrInspectionEntry:
    """
    1字幕について、OCR直後からNoise辞書適用後までを観測する。

    翻訳処理と同じように、明示的な話者が存在する場合は
    話者と本文を分離し、本文だけへOCRクリーンアップを適用する。
    """
    raw_text = block.text

    parsed = parse_speaker_from_text(
        raw_text
    )

    cleanup_input = (
        parsed.text
        if parsed.speaker is not None
        else raw_text
    )

    cleaned_text = cleanup_ocr_text(
        cleanup_input
    )

    noise_candidates = tuple(
        find_suspicious_latin_sequences(
            cleaned_text,
            noise_dictionary,
        )
    )

    short_uppercase_fragment_candidates = (
        find_suspicious_short_uppercase_fragments(
            cleaned_text
        )
    )

    noise_applied_text = (
        apply_noise_dictionary_to_text(
            cleaned_text,
            noise_dictionary,
        )
    )

    (
        status,
        reasons,
        resolved_text,
    ) = build_ocr_inspection_quality(
        cleaned_text=cleaned_text,
        noise_candidates=noise_candidates,
        short_uppercase_fragment_candidates=(
            short_uppercase_fragment_candidates
        ),
        noise_applied_text=noise_applied_text,
    )

    changed_steps: list[str] = []

    if parsed.speaker is not None:
        changed_steps.append(
            STEP_SPEAKER_PARSE
        )

    if cleaned_text != cleanup_input:
        changed_steps.append(
            STEP_OCR_CLEANUP
        )

    if noise_candidates:
        changed_steps.append(
            STEP_NOISE_DETECTED
        )

    if short_uppercase_fragment_candidates:
        changed_steps.append(
            STEP_SHORT_UPPERCASE_FRAGMENT_DETECTED
        )

    if noise_applied_text != cleaned_text:
        changed_steps.append(
            STEP_NOISE_DICTIONARY
        )

    return OcrInspectionEntry(
        subtitle_id=block.number,
        timestamp=block.timestamp,
        raw_text=raw_text,
        speaker=parsed.speaker,
        parsed_text=parsed.text,
        cleaned_text=cleaned_text,
        noise_candidates=noise_candidates,
        short_uppercase_fragment_candidates=(
            short_uppercase_fragment_candidates
        ),
        noise_applied_text=noise_applied_text,
        status=status,
        reasons=reasons,
        resolved_text=resolved_text,
        changed_steps=tuple(
            changed_steps
        ),
    )


def build_ocr_inspection_summary(
    entries: tuple[OcrInspectionEntry, ...],
) -> OcrInspectionSummary:
    return OcrInspectionSummary(
        subtitle_count=len(entries),
        speaker_detected_count=sum(
            entry.speaker is not None
            for entry in entries
        ),
        cleanup_changed_count=sum(
            STEP_OCR_CLEANUP
            in entry.changed_steps
            for entry in entries
        ),
        noise_candidate_subtitle_count=sum(
            bool(entry.noise_candidates)
            for entry in entries
        ),
        noise_candidate_count=sum(
            len(entry.noise_candidates)
            for entry in entries
        ),
        noise_applied_count=sum(
            STEP_NOISE_DICTIONARY
            in entry.changed_steps
            for entry in entries
        ),
        suspicious_subtitle_count=sum(
            entry.status
            == OcrInspectionStatus.SUSPICIOUS
            for entry in entries
        ),
        changed_subtitle_count=sum(
            entry.changed
            for entry in entries
        ),
    )


def inspect_ocr_blocks(
    blocks: list[SrtBlock],
    *,
    source_srt: str | Path,
    profile_name: str,
    noise_dictionary: NoiseDictionary,
) -> OcrInspectionReport:
    source_path = (
        Path(source_srt)
        .expanduser()
        .resolve()
    )

    entries = tuple(
        inspect_ocr_block(
            block,
            noise_dictionary,
        )
        for block in blocks
    )

    summary = (
        build_ocr_inspection_summary(
            entries
        )
    )

    return OcrInspectionReport(
        source_srt=source_path,
        profile_name=profile_name,
        summary=summary,
        entries=entries,
    )
