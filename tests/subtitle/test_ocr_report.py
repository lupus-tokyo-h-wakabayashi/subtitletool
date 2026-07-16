from __future__ import annotations

import json
from pathlib import Path

from lib.subtitle.ocr_inspection import (
    OcrInspectionEntry,
    OcrInspectionReport,
    OcrInspectionSummary,
)
from lib.subtitle.ocr_report import (
    write_ocr_json_report,
)


def build_report(
    tmp_path: Path,
) -> OcrInspectionReport:
    entry = OcrInspectionEntry(
        subtitle_id="101",
        timestamp=(
            "00:08:12,000 --> "
            "00:08:14,500"
        ),
        raw_text=(
            "DANIEL: | think VVNsKomCIAcM"
        ),
        speaker="DANIEL",
        parsed_text=(
            "| think VVNsKomCIAcM"
        ),
        cleaned_text=(
            "I think VVNsKomCIAcM"
        ),
        noise_candidates=(
            "VVNsKomCIAcM",
        ),
        noise_applied_text=(
            "I think （判読不能）"
        ),
        changed_steps=(
            "speaker_parse",
            "ocr_cleanup",
            "noise_detected",
            "noise_dictionary",
        ),
    )

    summary = OcrInspectionSummary(
        subtitle_count=1,
        speaker_detected_count=1,
        cleanup_changed_count=1,
        noise_candidate_subtitle_count=1,
        noise_candidate_count=1,
        noise_applied_count=1,
        changed_subtitle_count=1,
    )

    return OcrInspectionReport(
        source_srt=(
            tmp_path / "input.eng.srt"
        ),
        profile_name="stargate",
        summary=summary,
        entries=(entry,),
    )


def test_json_report_is_written(
    tmp_path: Path,
) -> None:
    report = build_report(
        tmp_path
    )

    output_path = (
        tmp_path
        / "nested"
        / "report.json"
    )

    result = write_ocr_json_report(
        output_path,
        report,
    )

    assert result == output_path.resolve()
    assert output_path.is_file()


def test_json_report_structure(
    tmp_path: Path,
) -> None:
    report = build_report(
        tmp_path
    )

    output_path = (
        tmp_path / "report.json"
    )

    write_ocr_json_report(
        output_path,
        report,
    )

    data = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    assert data["version"] == 1
    assert data["profile"] == "stargate"

    assert (
        data["summary"]["subtitle_count"]
        == 1
    )

    assert (
        data["entries"][0]["subtitle_id"]
        == "101"
    )

    assert (
        data["entries"][0]["speaker"]
        == "DANIEL"
    )

    assert (
        data["entries"][0]
        ["noise_candidates"]
        == ["VVNsKomCIAcM"]
    )


def test_japanese_is_not_ascii_escaped(
    tmp_path: Path,
) -> None:
    report = build_report(
        tmp_path
    )

    output_path = (
        tmp_path / "report.json"
    )

    write_ocr_json_report(
        output_path,
        report,
    )

    content = output_path.read_text(
        encoding="utf-8"
    )

    assert "（判読不能）" in content
    assert "\\u5224" not in content


def test_report_can_be_overwritten(
    tmp_path: Path,
) -> None:
    report = build_report(
        tmp_path
    )

    output_path = (
        tmp_path / "report.json"
    )

    write_ocr_json_report(
        output_path,
        report,
    )

    write_ocr_json_report(
        output_path,
        report,
    )

    assert output_path.is_file()

    assert not (
        tmp_path / "report.json.tmp"
    ).exists()
