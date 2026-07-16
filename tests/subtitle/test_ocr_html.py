from __future__ import annotations

from pathlib import Path

from lib.subtitle.ocr_html import (
    render_entry,
    render_ocr_html_report,
    write_ocr_html_report,
)
from lib.subtitle.ocr_inspection import (
    OcrInspectionEntry,
    OcrInspectionReport,
    OcrInspectionSummary,
)


def build_report(
    tmp_path: Path,
    *,
    raw_text: str = "DANIEL: Move away.",
) -> OcrInspectionReport:
    entry = OcrInspectionEntry(
        subtitle_id="1",
        timestamp=(
            "00:00:01,000 --> "
            "00:00:03,000"
        ),
        raw_text=raw_text,
        speaker="DANIEL",
        parsed_text="Move away.",
        cleaned_text="Move away.",
        noise_candidates=(
            "VVNsKomCIAcM",
        ),
        noise_applied_text=(
            "Move （判読不能） away."
        ),
        changed_steps=(
            "speaker_parse",
            "noise_detected",
            "noise_dictionary",
        ),
    )

    summary = OcrInspectionSummary(
        subtitle_count=1,
        speaker_detected_count=1,
        cleanup_changed_count=0,
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


def test_html_contains_report_data(
    tmp_path: Path,
) -> None:
    html = render_ocr_html_report(
        build_report(tmp_path)
    )

    assert "SubtitleTool OCR Inspection" in html
    assert "#1" in html

    assert (
        "00:00:01,000 --&gt; "
        "00:00:03,000"
        in html
    )

    assert "DANIEL" in html
    assert "Move away." in html
    assert "VVNsKomCIAcM" in html
    assert "（判読不能）" in html


def test_html_escapes_subtitle_content(
    tmp_path: Path,
) -> None:
    html = render_ocr_html_report(
        build_report(
            tmp_path,
            raw_text=(
                '<script>alert("x")</script>'
            ),
        )
    )

    assert (
        '<script>alert("x")</script>'
        not in html
    )

    assert (
        "&lt;script&gt;"
        in html
    )


def test_html_contains_filter_controls(
    tmp_path: Path,
) -> None:
    html = render_ocr_html_report(
        build_report(tmp_path)
    )

    assert 'id="search"' in html
    assert 'id="changed-only"' in html
    assert 'id="noise-only"' in html
    assert 'id="speaker-only"' in html
    assert 'id="reset"' in html


def test_html_does_not_use_external_assets(
    tmp_path: Path,
) -> None:
    html = render_ocr_html_report(
        build_report(tmp_path)
    )

    assert 'src="http://' not in html
    assert 'src="https://' not in html
    assert 'href="http://' not in html
    assert 'href="https://' not in html


def test_html_report_is_written(
    tmp_path: Path,
) -> None:
    output_path = (
        tmp_path
        / "nested"
        / "report.html"
    )

    result = write_ocr_html_report(
        output_path,
        build_report(tmp_path),
    )

    assert result == output_path.resolve()
    assert output_path.is_file()

    assert not (
        output_path.parent
        / "report.html.tmp"
    ).exists()


def test_speaker_only_entry_is_not_marked_as_changed() -> None:
    entry = OcrInspectionEntry(
        subtitle_id="1",
        timestamp=(
            "00:00:01,000 --> "
            "00:00:03,000"
        ),
        raw_text=(
            "[DANIEL] Normal dialogue."
        ),
        speaker="DANIEL",
        parsed_text="Normal dialogue.",
        cleaned_text="Normal dialogue.",
        noise_candidates=(),
        noise_applied_text="Normal dialogue.",
        changed_steps=(
            "speaker_parse",
        ),
    )

    html = render_entry(entry)

    assert 'data-changed="false"' in html
    assert 'data-speaker="true"' in html
    assert "entry-changed" not in html
