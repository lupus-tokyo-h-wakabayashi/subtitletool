from __future__ import annotations

from html import escape
from pathlib import Path

from lib.subtitle.ocr_inspection import (
    OcrInspectionEntry,
    OcrInspectionReport,
)


def render_text(
    text: str,
) -> str:
    if not text:
        return (
            '<span class="empty">'
            "（空）"
            "</span>"
        )

    return escape(
        text
    ).replace(
        "\n",
        "<br>",
    )


def render_noise_candidates(
    entry: OcrInspectionEntry,
) -> str:
    if not entry.noise_candidates:
        return (
            '<span class="empty">'
            "なし"
            "</span>"
        )

    return "".join(
        (
            '<span class="candidate">'
            f"{escape(candidate)}"
            "</span>"
        )
        for candidate
        in entry.noise_candidates
    )


def render_changed_steps(
    entry: OcrInspectionEntry,
) -> str:
    if not entry.changed_steps:
        return (
            '<span class="empty">'
            "なし"
            "</span>"
        )

    return "".join(
        (
            '<span class="step">'
            f"{escape(step)}"
            "</span>"
        )
        for step in entry.changed_steps
    )


def render_entry(
    entry: OcrInspectionEntry,
) -> str:
    searchable_text = " ".join(
        [
            entry.subtitle_id,
            entry.timestamp,
            entry.speaker or "",
            entry.raw_text,
            entry.parsed_text,
            entry.cleaned_text,
            entry.noise_applied_text,
            *entry.noise_candidates,
        ]
    ).lower()

    changed_value = (
        "true"
        if entry.changed
        else "false"
    )

    noise_value = (
        "true"
        if entry.noise_candidates
        else "false"
    )

    speaker_value = (
        "true"
        if entry.speaker is not None
        else "false"
    )

    card_classes = ["entry"]

    if entry.changed:
        card_classes.append(
            "entry-changed"
        )

    if entry.noise_candidates:
        card_classes.append(
            "entry-noise"
        )

    if (
        entry.noise_applied_text
        != entry.cleaned_text
    ):
        card_classes.append(
            "entry-applied"
        )

    return f"""
<article
    class="{escape(' '.join(card_classes))}"
    data-search="{escape(searchable_text, quote=True)}"
    data-changed="{changed_value}"
    data-noise="{noise_value}"
    data-speaker="{speaker_value}"
>
    <header class="entry-header">
        <strong>#{escape(entry.subtitle_id)}</strong>
        <span>{escape(entry.timestamp)}</span>
        <span>
            Speaker:
            {escape(entry.speaker or "なし")}
        </span>
    </header>

    <div class="comparison-grid">
        <section>
            <h3>OCR Raw</h3>
            <div class="text-value">
                {render_text(entry.raw_text)}
            </div>
        </section>

        <section>
            <h3>Parsed Text</h3>
            <div class="text-value">
                {render_text(entry.parsed_text)}
            </div>
        </section>

        <section>
            <h3>Cleaned</h3>
            <div class="text-value">
                {render_text(entry.cleaned_text)}
            </div>
        </section>

        <section>
            <h3>Noise Applied</h3>
            <div class="text-value">
                {render_text(entry.noise_applied_text)}
            </div>
        </section>
    </div>

    <div class="metadata-grid">
        <section>
            <h3>Noise Candidates</h3>
            <div>
                {render_noise_candidates(entry)}
            </div>
        </section>

        <section>
            <h3>Changed Steps</h3>
            <div>
                {render_changed_steps(entry)}
            </div>
        </section>
    </div>
</article>
"""


def render_ocr_html_report(
    report: OcrInspectionReport,
) -> str:
    summary = report.summary

    entries_html = "\n".join(
        render_entry(entry)
        for entry in report.entries
    )

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta
    name="viewport"
    content="width=device-width, initial-scale=1"
>
<title>SubtitleTool OCR Inspection</title>
<style>
:root {{
    color-scheme: dark;
    font-family:
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
    background: #111318;
    color: #e8eaf0;
}}

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    background: #111318;
    color: #e8eaf0;
}}

main {{
    width: min(1600px, 100%);
    margin: 0 auto;
    padding: 24px;
}}

h1,
h2,
h3 {{
    margin-top: 0;
}}

h1 {{
    margin-bottom: 8px;
}}

h3 {{
    margin-bottom: 8px;
    font-size: 14px;
    color: #b9c2d0;
}}

.source {{
    margin-bottom: 20px;
    color: #aeb7c5;
    overflow-wrap: anywhere;
}}

.summary-grid {{
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(170px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}}

.summary-item {{
    padding: 14px;
    border: 1px solid #303541;
    border-radius: 10px;
    background: #1a1e26;
}}

.summary-item strong {{
    display: block;
    margin-top: 6px;
    font-size: 24px;
}}

.controls {{
    position: sticky;
    top: 0;
    z-index: 10;
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
    margin-bottom: 20px;
    padding: 14px;
    border: 1px solid #303541;
    border-radius: 10px;
    background: rgba(26, 30, 38, 0.96);
    backdrop-filter: blur(8px);
}}

.controls input[type="search"] {{
    flex: 1 1 320px;
    min-width: 220px;
    padding: 10px 12px;
    border: 1px solid #454c5b;
    border-radius: 8px;
    background: #101218;
    color: #ffffff;
}}

.controls label {{
    display: flex;
    gap: 6px;
    align-items: center;
}}

.controls button {{
    padding: 9px 14px;
    border: 1px solid #596276;
    border-radius: 8px;
    background: #272d38;
    color: #ffffff;
    cursor: pointer;
}}

.result-count {{
    margin-bottom: 12px;
    color: #aeb7c5;
}}

.entry {{
    margin-bottom: 16px;
    border: 1px solid #303541;
    border-left: 5px solid #4b5362;
    border-radius: 10px;
    background: #1a1e26;
    overflow: hidden;
}}

.entry-changed {{
    border-left-color: #d6a84b;
}}

.entry-noise {{
    border-left-color: #df6666;
}}

.entry-applied {{
    box-shadow:
        inset 0 0 0 1px rgba(94, 189, 129, 0.35);
}}

.entry-header {{
    display: flex;
    flex-wrap: wrap;
    gap: 18px;
    padding: 12px 16px;
    background: #222731;
    color: #c9d0dc;
}}

.comparison-grid {{
    display: grid;
    grid-template-columns:
        repeat(4, minmax(0, 1fr));
    gap: 1px;
    background: #303541;
}}

.comparison-grid section {{
    min-width: 0;
    padding: 14px;
    background: #171a21;
}}

.text-value {{
    line-height: 1.55;
    overflow-wrap: anywhere;
    white-space: normal;
}}

.metadata-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: #303541;
}}

.metadata-grid section {{
    padding: 12px 14px;
    background: #1d2129;
}}

.candidate,
.step {{
    display: inline-block;
    margin: 2px 6px 2px 0;
    padding: 3px 7px;
    border-radius: 6px;
}}

.candidate {{
    background: #672f36;
    color: #ffd9dc;
}}

.step {{
    background: #303b50;
    color: #d9e4ff;
}}

.empty {{
    color: #777f8d;
}}

.hidden {{
    display: none;
}}

@media (max-width: 1100px) {{
    .comparison-grid {{
        grid-template-columns: 1fr 1fr;
    }}
}}

@media (max-width: 680px) {{
    main {{
        padding: 12px;
    }}

    .comparison-grid,
    .metadata-grid {{
        grid-template-columns: 1fr;
    }}
}}
</style>
</head>
<body>
<main>
    <h1>SubtitleTool OCR Inspection</h1>

    <div class="source">
        Source:
        {escape(str(report.source_srt))}
        <br>
        Profile:
        {escape(report.profile_name)}
    </div>

    <section class="summary-grid">
        <div class="summary-item">
            Subtitles
            <strong>{summary.subtitle_count}</strong>
        </div>

        <div class="summary-item">
            Speaker Detected
            <strong>{summary.speaker_detected_count}</strong>
        </div>

        <div class="summary-item">
            Cleanup Changed
            <strong>{summary.cleanup_changed_count}</strong>
        </div>

        <div class="summary-item">
            Noise Subtitles
            <strong>
                {summary.noise_candidate_subtitle_count}
            </strong>
        </div>

        <div class="summary-item">
            Noise Candidates
            <strong>{summary.noise_candidate_count}</strong>
        </div>

        <div class="summary-item">
            Noise Applied
            <strong>{summary.noise_applied_count}</strong>
        </div>

        <div class="summary-item">
            Changed Subtitles
            <strong>{summary.changed_subtitle_count}</strong>
        </div>
    </section>

    <section class="controls">
        <input
            id="search"
            type="search"
            placeholder="字幕ID・本文・話者を検索"
        >

        <label>
            <input
                id="changed-only"
                type="checkbox"
            >
            変更ありのみ
        </label>

        <label>
            <input
                id="noise-only"
                type="checkbox"
            >
            Noise候補ありのみ
        </label>

        <label>
            <input
                id="speaker-only"
                type="checkbox"
            >
            話者ありのみ
        </label>

        <button
            id="reset"
            type="button"
        >
            解除
        </button>
    </section>

    <div
        id="result-count"
        class="result-count"
    ></div>

    <section id="entries">
        {entries_html}
    </section>
</main>

<script>
(() => {{
    const search = document.getElementById("search");
    const changedOnly =
        document.getElementById("changed-only");
    const noiseOnly =
        document.getElementById("noise-only");
    const speakerOnly =
        document.getElementById("speaker-only");
    const reset =
        document.getElementById("reset");
    const resultCount =
        document.getElementById("result-count");

    const entries = Array.from(
        document.querySelectorAll(".entry")
    );

    const applyFilters = () => {{
        const query = search.value
            .trim()
            .toLowerCase();

        let visibleCount = 0;

        for (const entry of entries) {{
            const searchMatched =
                !query
                || entry.dataset.search.includes(query);

            const changedMatched =
                !changedOnly.checked
                || entry.dataset.changed === "true";

            const noiseMatched =
                !noiseOnly.checked
                || entry.dataset.noise === "true";

            const speakerMatched =
                !speakerOnly.checked
                || entry.dataset.speaker === "true";

            const visible =
                searchMatched
                && changedMatched
                && noiseMatched
                && speakerMatched;

            entry.classList.toggle(
                "hidden",
                !visible
            );

            if (visible) {{
                visibleCount += 1;
            }}
        }}

        resultCount.textContent =
            `${{visibleCount}} / ${{entries.length}} 件`;
    }};

    search.addEventListener(
        "input",
        applyFilters
    );

    changedOnly.addEventListener(
        "change",
        applyFilters
    );

    noiseOnly.addEventListener(
        "change",
        applyFilters
    );

    speakerOnly.addEventListener(
        "change",
        applyFilters
    );

    reset.addEventListener(
        "click",
        () => {{
            search.value = "";
            changedOnly.checked = false;
            noiseOnly.checked = false;
            speakerOnly.checked = false;
            applyFilters();
        }}
    );

    applyFilters();
}})();
</script>
</body>
</html>
"""


def write_ocr_html_report(
    output_path: str | Path,
    report: OcrInspectionReport,
) -> Path:
    path = (
        Path(output_path)
        .expanduser()
        .resolve()
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        render_ocr_html_report(
            report
        ),
        encoding="utf-8",
    )

    temporary_path.replace(path)

    return path
