from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

from .ffprobe import subtitle_count
from .mux_plan import (
    MuxPlan,
    build_mux_plan,
)
from .mux_validation import (
    validate_mux_output,
)


def build_mkvmerge_command(
    mux_plan: MuxPlan,
) -> list[str]:
    """
    元MKVの全トラックを維持したまま、
    日本語SRTを追加するmkvmergeコマンドを生成する。
    """
    return [
        "mkvmerge",
        "--output",
        str(mux_plan.temporary_output),
        str(mux_plan.input_mkv),
        "--language",
        f"0:{mux_plan.added_language}",
        "--track-name",
        f"0:{mux_plan.added_title}",
        "--default-track-flag",
        "0:no",
        "--forced-display-flag",
        "0:no",
        str(mux_plan.input_srt),
    ]


def run_mkvmerge(
    command: list[str],
) -> None:
    """
    mkvmergeを実行する。
    """
    print()
    print(shlex.join(command))

    subprocess.run(
        command,
        check=True,
    )


def print_validation_warnings(
    warnings: tuple[str, ...],
) -> None:
    if not warnings:
        return

    print()
    print("Mux Validation Warnings:")

    for warning in warnings:
        print(f"  - {warning}")


def build_validation_error(
    errors: tuple[str, ...],
) -> RuntimeError:
    details = "\n".join(
        f"  - {error}"
        for error in errors
    )

    return RuntimeError(
        "Mux validation failed:\n"
        f"{details}"
    )


def mux_japanese_srt(
    input_mkv: str | Path,
    ja_srt: str | Path,
    output_mkv: str | Path | None = None,
) -> Path:
    """
    mkvmergeを使用して日本語SRTをMKVへ追加する。

    正式な出力へ直接書き込まず、一時MKVを作成して
    検証に成功した場合だけ出力ファイルとして確定する。
    """
    input_path = (
        Path(input_mkv)
        .expanduser()
        .resolve()
    )

    subtitle_path = (
        Path(ja_srt)
        .expanduser()
        .resolve()
    )

    if output_mkv is None:
        output_path = input_path.with_name(
            f"{input_path.stem}.ja.mkv"
        )
    else:
        output_path = (
            Path(output_mkv)
            .expanduser()
            .resolve()
        )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"MKV not found: {input_path}"
        )

    if not subtitle_path.is_file():
        raise FileNotFoundError(
            f"SRT not found: {subtitle_path}"
        )

    if output_path.exists():
        print(f"Skip MUX: {output_path}")
        return output_path

    if shutil.which("mkvmerge") is None:
        raise RuntimeError(
            "mkvmerge command not found. "
            "Install MKVToolNix before running MUX."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_subtitle_count = subtitle_count(
        input_path
    )

    mux_plan = build_mux_plan(
        input_mkv=input_path,
        input_srt=subtitle_path,
        output_mkv=output_path,
        existing_subtitle_count=(
            existing_subtitle_count
        ),
    )

    # 前回失敗時の一時ファイルが残っていても
    # 新しいMUXを実行できるようにする。
    mux_plan.temporary_output.unlink(
        missing_ok=True
    )

    command = build_mkvmerge_command(
        mux_plan
    )

    run_mkvmerge(command)

    validation = validate_mux_output(
        mux_plan.input_mkv,
        mux_plan.temporary_output,
        added_language=(
            mux_plan.added_language
        ),
        added_title=(
            mux_plan.added_title
        ),
    )

    print_validation_warnings(
        validation.warnings
    )

    if not validation.valid:
        raise build_validation_error(
            validation.errors
        )

    print()
    print("Mux Validation: OK")

    if not mux_plan.temporary_output.is_file():
        raise FileNotFoundError(
            "Temporary mux output not found: "
            f"{mux_plan.temporary_output}"
        )

    mux_plan.temporary_output.replace(
        mux_plan.output_mkv
    )

    print(
        "Mux Output Finalized: "
        f"{mux_plan.output_mkv}"
    )

    return mux_plan.output_mkv
