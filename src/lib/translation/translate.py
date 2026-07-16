#!/usr/bin/env python3

from pathlib import Path

from lib.profile.config import (
    resolve_profile_config,
)
from lib.profile.noise import (
    load_noise_dictionary,
)
from lib.subtitle.srt import (
    SrtBlock,
    parse_srt,
)
from .translation_output import (
    print_translation_already_complete,
)
from .translation_resume import (
    load_resume_blocks,
)
from .translation_session import (
    run_translation_session,
)

MODEL = "qwen3:14b"

# 実際に翻訳する字幕数
CHUNK_SIZE = 10

# 翻訳対象の前後に参考として渡す字幕数
CONTEXT_SIZE = 15


def filter_empty_source_blocks(
    blocks: list[SrtBlock],
) -> tuple[list[SrtBlock], list[SrtBlock]]:
    """
    本文が空または空白だけの字幕を翻訳対象から除外する。

    戻り値:
        translation_blocks:
            LLMへ送信する有効な字幕

        skipped_blocks:
            本文が空のため除外した字幕
    """
    translation_blocks: list[SrtBlock] = []
    skipped_blocks: list[SrtBlock] = []

    for block in blocks:
        if block.text.strip():
            translation_blocks.append(block)
        else:
            skipped_blocks.append(block)

    return (
        translation_blocks,
        skipped_blocks,
    )


def resolve_requested_profile(
    profile_name: str | None,
    style_name: str | None,
    glossary_name: str | None,
) -> str | None:
    """
    profile指定と旧style/glossary指定を統一する。

    移行期間中のみstyle/glossaryも受け付ける。
    """
    requested_profile = profile_name

    legacy_profile_specified = (
        style_name is not None
        or glossary_name is not None
    )

    if not legacy_profile_specified:
        return requested_profile

    if style_name != glossary_name:
        raise ValueError(
            "Style and glossary profiles must match "
            "during migration: "
            f"style={style_name!r}, "
            f"glossary={glossary_name!r}"
        )

    legacy_profile = style_name

    if (
        requested_profile is not None
        and legacy_profile is not None
        and requested_profile != legacy_profile
    ):
        raise ValueError(
            "Profile conflicts with legacy options: "
            f"profile={requested_profile!r}, "
            f"style={style_name!r}, "
            f"glossary={glossary_name!r}"
        )

    if requested_profile is None:
        requested_profile = legacy_profile

    return requested_profile


def translate_srt(
    input_srt: str | Path,
    output_srt: str | Path,
    model: str = MODEL,
    chunk_size: int = CHUNK_SIZE,
    context_size: int = CONTEXT_SIZE,
    profile_name: str | None = None,
    style_name: str | None = None,
    glossary_name: str | None = None,
    inspect_request: bool = False,
) -> Path:
    input_path = (
        Path(input_srt)
        .expanduser()
        .resolve()
    )

    output_path = (
        Path(output_srt)
        .expanduser()
        .resolve()
    )

    if input_path == output_path:
        raise ValueError(
            "Input and output SRT paths must be different: "
            f"{input_path}"
        )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"SRT not found: {input_path}"
        )

    requested_profile = resolve_requested_profile(
        profile_name=profile_name,
        style_name=style_name,
        glossary_name=glossary_name,
    )

    profile_config = resolve_profile_config(
        requested_profile
    )

    resolved_profile = (
        profile_config.resolved_profile
    )

    noise_dictionary = load_noise_dictionary(
        profile_config
    )

    parsed_source_blocks = parse_srt(
        input_path
    )

    (
        source_blocks,
        skipped_empty_blocks,
    ) = filter_empty_source_blocks(
        parsed_source_blocks
    )

    if skipped_empty_blocks:
        print()
        print(
            "Skipped empty source subtitles:"
        )

        for block in skipped_empty_blocks:
            print(
                "  - "
                f"id={block.number}, "
                f"timestamp={block.timestamp}"
            )

    if not source_blocks:
        raise RuntimeError(
            "No translatable subtitle blocks: "
            f"{input_path}"
        )

    translated_blocks_all = load_resume_blocks(
        source_blocks,
        output_path,
    )

    total_blocks = len(source_blocks)
    resume_start = len(
        translated_blocks_all
    )

    if resume_start == total_blocks:
        if inspect_request:
            raise RuntimeError(
                "No translation request to inspect: "
                "all subtitles are already translated."
            )

        print_translation_already_complete(
            requested_profile=(
                profile_config.requested_profile
            ),
            resolved_profile=resolved_profile,
            fallback_used=(
                profile_config.fallback_used
            ),
            noise_dictionary=noise_dictionary,
            subtitle_count=total_blocks,
            output_path=output_path,
        )

        return output_path

    inspection_path = run_translation_session(
        source_blocks=source_blocks,
        translated_blocks_all=(
            translated_blocks_all
        ),
        output_path=output_path,
        model=model,
        chunk_size=chunk_size,
        context_size=context_size,
        profile_config=profile_config,
        noise_dictionary=noise_dictionary,
        inspect_request=inspect_request,
    )

    if inspection_path is not None:
        return inspection_path

    return output_path

    return output_path
