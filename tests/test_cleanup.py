from pathlib import Path

import pytest
from commands.make import (
    cleanup_make_intermediate_files,
)
from lib.infrastructure.cleanup import (
    cleanup_intermediate_files,
)


def test_cleanup_intermediate_files_deletes_existing_files(
    tmp_path: Path,
) -> None:
    sup_path = (
        tmp_path
        / "movie.eng.sup"
    )
    eng_srt = (
        tmp_path
        / "movie.eng.srt"
    )
    ja_srt = (
        tmp_path
        / "movie.ja.srt"
    )

    for path in (
            sup_path,
            eng_srt,
            ja_srt,
    ):
        path.write_text(
            "temporary",
            encoding="utf-8",
        )

    result = cleanup_intermediate_files(
        [
            sup_path,
            eng_srt,
            ja_srt,
        ]
    )

    assert result.deleted == (
        sup_path.resolve(),
        eng_srt.resolve(),
        ja_srt.resolve(),
    )

    assert result.missing == ()

    assert not sup_path.exists()
    assert not eng_srt.exists()
    assert not ja_srt.exists()


def test_cleanup_intermediate_files_records_missing_files(
    tmp_path: Path,
) -> None:
    missing_path = (
        tmp_path
        / "movie.eng.sup"
    )

    result = cleanup_intermediate_files(
        [
            missing_path,
        ]
    )

    assert result.deleted == ()

    assert result.missing == (
        missing_path.resolve(),
    )


def test_cleanup_intermediate_files_rejects_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        IsADirectoryError,
        match=(
            "Cleanup target is not a file"
        ),
    ):
        cleanup_intermediate_files(
            [
                tmp_path,
            ]
        )


def test_cleanup_make_intermediate_files_deletes_files(
    tmp_path: Path,
) -> None:
    sup_path = (
        tmp_path
        / "movie.eng.sup"
    )
    eng_srt = (
        tmp_path
        / "movie.eng.srt"
    )
    ja_srt = (
        tmp_path
        / "movie.ja.srt"
    )

    for path in (
            sup_path,
            eng_srt,
            ja_srt,
    ):
        path.write_text(
            "temporary",
            encoding="utf-8",
        )

    result = cleanup_make_intermediate_files(
        sup_path=sup_path,
        eng_srt=eng_srt,
        ja_srt=ja_srt,
        keep_intermediate=False,
    )

    assert result is not None

    assert result.deleted == (
        sup_path.resolve(),
        eng_srt.resolve(),
        ja_srt.resolve(),
    )

    assert result.missing == ()

    assert not sup_path.exists()
    assert not eng_srt.exists()
    assert not ja_srt.exists()


def test_cleanup_make_intermediate_files_keeps_files(
    tmp_path: Path,
) -> None:
    sup_path = (
        tmp_path
        / "movie.eng.sup"
    )
    eng_srt = (
        tmp_path
        / "movie.eng.srt"
    )
    ja_srt = (
        tmp_path
        / "movie.ja.srt"
    )

    for path in (
            sup_path,
            eng_srt,
            ja_srt,
    ):
        path.write_text(
            "temporary",
            encoding="utf-8",
        )

    result = cleanup_make_intermediate_files(
        sup_path=sup_path,
        eng_srt=eng_srt,
        ja_srt=ja_srt,
        keep_intermediate=True,
    )

    assert result is None

    assert sup_path.is_file()
    assert eng_srt.is_file()
    assert ja_srt.is_file()

    assert sup_path.read_text(
        encoding="utf-8",
    ) == "temporary"

    assert eng_srt.read_text(
        encoding="utf-8",
    ) == "temporary"

    assert ja_srt.read_text(
        encoding="utf-8",
    ) == "temporary"
