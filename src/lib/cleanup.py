from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CleanupResult:
    """
    中間ファイル削除結果。
    """

    deleted: tuple[Path, ...]
    missing: tuple[Path, ...]


def cleanup_intermediate_files(
    paths: list[str | Path],
) -> CleanupResult:
    """
    指定された中間ファイルを削除する。

    存在する通常ファイルは削除し、
    存在しないファイルはmissingとして返す。

    ディレクトリなど通常ファイル以外が指定された場合は、
    誤削除防止のため例外にする。
    """
    deleted: list[Path] = []
    missing: list[Path] = []

    for value in paths:
        path = (
            Path(value)
            .expanduser()
            .resolve()
        )

        if not path.exists():
            missing.append(
                path
            )
            continue

        if not path.is_file():
            raise IsADirectoryError(
                "Cleanup target is not a file: "
                f"{path}"
            )

        path.unlink()

        deleted.append(
            path
        )

    return CleanupResult(
        deleted=tuple(
            deleted
        ),
        missing=tuple(
            missing
        ),
    )
