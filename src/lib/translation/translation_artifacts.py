from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TranslationArtifactCleanupResult:
    """
    翻訳生成物の削除結果。
    """

    deleted_files: tuple[Path, ...]
    missing_files: tuple[Path, ...]
    deleted_directories: tuple[Path, ...]
    missing_directories: tuple[Path, ...]
    non_empty_directories: tuple[Path, ...]


@dataclass
class TranslationArtifactRegistry:
    """
    1回の翻訳実行が生成したファイルと
    ディレクトリを記録する。
    """

    root_directory: Path
    _files: set[Path] = field(
        default_factory=set,
        init=False,
        repr=False,
    )
    _directories: set[Path] = field(
        default_factory=set,
        init=False,
        repr=False,
    )

    def __post_init__(
        self,
    ) -> None:
        self.root_directory = (
            Path(
                self.root_directory
            )
            .expanduser()
            .resolve()
        )

    @property
    def files(
        self,
    ) -> tuple[Path, ...]:
        """
        登録済みファイルを安定した順序で返す。
        """
        return tuple(
            sorted(
                self._files,
                key=str,
            )
        )

    @property
    def directories(
        self,
    ) -> tuple[Path, ...]:
        """
        登録済みディレクトリを安定した順序で返す。
        """
        return tuple(
            sorted(
                self._directories,
                key=str,
            )
        )

    def register_file(
        self,
        path: str | Path,
    ) -> Path:
        """
        今回の翻訳実行が生成したファイルを登録する。
        """
        resolved_path = self._resolve_artifact_path(
            path
        )

        self._files.add(
            resolved_path
        )

        return resolved_path

    def register_files(
        self,
        paths: tuple[Path, ...] | list[Path],
    ) -> tuple[Path, ...]:
        """
        複数の生成ファイルを登録する。
        """
        return tuple(
            self.register_file(
                path
            )
            for path in paths
        )

    def register_directory(
        self,
        path: str | Path,
    ) -> Path:
        """
        今回の翻訳実行専用ディレクトリを登録する。
        """
        resolved_path = self._resolve_artifact_path(
            path
        )

        self._directories.add(
            resolved_path
        )

        return resolved_path

    def cleanup(
        self,
    ) -> TranslationArtifactCleanupResult:
        """
        登録済み生成物だけを削除する。

        ファイルを先に削除し、
        ディレクトリは空の場合だけ削除する。
        """
        deleted_files: list[Path] = []
        missing_files: list[Path] = []

        for path in self.files:
            if not path.exists():
                missing_files.append(
                    path
                )
                continue

            if not path.is_file():
                raise IsADirectoryError(
                    "Translation artifact is not "
                    f"a file: {path}"
                )

            path.unlink()

            deleted_files.append(
                path
            )

        deleted_directories: list[Path] = []
        missing_directories: list[Path] = []
        non_empty_directories: list[Path] = []

        directories = sorted(
            self._directories,
            key=lambda path: (
                len(path.parts),
                str(path),
            ),
            reverse=True,
        )

        for path in directories:
            if not path.exists():
                missing_directories.append(
                    path
                )
                continue

            if not path.is_dir():
                raise NotADirectoryError(
                    "Translation artifact is not "
                    f"a directory: {path}"
                )

            if any(
                path.iterdir()
            ):
                non_empty_directories.append(
                    path
                )
                continue

            path.rmdir()

            deleted_directories.append(
                path
            )

        return TranslationArtifactCleanupResult(
            deleted_files=tuple(
                deleted_files
            ),
            missing_files=tuple(
                missing_files
            ),
            deleted_directories=tuple(
                deleted_directories
            ),
            missing_directories=tuple(
                missing_directories
            ),
            non_empty_directories=tuple(
                non_empty_directories
            ),
        )

    def _resolve_artifact_path(
        self,
        path: str | Path,
    ) -> Path:
        resolved_path = (
            Path(
                path
            )
            .expanduser()
            .resolve()
        )

        if resolved_path == self.root_directory:
            raise ValueError(
                "Translation artifact must not "
                "be the registry root directory: "
                f"{resolved_path}"
            )

        if (
            self.root_directory
            not in resolved_path.parents
        ):
            raise ValueError(
                "Translation artifact is outside "
                "the registry root directory: "
                f"{resolved_path}"
            )

        return resolved_path
