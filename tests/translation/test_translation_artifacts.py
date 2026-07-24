from pathlib import Path

import pytest
from lib.translation.translation_artifacts import (
    TranslationArtifactRegistry,
)


def test_registry_records_files_without_duplicates(
    tmp_path: Path,
) -> None:
    registry = TranslationArtifactRegistry(
        root_directory=tmp_path
    )

    artifact_path = (
        tmp_path
        / "failed-translation.txt"
    )

    first = registry.register_file(
        artifact_path
    )

    second = registry.register_file(
        artifact_path
    )

    assert first == artifact_path.resolve()
    assert second == artifact_path.resolve()
    assert registry.files == (
        artifact_path.resolve(),
    )


def test_registry_rejects_artifact_outside_root(
    tmp_path: Path,
) -> None:
    registry_root = (
        tmp_path
        / "translation"
    )

    registry = TranslationArtifactRegistry(
        root_directory=registry_root
    )

    outside_path = (
        tmp_path
        / "outside.json"
    )

    with pytest.raises(
        ValueError,
        match=(
            "Translation artifact is outside "
            "the registry root directory"
        ),
    ):
        registry.register_file(
            outside_path
        )


def test_registry_rejects_root_directory(
    tmp_path: Path,
) -> None:
    registry = TranslationArtifactRegistry(
        root_directory=tmp_path
    )

    with pytest.raises(
        ValueError,
        match=(
            "Translation artifact must not "
            "be the registry root directory"
        ),
    ):
        registry.register_directory(
            tmp_path
        )


def test_cleanup_deletes_registered_files_and_directories(
    tmp_path: Path,
) -> None:
    registry = TranslationArtifactRegistry(
        root_directory=tmp_path
    )

    session_directory = (
        tmp_path
        / "translation-metrics"
        / "session"
    )

    session_directory.mkdir(
        parents=True
    )

    chunk_path = (
        session_directory
        / "chunk-000001-000010.json"
    )

    summary_path = (
        session_directory
        / "summary.json"
    )

    chunk_path.write_text(
        "{}",
        encoding="utf-8",
    )

    summary_path.write_text(
        "{}",
        encoding="utf-8",
    )

    registry.register_files(
        [
            chunk_path,
            summary_path,
        ]
    )

    registry.register_directory(
        session_directory
    )

    result = registry.cleanup()

    assert result.deleted_files == (
        chunk_path.resolve(),
        summary_path.resolve(),
    )

    assert result.missing_files == ()
    assert result.deleted_directories == (
        session_directory.resolve(),
    )
    assert result.missing_directories == ()
    assert result.non_empty_directories == ()

    assert not chunk_path.exists()
    assert not summary_path.exists()
    assert not session_directory.exists()


def test_cleanup_reports_missing_artifacts(
    tmp_path: Path,
) -> None:
    registry = TranslationArtifactRegistry(
        root_directory=tmp_path
    )

    missing_file = (
        tmp_path
        / "missing.txt"
    )

    missing_directory = (
        tmp_path
        / "missing-directory"
    )

    registry.register_file(
        missing_file
    )

    registry.register_directory(
        missing_directory
    )

    result = registry.cleanup()

    assert result.deleted_files == ()
    assert result.missing_files == (
        missing_file.resolve(),
    )
    assert result.deleted_directories == ()
    assert result.missing_directories == (
        missing_directory.resolve(),
    )
    assert result.non_empty_directories == ()


def test_cleanup_preserves_directory_with_unregistered_file(
    tmp_path: Path,
) -> None:
    registry = TranslationArtifactRegistry(
        root_directory=tmp_path
    )

    session_directory = (
        tmp_path
        / "translation-metrics"
        / "session"
    )

    session_directory.mkdir(
        parents=True
    )

    registered_path = (
        session_directory
        / "summary.json"
    )

    unregistered_path = (
        session_directory
        / "manual-note.txt"
    )

    registered_path.write_text(
        "{}",
        encoding="utf-8",
    )

    unregistered_path.write_text(
        "keep",
        encoding="utf-8",
    )

    registry.register_file(
        registered_path
    )

    registry.register_directory(
        session_directory
    )

    result = registry.cleanup()

    assert result.deleted_files == (
        registered_path.resolve(),
    )

    assert result.deleted_directories == ()
    assert result.non_empty_directories == (
        session_directory.resolve(),
    )

    assert not registered_path.exists()
    assert unregistered_path.exists()
    assert session_directory.exists()
