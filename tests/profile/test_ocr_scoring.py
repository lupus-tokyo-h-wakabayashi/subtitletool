from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from lib.profile.ocr_scoring import (
    OCR_SCORING_PATH,
    load_ocr_scoring_config,
)


def load_valid_payload() -> dict[str, Any]:
    return json.loads(
        OCR_SCORING_PATH.read_text(
            encoding="utf-8",
        )
    )


def write_payload(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def test_load_ocr_scoring_config() -> None:
    config = load_ocr_scoring_config()

    assert config.version == 1

    assert (
        config.get_value(
            "thresholds",
            "high_confidence",
        )
        == 12
    )

    assert (
        config.get_value(
            "weights",
            "unbalanced_parentheses",
        )
        == 3
    )

    assert (
        config.thresholds[
            "high_confidence"
        ].description
    )

    assert (
        config.weights[
            "unbalanced_parentheses"
        ].description
    )


def test_load_ocr_scoring_config_rejects_unsupported_version(
    tmp_path: Path,
) -> None:
    payload = load_valid_payload()
    payload["version"] = 2

    config_path = (
        tmp_path
        / "ocr-scoring.json"
    )

    write_payload(
        config_path,
        payload,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Unsupported OCR scoring version"
        ),
    ):
        load_ocr_scoring_config(
            config_path
        )


def test_load_ocr_scoring_config_rejects_missing_root_key(
    tmp_path: Path,
) -> None:
    payload = load_valid_payload()
    del payload["weights"]

    config_path = (
        tmp_path
        / "ocr-scoring.json"
    )

    write_payload(
        config_path,
        payload,
    )

    with pytest.raises(
        ValueError,
        match="missing keys",
    ):
        load_ocr_scoring_config(
            config_path
        )


def test_load_ocr_scoring_config_rejects_missing_group_item(
    tmp_path: Path,
) -> None:
    payload = load_valid_payload()

    del payload["weights"][
        "unbalanced_parentheses"
    ]

    config_path = (
        tmp_path
        / "ocr-scoring.json"
    )

    write_payload(
        config_path,
        payload,
    )

    with pytest.raises(
        ValueError,
        match="missing keys",
    ):
        load_ocr_scoring_config(
            config_path
        )


def test_load_ocr_scoring_config_rejects_missing_item_description(
    tmp_path: Path,
) -> None:
    payload = load_valid_payload()

    del payload["weights"][
        "unbalanced_parentheses"
    ]["description"]

    config_path = (
        tmp_path
        / "ocr-scoring.json"
    )

    write_payload(
        config_path,
        payload,
    )

    with pytest.raises(
        ValueError,
        match="missing keys",
    ):
        load_ocr_scoring_config(
            config_path
        )


def test_load_ocr_scoring_config_rejects_boolean_value(
    tmp_path: Path,
) -> None:
    payload = load_valid_payload()

    payload["weights"][
        "unbalanced_parentheses"
    ]["value"] = True

    config_path = (
        tmp_path
        / "ocr-scoring.json"
    )

    write_payload(
        config_path,
        payload,
    )

    with pytest.raises(
        ValueError,
        match="must be numeric",
    ):
        load_ocr_scoring_config(
            config_path
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        -0.01,
        1.01,
    ],
)
def test_load_ocr_scoring_config_rejects_invalid_ratio(
    tmp_path: Path,
    invalid_value: float,
) -> None:
    payload = load_valid_payload()

    payload["limits"][
        "minimum_glossary_similarity"
    ]["value"] = invalid_value

    config_path = (
        tmp_path
        / "ocr-scoring.json"
    )

    write_payload(
        config_path,
        payload,
    )

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        load_ocr_scoring_config(
            config_path
        )


def test_load_ocr_scoring_config_rejects_invalid_threshold_order(
    tmp_path: Path,
) -> None:
    payload = load_valid_payload()

    payload["thresholds"][
        "failed_with_normal_sibling"
    ]["value"] = 9

    payload["thresholds"][
        "failed_subtitle"
    ]["value"] = 8

    config_path = (
        tmp_path
        / "ocr-scoring.json"
    )

    write_payload(
        config_path,
        payload,
    )

    with pytest.raises(
        ValueError,
        match=(
            "failed_with_normal_sibling <= "
            "failed_subtitle <= "
            "high_confidence"
        ),
    ):
        load_ocr_scoring_config(
            config_path
        )


def test_load_ocr_scoring_config_rejects_invalid_limit_order(
    tmp_path: Path,
) -> None:
    payload = load_valid_payload()

    payload["limits"][
        "minimum_short_token_ratio"
    ]["value"] = 0.9

    payload["limits"][
        "minimum_high_short_token_ratio"
    ]["value"] = 0.8

    config_path = (
        tmp_path
        / "ocr-scoring.json"
    )

    write_payload(
        config_path,
        payload,
    )

    with pytest.raises(
        ValueError,
        match=(
            "minimum_short_token_ratio "
            "must not exceed "
            "minimum_high_short_token_ratio"
        ),
    ):
        load_ocr_scoring_config(
            config_path
        )


def test_load_ocr_scoring_config_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    config_path = (
        tmp_path
        / "ocr-scoring.json"
    )

    config_path.write_text(
        '{"version":',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid OCR scoring JSON",
    ):
        load_ocr_scoring_config(
            config_path
        )


def test_loaded_payload_is_independent_between_tests() -> None:
    first_payload = load_valid_payload()
    second_payload = load_valid_payload()

    modified_payload = copy.deepcopy(
        first_payload
    )

    modified_payload["version"] = 999

    assert second_payload["version"] == 1
