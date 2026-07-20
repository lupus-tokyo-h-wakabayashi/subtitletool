from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import CONFIG_DIR

SUPPORTED_OCR_SCORING_VERSION = 1

OCR_SCORING_PATH = (
    CONFIG_DIR
    / "ocr-scoring.json"
)

ROOT_KEYS = frozenset(
    {
        "version",
        "description",
        "thresholds",
        "limits",
        "weights",
    }
)

ITEM_KEYS = frozenset(
    {
        "value",
        "description",
    }
)

EXPECTED_GROUP_KEYS = {
    "thresholds": frozenset(
        {
            "high_confidence",
            "failed_subtitle",
            "failed_with_normal_sibling",
        }
    ),
    "limits": frozenset(
        {
            "maximum_short_line_length",
            "minimum_ascii_letters",
            "minimum_tokens",
            "minimum_many_tokens",
            "maximum_short_token_length",
            "minimum_short_token_ratio",
            "minimum_high_short_token_ratio",
            "minimum_suspicious_tokens",
            "minimum_structural_symbols",
            "minimum_dense_structural_symbols",
            "minimum_glossary_similarity",
            "minimum_glossary_source_length",
            "maximum_glossary_length_difference",
            "maximum_glossary_exact_protection",
            "minimum_damage_score_for_glossary_miss",
            "minimum_natural_sentence_tokens",
        }
    ),
    "weights": frozenset(
        {
            "short_line",
            "letters_and_digits",
            "minimum_ascii_letters",
            "minimum_tokens",
            "many_tokens",
            "multiple_structural_symbols",
            "dense_structural_symbols",
            "strong_corruption_symbol",
            "unbalanced_parentheses",
            "unbalanced_square_brackets",
            "unbalanced_braces",
            "unbalanced_angle_brackets",
            "unbalanced_double_quotes",
            "multiple_unbalanced_delimiters",
            "short_token_ratio",
            "high_short_token_ratio",
            "suspicious_tokens",
            "invalid_single_letter",
            "vowelless_uppercase_token",
            "irregular_mixed_case",
            "symbol_dense_structure",
            "low_symbol_word_salad",
            "short_mixed_case",
            "damaged_alphanumeric_structure",
            "glossary_exact_match",
            "glossary_similar_match",
            "damaged_without_glossary_match",
            "identifier_like",
            "equation_like",
            "time_like",
            "natural_sentence",
        }
    ),
}

RATIO_LIMIT_KEYS = frozenset(
    {
        "minimum_short_token_ratio",
        "minimum_high_short_token_ratio",
        "minimum_glossary_similarity",
    }
)


@dataclass(frozen=True)
class OcrScoringItem:
    value: int | float
    description: str


@dataclass(frozen=True)
class OcrScoringConfig:
    version: int
    description: str
    thresholds: Mapping[
        str,
        OcrScoringItem,
    ]
    limits: Mapping[
        str,
        OcrScoringItem,
    ]
    weights: Mapping[
        str,
        OcrScoringItem,
    ]
    path: Path

    def get_value(
        self,
        group_name: str,
        item_name: str,
    ) -> int | float:
        groups = {
            "thresholds": self.thresholds,
            "limits": self.limits,
            "weights": self.weights,
        }

        try:
            group = groups[group_name]
        except KeyError as error:
            raise KeyError(
                "Unknown OCR scoring group: "
                f"{group_name!r}"
            ) from error

        try:
            return group[item_name].value
        except KeyError as error:
            raise KeyError(
                "Unknown OCR scoring item: "
                f"group={group_name!r}, "
                f"item={item_name!r}"
            ) from error


def require_object(
    value: Any,
    *,
    label: str,
    path: Path,
) -> dict[str, Any]:
    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            f"{label} must be an object: "
            f"path={path}"
        )

    return value


def require_exact_keys(
    payload: Mapping[str, Any],
    expected_keys: frozenset[str],
    *,
    label: str,
    path: Path,
) -> None:
    actual_keys = frozenset(
        payload.keys()
    )

    missing_keys = sorted(
        expected_keys
        - actual_keys
    )

    unexpected_keys = sorted(
        actual_keys
        - expected_keys
    )

    if missing_keys:
        raise ValueError(
            f"{label} has missing keys: "
            f"keys={missing_keys}, "
            f"path={path}"
        )

    if unexpected_keys:
        raise ValueError(
            f"{label} has unexpected keys: "
            f"keys={unexpected_keys}, "
            f"path={path}"
        )


def require_description(
    value: Any,
    *,
    label: str,
    path: Path,
) -> str:
    if (
        not isinstance(
            value,
            str,
        )
        or not value.strip()
    ):
        raise ValueError(
            f"{label} must be a non-empty string: "
            f"path={path}"
        )

    return value


def require_number(
    value: Any,
    *,
    label: str,
    path: Path,
) -> int | float:
    if type(value) not in {
        int,
        float,
    }:
        raise ValueError(
            f"{label} must be numeric: "
            f"path={path}"
        )

    if not math.isfinite(
        value
    ):
        raise ValueError(
            f"{label} must be finite: "
            f"path={path}"
        )

    return value


def parse_scoring_item(
    value: Any,
    *,
    group_name: str,
    item_name: str,
    path: Path,
) -> OcrScoringItem:
    label = (
        "OCR scoring item "
        f"{group_name}.{item_name}"
    )

    payload = require_object(
        value,
        label=label,
        path=path,
    )

    require_exact_keys(
        payload,
        ITEM_KEYS,
        label=label,
        path=path,
    )

    item_value = require_number(
        payload["value"],
        label=f"{label}.value",
        path=path,
    )

    description = require_description(
        payload["description"],
        label=f"{label}.description",
        path=path,
    )

    return OcrScoringItem(
        value=item_value,
        description=description,
    )


def parse_scoring_group(
    value: Any,
    *,
    group_name: str,
    path: Path,
) -> dict[str, OcrScoringItem]:
    payload = require_object(
        value,
        label=(
            "OCR scoring group "
            f"{group_name!r}"
        ),
        path=path,
    )

    expected_keys = (
        EXPECTED_GROUP_KEYS[
            group_name
        ]
    )

    require_exact_keys(
        payload,
        expected_keys,
        label=(
            "OCR scoring group "
            f"{group_name!r}"
        ),
        path=path,
    )

    return {
        item_name: parse_scoring_item(
            payload[item_name],
            group_name=group_name,
            item_name=item_name,
            path=path,
        )
        for item_name in sorted(
            expected_keys
        )
    }


def validate_thresholds(
    thresholds: Mapping[
        str,
        OcrScoringItem,
    ],
    *,
    path: Path,
) -> None:
    for item_name, item in (
        thresholds.items()
    ):
        if (
            type(item.value) is not int
            or item.value < 0
        ):
            raise ValueError(
                "OCR scoring threshold must "
                "be a non-negative integer: "
                f"item={item_name!r}, "
                f"value={item.value!r}, "
                f"path={path}"
            )

    sibling_threshold = thresholds[
        "failed_with_normal_sibling"
    ].value

    failed_threshold = thresholds[
        "failed_subtitle"
    ].value

    high_threshold = thresholds[
        "high_confidence"
    ].value

    if not (
        sibling_threshold
        <= failed_threshold
        <= high_threshold
    ):
        raise ValueError(
            "OCR scoring thresholds must satisfy "
            "failed_with_normal_sibling <= "
            "failed_subtitle <= high_confidence: "
            f"path={path}"
        )


def validate_limits(
    limits: Mapping[
        str,
        OcrScoringItem,
    ],
    *,
    path: Path,
) -> None:
    for item_name, item in (
        limits.items()
    ):
        if item_name in RATIO_LIMIT_KEYS:
            if not (
                0
                <= item.value
                <= 1
            ):
                raise ValueError(
                    "OCR scoring ratio limit must "
                    "be between 0 and 1: "
                    f"item={item_name!r}, "
                    f"value={item.value!r}, "
                    f"path={path}"
                )

            continue

        if (
            type(item.value) is not int
            or item.value < 0
        ):
            raise ValueError(
                "OCR scoring limit must be a "
                "non-negative integer: "
                f"item={item_name!r}, "
                f"value={item.value!r}, "
                f"path={path}"
            )

    if (
        limits[
            "minimum_tokens"
        ].value
        > limits[
        "minimum_many_tokens"
    ].value
    ):
        raise ValueError(
            "minimum_tokens must not exceed "
            "minimum_many_tokens: "
            f"path={path}"
        )

    if (
        limits[
            "minimum_short_token_ratio"
        ].value
        > limits[
        "minimum_high_short_token_ratio"
    ].value
    ):
        raise ValueError(
            "minimum_short_token_ratio must not "
            "exceed "
            "minimum_high_short_token_ratio: "
            f"path={path}"
        )

    if (
        limits[
            "minimum_structural_symbols"
        ].value
        > limits[
        "minimum_dense_structural_symbols"
    ].value
    ):
        raise ValueError(
            "minimum_structural_symbols must not "
            "exceed "
            "minimum_dense_structural_symbols: "
            f"path={path}"
        )


def validate_weights(
    weights: Mapping[
        str,
        OcrScoringItem,
    ],
    *,
    path: Path,
) -> None:
    for item_name, item in (
        weights.items()
    ):
        if type(
            item.value
        ) is not int:
            raise ValueError(
                "OCR scoring weight must be "
                "an integer: "
                f"item={item_name!r}, "
                f"value={item.value!r}, "
                f"path={path}"
            )


def parse_ocr_scoring_config(
    payload: Any,
    *,
    path: Path,
) -> OcrScoringConfig:
    root = require_object(
        payload,
        label="OCR scoring configuration",
        path=path,
    )

    require_exact_keys(
        root,
        ROOT_KEYS,
        label="OCR scoring configuration",
        path=path,
    )

    version = root["version"]

    if type(
        version
    ) is not int:
        raise ValueError(
            "OCR scoring version must be "
            f"an integer: path={path}"
        )

    if (
        version
        != SUPPORTED_OCR_SCORING_VERSION
    ):
        raise ValueError(
            "Unsupported OCR scoring version: "
            f"version={version!r}, "
            "supported="
            f"{SUPPORTED_OCR_SCORING_VERSION}, "
            f"path={path}"
        )

    description = require_description(
        root["description"],
        label=(
            "OCR scoring configuration "
            "description"
        ),
        path=path,
    )

    thresholds = parse_scoring_group(
        root["thresholds"],
        group_name="thresholds",
        path=path,
    )

    limits = parse_scoring_group(
        root["limits"],
        group_name="limits",
        path=path,
    )

    weights = parse_scoring_group(
        root["weights"],
        group_name="weights",
        path=path,
    )

    validate_thresholds(
        thresholds,
        path=path,
    )

    validate_limits(
        limits,
        path=path,
    )

    validate_weights(
        weights,
        path=path,
    )

    return OcrScoringConfig(
        version=version,
        description=description,
        thresholds=thresholds,
        limits=limits,
        weights=weights,
        path=path,
    )


def load_ocr_scoring_config(
    path: Path = OCR_SCORING_PATH,
) -> OcrScoringConfig:
    try:
        raw_text = path.read_text(
            encoding="utf-8"
        )
    except OSError as error:
        raise RuntimeError(
            "Failed to read OCR scoring "
            f"configuration: path={path}"
        ) from error

    try:
        payload = json.loads(
            raw_text
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            "Invalid OCR scoring JSON: "
            f"path={path}, "
            f"line={error.lineno}, "
            f"column={error.colno}"
        ) from error

    return parse_ocr_scoring_config(
        payload,
        path=path,
    )
