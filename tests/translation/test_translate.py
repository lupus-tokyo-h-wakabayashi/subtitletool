import pytest

from lib.translation.translate import (
    resolve_requested_profile,
)


def test_resolve_requested_profile_with_profile() -> None:
    result = resolve_requested_profile(
        profile_name="stargate",
        style_name=None,
        glossary_name=None,
    )

    assert result == "stargate"


def test_resolve_requested_profile_with_legacy_options() -> None:
    result = resolve_requested_profile(
        profile_name=None,
        style_name="stargate",
        glossary_name="stargate",
    )

    assert result == "stargate"


def test_resolve_requested_profile_rejects_legacy_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Style and glossary profiles "
            "must match"
        ),
    ):
        resolve_requested_profile(
            profile_name=None,
            style_name="stargate",
            glossary_name="default",
        )


def test_resolve_requested_profile_rejects_conflict() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Profile conflicts with "
            "legacy options"
        ),
    ):
        resolve_requested_profile(
            profile_name="default",
            style_name="stargate",
            glossary_name="stargate",
        )
