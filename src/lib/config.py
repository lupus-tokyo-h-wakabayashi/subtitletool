from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

DEFAULT_PROFILE_NAME = "default"


@dataclass(frozen=True)
class ProfileConfig:
    """
    解決済みの翻訳profile設定パス。
    """

    requested_profile: str | None
    resolved_profile: str
    profile_dir: Path

    prompt_path: Path
    glossary_path: Path
    style_path: Path
    noise_path: Path
    noise_local_path: Path

    fallback_used: bool


def normalize_profile_name(
    profile_name: str | None,
) -> str | None:
    """
    CLIなどから受け取ったprofile名を正規化する。

    Noneまたは空文字はprofile未指定として扱う。
    """
    if profile_name is None:
        return None

    normalized = profile_name.strip()

    if not normalized:
        return None

    return normalized


def resolve_profile_name(
    profile_name: str | None,
) -> tuple[str | None, str, bool]:
    """
    要求されたprofile名から使用profileを決定する。

    戻り値:
        requested_profile
        resolved_profile
        fallback_used
    """
    requested_profile = normalize_profile_name(
        profile_name
    )

    if requested_profile is None:
        return (
            None,
            DEFAULT_PROFILE_NAME,
            False,
        )

    requested_dir = (
        CONFIG_DIR / requested_profile
    )

    if requested_dir.is_dir():
        return (
            requested_profile,
            requested_profile,
            False,
        )

    return (
        requested_profile,
        DEFAULT_PROFILE_NAME,
        True,
    )


def validate_profile_config(
    config: ProfileConfig,
) -> None:
    """
    共通promptとprofile必須ファイルの存在を検証する。

    profileディレクトリが存在するにもかかわらず、
    必須ファイルが不足している場合はエラーにする。
    """
    legacy_glossary_path = (
        config.profile_dir
        / "glossary.txt"
    )

    if (
        not config.glossary_path.is_file()
        and legacy_glossary_path.is_file()
    ):
        raise FileNotFoundError(
            "Glossary JSON is missing: "
            f"profile={config.resolved_profile!r}, "
            f"expected={config.glossary_path}, "
            f"legacy={legacy_glossary_path}. "
            "Convert glossary.txt to glossary.json."
        )

    required_paths = {
        "prompt": config.prompt_path,
        "glossary": config.glossary_path,
        "style": config.style_path,
        "noise": config.noise_path,
    }

    missing = {
        name: path
        for name, path in required_paths.items()
        if not path.is_file()
    }

    if not missing:
        return

    details = ", ".join(
        f"{name}={path}"
        for name, path in missing.items()
    )

    raise FileNotFoundError(
        "Profile configuration is incomplete: "
        f"profile={config.resolved_profile!r}, "
        f"missing={details}"
    )


def resolve_profile_config(
    profile_name: str | None,
) -> ProfileConfig:
    """
    profile名を解決し、設定ファイルのパスを返す。

    profile未指定:
        defaultを使用

    指定profileが存在しない:
        defaultへフォールバック

    profileディレクトリは存在するが
    必須ファイルが不足:
        FileNotFoundError
    """
    (
        requested_profile,
        resolved_profile,
        fallback_used,
    ) = resolve_profile_name(
        profile_name
    )

    profile_dir = (
        CONFIG_DIR / resolved_profile
    )

    config = ProfileConfig(
        requested_profile=requested_profile,
        resolved_profile=resolved_profile,
        profile_dir=profile_dir,
        prompt_path=CONFIG_DIR / "prompt.txt",
        glossary_path=(
            profile_dir / "glossary.json"
        ),
        style_path=(
            profile_dir / "style.txt"
        ),
        noise_path=(
            profile_dir / "noise.json"
        ),
        noise_local_path=(
            profile_dir / "noise.local.json"
        ),
        fallback_used=fallback_used,
    )

    validate_profile_config(config)

    return config
