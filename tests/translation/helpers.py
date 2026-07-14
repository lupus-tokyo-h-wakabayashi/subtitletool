from pathlib import Path

from lib.profile.noise import NoiseEntry, NoiseDictionary


def build_test_noise_dictionary(
    entries: list[NoiseEntry],
) -> NoiseDictionary:
    return NoiseDictionary(
        profile_name="test",
        entries={
            entry.source: entry
            for entry in entries
        },
        official_path=Path("noise.json"),
        local_path=Path("noise.local.json"),
        local_loaded=False,
    )
