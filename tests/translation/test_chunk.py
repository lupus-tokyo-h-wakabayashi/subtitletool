from lib.translation.translation_chunk import (
    normalize_translation_text,
)


def test_normalize_translation_text() -> None:
    result = normalize_translation_text(
        "  スターゲイトです。  "
    )

    assert isinstance(
        result,
        str,
    )
    assert result
    assert (
        result.strip()
        == result
    )
