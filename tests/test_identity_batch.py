from __future__ import annotations

import pytest

from wic_history.identity_batch import character_jaccard, normalize_identity_surface


def test_identity_normalization_preserves_names_without_declaring_aliases() -> None:
    assert normalize_identity_surface(" Hans Holbein （霍爾平） ") == "hansholbein霍爾平"
    assert normalize_identity_surface("孫文") != normalize_identity_surface("孫中山")


def test_character_blocking_is_high_recall_signal_not_merge_decision() -> None:
    assert character_jaccard("孫文", "孫中山") == pytest.approx(0.25)
    assert character_jaccard("霍爾平", "霍") == pytest.approx(1 / 3)
