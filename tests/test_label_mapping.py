"""Tests for the explicit MedleyDB coarse taxonomy."""

import pytest

from src.data.label_mapping import OTHER_LABEL, normalize_instrument_label, normalize_label


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("female singer", "vocals"),
        ("  MALE   RAPPER ", "vocals"),
        ("clean electric guitar", "guitar"),
        ("double_bass", "bass"),
        ("drum set", "drums"),
        ("tack piano", "keyboards"),
        ("violin section", "strings"),
        ("tenor saxophone", "winds_brass"),
        ("tabla", "percussion"),
        ("fx/processed sound", "electronic"),
    ],
)
def test_known_labels(raw: str, expected: str) -> None:
    assert normalize_instrument_label(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "Main System", "imaginary lute"])
def test_unknown_labels_are_not_guessed(raw: str | None) -> None:
    assert normalize_instrument_label(raw) == OTHER_LABEL


def test_label_granularity_dispatch() -> None:
    assert normalize_label("Clean Electric Guitar", "coarse_family") == "guitar"
    assert normalize_label("Clean Electric Guitar", "medleydb_instrument") == "clean_electric_guitar"
    assert normalize_label(" hi   hat ", "medleydb_instrument") == "high_hat"
    assert normalize_label("", "medleydb_instrument") == OTHER_LABEL
