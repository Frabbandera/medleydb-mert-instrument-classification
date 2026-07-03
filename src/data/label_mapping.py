"""Map detailed MedleyDB instrument names to a compact first taxonomy.

The official metadata contains many labels with very few examples.  The first
debug model therefore keeps musically meaningful families while preserving
separate guitar, bass, drum-kit, keyboard, and non-kit percussion classes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

OTHER_LABEL = "other_unknown"


_LABEL_GROUPS: Mapping[str, frozenset[str]] = {
    "vocals": frozenset(
        {
            "male singer",
            "female singer",
            "male speaker",
            "female speaker",
            "male rapper",
            "female rapper",
            "beatboxing",
            "vocalists",
            "choir",
            "crowd",
            "male screamer",
            "female screamer",
        }
    ),
    "guitar": frozenset(
        {
            "acoustic guitar",
            "clean electric guitar",
            "distorted electric guitar",
            "slide guitar",
            "lap steel guitar",
        }
    ),
    "bass": frozenset({"electric bass", "double bass"}),
    "drums": frozenset({"drum set", "snare drum", "kick drum", "bass drum", "toms"}),
    "keyboards": frozenset(
        {
            "piano",
            "tack piano",
            "electric piano",
            "harpsichord",
            "electronic organ",
            "pipe organ",
            "harmonium",
        }
    ),
    "strings": frozenset(
        {
            "erhu",
            "violin",
            "viola",
            "cello",
            "violin section",
            "viola section",
            "cello section",
            "string section",
            "dilruba",
            "banjo",
            "guzheng",
            "harp",
            "liuqin",
            "mandolin",
            "oud",
            "ukulele",
            "zhongruan",
            "sitar",
            "dulcimer",
            "yangqin",
        }
    ),
    "winds_brass": frozenset(
        {
            "dizi",
            "flute",
            "flute section",
            "piccolo",
            "bamboo flute",
            "panpipes",
            "recorder",
            "alto saxophone",
            "baritone saxophone",
            "bass clarinet",
            "clarinet",
            "clarinet section",
            "tenor saxophone",
            "soprano saxophone",
            "oboe",
            "english horn",
            "bassoon",
            "bagpipe",
            "trumpet",
            "cornet",
            "trombone",
            "french horn",
            "euphonium",
            "tuba",
            "brass section",
            "french horn section",
            "trombone section",
            "horn section",
            "trumpet section",
            "harmonica",
            "concertina",
            "accordion",
            "bandoneon",
            "melodica",
        }
    ),
    "percussion": frozenset(
        {
            "triangle",
            "sleigh bells",
            "cowbell",
            "cabasa",
            "high hat",
            "hi hat",
            "gong",
            "guiro",
            "gu",
            "cymbal",
            "chimes",
            "castanet",
            "claps",
            "rattle",
            "shaker",
            "maracas",
            "xylophone",
            "vibraphone",
            "marimba",
            "glockenspiel",
            "whistle",
            "snaps",
            "timpani",
            "bongo",
            "conga",
            "tambourine",
            "darbuka",
            "doumbek",
            "tabla",
            "auxiliary percussion",
        }
    ),
    "electronic": frozenset(
        {
            "synthesizer",
            "drum machine",
            "theremin",
            "fx/processed sound",
            "scratches",
            "sampler",
        }
    ),
}

_NORMALIZED_TO_GROUP = {
    instrument: group for group, instruments in _LABEL_GROUPS.items() for instrument in instruments
}


def clean_raw_label(raw_label: str | None) -> str:
    """Normalize spelling-neutral formatting before taxonomy lookup."""

    if raw_label is None:
        return ""
    label = str(raw_label).strip().lower().replace("_", " ")
    label = re.sub(r"\s+", " ", label)
    return label


def normalize_instrument_label(raw_label: str | None) -> str:
    """Return a stable coarse label or ``other_unknown``.

    The mapping is deliberately explicit. Substring matching would silently
    turn new or ambiguous metadata into a confident target class.
    """

    return _NORMALIZED_TO_GROUP.get(clean_raw_label(raw_label), OTHER_LABEL)


_INSTRUMENT_SYNONYMS: Mapping[str, str] = {
    # MedleyDB metadata occasionally uses both spellings.  These are spelling
    # normalizations, not broad family merges.
    "hi hat": "high_hat",
    "fx/processed sound": "fx_processed_sound",
}


def normalize_medleydb_instrument_label(raw_label: str | None) -> str:
    """Return a lightly normalized MedleyDB instrument label.

    This protocol keeps the metadata label as close as possible to the original
    annotation.  It lowercases, trims whitespace, replaces separators with
    underscores, and applies only explicit spelling-level synonym merges.
    """

    cleaned = clean_raw_label(raw_label)
    if not cleaned:
        return OTHER_LABEL
    normalized = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
    if not normalized:
        return OTHER_LABEL
    return _INSTRUMENT_SYNONYMS.get(cleaned, normalized)


def normalize_label(raw_label: str | None, granularity: str = "coarse_family") -> str:
    """Normalize one raw MedleyDB label for a selected experimental protocol."""

    if granularity == "coarse_family":
        return normalize_instrument_label(raw_label)
    if granularity == "medleydb_instrument":
        return normalize_medleydb_instrument_label(raw_label)
    raise ValueError("granularity must be 'coarse_family' or 'medleydb_instrument'")


def taxonomy_groups() -> dict[str, tuple[str, ...]]:
    """Return a serializable copy of the documented taxonomy."""

    return {name: tuple(sorted(labels)) for name, labels in _LABEL_GROUPS.items()}
