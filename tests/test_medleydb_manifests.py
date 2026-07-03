from pathlib import Path

from src.data.build_stem_index import _tracklist_coverage


def _names(filename: str) -> set[str]:
    path = Path(__file__).parents[1] / "src" / "data" / "resources" / filename
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def test_bundled_release_manifests_have_expected_counts() -> None:
    v1 = _names("tracklist_v1.txt")
    v2 = _names("tracklist_v2.txt")

    assert len(v1) == 122
    assert len(v2) == 74
    assert v1.isdisjoint(v2)


def test_tracklist_coverage_does_not_need_external_code_repo() -> None:
    local_tracks = {"AClassicEducation_NightOwl", "Allegria_MendelssohnMovement1"}
    coverage = {name: (expected, present) for name, expected, present in _tracklist_coverage(local_tracks)}

    assert coverage["tracklist_v1"] == (122, 1)
    assert coverage["tracklist_v2"] == (74, 1)
