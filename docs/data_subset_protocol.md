# 03 - MedleyDB subset protocol

This methodology document defines exactly how the isolated-stem subset CSV files
are produced. It applies equally to local and Colab runs. The generated
`subset_report_<profile>_<granularity>.md` records the outcome for the
particular MedleyDB copy used.

The supported subset profiles are:

- `debug`: small sanity checks; useful for testing the pipeline quickly, not for
  final report numbers;
- `largest_balanced`: the largest balanced subset possible under the configured
  class, segment, track-diversity, split, and RMS constraints.

The final isolated-stem report should prioritize
`largest_balanced + medleydb_instrument`. `largest_balanced + coarse_family` is
a controlled comparison with broader labels.

## 1. Index only official stem entries

The indexer prefers
`<root>/Audio/<track>/<track>_METADATA.yaml` or the equivalent direct-layout
metadata beside each track. It reads the metadata's `stem_dir`, stem filename,
instrument, genre, and bleed flag. Mixes and raw microphone tracks are never
training examples. A separate `medleydb-master` checkout is not required;
`--medleydb-code-root` exists only as an optional fallback for unusual copies
whose local YAML metadata is missing.

Every declared stem receives an index row. Missing, empty, unreadable, truncated, or malformed files are marked invalid with an error. Stem WAV files that are present but absent from metadata are reported as orphans and receive `other_unknown`.

Paths in CSV files are relative to `--medleydb-root`, so generated metadata is
portable between Windows, Colab, Lightning AI, and other machines when the
correct dataset root is supplied.

## 2. Normalize labels explicitly

Two label granularities are supported:

- `coarse_family`: broad instrument families used for controlled debugging and
  cheaper ablations;
- `medleydb_instrument`: lightly normalized original MedleyDB instrument labels
  used for the more meaningful instrument-classification protocol.

`normalize_label(raw_label, "coarse_family")` uses exact, documented label
sets. It does not use substring guesses. The families are:

- `vocals`: singers, speakers, rappers, choirs, and related voice labels;
- `guitar`: acoustic, clean/distorted electric, slide, and lap-steel guitar;
- `bass`: electric and double bass;
- `drums`: drum set, kick, snare, bass drum, and toms;
- `keyboards`: acoustic/electric/tack piano, harpsichord, and organs;
- `strings`: bowed strings and non-guitar plucked/struck string instruments;
- `winds_brass`: flutes, reeds, brass, and free reeds;
- `percussion`: non-kit drums, idiophones, and auxiliary percussion;
- `electronic`: synthesizer, drum machine, sampler, scratches, and processed sound.

Anything else is `other_unknown` and excluded from model classes.

`normalize_label(raw_label, "medleydb_instrument")` preserves the metadata label
as much as possible: it lowercases, strips whitespace, replaces separators with
underscores, and merges only explicit spelling-level variants such as
`hi hat -> high_hat`. It does not collapse guitar, bass, strings, brass, or
voice labels into broad families.

## 3. Filter files and enumerate candidates

By default the script keeps only valid, known-label stems from tracks with `has_bleed: no`. `--allow-bleed` includes metadata-labelled bleed tracks. Unknown bleed metadata is not itself a rejection, but normally coincides with unusable metadata.

The script enumerates only complete windows. With the defaults, starts are `0, 5, 10, ...` seconds and no short tail is padded into the subset.

Classes need at least three contributing tracks and at least
`--min-segments-per-class` candidate windows in the selected label granularity.
Eligible classes are ordered by distinct track count, candidate count, then
label name. `--max-classes` is a maximum rather than a promise that weak classes
will be forced into the dataset.

## 4. Split songs before checking audio segments

The unique tracks contributing to selected classes are repeatedly assigned to train, validation, and test using the requested ratios and seed. The search rejects any mapping that leaves a class absent from a split and scores candidate capacity and distribution error. One mapping is applied globally, including tracks that contain stems from several classes.

This order is essential: no later balancing step may move an individual segment to another split.

## 5. Reject silence and balance exactly

Within each class and split, candidate offsets are shuffled deterministically and visited round-robin across tracks. Audio is loaded without peak normalization and its whole-segment RMS is measured. Segments below `-50 dBFS` are considered mostly silent.

The script collects up to the requested split quota, then finds the largest common class total between the configured minimum and maximum that every class can support in every split. It samples exact per-split quotas for every class. If a class cannot support the minimum, it is replaced by the next ranked eligible family and the track split is rebuilt.

The final label IDs are assigned alphabetically. Profile-specific maps are
written, for example
`labels_largest_balanced_medleydb_instrument_label_to_id.json`. Re-running with
identical inputs and seed produces the same table.

## 6. Required audit checks

Before trusting a subset, read the relevant report, for example
`data/reports/subset_report_largest_balanced_coarse_family.md` or
`data/reports/subset_report_largest_balanced_medleydb_instrument.md`, and
confirm:

- the expected number of files is valid;
- selected classes are appropriate;
- every class has identical total size;
- all three splits contain every class;
- train/validation, train/test, and validation/test track overlaps are zero;
- silence and decode rejection counts are plausible.

Also read `data/reports/label_mapping_report.md`. It lists each raw metadata
label, its `medleydb_instrument` label, its `coarse_family` label, and how many
stems, tracks, and candidate segments support it. This is the quick audit that
the instrument-level protocol has not accidentally been collapsed into broad
families.

## 7. Capped-natural alternative

`--sampling-strategy capped_natural` preserves the same class eligibility and
global track split. It first reserves the configured minimum class/split quotas,
then fills one global budget (`max_classes x max_segments_per_class`) by visiting
tracks deterministically without forcing equal class totals. This makes class
frequency less artificial while preventing an unbounded dataset expansion.

Use a separate metadata/report/cache directory for this strategy. The training
config may then use ordinary shuffling, inverse-frequency loss, or a weighted
train sampler. Validation and test loaders are never weighted.

## 8. Comparing protocols

Do not compare scores from `coarse_family` and `medleydb_instrument` as if they were the same task. Coarse-family classification has fewer and broader classes; MedleyDB-instrument classification has a more detailed and harder label space. Macro-F1 remains the primary metric in both settings because it gives equal importance to every selected class.
