# 05 - Running the pipeline on Google Colab

This guide follows the verified notebook in
`notebooks/pipeline_medleydb.ipynb`. It is intended for team members using:

- the private GitHub repository `filippoLonghi/medleydb-mert-instrument-classification`;
- a shared project folder visible inside Google Drive;
- Colab Secrets for GitHub authentication;
- the full MedleyDB stem copy for the real frozen-MERT experiment, with a small
  debug option only for checking a partial upload.

The repository is cloned into Colab's temporary `/content` storage for speed.
MedleyDB and generated experiment artifacts are synchronized to Google Drive so
they can survive runtime resets and be inspected by teammates.

This is the Colab execution guide. Local Windows, Linux, or macOS users should
follow [the local CPU/GPU commands](commands.md). The shared conceptual
background is in the [theory document](theory_mert_stem_classification.md), and
the exact data methodology is in the [subset protocol](data_subset_protocol.md).

## 1. One-time setup

### Google Drive layout

Each team member must be able to see this folder structure:

```text
MyDrive/
  medleydb_mert_project/
    archive_debug_runs/
    MedleyDB/
      Audio/
        <track_id>/
          <track_id>_METADATA.yaml
          <track_id>_STEMS/
            *.wav
    isolated_stem_v1/
      data/
        metadata/
        cache/
        reports/
      checkpoints/
      results/
```

The indexer also supports a direct `MedleyDB/<track_id>/...` layout. The
`Audio/<track_id>/...` layout above matches the verified setup.

If another person shared the project folder with you, add a shortcut to it in
**My Drive** and keep the name `medleydb_mert_project`.
Otherwise the verified `/content/drive/MyDrive/...` paths will not exist.

Do not upload MedleyDB audio, cached embeddings, or checkpoints to GitHub.

### Private GitHub access

Each team member should use their own GitHub token:

1. Create a fine-grained personal access token on GitHub.
2. Grant it access to this private repository with read-only **Contents**
   permission. Add write permission only if that person must push changes.
3. In Colab, open the key icon named **Secrets** in the left sidebar.
4. Add a secret named exactly `GITHUB_TOKEN`.
5. Enable notebook access for that secret.

Never paste a token directly into a notebook cell, commit it, print it, or share
it with another team member.

## 2. Start the Colab runtime

Open `notebooks/pipeline_medleydb.ipynb` in Colab. Select
**Runtime -> Change runtime type -> GPU**, then run the following cells in order.

### Mount Drive and define the verified paths

```python
from google.colab import drive
drive.mount("/content/drive")

import os
from pathlib import Path

MEDLEYDB_ROOT = "/content/drive/MyDrive/medleydb_mert_project/MedleyDB"
RUN_ROOT = "/content/drive/MyDrive/medleydb_mert_project/isolated_stem_v1"
PERSIST_ROOT = RUN_ROOT
os.environ["HF_HOME"] = f"{PERSIST_ROOT}/huggingface"

SUBSET_PROFILE = "largest_balanced"
LABEL_GRANULARITY = "medleydb_instrument"
RUN_PROFILE = "largest_balanced_medleydb_t4"
EXPERIMENT_CONFIG = "configs/experiments/isolated_largest_balanced_medleydb_mert95_last_mean_mlp_h256_d02.yaml"
EXPERIMENT_ID = "isolated_largest_balanced_medleydb_mert95_last_mean_mlp_h256_d02_colab"
REPLACE_EXISTING = False
OVERWRITE_EMBEDDINGS = True

print("MedleyDB:", MEDLEYDB_ROOT)
print("Run root:", RUN_ROOT)
```

The notebook cells use Python variables and `subprocess.run([...])` for paths.
If a command is rewritten as a shell command with `!python ...`, keep Drive paths
inside quotes because Google Drive folder names can contain spaces.

For normal Colab testing on a T4, keep `RUN_PROFILE =
"largest_balanced_medleydb_t4"` and use the YAML config
`configs/datasets/subset_largest_balanced_medleydb_instrument.yaml`
as the source of truth for the final subset. The final subset is the
largest balanced MedleyDB-instrument protocol, not a hardcoded
10-class/80-segment debug setting. If a teammate has already
used `EXPERIMENT_ID`, choose a new ID such as
`isolated_largest_balanced_medleydb_mert95_last_mean_mlp_h256_d02_colab_filippo` or deliberately set
`REPLACE_EXISTING = True`.

Use this cache checklist before running expensive cells:

| Situation | What to change |
| --- | --- |
| First full run after a debug cache or fresh Drive folder | `OVERWRITE_EMBEDDINGS = True` |
| Same dataset, same subset, same MERT model/layer/pooling | `OVERWRITE_EMBEDDINGS = False` |
| New classifier-head experiment on the same cache | Change `EXPERIMENT_ID`; keep `OVERWRITE_EMBEDDINGS = False` |
| Changed subset size, segment length, class selection, MERT layer, pooling, or model | Use a new cache directory or set `OVERWRITE_EMBEDDINGS = True` deliberately |
| Replacing an existing registered result | Set `REPLACE_EXISTING = True` only after deciding to overwrite that result |

Optionally verify that Drive contains the expected files:

```python
from pathlib import Path

dataset_path = Path(MEDLEYDB_ROOT)
assert dataset_path.is_dir(), f"MedleyDB directory not found: {dataset_path}"
print("Dataset directory found.")
```

## 3. Clone or update the private repository

This cell reads `GITHUB_TOKEN` from Colab Secrets. On a fresh runtime it clones
the repository. If the repository is already present in the active runtime, it
runs `git pull --ff-only` instead. Authentication is passed only through the Git
subprocess environment and is never embedded in a URL or saved in `.git/config`.

```python
from google.colab import userdata
from pathlib import Path
import base64
import os
import subprocess

token = userdata.get("GITHUB_TOKEN")
if not token:
    raise RuntimeError(
        "GITHUB_TOKEN is unavailable. Add it in Colab Secrets and enable "
        "notebook access."
    )

username = "filippoLonghi"
repo_name = "medleydb-mert-instrument-classification"
REPO_DIR = f"/content/{repo_name}"

public_url = f"https://github.com/{username}/{repo_name}.git"
authorization = base64.b64encode(f"x-access-token:{token}".encode()).decode()
git_env = os.environ.copy()
git_env["GIT_CONFIG_COUNT"] = "1"
git_env["GIT_CONFIG_KEY_0"] = "http.https://github.com/.extraheader"
git_env["GIT_CONFIG_VALUE_0"] = f"AUTHORIZATION: basic {authorization}"

repo_path = Path(REPO_DIR)
if (repo_path / ".git").is_dir():
    subprocess.run(
        ["git", "-C", REPO_DIR, "pull", "--ff-only"],
        env=git_env,
        check=True,
    )
    print("Repository updated.")
else:
    subprocess.run(
        ["git", "clone", public_url, REPO_DIR],
        env=git_env,
        check=True,
    )
    print("Repository cloned.")

del token, authorization
git_env.pop("GIT_CONFIG_VALUE_0", None)

%cd {REPO_DIR}
!python -m pip install -q -r requirements.txt
```

The pull uses `--ff-only`: it updates a clean checkout but refuses to create an
unexpected merge. Commit or discard intentional local edits before pulling.
Remember that the complete `/content` checkout disappears when Colab resets.
Because the token is not stored in `.git/config`, rerun this same secure
clone/update cell for later pulls in the active runtime; a plain `!git pull`
may fail for the private repository.

### Verify GPU support

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

Use Colab's preinstalled CUDA-compatible PyTorch unless an actual compatibility
error occurs. Installing an arbitrary CUDA build can break the matching
`torch` and `torchaudio` packages.

## 4. Run the frozen-MERT pipeline

All commands must run from `REPO_DIR`. The notebook uses
`subprocess.run(..., check=True)` so it stops immediately if a stage fails.
Generated metadata, caches, checkpoints, and results should be written under
`RUN_ROOT`:

```text
/content/drive/MyDrive/medleydb_mert_project/isolated_stem_v1/
```

### Stage 1 - build the stem index

```python
!python -m src.data.build_stem_index \
  --medleydb-root "{MEDLEYDB_ROOT}" \
  --out "{RUN_ROOT}/data/metadata/stem_index.csv" \
  --report-dir "{RUN_ROOT}/data/reports"
```

Read `RUN_ROOT/data/reports/data_health_report.md` and
`RUN_ROOT/data/reports/bad_files.csv`. Missing stems are reported without stopping the
scan. A separate checkout of `medleydb-master` is not required.

### Stage 2 - create the largest-balanced instrument-level subset

For the real isolated-stem final experiment on a full MedleyDB stem upload, use:

```python
!python -m src.data.create_balanced_subset \
  --config configs/datasets/subset_largest_balanced_medleydb_instrument.yaml
```

Whenever the available MedleyDB files or Stage 2 parameters change, rerun
Stages 1 and 2. Then rerun Stage 3 with `--overwrite`, because the selected
segment fingerprint has changed.

Only for a tiny partial upload, switch `DATASET_CONFIG` to a debug dataset YAML
such as `configs/datasets/subset_debug_medleydb_instrument.yaml`. Do not report
that debug run as the real experiment.

### Stage 3 - extract frozen MERT embeddings

```python
!python -m src.features.extract_mert_embeddings \
  --experiment-config configs/experiments/isolated_largest_balanced_medleydb_mert95_last_mean_mlp_h256_d02.yaml \
  --batch-size 1 --device auto
```

Add `--overwrite` when intentionally replacing caches from an earlier subset,
layer, pooling method, or model configuration.

The notebook exposes this as `OVERWRITE_EMBEDDINGS`. Keep it `True` for the
first full run after any debug cache; set it to `False` later to reuse a
matching full cache.

For final reported runs, prefer pinning the MERT revision once a cache has been
created and verified. `model_revision: main` follows the current Hugging Face
default branch; a resolved commit hash is more reproducible across machines and
dates.

### Stage 4 and 5 - train, evaluate, and register

```python
cmd = [
    "python", "-m", "src.experiments.run_experiment",
    "--config", EXPERIMENT_CONFIG,
    "--experiment-id", EXPERIMENT_ID,
]
if REPLACE_EXISTING:
    cmd.append("--replace-existing")
subprocess.run(cmd, check=True)
```

This writes:

```text
results/<EXPERIMENT_ID>/
checkpoints/<EXPERIMENT_ID>/best.ckpt
results/experiment_registry.csv
```

The result directory contains metrics, classification report, raw and
normalized confusion matrices, plots, predictions, and the resolved config.

## 5. Plot and save generated artifacts

The notebook plots the main result at the end:

- test accuracy, macro-F1, and weighted-F1;
- per-class classification report;
- raw confusion matrix;
- normalized confusion matrix;
- per-class F1 bar plot;
- a preview of `predictions.csv`.

The notebook also defines `sync_artifacts_to_drive()` and calls it after the
expensive stages. You can run it manually at the end:

```python
sync_artifacts_to_drive()
```

This saves the selected subset, reports, frozen embeddings, trained head, and
evaluation results in the shared project folder.

## 6. Running the other notebooks

After the first pipeline run has saved artifacts to Drive, the other notebooks
can be opened from GitHub with their **Open in Colab** badge. Their setup cell:

1. mounts Google Drive;
2. clones the private repository into `/content` if needed;
3. restores `data/metadata`, `data/reports`, `data/cache`, `checkpoints`, and
   `results` from `isolated_stem_v1`;
4. syncs new outputs back to Drive after experiment cells are enabled and run.

The follow-up notebooks deliberately keep expensive execution flags such as
`RUN_EXPERIMENTS`, `RUN_EXTRACTION`, and `RUN_FINETUNING` set to `False` by
default. Change only the specific flag needed for the experiment being run.

## 7. Restore a previous experiment

After mounting Drive, cloning the repository, and installing dependencies in a
new runtime, restore the generated files with:

```python
!mkdir -p data/metadata data/reports data/cache checkpoints results
!rsync -a "{PERSIST_ROOT}/data/metadata/" data/metadata/
!rsync -a "{PERSIST_ROOT}/data/reports/" data/reports/
!rsync -a "{PERSIST_ROOT}/data/cache/" data/cache/
!rsync -a "{PERSIST_ROOT}/checkpoints/" checkpoints/
!rsync -a "{PERSIST_ROOT}/results/" results/
```

If the subset and embedding settings match, Stage 3 detects and reuses the
restored cache. Stages 4 and 5 then run without reading audio.

## Team workflow

- Pull repository updates at the beginning of every active Colab session.
- Each person uses their own `GITHUB_TOKEN`; tokens are never shared.
- Treat `isolated_stem_v1` as shared experiment state. Coordinate before
  overwriting caches, checkpoints, or result folders.
- Use a separate `EXPERIMENT_ID` when two people run different configurations
  concurrently. The experiment runner refuses to overwrite an existing ID unless
  `REPLACE_EXISTING = True`.
- Commit only code, configuration, documentation, tests, and small reviewed
  summaries. Never commit Drive paths, audio, downloaded MERT weights, caches,
  checkpoints, or tokens.

## Common problems

### `GITHUB_TOKEN` is unavailable

Check the secret name, enable notebook access, and rerun the authentication
cell. Secret names are case-sensitive.

### Repository not found or authentication failed

Confirm that the token owner can open the private repository and that the token
was granted access to this repository. Organization policies may also require
token approval.

### MedleyDB directory not found

Confirm that the shared folder or its shortcut appears under **My Drive** with
the exact expected name. If the team uses a real Google Shared Drive instead,
change both paths to its mounted location under
`/content/drive/Shareddrives/<drive-name>/...`.

### Too few balanced classes

For a partial debug upload, switch to a debug dataset YAML such as
`configs/datasets/subset_debug_medleydb_instrument.yaml`. The subset still
needs sufficient distinct tracks to create leakage-free train, validation, and
test splits. Inspect
`RUN_ROOT/data/reports/subset_report_largest_balanced_medleydb_instrument.md`
for the exact exclusion reason.

### Existing embedding cache does not match

The dataset or Stage 2 selection changed. Rerun Stage 3 with `--overwrite`,
then retrain and reevaluate.

### Experiment ID already exists

That is a safety feature. Either choose a new `EXPERIMENT_ID` or set
`REPLACE_EXISTING = True` only when the team agrees to replace that result.

### Runtime disconnected

Files already copied to `PERSIST_ROOT` remain in Drive. Files left only under
`/content` are lost, so save artifacts after important runs.
