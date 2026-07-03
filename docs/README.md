# Documentation reading order

Start with the repository [README](../README.md), then choose the documents
needed for your role or environment.

| Order | Document | Purpose | Who should read it |
|---:|---|---|---|
| 01 | [Project README](../README.md) | Project goal, pipeline overview, outputs, and next experiment | Everyone |
| 02 | [Theory: frozen MERT stem classification](theory_mert_stem_classification.md) | What is being classified and why isolated stems, frozen MERT, caching, balancing, and track-level splitting are used | Everyone before interpreting results |
| 03 | [Subset protocol](data_subset_protocol.md) | Exact indexing, label normalization, filtering, splitting, silence rejection, balanced sampling, and capped-natural sampling | Anyone reviewing data preparation or methodology |
| 04 | [Local CPU/GPU commands](commands.md) | Installation and the five terminal commands for Windows, Linux, macOS, or a local workstation | Anyone running locally |
| 05 | [Google Colab team guide](google_colab.md) | Private GitHub access, shared Drive paths, Colab GPU execution, persistence, and collaboration | Anyone running on Colab |
| 05A | [Executable Colab notebook](../notebooks/pipeline_medleydb.ipynb) | Verified notebook cells corresponding to the Colab guide | Colab users |
| 06 | [Isolated-stem experiments](experiments_isolated_stem.md) | Research questions, controlled variables, improvement criteria, and cost | Anyone running experiments |
| 07 | [Notebooks guide](notebooks_guide.md) | Notebook order, flags, paths, and result conventions | Notebook users |

## Which execution guide should I use?

- **Local Windows/Linux/macOS computer:** use
  [Local CPU/GPU commands](commands.md). A GPU is recommended for Stage 3, but
  the pipeline can use CPU when necessary.
- **Google Colab GPU:** use the
  [Google Colab team guide](google_colab.md) and its notebook. Do not substitute
  local Windows paths into the notebook.
- **Lightning AI or another cloud machine:** start from the local commands, then
  replace `--medleydb-root` with the mounted dataset path and place caches and
  checkpoints on persistent storage.

The theory and subset-protocol documents describe the same model and data
procedure in every environment. Only installation, paths, authentication, and
artifact storage differ.
