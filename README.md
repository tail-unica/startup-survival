# The Future Is Not a Feature: A Look-Ahead Bias-Free Evaluation Framework for Startup Success Prediction through Machine Learning

This repository contains the code, the released datasets, and the reproducible
pipeline accompanying the paper *"The Future Is Not a Feature: A Look-Ahead
Bias-Free Evaluation Framework for Startup Success Prediction through Machine
Learning."*

## Overview

Predicting startup success from early-stage signals is a popular machine-learning
task, but many published pipelines are contaminated by **look-ahead bias**:
features computed over the whole life of a company leak information that would not
have been available at prediction time, inflating the reported performance.

This project proposes and empirically validates a **bias-free evaluation
framework**. Starting from a firm-level *panel* dataset, we engineer features and
the prediction target under a strict temporal constraint (a fixed look-back
**time window**), so that every feature is computed only from information
available up to the decision point. We then quantify the impact of look-ahead
bias by comparing, on the same firms and models, two dataset constructions:

- **`window`** — the bias-controlled dataset (features restricted to a time
  window, no look-ahead);
- **`nowindow`** — the same firms with cumulative, full-history features (i.e.
  with look-ahead bias).

`window` and `nowindow` differ in the features, in the definition of the target
and, as a consequence of the latter, in the base rate (0.321 against 0.405), so
the gap between them is the sum of two leaks. Two control settings complete the
2x2 and separate them, each derived from the two processed datasets by swapping
the target on `CompanyID` (same firms, same columns):

- **`leaklabel`** — bias-free features, leaked target ("does the firm ever reach
  the next stage" instead of "within 7 years of `StartingAge`");
- **`leakfeat`** — leaked features, bias-free target.

AUC is the metric to read across the label axis: no model tunes its decision
threshold, so F1, precision and recall move with the base rate on their own.

Two further ablation settings probe the contribution of specific feature
families:

- **`noteam`** — team-related features removed;
- **`nocompetitors`** — competition-related features removed.

For every setting we train seven model families — Random Forest (`rf`), LightGBM
(`lgb`), a Multi-Layer Perceptron (`mlp`), Decision Tree (`dt`), Logistic
Regression (`lr`), an RBF-kernel Support Vector Machine (`svm`), and TabPFN
(`tabpfn`), tune them with a Weights & Biases sweep, and interpret them
with **SHAP**. Differences in feature importance across settings are tested for
significance with the **Wilcoxon signed-rank test**.

## Data availability

The original initial panel is derived from **PitchBook** data. Due to licensing
and intellectual-property restrictions, **the original panel cannot be
redistributed**. To keep the repository fully runnable and transparent, we
release the following:

| File | Content | Status |
|------|---------|--------|
| `data/raw/example_panel.csv` | **Example panel with non-real data.** Synthetic records that reproduce the schema, column types, value ranges, and the panel (multi-row-per-firm) structure of the original input, so that the feature/target-engineering code can be executed end to end. The values do **not** correspond to any real company. | Released |
| `data/processed/dataset_window.csv` | **Final bias-controlled dataset** (`window`) used in all experiments. | Released |
| `data/processed/dataset_nowindow.csv` | **Final dataset with look-ahead bias** (`nowindow`) used in all experiments. | Released |
| `data/raw/panel.csv.gz` | Original PitchBook-derived panel. | **Not released** |
| `data/raw/QS_World_Rankings.csv` | QS World University Rankings, used to flag top-tier institutes. | See QS terms |

The two released **final datasets** are the exact inputs to every model and
experiment reported in the paper, so all quantitative results can be reproduced
without access to the original panel. The example panel is provided only to
demonstrate and re-run the upstream feature-engineering pipeline.


## Repository structure

```
.
├── config/
│   └── config.yaml               # Paths, time window, split, seeds, frequency encoding, sweep
├── data/
│   ├── raw/
│   │   ├── example_panel.csv     # Synthetic example panel (released)
│   │   ├── QS_World_Rankings.csv # University ranking, to flag top-tier institutes
│   │   └── pitchbook/            # PitchBook extraction (not released)
│   ├── reference/                # R intermediates, ground truth for the port (not released)
│   ├── interim/                  # Per-stage outputs of the panel pipeline
│   └── processed/
│       ├── dataset_window.csv    # Final bias-controlled dataset (released)
│       └── dataset_nowindow.csv  # Final look-ahead-biased dataset (released)
├── docs/                         # Design specs and implementation plans
├── scripts/
│   ├── build_panel.py            # Builds the panel from the raw PitchBook CSVs
│   ├── build_datasets.py         # Rebuilds the two processed datasets from the panel
│   ├── check_extraction.py       # Checks a candidate extraction against the reference
│   └── derive_europe_mapping.py  # Recovers the country->continent verdict the R used
├── src/
│   ├── preprocessing.py          # Feature & target engineering, time window, imputation
│   ├── encoding.py               # Frequency encoding, fitted per split (no leakage)
│   ├── utils.py                  # Splits, plotting, SHAP comparison, Wilcoxon test, metrics
│   ├── training.py               # One sweep run: build, fit, log, explain
│   ├── models/                   # One class per model family
│   │   ├── base.py               # The interface: matrices, fit, score, explain
│   │   ├── sklearn_models.py     # RF, LightGBM, Decision Tree, LR, SVM
│   │   ├── mlp.py                # The network and its training loop
│   │   └── tabpfn.py             # TabPFN v2, run locally
│   ├── panel/                    # Panel construction pipeline, ported from R
│   │   ├── config.py             # Paths and the bug flags, all defaulting to R behaviour
│   │   ├── rutils.py             # R/dplyr semantics that polars does not share
│   │   ├── io.py                 # Schema-explicit readers, row-count guards
│   │   ├── validate.py           # Per-column verification against the R intermediates
│   │   └── stage1..stage7_*.py   # The seven stages, one parquet each
│   └── RCode/                    # Original R scripts, kept for reference (not released)
├── tests/                        # pytest suite for src/
├── notebook.ipynb                # Main reproducible pipeline (end to end)
├── pyproject.toml                # Dependencies, ruff configuration
├── uv.lock                       # Exact resolution, committed
├── .env                          # Template for the environment variables
└── README.md
```

`src/RCode/` holds the original R scripts the pipeline was translated from and
`data/reference/` the intermediates its output is checked against; neither is
redistributable, so both are absent from a fresh clone.

## Panel construction

`scripts/build_panel.py` rebuilds `data/interim/panel.csv.gz` from the raw
PitchBook extraction. It is a faithful port of the two R scripts that used to
produce the panel, plus the two post-processing steps that followed them.

```bash
uv run python scripts/check_extraction.py data/raw/pitchbook   # right vintage?
uv run python scripts/build_panel.py --verify                  # ~8 minutes
uv run python scripts/build_panel.py --from 4                  # resume at a stage
```

Seven stages, each reading the previous one's parquet from `data/interim/` and
writing its own, so re-running one does not force the others. Each stage runs in
its own interpreter: end to end in a single process the pipeline peaks past this
machine's memory.

Nothing writes to `data/raw/` or `data/reference/`. The finished panel lands in
`data/interim/panel.csv.gz`; `data/raw/panel.csv.gz` is the reference it is
compared against and stays untouched.

### Verification

`--verify` checks each stage against the file the R produced, column by column,
aligned by key. `uv run python -m src.panel.validate --checkpoint C` runs one on
its own.

| checkpoint | stage | reference | rows | result |
|---|---|---|---|---|
| A | 2 | `db3.csv` | 534,851 | 52/52 columns identical |
| B | 3 | `db_master_1.csv` | 116,920 | 34/34 identical |
| C | 5 | `db_master_2.csv` | 1,001,625 | 107/107 identical |
| D | 5 | `db_selected.csv` | 1,001,625 | 89/89 identical |
| E | 6 | `db_master_panel.csv.gz` | 882,324 | 112/113 identical |
| F | 7 | `data/raw/panel.csv.gz` | 882,324 | 107/114 identical |

Three groups of columns are declared rather than reproduced, and every run
prints them:

- **The six `TotalRaised_Est*` columns are not produced.** The R filled missing
  deal amounts with a `randomForest` fitted with no seed, across all years, on
  the whole dataset before any split, using the deal type — which determines the
  target — as a predictor. `src/panel/stage4_deals.py` carries a comment at the
  exact line naming the seven columns it created; missing amounts are now left
  missing for the imputation that already runs before training.
- **`StageBlock` at E and F.** Its value in those files matches neither a
  recomputation on `GrowthStage` nor one on the grouped stage nor
  `db_selected`'s own. Nothing downstream reads it.
- **The six competitor columns at F.** They were computed from a different
  download of `CompanySimilarRelation.csv`: checkpoint B reproduces every
  competitor aggregate in `db_master_1.csv` exactly from the extraction we have,
  yet 655,869 rows over 84,138 companies carry the same competitor count as the
  published panel and a different mean similarity, in both directions.

### Reproducing the R's defects

The R has ten known defects. All are reproduced by default, each behind a
`fix_*` flag in `PanelConfig` that defaults to `False`, where `False` means "do
what the R did". `docs/superpowers/specs/` holds the register with the measured
impact of each. The largest is B1: the team panel and the deal table cut at
`YearFounded > 2000` while the rest of the pipeline cuts at `> 1999`, so the
entire 2000 founding cohort reaches the models with no team data at all.

## Installation

The code was developed and tested with **Python 3.12**. Dependencies are managed
with [uv](https://docs.astral.sh/uv/); install it first if you do not have it
(`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
git clone https://github.com/tail-unica/startup-survival
cd startup-survival
uv sync
```

`uv sync` creates `.venv`, installs the exact versions recorded in `uv.lock`,
and installs the project itself, so `import src...` works from any directory.
There is no separate virtual-environment step and nothing to activate: prefix
commands with `uv run`, or activate `.venv` by hand if you prefer.

Key dependencies (`uv.lock` holds the exact resolution): `polars`, `pandas`,
`numpy`, `scikit-learn`, `lightgbm`, `torch`, `tabpfn`, `shap`, `matplotlib`,
`seaborn`, `scipy`, `statsmodels`, `wandb`, `PyYAML`, `python-dotenv`,
`joblib`.

## Development

```bash
uv run pytest            # test suite
uv run ruff check .      # lint
uv run ruff format .     # format
uv run jupyter lab       # open notebook.ipynb
```

`tests/` covers the feature and target engineering (`src/preprocessing.py`), the
per-split frequency encoding (`src/encoding.py`), the split/imputation/scaling
cache and the SHAP comparison and Wilcoxon machinery (`src/utils.py`), the seven
model families (`src/models/`), and the panel pipeline configuration
(`src/panel/`). The model tests run without W&B, without the datasets and
without a GPU.

### Adding a model family

Write a class in `src/models/` answering the four questions the training loop
asks — which matrix it is fitted on (`wants_scaled`), how it is built from the
sweep config (`from_sweep`), how it turns probabilities into labels (`score`),
and which SHAP explainer stays tractable on it (`explain`) — then register it in
`src/models/__init__.py` and add its hyperparameters to `sweep_settings` in
`config/config.yaml` under a `<family>_` prefix. `src/training.py` does not
change.

`tests/test_preprocessing.py` is currently skipped at module level: it targets a
function that was removed when category collapsing moved per split, and it is
re-pointed together with the panel pipeline port. The skip message says so, and
the suite reports it on every run.

Lint and format rules live in `pyproject.toml` (`ruff`, line length 100).

## Configuration

Experiment tracking and hyperparameter sweeps use **Weights & Biases**. 

If you don't have an account, create one at https://wandb.ai/site

Then follow the instructions on the website to create an **entity** and a **project**

Fill the `.env` with your own entity and project.

Then log in once:

```bash
wandb login
```

All paths, the look-back window (`time_window`), the final year
(`last_year`), the test split (`test_size`), the seeds (`seeds`), the
frequency-encoding settings (`frequency_encoding`) and the hyperparameter grids
are defined in `config/config.yaml`.

## Step-by-step usage

The workflow is driven by **`notebook.ipynb`**, which orchestrates the code in
`src/`: run its sections in order.

```bash
uv run jupyter lab notebook.ipynb
```

### 1. Dataset creation (optional, from the panel)

The first cells of the notebook rebuild the two final datasets from the input
panel via the functions in `src/preprocessing.py` (`build_windowed_dataset`,
`build_full_history_dataset`, `preprocess_dataset`). They ship commented out,
because without the original panel there is nothing to rebuild. The same
pipeline runs as a script:

```bash
uv run python scripts/build_datasets.py
```

The processed datasets carry `HQCountry` and `PrimaryIndustrySector` as **raw
categories**: they are collapsed and frequency-encoded per split (see below), so
the categories themselves have to survive preprocessing.

- With the original panel this regenerates `dataset_window.csv` and
  `dataset_nowindow.csv`.
- Without the original panel, point `config['paths']['raw_dataset']` to
  `data/raw/example_panel.csv` to run the pipeline on the synthetic example.

> **Note.** The released `data/processed/dataset_window.csv` and
> `data/processed/dataset_nowindow.csv` already contain the final datasets, so
> this step can be **skipped** to reproduce the paper's results directly.

### 2. Dataset selection

Choose one experiment by running the corresponding cell, which loads the dataset
and sets a `tag`:

- `window` — bias-controlled (loads `dataset_window.csv`);
- `nowindow` — look-ahead bias (loads `dataset_nowindow.csv`);
- `noteam` — `dataset_window.csv` with team features dropped;
- `nocompetitors` — `dataset_window.csv` with competition features dropped;
- `leaklabel` — bias-free features with the leaked target merged in on `CompanyID`;
- `leakfeat` — leaked features with the bias-free target.

Which columns count as team features and which as competition features is set by
`ablations` in `config/config.yaml`.

### 3. Split, imputation and scaling

Run the split/imputation/scaling cell. It separates `CompanyID` and `Target`,
builds **one train/validation/test split per evaluation seed** (`seeds` in
`config.yaml`, currently `[1, 2, 3, 4, 5]`), each with its own frequency
encoding, KNN imputer and `RobustScaler` fitted on that seed's training set
alone. Splits are cached under `tmp/splits`, so the cell is slow only the first
time for a given experiment.

#### Frequency encoding

`HQCountry` and `PrimaryIndustrySector` are turned into the **share of the
training rows** their category holds, by `src/encoding.py`. The encoding is
fitted on the training split and then *looked up* for validation and test, so a
held-out row never contributes to its own encoding — the same look-ahead
leakage the time window exists to avoid, applied to the features instead of the
target.

Categories below `frequency_encoding.min_frequency` (a share, default `0.05`)
are pooled into a single `Others` bucket, and so are the categories a held-out
split shows for the first time and the missing values. Shares rather than raw
counts, because the four experiments have different row counts and the paper
compares them — including their SHAP importances — so the feature has to carry
the same units in all of them.

#### Regenerating the cached splits

The cache file name carries a fingerprint of the dataset columns and of the
encoding settings, so editing `frequency_encoding` or regenerating the processed
CSVs invalidates the cache on its own. To rebuild on purpose — after changing
`prepare_splits` itself, say — set `REBUILD_SPLITS = True` in the split cell; it
calls `clear_split_cache(tag=tag)` and recomputes. `get_split(..., force=True)`
does the same for a single split.

### 4. Hyperparameter sweep and training

Initialize the W&B sweep, then start the agent:

- Set `number_of_runs = 35` to evaluate **all seven models with the best
  configuration** already stored in `config/config.yaml` (make sure all model
  types are enabled in `model_type`). The grid crosses the 7 models with the 5
  evaluation seeds, so each model is replicated on five independent splits.
- Set `number_of_runs = 70` and a single fixed `model_type` to **search** for the
  best hyperparameters.

### Seeds

`seeds` in `config.yaml` is the **only** place a seed is written. The sweep
block references it (`values: *seeds`) and the split cell prepares exactly those
splits, so the two can never drift apart. Set it to match what you are running:

| Running | `seeds` |
|---|---|
| Evaluation (`number_of_runs = 35`) | `[1, 2, 3, 4, 5]` |
| Hyperparameter search (`number_of_runs = 70`) | `[12]` |

Each run draws one seed, and it drives **both** the split and the model's
randomness, so a model's five evaluation runs are five independent replications
rather than five reruns of one partition. Result tables report **mean ± std**
across them.

Hyperparameter search is deliberately *not* multi-seed, and its seed is
deliberately not one of the evaluation five: several seeds would let the
optimiser treat the seed as a hyperparameter and report the luckiest split, and
tuning on a split you later report on inflates it.

Each run trains the model, logs validation/test metrics and plots to W&B, and
computes SHAP values (stored under the current `tag` for later comparison).
`rf`/`lgb`/`dt` use `TreeExplainer`, `lr` uses `LinearExplainer`, `mlp` uses
`KernelExplainer`, and `svm`/`tabpfn` use `PermutationExplainer` — the only
explainer that stays tractable on an RBF kernel and on a model that runs a
full forward pass per evaluation. Its budget is set by `shap_permutation` in
`config/config.yaml`.

SHAP is computed on the **first evaluation seed only**: these tables ask which
features move when the window is removed, not how much the metrics vary, and
`compute_wilcoxon_table` pairs rows within a single explained sample, so pooling
seeds would change what the test measures.

`tabpfn` runs TabPFN v2 locally, on the same `tabpfn-v2-classifier-v2_default`
checkpoint the Prior Labs API used to serve; it is downloaded to the TabPFN
cache on first use (override the location with `TABPFN_MODEL_CACHE_DIR`).

It needs a **CUDA GPU**: `device="auto"` falls back to CPU, where TabPFN past a
thousand rows is impractical. The v2 checkpoint declares a 10k-row pretraining
limit and the training split holds ~18k, so the notebook passes
`ignore_pretraining_limits=True` — the same 50k allowance the API applied to
this model, which keeps the training set identical to the one the other models
see instead of subsampling it.

### 5. Cross-experiment comparison

After running the relevant settings (e.g. both `window` and `nowindow`), the
final cells produce:

- **Metric comparison tables** (`compare_metrics`) reporting each metric for two
  settings and the percent change relative to `window`;
- **Metric tables** (`compare_metrics`) reporting mean ± std across seeds, with
  the percent change computed on the means;
- **SHAP comparison plots** (`plot_shap_comparison`) across settings;
- **Wilcoxon signed-rank tests** (`compute_wilcoxon_table`) on the SHAP feature
  importances, to assess whether the differences are statistically significant.

To reproduce a comparison you must run both settings first, so that the in-memory
`metrics_store` and `shap_store` contain the data for both.

## Reproducing the paper's results

1. Install the environment and configure W&B (see above).
2. Skip dataset creation and use the released
   `data/processed/dataset_{window,nowindow}.csv`.
3. Run each of the four settings (`window`, `nowindow`, `noteam`,
   `nocompetitors`) with `number_of_runs = 5`.
4. Execute the comparison cells to obtain the metric tables, SHAP plots, and
   Wilcoxon tests.

## Citation

If you use this code or the released datasets, please cite the accompanying
paper:

```bibtex
@article{startup_survival,
  title   = {The Future Is Not a Feature: A Look-Ahead Bias-Free Evaluation
             Framework for Startup Success Prediction through Machine Learning},
  authors  = {Casti Giulio and Mandas Marco and Piras Luca and Marras Mirko and Fenu Gianni},
  year    = {2026}
}
```

## License and data rights

The source code is released for academic and research use. The original
PitchBook-derived panel is **not** included and remains subject to PitchBook's
licensing terms; the QS World University Rankings are subject to QS's terms of
use. Only the synthetic example panel and the two final, de-identified datasets
are distributed with this repository.
