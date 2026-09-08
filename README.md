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
│   └── config.yaml              # Paths, time window, split, seeds, frequency encoding, sweep
├── data/
│   ├── raw/
│   │   └── example_panel.csv     # Synthetic example panel (released)
│   └── processed/
│       ├── dataset_window.csv    # Final bias-controlled dataset (released)
│       └── dataset_nowindow.csv  # Final look-ahead-biased dataset (released)
├── scripts/
│   └── build_datasets.py         # Rebuilds the two processed datasets from the panel
├── src/
│   ├── preprocessing.py          # Feature & target engineering, time window, imputation
│   ├── encoding.py               # Frequency encoding, fitted per split (no leakage)
│   ├── utils.py                  # Splits, plotting, SHAP comparison, Wilcoxon test, metrics
│   └── models/
│       └── MLP.py                # Multi-Layer Perceptron model
├── tests/                        # pytest suite for src/
├── notebook.ipynb                # Main reproducible pipeline (end to end)
├── requirements.txt
├── .env                         # Template for the environment variables
└── README.md
```

## Installation

The code was developed and tested with **Python 3.12**.

```bash
# 1. Clone the repository
git clone https://github.com/tail-unica/startup-survival
cd startup-survival

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install the dependencies
pip install -r requirements.txt
```

Key dependencies (see `requirements.txt` for exact versions): `polars`,
`pandas`, `numpy`, `scikit-learn`, `scikit-survival`, `shap`, `torch`,
`matplotlib`, `seaborn`, `scipy`, `wandb`, `PyYAML`, `tabpfn`.

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

The entire workflow is driven by **`notebook.ipynb`**. 

Then proceed through the sections in order.

### 1. Dataset creation (optional, from the panel)

The first part of the notebook rebuilds the two final datasets from the input
panel via the functions in `src/preprocessing.py`
(`getCompleteDatasetWithTimeWindow`, `getCompleteDatasetWithoutTimeWindow`,
`preprocessDataset`). The same pipeline is available as a script:

```bash
python scripts/build_datasets.py
```

The processed datasets carry `HQCountry` and `PrimaryIndustrySector` as **raw
categories**. They used to be collapsed and frequency-encoded here, on the whole
dataset; that step now runs per split (see below), which is why the categories
themselves have to survive preprocessing.

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
- `nocompetitors` — `dataset_window.csv` with competition features dropped.

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
