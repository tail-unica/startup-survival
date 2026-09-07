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

Two additional ablation settings probe the contribution of specific feature
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
│   └── config.yaml              # Paths, time window, split, seed, hyperparameter sweep
├── data/
│   ├── raw/
│   │   └── example_panel.csv     # Synthetic example panel (released)
│   └── processed/
│       ├── dataset_window.csv    # Final bias-controlled dataset (released)
│       └── dataset_nowindow.csv  # Final look-ahead-biased dataset (released)
├── src/
│   ├── preprocessing.py          # Feature & target engineering, time window, imputation
│   ├── utils.py                  # Plotting, SHAP comparison, Wilcoxon test, metrics
│   └── models/
│       └── MLP.py                # Multi-Layer Perceptron model
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
`matplotlib`, `seaborn`, `scipy`, `wandb`, `PyYAML`, `tabpfn-client`.

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
(`last_year`), the test split (`test_size`), the random seed
(`random_seed`) and the hyperparameter grids are defined in
`config/config.yaml`.

## Step-by-step usage

The entire workflow is driven by **`notebook.ipynb`**. 

Then proceed through the sections in order.

### 1. Dataset creation (optional, from the panel)

The first part of the notebook rebuilds the two final datasets from the input
panel via the functions in `src/preprocessing.py`
(`getCompleteDatasetWithTimeWindow`, `getCompleteDatasetWithoutTimeWindow`,
`preprocessDataset`).

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
`config.yaml`, currently `[1, 2, 3, 4, 5]`), each with its own KNN imputer and
`RobustScaler` fitted on that seed's training set alone. Splits are cached under
`tmp/splits`, so the cell is slow only the first time for a given experiment.

### 4. Hyperparameter sweep and training

Initialize the W&B sweep, then start the agent:

- Set `number_of_runs = 35` to evaluate **all seven models with the best
  configuration** already stored in `config/config.yaml` (make sure all model
  types are enabled in `model_type`). The grid crosses the 7 models with the 5
  evaluation seeds, so each model is replicated on five independent splits.
- Set `number_of_runs = 70` and a single fixed `model_type` to **search** for the
  best hyperparameters.

### Seeds

Each sweep run draws a `seed` that drives **both** the split and the model's
randomness, so a model's five runs are five independent replications rather than
five reruns of one partition. Result tables report **mean ± std** across them.

Hyperparameter search is deliberately *not* multi-seed. The bayes block pins
`seed` to the single tuning seed (`random_seed`, `12`), which is kept out of the
five evaluation seeds: tuning and reporting never share a split. Listing several
seeds in the bayes block would let the optimiser treat the seed as a
hyperparameter and report the luckiest split — the very noise the multi-seed
evaluation exists to expose.

Each run trains the model, logs validation/test metrics and plots to W&B, and
computes SHAP values (stored under the current `tag` for later comparison).
`rf`/`lgb`/`dt` use `TreeExplainer`, `lr` uses `LinearExplainer`, `mlp` uses
`KernelExplainer`, and `svm`/`tabpfn` use `PermutationExplainer` — the only
explainer that stays tractable on an RBF kernel and on a model served over the
network. Its budget is set by `shap_permutation` in `config/config.yaml`.

SHAP is computed on the **first evaluation seed only**: these tables ask which
features move when the window is removed, not how much the metrics vary, and
`compute_wilcoxon_table` pairs rows within a single explained sample, so pooling
seeds would change what the test measures.

`tabpfn` runs TabPFN v2 through the Prior Labs API, which needs an access token:

```bash
export TABPFN_TOKEN="<your Prior Labs token>"
```

Do **not** put the token in `.env` — that file is tracked by git. The notebook
falls back to `~/.cache/tabpfn/auth_token` when the variable is unset.

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
