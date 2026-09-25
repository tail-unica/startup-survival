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
available up to the decision point. The target carries a single horizon, **T**:
a firm counts as having grown if it reaches the next growth stage **within T years**
of its `StartingAge`, the age at which it first reached an early stage.
We then quantify the impact of look-ahead bias by
comparing, on the same firms and models, two dataset constructions:

- **`controlled`**: the bias-controlled dataset, attributes as of each row's own
  year, features read at the age the firm first reached an early stage;
- **`leakboth`**: the same firms with the attributes declared at extraction time
  and cumulative, full-history features (i.e. with look-ahead bias).

Each comes from its own panel: `1_panel_construction.ipynb` runs once with its
temporization switches on and once with them off, writing `panel_timed.csv.gz` and
`panel_snapshot.csv.gz`.

`controlled` and `leakboth` differ in the features, in the definition of the target
and, as a consequence of the latter, in the positive class prevalence, so the gap
between them is the sum of two leaks. Two control settings complete the 2x2 and
separate them, each derived from the two processed datasets by swapping
the target on `CompanyID` (same firms, same columns):

- **`leaklabel`**: bias-free features, leaked target ("does the firm ever reach
  the next stage" instead of "within T years of `StartingAge`");
- **`leakfeat`**: leaked features, bias-free target.

<!-- AUC is the metric to read across the label axis: no model tunes its decision
threshold, so F1, precision and recall move with the prevalence on their own. -->

Two further ablation settings probe the contribution of specific feature
families:

- **`noteam`**: team-related features removed;
- **`nocompetitors`**: competition-related features removed.

For every setting we train seven model families: Random Forest (`rf`), LightGBM
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
| `data/example/pitchbook/` | **A synthetic extraction.** Fourteen invented tables in the shape PitchBook delivers, carrying only the columns the pipeline reads and a cast of eight companies chosen so that every branch of the panel construction fires at least once. `scripts/make_example_data.py` writes them and documents each company's story. | Released |
| `data/example/panel_timed.csv` | **The timed panel those tables produce**, 79 rows by 52 columns: every attribute is the one of the row's own year, so a firm's team, investors and competitors are the ones it had at that age. It is the example of the real thing, and the schema the notebook checks itself against. The values do **not** correspond to any real company. | Released |
| `data/example/panel_snapshot.csv` | **The snapshot panel of the same tables**, the same 79 rows by 52 columns with the temporization switches off: every attribute is the one declared at extraction time, repeated on every year of the firm's life. 13 of the 52 columns differ from the timed panel (the team, education, competitor and cumulative-investment ones) and that difference is the look-ahead bias the paper measures. | Released |
| `data/processed/dataset_controlled.csv` | **Final bias-controlled dataset** (`controlled`), built from the timed panel. The exact input of every experiment. | Released |
| `data/processed/dataset_leakboth.csv` | **Final dataset with look-ahead bias** (`leakboth`), built from the snapshot panel on the same firms. | Released |
| `data/raw/pitchbook/` | The PitchBook extraction the two panels are built from. | **Not released** |
| `data/interim/panel_timed.csv.gz` | **The real timed panel**, what `1_panel_construction.ipynb` produces from that extraction with the switches on. Same construction as `data/example/panel_timed.csv`, on the full sample. | **Not released** |
| `data/interim/panel_snapshot.csv.gz` | **The real snapshot panel**, the same run with the switches off. Same construction as `data/example/panel_snapshot.csv`, on the full sample. | **Not released** |
| `data/raw/QS_World_Rankings.csv` | QS World University Rankings, used to flag top-tier institutes. | See QS terms |

The two released **final datasets** are the exact inputs to every model and
experiment reported in the paper, so all quantitative results can be reproduced
without access to the original panels. The two example panels are provided only to
demonstrate and re-run the upstream feature-engineering pipeline, and to show, on
data that can be published, what the two temporization settings do to the same
firms.


## Repository structure

```
.
├── config/
│   └── config.yaml               # Every setting and every domain rule, in one place
├── data/
│   ├── raw/
│   │   ├── QS_World_Rankings.csv # University ranking, to flag top-tier institutes
│   │   └── pitchbook/            # PitchBook extraction (not released)
│   ├── example/
│   │   ├── pitchbook/            # Synthetic extraction, 14 tables (released)
│   │   ├── panel_timed.csv       # The timed panel those tables produce (released)
│   │   └── panel_snapshot.csv    # The same tables with the switches off (released)
│   ├── interim/                  # The two real panels and the per-phase outputs
│   └── processed/
│       ├── dataset_controlled.csv  # Final bias-controlled dataset (released)
│       └── dataset_leakboth.csv    # Final look-ahead-biased dataset (released)
├── scripts/
│   ├── pipeline.py               # The command line: tables, panel, datasets, experiments
│   └── make_example_data.py      # Writes the synthetic extraction, one story per company
├── src/
│   ├── panel/                    # The panel pipeline, one module per phase
│   │   ├── companies.py          # Phase 1: the skeleton
│   │   ├── people.py             # Phase 2: one row per (company, person)
│   │   ├── team.py               # Phase 3: the team columns
│   │   ├── deals.py              # Phase 4: the funding rounds
│   │   ├── stages.py             # Phase 5: stages, cumulative totals, chief executive
│   │   ├── target.py             # Phase 6: the groups, the future stage, the truncation
│   │   ├── competitors.py        # Phase 7: the competitors, and the final shape
│   │   ├── pipeline.py           # The seven phases chained: build_panel
│   │   ├── checks.py             # The invariants, shared by notebooks, CLI and tests
│   │   ├── expressions.py        # Expressions with the pipeline's missing-value rules
│   │   ├── io.py                 # Schema-explicit readers, no type inference
│   │   ├── expansions.py         # The two memory-heavy expansions
│   │   └── config.py             # Paths, and the domain rules read from config.yaml
│   ├── preprocessing.py          # Feature and target engineering, the two datasets
│   ├── experiments.py            # The six experiments, and the sweep grid
│   ├── encoding.py               # Frequency encoding, fitted per split (no leakage)
│   ├── utils.py                  # Splits, plotting, SHAP comparison, Wilcoxon, metrics
│   ├── training.py               # One run: build, fit, score, explain, log
│   └── models/                   # One class per model family
├── tests/                        # Unit tests, the example end-to-end, the contract
├── .github/workflows/tests.yml   # Lint, format check and tests on every push
├── 1_panel_construction.ipynb    # From the raw tables to the panel, phase by phase
├── 2_dataset_construction.ipynb  # From the panels to the two datasets
├── 3_experiments.ipynb           # From the datasets to the results
├── pyproject.toml                # Dependencies, ruff and pytest configuration
├── uv.lock                       # Exact resolution, committed
├── .env.example                  # Template for .env, the W&B entity and project
└── README.md
```


## Getting started

**Python 3.12**, and [uv](https://docs.astral.sh/uv/) to manage the environment. If
you do not have uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone the repository and create the environment:

```bash
git clone https://github.com/tail-unica/startup-survival
cd startup-survival
uv sync
```

`uv sync` creates `.venv/`, installs the exact versions recorded in `uv.lock`, and
installs the project itself, so `import src...` works from any directory. There is
no separate virtual-environment step and nothing to activate: prefix commands with
`uv run`, which uses that environment, or activate `.venv` by hand if you prefer.

Add `--group dev` to get the tools too (pytest, ruff, JupyterLab and its kernel)
which is what the notebooks and the test suite need:

```bash
uv sync --group dev
```

**Check that it works**, without any data and in a few seconds:

```bash
uv run pytest -q tests/panel tests/test_example_pipeline.py
```

Those tests build the rows they need and run the whole panel pipeline over the
synthetic extraction, so a green run means the environment is complete.

Key dependencies (`uv.lock` holds the exact resolution): `polars`, `pandas`,
`numpy`, `scikit-learn`, `lightgbm`, `torch`, `tabpfn`, `shap`, `matplotlib`,
`seaborn`, `scipy`, `wandb`, `PyYAML`, `python-dotenv`, `joblib`.

### Weights & Biases, only if you want it

Tracking is **optional**: without it the runs happen locally and the metrics are
printed. To use it, create an account at <https://wandb.ai/site>, create an entity
and a project, copy `.env.example` to `.env` and fill them in, and log in once:

```bash
cp .env.example .env
wandb login
```

`.env` is ignored by git, so your values stay on your machine.

Then pass `--wandb` on the command line, or set `USE_WANDB = True` in
`3_experiments.ipynb`.

### Where the data has to be

| you have | put it in | what you can run |
|---|---|---|
| nothing | no data needed | everything, on the synthetic extraction already in `data/example/pitchbook/` |
| the released datasets (in the repository) | `data/processed/` | the experiments, and every comparison of the paper |
| a PitchBook extraction | `data/raw/pitchbook/` | the whole pipeline, from the raw tables |

The QS ranking (`data/raw/QS_World_Rankings.csv`) is read when the datasets are
built, to flag the top-tier institutes.

## Running the pipeline

Three stages, and two ways to run each of them. The **notebooks** walk through the
same functions one call at a time, with the explanation of every step and the
intermediate result printed underneath; the **command line** runs them without the
narration. There is no second implementation: `scripts/pipeline.py` calls exactly
what the notebooks call.

| stage | notebook | command |
|---|---|---|
| raw tables → panel | `1_panel_construction.ipynb` | `pipeline.py panel` |
| panels → datasets | `2_dataset_construction.ipynb` | `pipeline.py datasets` |
| datasets → results | `3_experiments.ipynb` | `pipeline.py experiments` |

### From zero to results, with the notebooks

```bash
uv sync --group dev
uv run jupyter lab
```

JupyterLab opens in the project's environment, so the notebooks find `src/` without
any further setup. In an editor, open the repository and pick the interpreter in
`.venv/bin/python` instead; `uv sync --group dev` has already installed the kernel. Then, in order:

1. **`1_panel_construction.ipynb`.** Two switches in the first cell: `EXAMPLE`, which
   reads the synthetic extraction instead of the PitchBook one, and `TIMED`, which
   chooses the configuration. Run it top to bottom, then set `TIMED = False` and run
   it again: the two runs write `panel_timed.*` and `panel_snapshot.*` under
   `data/interim/`, and the comparison of the paper needs both.
2. **`2_dataset_construction.ipynb`.** Reads those two panels and writes
   `dataset_controlled.csv` and `dataset_leakboth.csv` under `data/processed/`. The
   same `EXAMPLE` switch is in its first cell. Skip this notebook entirely if you are
   using the released datasets.
3. **`3_experiments.ipynb`.** Set `SETTING` to the experiment you want, run down to
   the results table, then change `SETTING` and run again: the stores accumulate, and
   the comparison cells at the bottom read them. `USE_WANDB` decides whether the runs
   are logged or only printed.

Every phase prints what it produced, so a notebook can be read as a report of the
run as well as executed.

### From zero to results, with the command line

```bash
uv sync
uv run python scripts/pipeline.py panel --both      # the two panels
uv run python scripts/pipeline.py datasets          # the two datasets
uv run python scripts/pipeline.py experiments --setting all
```

Or, to see the whole thing work on data that ships with the repository:

```bash
uv run python scripts/pipeline.py all --example
```

### Without the PitchBook data

The extraction cannot be redistributed, so **`data/example/pitchbook/` ships a
synthetic one**: fourteen tables in the shape PitchBook delivers, with only the
columns the pipeline reads, and eight invented companies chosen so that every
branch fires at least once: the undated rounds, the person who joins two years in,
the competitor that dies halfway, the company acquired and the one that closes.

```bash
uv run python scripts/pipeline.py all --example
```

That builds the tables, the two panels, and the two datasets, in seconds, writing
everything under `data/example/`. In the notebooks the same route is `EXAMPLE =
True` in the first cell. It is the shortest way to see what each phase does, and
`tests/test_example_pipeline.py` checks the result value by value.

The **experiments** need no licensed data either: `data/processed/dataset_*.csv`
are released, so `pipeline.py experiments` runs on the real ones out of the box.

### `pipeline.py tables`

Writes the synthetic extraction into `data/example/pitchbook/`. Every company's
story is documented in `scripts/make_example_data.py`, above its rows.

### `pipeline.py panel`

Builds the panel from the raw tables: one row per year of life of every company.

| option | meaning |
|---|---|
| `--timed` | attributes as of each row's own year (the default). Writes `panel_timed.parquet` and `panel_timed.csv.gz` |
| `--snapshot` | attributes as declared at extraction time, the panel that carries the look-ahead. Writes `panel_snapshot.*` |
| `--both` | one after the other, which is what the comparison needs |
| `--example` | read the synthetic extraction, and write beside it. It also refreshes the released copies, `data/example/panel_timed.csv` and `panel_snapshot.csv` |
| `--raw-dir`, `--interim-dir` | override where it reads from and writes to |

Roughly two minutes and 1.8 GB of memory per panel on the full extraction; a couple
of seconds on the example. Each phase hands the next a frame, and the panel is
written at the end together with the report of the run and the checks.

```bash
uv run python scripts/pipeline.py panel --both
```

### `pipeline.py datasets`

Turns the two panels into the two datasets the models read: the bias-controlled one
from the timed panel, the biased one from the snapshot panel, with the same firms
and the same columns.

| option | meaning |
|---|---|
| `--missing-threshold` | share of missing values past which a feature is dropped (`config.yaml`: 0.5) |
| `--max-missing-per-row` | a row with this many missing values or more is dropped (`config.yaml`: 6) |
| `--example` | read the example panels, and write under `data/example/processed/` |

The two thresholds go together: the looser the column threshold, the more sparse
features stay in, and the more missing values an ordinary row carries, so the
per-row budget has to grow with it.

```bash
uv run python scripts/pipeline.py datasets
uv run python scripts/pipeline.py datasets --missing-threshold 0.4 --max-missing-per-row 4
```

### `pipeline.py experiments`

Trains the seven model families on one of the six experiments.

| option | meaning |
|---|---|
| `--setting` | `controlled`, `leakboth`, `noteam`, `nocompetitors`, `leaklabel`, `leakfeat`, or `all` (default: `controlled`) |
| `--wandb` | log to Weights & Biases and let a sweep agent drive the runs. **Without it** the same runs happen locally and the metrics are printed |
| `--runs` | how many runs the sweep agent draws (default: 35, that is 7 families × 5 seeds) |

```bash
uv run python scripts/pipeline.py experiments --setting controlled
uv run python scripts/pipeline.py experiments --setting all --wandb
```

The six settings are two datasets, two ablations of the bias-controlled one, and
two controls that take the 2×2 apart by swapping the target between the two
datasets on `CompanyID`, legitimate because both carry the same firms and the same
columns. Every setting also draws the same split for a given seed, stratified on the
controlled target, so two settings are always compared on the same firms.

### `pipeline.py all`

Every stage in order: the tables when `--example` is given, both panels, the two
datasets, and one experiment at the end. `--skip-experiments` stops after the
datasets.

```bash
uv run python scripts/pipeline.py all --example
uv run python scripts/pipeline.py all --skip-experiments
```

### The two configurations, and why there are two panels

Every attribute of a person, an investor or a competitor can be read two ways: as it
was **in the year of the row**, or as it was **declared at extraction time**. The
first is what a model could have known at the time; the second is the look-ahead the
paper measures. The two panels are the same pipeline with that one switch flipped,
so everything else is held constant, and they carry the same firm-years in the same
order, which is what makes swapping the target between the two datasets legitimate.
The dataset builder checks it rather than assuming it.

### Verification

The checks are structural: they hold of any extraction, because they are the rules
the pipeline enforces. No year precedes a founding year, no company-year appears
twice, the columns are the declared schema, nothing survives past an exit, and the
cumulative columns never go backwards. One of those failing is a bug, so it raises.
The same functions run in the notebooks, in the command line and in the test suite.

## Development

```bash
uv run pytest            # test suite
uv run ruff check .      # lint
uv run ruff format .     # format
uv run jupyter lab       # open the three notebooks
```

On every push and pull request `.github/workflows/tests.yml` runs the tests, the
lint, the format check (`ruff format --check .`) and the type check
(`uv run ty check src scripts tests`).

`tests/` is in three layers.

**Unit tests**, one file per phase of the panel and one per module downstream: the
shape of the skeleton, the two merges of the appointments, the window each person
spans, the four steps that repair a date, the growth-stage cascade, the truncation,
the target engineering, the per-split frequency encoding, the split cache, and the
SHAP and Wilcoxon machinery.

**The example end-to-end**, `tests/test_example_pipeline.py`, runs the *whole* panel
pipeline over the synthetic extraction and states what it must say about each
company: the acquisition that takes the date of the ownership change, the rounds
spread over a gap, the founder counted from year zero, the person with no start date
never counted, the competitor that stops counting when it dies. It calls
`build_panel`, the same function the notebook and the command line call.

**The contract**, `tests/test_integration.py`: every column `src/preprocessing.py`
selects has to be one the panel produces, every ablation has to name columns that
survive the missing-value threshold, every column the panel carries has to reach a
dataset, and the two panels have to have distinct names.

**No test reads the PitchBook data**, and none of them parses a notebook: every one
builds the handful of rows it needs, so what fails is the logic.

### Adding a model family

Write a class in `src/models/` answering the four questions the training loop
asks: which matrix it is fitted on (`wants_scaled`), how it is built from the
sweep config (`from_sweep`), how it turns probabilities into labels (`score`),
and which SHAP explainer stays tractable on it (`explain`); then register it in
`src/models/__init__.py` and add its hyperparameters to `sweep_settings` in
`config/config.yaml` under a `<family>_` prefix. `src/training.py` does not
change.

Lint and format rules live in `pyproject.toml` (`ruff`, line length 100).

## Configuration

`config/config.yaml` holds every setting and every domain rule, so that a choice is
made in one place and read everywhere:

| block | what it decides |
|---|---|
| `paths` | where the extraction, the two panels, the ranking and the two datasets live |
| `first_year` | the oldest founding year admitted into the sample |
| `first_decision_year` | the oldest decision year (founding year plus starting age) a dataset row can have |
| `max_starting_age` | the oldest age at which a firm can first reach an early stage and still enter the datasets |
| `T`, `last_year` | T, the horizon in years the target is read over, and the last year the data covers |
| `preprocessing` | the missing-value policy: the share past which a feature is dropped, and the per-row budget |
| `test_size`, `seeds`, `shap_seed` | the split geometry, the evaluation seeds, and the seed SHAP is computed on |
| `frequency_encoding` | which categories are encoded per split, and the threshold below which they are pooled |
| `ablations` | which columns the two ablation experiments remove |
| `sweep_settings` | the model families, and the hyperparameters of each |
| `shap_permutation` | the row budget of the two models that need a permutation explainer |
| `figures` | where the figures are written, and which model and experiments the SHAP comparison shows |
| `panel` | the domain rules of the panel: degrees, fields of study, founders, investor categories, deal types, the repair of the dates, the stage groups and the released schema |

The command line can override the two thresholds for one run
(`--missing-threshold`, `--max-missing-per-row`); everything else is read as it is
written there.

## The experiments in detail

`3_experiments.ipynb` drives them, and `pipeline.py experiments` runs the same
thing headless. What follows is what happens inside, and which knobs exist.

### Choosing an experiment

One line in the notebook (`SETTING = ...`), or `--setting` on the command line.
`src/experiments.py` assembles each one:

- `controlled`: bias-controlled (loads `dataset_controlled.csv`);
- `leakboth`: look-ahead bias (loads `dataset_leakboth.csv`);
- `noteam`: `dataset_controlled.csv` with team features dropped;
- `nocompetitors`: `dataset_controlled.csv` with competition features dropped;
- `leaklabel`: bias-free features with the leaked target merged in on `CompanyID`;
- `leakfeat`: leaked features with the bias-free target.

Which columns count as team features and which as competition features is set by
`ablations` in `config/config.yaml`.

### Split, imputation and scaling

`CompanyID` and `Target` are separated, and **one train/validation/test split per
evaluation seed** is built (`seeds` in `config.yaml`, currently `[1, 2, 3, 4, 5]`),
stratified on the target of `controlled` whatever the setting,
each with its own frequency encoding, KNN imputer and `RobustScaler` fitted on that
seed's training set alone. Splits are cached under `tmp/splits`, so this is slow
only the first time for a given experiment.

### Frequency encoding

`HQCountry` and `PrimaryIndustrySector` are turned into the **share of the
training rows** their category holds, by `src/encoding.py`. The encoding is
fitted on the training split and then *looked up* for validation and test, so a
held-out row never contributes to its own encoding, the same look-ahead
leakage the time window exists to avoid, applied to the features instead of the
target.

Categories below `frequency_encoding.min_frequency` (a share, default `0.05`)
are pooled into a single `Others` bucket, and so are the categories a held-out
split shows for the first time and the missing values. Shares rather than raw
counts, so that the value does not depend on how many rows the training split
has.

### Regenerating the cached splits

The cache file name carries a fingerprint of the dataset (columns and contents), of
the labels the split is stratified on and of the encoding settings, so editing
`frequency_encoding` or regenerating the processed CSVs invalidates the cache on its
own. To rebuild on purpose (after changing
`prepare_splits` itself, say) set `REBUILD_SPLITS = True` in the notebook; it calls
`clear_split_cache(tag=SETTING)` and recomputes. `get_split(..., force=True)` does
the same for a single split.

### Training, with or without Weights & Biases

**Weights & Biases is optional.** With `USE_WANDB = False` in the notebook, or
without `--wandb` on the command line, the runs happen locally: the same models on
the same splits, the metrics printed and kept in memory, which is all the tables and
the plots below need. With it on, a sweep agent drives the runs and everything is
logged to the entity and project read from `.env`.

The grid crosses the seven families with the five evaluation seeds, so each model is
replicated on five independent splits: 35 runs. To **search** for hyperparameters
instead, fix a single `model_type` in `config.yaml`, put the search space back (the
bayes block is there, commented), set `seeds: [12]` (deliberately not one of the
five) and raise `--runs`.

### Seeds

`seeds` in `config.yaml` is the **only** place a seed is written. The sweep
block references it (`values: *seeds`) and the split cell prepares exactly those
splits, so the two can never drift apart. Set it to match what you are running:

| Running | `seeds` |
|---|---|
| Evaluation (35 runs) | `[1, 2, 3, 4, 5]` |
| Hyperparameter search (one family, `--runs 70`) | `[12]` |

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
`KernelExplainer`, and `svm`/`tabpfn` use `PermutationExplainer`, the only
explainer that stays tractable on an RBF kernel and on a model that runs a
full forward pass per evaluation. Its budget is set by `shap_permutation` in
`config/config.yaml`.

SHAP is computed on the **first evaluation seed only**: these tables ask which
features move when the leakage is removed, not how much the metrics vary, and
`compute_wilcoxon_table` pairs rows within a single explained sample, so pooling
seeds would change what the test measures.

`tabpfn` runs TabPFN v2 locally, on the same `tabpfn-v2-classifier-v2_default`
checkpoint the Prior Labs API used to serve; it is downloaded to the TabPFN
cache on first use (override the location with `TABPFN_MODEL_CACHE_DIR`).

It needs a **CUDA GPU**: `device="auto"` falls back to CPU, where TabPFN past a
thousand rows is impractical. The v2 checkpoint declares a 10k-row pretraining
limit and the training split of the full dataset is larger, so
`src/models/tabpfn.py` passes `ignore_pretraining_limits=True`, the same 50k allowance the API applied to
this model, which keeps the training set identical to the one the other models
see instead of subsampling it.

### Comparing the experiments

After running the settings you want to compare (`controlled` and `leakboth`, say) the
last cells of `3_experiments.ipynb` produce:

- **Metric comparison tables** (`compare_metrics`) reporting each metric for two
  settings as mean ± std across seeds, and the percent change relative to
  `controlled`, computed on the means;
- **SHAP comparison plots** (`plot_shap_comparison`) across settings;
- **Wilcoxon signed-rank tests** (`compute_wilcoxon_table`) on the SHAP feature
  importances, to assess whether the differences are statistically significant.

Both settings have to have been run first, so that `metrics_store` and `shap_store`
hold the data for both. On the command line, `--setting all` runs the six in one go
and prints each one's metric table; the comparisons, the SHAP plots and the Wilcoxon
tables are in the notebook.

## Reproducing the paper's results

The two processed datasets are released, so the results can be reproduced without
the PitchBook extraction:

```bash
uv sync
uv run python scripts/pipeline.py experiments --setting all
```

That trains the seven families on each of the six experiments and prints the metric
tables. Add `--wandb` to log the runs, after filling `.env` (from `.env.example`)
and `wandb login`.

In the notebook the same thing is `3_experiments.ipynb`: set `SETTING`, run down to
the comparison cells, then change `SETTING` and run again: the stores keep what each
experiment produced, and the comparisons read them.

To rebuild everything from the extraction, in order:

```bash
uv run python scripts/pipeline.py panel --both      # the two panels
uv run python scripts/pipeline.py datasets         # the two datasets
uv run python scripts/pipeline.py experiments --setting all
```

## Citation

If you use this code or the released datasets, please cite the accompanying
paper:

```bibtex
@article{startup_survival,
  title   = {The Future Is Not a Feature: A Look-Ahead Bias-Free Evaluation
             Framework for Startup Success Prediction through Machine Learning},
  author  = {Casti, Giulio and Mandas, Marco and Piras, Luca and Marras, Mirko and Fenu, Gianni},
  year    = {2026}
}
```

## License and data rights

The source code is released for academic and research use. The original
PitchBook-derived panel is **not** included and remains subject to PitchBook's
licensing terms; the QS World University Rankings are subject to QS's terms of
use. What this repository distributes is the synthetic extraction, the two example
panels it produces, and the two final, de-identified datasets.
