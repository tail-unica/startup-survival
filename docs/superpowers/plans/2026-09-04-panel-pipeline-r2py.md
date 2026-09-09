# Panel Pipeline R→Python Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reimplement in Python/polars the R pipeline that builds the startup panel from the raw PitchBook CSVs, reproducing the R output exactly — bugs included — and proving it with automated per-column verification against the R intermediate files.

**Architecture:** Five sequential stages, each a pure `run(cfg)` function reading the previous stage's parquet from `data/interim/` and writing its own. Shared infrastructure holds R→polars semantic primitives (`rutils.py`), schema-explicit readers (`io.py`) and a verification engine (`validate.py`). Every R defect is reproduced by default and sits behind a `fix_*` flag in `PanelConfig` that defaults to `False`. A narrative notebook drives the stages and prints the verification reports.

**Tech Stack:** Python 3.12, polars 1.38, numpy 2.3, scikit-learn 1.6, pytest 9.1.

**Spec:** `docs/superpowers/specs/2026-09-04-panel-pipeline-r2py-design.md`

## Global Constraints

- **Reproduce R behaviour by default.** Every `fix_*` flag in `PanelConfig` defaults to `False`, and `False` means "behave exactly like the R script". Never fix a bug inline; add a flag.
- **polars only.** No pandas, no SQL, no SQLite. `src/preprocessing.py` already uses polars.
- **Memory ceiling: 7 GB RAM against 5.6 GB of source CSVs.** Read with `pl.scan_csv` in lazy mode, project only the needed columns, and `.collect(engine="streaming")`. Never materialise `Person.csv` (956 MB), `CompanySimilarRelation.csv` (836 MB), `Deal.csv` (293 MB) or `PersonPositionRelation.csv` (289 MB) in full.
- **Explicit schemas.** Always pass `infer_schema_length=0` (everything String) and cast deliberately. `CompanyID` is a string (`"100020-70"`); inferring it as anything else silently breaks joins.
- **R source of truth:** `src/RCode/1_Arrange_DB.R` (1257 lines) and `src/RCode/2_Arrange_Final.R` (269 lines). Line references in this plan point at those files.
- **Reference data:** `data/raw/pitchbook/*.csv` (raw), `data/reference/*.csv` (R outputs, ground truth). Both gitignored.
- **Run everything with `.venv/bin/python`**, from the repository root.
- `db_master.csv` is **not** a reference file — it comes from an older pipeline version. Ignore it.

**How the stage tasks are specified.** Tasks 1-5, 9 and 12 build infrastructure and carry complete code. The five stage tasks (6, 7, 8, 10, 11, 13, 14) are *transcriptions* of specific R line ranges, and are specified as: the exact R lines to translate, explicit notes on every semantic a naive translation would get wrong, complete code for the individual expressions that are genuinely tricky, and an objective acceptance criterion — the checkpoint report must say `PASS`. The R script in the repository is the specification for those steps; do not paraphrase it from memory, open it at the cited lines. When a checkpoint fails, the report names the column and prints the first ten divergent rows with their keys: re-read the R lines for *that column* before changing anything.

---

## Amendment 2026-09-09 — supersedes parts of the plan below

Three decisions taken after the plan was written. **Where this section and a
task below disagree, this section wins.**

### A1 — The RandomForest imputation is dropped, not ported

`TotalInvestedCapital_Est` and the six `TotalRaised_Est*` columns are **not
produced at all**. Missing amounts stay missing and are filled by the
imputation step that already runs before model training.

- **Task 12 is cancelled.** No `src/panel/imputation.py`, no
  `tests/panel/test_imputation.py`.
- `PanelConfig` loses `imputation_strategy`, `IMPUTATION_STRATEGIES`, the
  `__post_init__` validation and `seed` (nothing is random any more). The two
  imputation tests in `tests/panel/test_config.py` are deleted.
- Stage 4 carries, at the exact point where the R runs the model
  (`1_Arrange_DB.R:1037-1119`), a comment block naming the seven columns the
  RF created and why it is suspended. That comment is a deliverable, not
  decoration.
- Stage 5 drops the six `_Est` columns from the cumulative block and from
  `vars_selected`.
- Consequence for verification: at checkpoints C, D, E and F those columns
  exist in the reference and not in our output. They are declared
  **expected-missing** and excluded from the comparison; every other column
  must still match exactly.
- Consequence downstream: `src/preprocessing.py` currently selects
  `TotalRaised_Est` (line 506) and sums it (lines 303/311). Which column
  replaces it — `TotalRaised` (zero where the amount was undisclosed) or
  `TotalRaised_NA` (null there, so the downstream imputer sees the hole) — is
  **an open question, to be asked before the final task**, not decided here.
  Both columns are produced either way.

### A2 — `TR_D` becomes a kept column

`TR_D` is computed by the R at `1_Arrange_DB.R:1246` and then dropped by
`vars_selected` in `2_Arrange_Final.R`. It is 1 exactly when the company-year
has no deal at all, and is the honest signal that replaces the imputation.
Add `TR_D` to `vars_selected` in stage 5.

Consequence for verification: `TR_D` is present in `db_master_2.csv`
(checkpoint C, so it is verified there) but **absent from `db_selected.csv`,
`db_master_panel.csv.gz` and `panel.csv.gz`**. At checkpoints D, E and F it is
declared **expected-extra** and excluded from the comparison.

### A3 — Two new stages, two new checkpoints

The post-R processing is no longer out of scope: the missing intermediate
`db_master_panel.csv.gz` has been supplied and its four rules were
reverse-engineered and verified to the row. See new Tasks 16 and 17 at the end
of this plan.

| checkpoint | stage | reference | rows | key |
|---|---|---|---|---|
| E | 6 | `db_master_panel.csv.gz` | 882.324 | `(CompanyID, Year_Delta)` |
| F | 7 | `data/raw/panel.csv.gz` | 882.324 | `(CompanyID, Year_Delta)` |

`CHECKPOINTS` in Task 5 therefore holds six entries, not four, and the
`test_four_checkpoints_are_registered_with_the_expected_shapes` test becomes
`test_six_checkpoints...`.

### A4 — `r_injected` is no longer needed

It existed to isolate the RF as the only divergence at C and D. With A1 the
columns are not produced, so there is nothing to inject: they are simply
excluded from the comparison and everything else must match exactly.

---

### Task 1: Package scaffolding and configuration

**Files:**
- Create: `src/panel/__init__.py`
- Create: `src/panel/config.py`
- Create: `tests/__init__.py`
- Create: `tests/panel/__init__.py`
- Test: `tests/panel/test_config.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: `PanelConfig` dataclass, imported by every later task as `from src.panel.config import PanelConfig`.

- [ ] **Step 1: Write the failing test**

```python
# tests/panel/test_config.py
from pathlib import Path

from src.panel.config import BUG_FLAGS, PanelConfig


def test_all_bug_flags_default_to_r_behaviour():
    cfg = PanelConfig()
    for flag in BUG_FLAGS:
        assert getattr(cfg, flag) is False, f"{flag} must default to False (R behaviour)"


def test_bug_flag_registry_matches_dataclass_fields():
    cfg = PanelConfig()
    declared = {f for f in vars(cfg) if f.startswith("fix_")}
    assert declared == set(BUG_FLAGS)


def test_paths_are_relative_to_repo_root():
    cfg = PanelConfig()
    assert cfg.raw_dir == Path("data/raw/pitchbook")
    assert cfg.ref_dir == Path("data/reference")
    assert cfg.interim_dir == Path("data/interim")


def test_default_imputation_strategy_is_standalone():
    assert PanelConfig().imputation_strategy == "r_legacy"


def test_rejects_unknown_imputation_strategy():
    import pytest

    with pytest.raises(ValueError, match="unknown imputation strategy"):
        PanelConfig(imputation_strategy="magic")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/panel/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.panel'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/panel/__init__.py
"""Python reimplementation of the R panel-construction pipeline."""
```

```python
# src/panel/config.py
"""Configuration for the panel pipeline.

Every ``fix_*`` flag defaults to ``False``, and ``False`` means "reproduce
the R behaviour, bug included". Fixing a defect is always opt-in: see
section 6 of the design spec for the registry and the measured impact of
each one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Every bug flag, in the order of the spec's registry (B1..B10).
BUG_FLAGS: tuple[str, ...] = (
    "fix_founding_year_threshold",       # B1
    "fix_is_other_label",                # B2
    "fix_stageblock_na",                 # B3
    "fix_same_country_narm",             # B4
    "fix_permanenza_media_per_company",  # B5
    "fix_negative_delta",                # B6
    "fix_institute_na_literal",          # B7
    "fix_europe_asymmetry",              # B8
    "fix_dup_coalesce",                  # B9
    "fix_is_out_na",                     # B10
)

IMPUTATION_STRATEGIES: tuple[str, ...] = ("r_legacy", "r_injected", "leakage_free")


@dataclass(frozen=True)
class PanelConfig:
    """Paths, bug flags and imputation strategy for one pipeline run."""

    raw_dir: Path = Path("data/raw/pitchbook")
    ref_dir: Path = Path("data/reference")
    interim_dir: Path = Path("data/interim")

    seed: int = 12
    imputation_strategy: str = "r_legacy"
    ignore_failed_checks: bool = False
    n_examples: int = 10
    rtol: float = 1e-9

    fix_founding_year_threshold: bool = False
    fix_is_other_label: bool = False
    fix_stageblock_na: bool = False
    fix_same_country_narm: bool = False
    fix_permanenza_media_per_company: bool = False
    fix_negative_delta: bool = False
    fix_institute_na_literal: bool = False
    fix_europe_asymmetry: bool = False
    fix_dup_coalesce: bool = False
    fix_is_out_na: bool = False

    def __post_init__(self) -> None:
        if self.imputation_strategy not in IMPUTATION_STRATEGIES:
            raise ValueError(
                f"unknown imputation strategy {self.imputation_strategy!r}; "
                f"expected one of {IMPUTATION_STRATEGIES}"
            )

    def interim(self, name: str) -> Path:
        """Path of an interim parquet, creating the directory if needed."""
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        return self.interim_dir / name

    def reference(self, name: str) -> Path:
        return self.ref_dir / name

    def raw(self, name: str) -> Path:
        return self.raw_dir / name

    def active_fixes(self) -> tuple[str, ...]:
        """Flags currently enabled — printed by the notebook on every run."""
        return tuple(f for f in BUG_FLAGS if getattr(self, f))
```

Create `tests/__init__.py` and `tests/panel/__init__.py` as empty files.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/panel/test_config.py -v`
Expected: 5 passed

- [ ] **Step 5: Add pytest to requirements**

`pytest` is installed in the venv but missing from `requirements.txt`. Add the line `pytest==9.1.1` in alphabetical position (between `pyppmd` and `python-dateutil`).

- [ ] **Step 6: Commit**

```bash
git add src/panel tests requirements.txt
git commit -m "feat(panel): add package scaffolding and PanelConfig"
```

---

### Task 2: R semantics — NA replacement and date parsing

**Files:**
- Create: `src/panel/rutils.py`
- Test: `tests/panel/test_rutils_na_dates.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `R_NA`, `R_NA_NAN`, `R_NA_INF`: `tuple[str, ...]` token sets.
  - `as_na(df: pl.DataFrame, tokens: Sequence[str]) -> pl.DataFrame`
  - `parse_date_r(col: pl.Expr) -> pl.Expr` returning `pl.Date`.

**Background — what the R code does.** `1_Arrange_DB.R:42-44` runs
`db1[db1[[i]] %in% c("", "NA", "N/A", "NULL", "NaN"), i] <- NA` over every
column. `%in%` coerces numerics to character, so on a float column
`as.character(NaN) == "NaN"`, `as.character(Inf) == "Inf"` and
`as.character(-Inf) == "-Inf"` do match. That is what turns `max()` over an
all-NA group (which R returns as `-Inf`) into a real NA at lines 650 and
1249. The token set differs between call sites — line 113 omits `"NaN"` —
and that difference is real and must be preserved.

Dates use `case_when` on `nchar` (`1_Arrange_DB.R:65-84`): 10 characters →
`%m/%d/%Y`; 8 characters → `parse_date_time2(orders = "mdy", cutoff_2000 = 24)`;
**any other length → NA**. `cutoff_2000 = 24` means a two-digit year `00..24`
becomes `2000..2024` and `25..99` becomes `1925..1999`. Note an 8-character
string can also be `1/5/2024`, which has a four-digit year.

- [ ] **Step 1: Write the failing test**

```python
# tests/panel/test_rutils_na_dates.py
import datetime as dt

import polars as pl

from src.panel.rutils import R_NA, R_NA_INF, R_NA_NAN, as_na, parse_date_r


def test_as_na_nulls_string_tokens():
    df = pl.DataFrame({"a": ["x", "", "NA", "N/A", "NULL", "NaN"]})
    assert as_na(df, R_NA)["a"].to_list() == ["x", None, None, None, None, "NaN"]
    assert as_na(df, R_NA_NAN)["a"].to_list() == ["x", None, None, None, None, None]


def test_as_na_matches_r_coercion_on_float_columns():
    df = pl.DataFrame({"a": [1.0, float("nan"), float("inf"), float("-inf"), None]})
    # R's %in% coerces to character, so "NaN"/"Inf"/"-Inf" match the numerics.
    assert as_na(df, R_NA_NAN)["a"].to_list() == [1.0, None, float("inf"), float("-inf"), None]
    out = as_na(df, R_NA_INF)["a"].to_list()
    assert out == [1.0, None, None, None, None]


def test_as_na_leaves_integers_untouched():
    df = pl.DataFrame({"a": [0, 1, None]})
    assert as_na(df, R_NA_INF)["a"].to_list() == [0, 1, None]


def test_parse_date_r_ten_character_format():
    df = pl.DataFrame({"d": ["05/13/2020", "12/01/1999"]})
    out = df.select(parse_date_r(pl.col("d")))["d"].to_list()
    assert out == [dt.date(2020, 5, 13), dt.date(1999, 12, 1)]


def test_parse_date_r_eight_character_two_digit_year_uses_cutoff_24():
    df = pl.DataFrame({"d": ["05/13/20", "05/13/24", "05/13/25", "05/13/99"]})
    out = df.select(parse_date_r(pl.col("d")))["d"].to_list()
    assert out == [dt.date(2020, 5, 13), dt.date(2024, 5, 13),
                   dt.date(1925, 5, 13), dt.date(1999, 5, 13)]


def test_parse_date_r_eight_character_four_digit_year():
    df = pl.DataFrame({"d": ["1/5/2024"]})
    assert df.select(parse_date_r(pl.col("d")))["d"].to_list() == [dt.date(2024, 1, 5)]


def test_parse_date_r_rejects_other_lengths():
    df = pl.DataFrame({"d": ["1/5/24", "2020-05-13", "", None]})
    assert df.select(parse_date_r(pl.col("d")))["d"].to_list() == [None, None, None, None]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/panel/test_rutils_na_dates.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.panel.rutils'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/panel/rutils.py
"""Primitives that reproduce R/dplyr semantics in polars.

Each function here exists because a naive polars translation of the R code
would differ in a way that is silent and hard to spot. Read the docstrings
before changing anything.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

#: Token set of `1_Arrange_DB.R:113` (db2) — no "NaN".
R_NA: tuple[str, ...] = ("", "NA", "N/A", "NULL")
#: Token set of `1_Arrange_DB.R:43` (db1 and most others).
R_NA_NAN: tuple[str, ...] = R_NA + ("NaN",)
#: Token set of `1_Arrange_DB.R:650` and `:1249`, applied *after* aggregation
#: so that max()/mean() over all-NA groups (-Inf / NaN in R) become NA.
R_NA_INF: tuple[str, ...] = R_NA_NAN + ("-Inf", "Inf")


def as_na(df: pl.DataFrame, tokens: Sequence[str]) -> pl.DataFrame:
    """Reproduce ``df[df[[i]] %in% tokens, i] <- NA`` over every column.

    On String columns this is a plain membership test. On Float columns R
    coerces the values to character first, so ``NaN``, ``Inf`` and ``-Inf``
    match the corresponding tokens. Integer and Boolean columns can never
    match any token and are left alone.
    """
    tok = set(tokens)
    exprs: list[pl.Expr] = []
    for name, dtype in df.schema.items():
        col = pl.col(name)
        if dtype == pl.String:
            exprs.append(
                pl.when(col.is_in(list(tok))).then(None).otherwise(col).alias(name)
            )
        elif dtype in (pl.Float32, pl.Float64):
            bad: pl.Expr | None = None
            if "NaN" in tok:
                bad = col.is_nan()
            if "Inf" in tok:
                pos = col.is_infinite() & (col > 0)
                bad = pos if bad is None else (bad | pos)
            if "-Inf" in tok:
                neg = col.is_infinite() & (col < 0)
                bad = neg if bad is None else (bad | neg)
            if bad is not None:
                exprs.append(
                    pl.when(bad.fill_null(False)).then(None).otherwise(col).alias(name)
                )
    return df.with_columns(exprs) if exprs else df


_DMY = r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$"


def parse_date_r(col: pl.Expr) -> pl.Expr:
    """Reproduce the two-branch ``case_when(nchar(...))`` date parsing.

    10 characters  -> ``%m/%d/%Y``
    8 characters   -> month/day/year, two-digit years resolved with
                      ``cutoff_2000 = 24`` (00..24 -> 2000s, 25..99 -> 1900s)
    anything else  -> NA, including 6-character dates such as ``1/5/24``.
    """
    n = col.str.len_chars()
    month = col.str.extract(_DMY, 1).cast(pl.Int32, strict=False)
    day = col.str.extract(_DMY, 2).cast(pl.Int32, strict=False)
    year_txt = col.str.extract(_DMY, 3)
    year_num = year_txt.cast(pl.Int32, strict=False)
    year = (
        pl.when(year_txt.str.len_chars() == 4)
        .then(year_num)
        .when(year_num <= 24)
        .then(year_num + 2000)
        .otherwise(year_num + 1900)
    )
    return (
        pl.when(n == 10)
        .then(col.str.to_date("%m/%d/%Y", strict=False))
        .when(n == 8)
        .then(pl.date(year, month, day))
        .otherwise(None)
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/panel/test_rutils_na_dates.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/panel/rutils.py tests/panel/test_rutils_na_dates.py
git commit -m "feat(panel): add R NA-token and date-parsing primitives"
```

---

### Task 2b: R semantics — aggregation and window primitives

**Files:**
- Modify: `src/panel/rutils.py`
- Test: `tests/panel/test_rutils_agg.py`

**Interfaces:**
- Consumes: `src.panel.rutils` from Task 2.
- Produces:
  - `cumany(col: pl.Expr) -> pl.Expr`
  - `rle_sequence(df, value_col: str, group_cols: list[str], out_col: str) -> pl.DataFrame`
  - `stage_block(df, value_col, group_cols, out_col, *, fix: bool) -> pl.DataFrame`
  - `weighted_cumulative(value_col: str, weight_col: str, group_cols: list[str]) -> pl.Expr`
  - `next_different(df, value_col, group_cols, stage_col, time_col) -> pl.DataFrame`
  - `quantile_type7(values, p: float) -> float`
  - `scale_r(col: pl.Expr) -> pl.Expr`
  - `coalesce_first_last(name: str) -> pl.Expr`
  - `tail_na_omit(name: str) -> pl.Expr`

**Background.** Five R behaviours that a naive translation gets wrong:

1. `rle()` (`2_Arrange_Final.R:214`) compares consecutive elements and treats
   an NA comparison as "different", so **every NA is its own run** and the
   element after an NA also starts a new run. Verified on company `100026-46`.
2. `cumsum` propagates NA to the end of the group. `StageBlock`
   (`2_Arrange_Final.R:211`) therefore becomes NA for the whole company from
   the first missing `GrowthStage` onward.
3. `weighted_cumulative` (`2_Arrange_Final.R:127-136`) is O(n²) in R; the
   algebraic equivalent `cumsum(x*w)/cumsum(w)` over valid rows is exact and
   O(n). Valid means `x` and `w` both present and `w > 0`.
4. `GrowthNextStage` / `TimeNextStage` (`2_Arrange_Final.R:218-229`) take the
   first *later* stage that differs from the current one, skipping NAs.
   R's `(i+1):n()` produces a reversed vector on the last row of each group,
   but the result is NA regardless, so reproduce the result, not the quirk.
5. `scale()` uses the sample standard deviation (`n-1`); numpy and polars
   default to `n`.

- [ ] **Step 1: Write the failing test**

```python
# tests/panel/test_rutils_agg.py
import math

import polars as pl

from src.panel.rutils import (
    coalesce_first_last,
    cumany,
    next_different,
    quantile_type7,
    rle_sequence,
    scale_r,
    stage_block,
    tail_na_omit,
    weighted_cumulative,
)


def _stages() -> pl.DataFrame:
    # Mirrors company 100026-46 from db_selected.csv: a leading NA, then runs.
    return pl.DataFrame(
        {
            "CompanyID": ["c"] * 6,
            "Age": [0, 1, 2, 3, 4, 5],
            "GrowthStage": [None, "Seed", "EarlyVC", "EarlyVC", "LaterVC", "LaterVC"],
        }
    )


def test_rle_sequence_treats_each_na_as_its_own_run():
    out = rle_sequence(_stages(), "GrowthStage", ["CompanyID"], "YearsInStage")
    assert out["YearsInStage"].to_list() == [1, 1, 1, 2, 1, 2]


def test_rle_sequence_restarts_per_group():
    df = pl.DataFrame({"g": ["a", "a", "b", "b"], "v": ["x", "x", "x", "x"]})
    out = rle_sequence(df, "v", ["g"], "n")
    assert out["n"].to_list() == [1, 2, 1, 2]


def test_stage_block_propagates_na_like_r_cumsum():
    out = stage_block(_stages(), "GrowthStage", ["CompanyID"], "StageBlock", fix=False)
    assert out["StageBlock"].to_list() == [None] * 6


def test_stage_block_without_na_counts_transitions():
    df = pl.DataFrame({"g": ["a"] * 4, "v": ["x", "x", "y", "z"]})
    out = stage_block(df, "v", ["g"], "b", fix=False)
    assert out["b"].to_list() == [0, 0, 1, 2]


def test_stage_block_fix_flag_stops_the_na_propagation():
    out = stage_block(_stages(), "GrowthStage", ["CompanyID"], "StageBlock", fix=True)
    assert out["StageBlock"].to_list() == [0, 1, 2, 2, 3, 3]


def test_next_different_skips_nulls_and_reports_distance():
    # CORRECTED 2026-09-09: the plan originally expected "Seed"/1 on row 0.
    # Row 0's own stage is null, and in R `which(future != NA)[1]` is NA, so
    # both outputs are null. Confirmed against company 100026-46 in
    # db_selected.csv, whose 2013 row has a null stage and null Next/Time.
    out = next_different(_stages(), "GrowthStage", ["CompanyID"], "Next", "Time")
    assert out["Next"].to_list() == [None, "EarlyVC", "LaterVC", "LaterVC", None, None]
    assert out["Time"].to_list() == [None, 1, 2, 1, None, None]


def test_cumany_is_monotone_within_group():
    df = pl.DataFrame({"g": ["a"] * 4, "v": [False, True, False, False]})
    out = df.with_columns(cumany(pl.col("v")).over("g").alias("c"))
    assert out["c"].to_list() == [False, True, True, True]


def test_weighted_cumulative_matches_r_definition():
    df = pl.DataFrame(
        {
            "g": ["a"] * 4,
            "x": [10.0, None, 20.0, 40.0],
            "w": [0.0, 5.0, 1.0, 3.0],
        }
    )
    out = df.with_columns(weighted_cumulative("x", "w", ["g"]).alias("c"))
    # row0: w == 0 -> no valid entry yet -> NA
    # row1: x is null -> still no valid entry -> NA
    # row2: (20*1)/1 = 20
    # row3: (20*1 + 40*3)/4 = 35
    assert out["c"].to_list() == [None, None, 20.0, 35.0]


def test_quantile_type7_matches_r():
    # R: quantile(c(1,2,3,4), 0.95) -> 3.85 ; quantile(c(1,2,3,4), 0.75) -> 3.25
    assert math.isclose(quantile_type7([1, 2, 3, 4], 0.95), 3.85)
    assert math.isclose(quantile_type7([1, 2, 3, 4], 0.75), 3.25)


def test_quantile_type7_ignores_nulls_and_handles_empty():
    assert math.isclose(quantile_type7([1, None, 3], 0.5), 2.0)
    assert math.isnan(quantile_type7([], 0.5))


def test_scale_r_uses_sample_standard_deviation():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    # sd with n-1 is 1.0, so the scaled values are exactly -1, 0, 1.
    assert df.select(scale_r(pl.col("x")))["x"].to_list() == [-1.0, 0.0, 1.0]


def test_coalesce_first_last_and_tail_na_omit():
    df = pl.DataFrame({"g": ["a"] * 3, "v": [None, "mid", "last"]})
    out = df.group_by("g", maintain_order=True).agg(
        coalesce_first_last("v").alias("cfl"), tail_na_omit("v").alias("tail")
    )
    # coalesce(first, last) ignores the middle row -> "last"
    assert out["cfl"].to_list() == ["last"]
    assert out["tail"].to_list() == ["last"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/panel/test_rutils_agg.py -v`
Expected: FAIL with `ImportError: cannot import name 'cumany'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/panel/rutils.py`. Put `import numpy as np` **at the top of the
module** with the existing imports — only the functions go at the end:

```python
def cumany(col: pl.Expr) -> pl.Expr:
    """dplyr ``cumany``. Callers apply ``.over(group)`` themselves.

    Script 2 replaces NA with FALSE before every cumany call, so nulls are
    already gone; ``fill_null(False)`` keeps the primitive total anyway.
    """
    return col.fill_null(False).cast(pl.Int8).cum_max().cast(pl.Boolean)


def _run_id(value_col: str, group_cols: list[str]) -> pl.Expr:
    """Run identifier reproducing R's ``rle``: an NA comparison counts as
    'different', so every null opens a new run and so does its successor."""
    prev = pl.col(value_col).shift(1).over(group_cols)
    is_new = (
        (pl.col(value_col) != prev)
        | pl.col(value_col).is_null()
        | prev.is_null()
    ).fill_null(True)
    return is_new.cast(pl.Int64).cum_sum().over(group_cols)


def rle_sequence(
    df: pl.DataFrame, value_col: str, group_cols: list[str], out_col: str
) -> pl.DataFrame:
    """``sequence(rle(x)$lengths)`` — position within the current run, 1-based."""
    return (
        df.with_columns(_run_id(value_col, group_cols).alias("__run"))
        .with_columns(
            (pl.int_range(pl.len()).over(group_cols + ["__run"]) + 1).alias(out_col)
        )
        .drop("__run")
    )


def stage_block(
    df: pl.DataFrame,
    value_col: str,
    group_cols: list[str],
    out_col: str,
    *,
    fix: bool,
) -> pl.DataFrame:
    """``cumsum(lag(x, default = first(x)) != x)``.

    With ``fix=False`` an NA comparison poisons the cumulative sum and the
    whole company becomes NA from the first missing value onward — the R
    behaviour, verified on company 100026-46. With ``fix=True`` an NA is
    treated as a transition and the counter keeps running.
    """
    prev = pl.col(value_col).shift(1).over(group_cols)
    first = pl.col(value_col).first().over(group_cols)
    lagged = pl.when(prev.is_null() & (pl.int_range(pl.len()).over(group_cols) == 0)) \
        .then(first).otherwise(prev)
    changed = lagged != pl.col(value_col)  # null when either side is null
    if fix:
        changed = (
            (lagged != pl.col(value_col))
            | (lagged.is_null() != pl.col(value_col).is_null())
        ).fill_null(False)
    return df.with_columns(
        changed.cast(pl.Int64).cum_sum().over(group_cols).alias(out_col)
    )


def weighted_cumulative(
    value_col: str, weight_col: str, group_cols: list[str]
) -> pl.Expr:
    """Cumulative weighted mean over rows where value and weight are present
    and the weight is positive. Algebraically identical to the R loop, O(n)."""
    valid = (
        pl.col(value_col).is_not_null()
        & pl.col(weight_col).is_not_null()
        & (pl.col(weight_col) > 0)
    )
    num = (
        pl.when(valid).then(pl.col(value_col) * pl.col(weight_col)).otherwise(0.0)
    ).cum_sum().over(group_cols)
    den = (
        pl.when(valid).then(pl.col(weight_col)).otherwise(0.0)
    ).cum_sum().over(group_cols)
    return pl.when(den > 0).then(num / den).otherwise(None)


def next_different(
    df: pl.DataFrame,
    value_col: str,
    group_cols: list[str],
    stage_col: str,
    time_col: str,
) -> pl.DataFrame:
    """First later value different from the current one, skipping nulls,
    plus its distance in rows. Both are null when no such value exists."""
    ordered = df.with_row_index("__i")
    later = (
        ordered.select(group_cols + ["__i", value_col])
        .filter(pl.col(value_col).is_not_null())
        .rename({"__i": "__j", value_col: "__cand"})
    )
    joined = (
        ordered.join(later, on=group_cols, how="left")
        .filter((pl.col("__j") > pl.col("__i")) | pl.col("__j").is_null())
        .filter(
            pl.col("__cand").is_null()
            | (pl.col("__cand") != pl.col(value_col))
            | pl.col(value_col).is_null()
        )
    )
    picked = (
        joined.sort(["__i", "__j"])
        .group_by("__i", maintain_order=True)
        .agg(
            pl.col("__cand").first().alias(stage_col),
            pl.col("__j").first().alias("__jj"),
        )
    )
    out = (
        ordered.join(picked, on="__i", how="left")
        .with_columns((pl.col("__jj") - pl.col("__i")).cast(pl.Int64).alias(time_col))
        .drop("__i", "__jj")
    )
    # A current stage that is itself null has no "next different" in R either.
    return out.with_columns(
        pl.when(pl.col(value_col).is_null()).then(None).otherwise(pl.col(stage_col)).alias(stage_col),
        pl.when(pl.col(value_col).is_null()).then(None).otherwise(pl.col(time_col)).alias(time_col),
    )


def quantile_type7(values, p: float) -> float:
    """R's default ``quantile()`` (type 7), which is numpy's default 'linear'
    interpolation. Nulls and NaNs are dropped, as with ``na.rm = TRUE``."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, p, method="linear"))


def scale_r(col: pl.Expr) -> pl.Expr:
    """``scale()`` with the sample standard deviation (denominator n-1)."""
    return (col - col.mean()) / col.std(ddof=1)


def coalesce_first_last(name: str) -> pl.Expr:
    """``coalesce(first(x), last(x))`` — deliberately ignores middle rows."""
    return pl.coalesce(pl.col(name).first(), pl.col(name).last())


def tail_na_omit(name: str) -> pl.Expr:
    """``tail(na.omit(x), 1)`` — last non-null in row order."""
    return pl.col(name).drop_nulls().last()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/panel/test_rutils_agg.py -v`
Expected: 12 passed

If `next_different` fails on the distance for row 2 (`EarlyVC` at index 2
should reach `LaterVC` at index 4, distance 2), check the `__j > __i` filter:
the candidate set must exclude the current row itself but keep every later
row, and `sort(["__i", "__j"])` must run before the `group_by`.

- [ ] **Step 5: Commit**

```bash
git add src/panel/rutils.py tests/panel/test_rutils_agg.py
git commit -m "feat(panel): add R aggregation and window primitives"
```

---

### Task 3: Schema-explicit CSV readers

**Files:**
- Create: `src/panel/io.py`
- Test: `tests/panel/test_io.py`

**Interfaces:**
- Consumes: `PanelConfig`.
- Produces:
  - `scan_raw(cfg, table: str, columns: list[str]) -> pl.LazyFrame` — every column String.
  - `read_raw(cfg, table, columns, *, expect_rows: int | None = None) -> pl.DataFrame`
  - `to_num(name: str) -> pl.Expr` — cast to Float64, non-numeric to null.
  - `to_int(name: str) -> pl.Expr`
  - `RAW_ROW_COUNTS: dict[str, int]` — expected row count per raw table.

**Background.** Free-text columns (`Description`, `DealSynopsis`, `Biography`)
contain quoted newlines. A parser that mis-handles them silently produces the
wrong number of rows, and every downstream count is then wrong for a reason
nobody suspects. `read_raw` therefore asserts the row count against a recorded
constant. Populate `RAW_ROW_COUNTS` in Step 3 by running the command given
there — do not guess the numbers.

- [ ] **Step 1: Write the failing test**

```python
# tests/panel/test_io.py
import polars as pl
import pytest

from src.panel.config import PanelConfig
from src.panel.io import RAW_ROW_COUNTS, read_raw, scan_raw, to_int, to_num


def test_scan_raw_returns_all_string_columns():
    cfg = PanelConfig()
    lf = scan_raw(cfg, "CompanyAffiliateRelation", ["CompanyID", "AffiliateType"])
    assert set(lf.collect_schema().values()) == {pl.String}


def test_read_raw_projects_only_requested_columns():
    cfg = PanelConfig()
    df = read_raw(cfg, "CompanyAffiliateRelation", ["CompanyID", "AffiliateType"])
    assert df.columns == ["CompanyID", "AffiliateType"]


def test_read_raw_row_count_matches_recorded_constant():
    cfg = PanelConfig()
    table = "CompanyAffiliateRelation"
    df = read_raw(cfg, table, ["CompanyID"], expect_rows=RAW_ROW_COUNTS[table])
    assert df.height == RAW_ROW_COUNTS[table]


def test_read_raw_raises_on_row_count_mismatch():
    cfg = PanelConfig()
    with pytest.raises(ValueError, match="row count mismatch"):
        read_raw(cfg, "CompanyAffiliateRelation", ["CompanyID"], expect_rows=1)


def test_to_num_and_to_int_null_out_non_numeric():
    df = pl.DataFrame({"a": ["1.5", "NA", "", None, "x"]})
    assert df.select(to_num("a"))["a"].to_list() == [1.5, None, None, None, None]
    assert df.select(to_int("a"))["a"].to_list() == [None, None, None, None, None]
    df2 = pl.DataFrame({"a": ["3", "NA"]})
    assert df2.select(to_int("a"))["a"].to_list() == [3, None]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/panel/test_io.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.panel.io'`

- [ ] **Step 3: Write minimal implementation**

First record the true row counts:

```bash
.venv/bin/python - <<'EOF'
import polars as pl
from pathlib import Path
base = Path("data/raw/pitchbook")
tables = ["Company", "CompanyAffiliateRelation", "CompanyBoardTeamRelation",
          "CompanyEmployeeHistoryRelation", "CompanyFinancialRelation",
          "CompanyNewsRelation", "CompanySimilarRelation", "Deal",
          "DealInvestorRelation", "Investor", "Person",
          "PersonEducationRelation", "PersonPositionRelation"]
for t in tables:
    lf = pl.scan_csv(base / f"{t}.csv", infer_schema_length=0)
    n = lf.select(pl.len()).collect(engine="streaming").item()
    print(f'    "{t}": {n},')
EOF
```

Paste the printed lines verbatim into `RAW_ROW_COUNTS`.

```python
# src/panel/io.py
"""Readers for the raw PitchBook CSVs.

Everything is read as String and cast deliberately. Type inference is not
used anywhere: `CompanyID` looks numeric in some files ("100020-70" does
not, but others do) and an inferred integer key breaks joins silently.
"""

from __future__ import annotations

import polars as pl

from src.panel.config import PanelConfig

#: Row count of each raw table, recorded once so that a parsing regression on
#: the quoted free-text columns fails loudly instead of skewing every count
#: downstream. Fill these in with the snippet in the plan (Task 3, Step 3).
RAW_ROW_COUNTS: dict[str, int] = {}


def scan_raw(cfg: PanelConfig, table: str, columns: list[str]) -> pl.LazyFrame:
    """Lazy scan of one raw table, projected to `columns`, all String."""
    return pl.scan_csv(
        cfg.raw(f"{table}.csv"),
        infer_schema_length=0,
        null_values=[],
        quote_char='"',
    ).select(columns)


def read_raw(
    cfg: PanelConfig,
    table: str,
    columns: list[str],
    *,
    expect_rows: int | None = None,
) -> pl.DataFrame:
    """Collect a projected raw table in streaming mode, asserting its height."""
    df = scan_raw(cfg, table, columns).collect(engine="streaming")
    if expect_rows is not None and df.height != expect_rows:
        raise ValueError(
            f"row count mismatch for {table}: expected {expect_rows}, got {df.height}. "
            "A quoted free-text column is probably being mis-parsed."
        )
    return df


def to_num(name: str) -> pl.Expr:
    """Cast a String column to Float64; anything non-numeric becomes null."""
    return pl.col(name).cast(pl.Float64, strict=False).alias(name)


def to_int(name: str) -> pl.Expr:
    """Cast a String column to Int64; anything non-integer becomes null."""
    return pl.col(name).cast(pl.Int64, strict=False).alias(name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/panel/test_io.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/panel/io.py tests/panel/test_io.py
git commit -m "feat(panel): add schema-explicit raw CSV readers"
```

---

### Task 4: Verification engine

**Files:**
- Create: `src/panel/validate.py`
- Test: `tests/panel/test_validate.py`

**Interfaces:**
- Consumes: `PanelConfig`.
- Produces:
  - `ColumnDiff` dataclass: `column, dtype_ref, dtype_act, n_compared, n_diff, pct_diff, n_na_only_ref, n_na_only_act, max_abs_diff, n_ambiguous_na, expected, examples`.
  - `VerificationReport` dataclass with `passed() -> bool`, `assert_clean()`, `render() -> str`, `to_json(path)`.
  - `verify(actual, reference, key, name, *, expected_diff=frozenset(), rtol=1e-9, n_examples=10) -> VerificationReport`

**Background — the `NA` token trap.** `write.csv` writes a missing value as
unquoted `NA` and the literal string `"NA"` as quoted `NA`, but a CSV parser
strips the quotes before either reaches us: **the two are indistinguishable
in the reference file.** This is not hypothetical — `Institute` is built with
`paste(unique(...))`, which stringifies NAs, so a person whose institutes are
all missing produces a cell equal to exactly `NA`. The resolution is to
compare string columns in R-serialized form: read the reference with
`null_values=[]` so nothing is nulled, and map nulls on the actual side to the
token `NA` before comparing. Cells equal to exactly `NA` are counted in
`n_ambiguous_na` and reported, because on those the check genuinely cannot
distinguish a null from the string.

Comparison is by key, never by position: sort both sides, confirm the key
sequences are identical, then compare column by column. Never join the two
full frames — 1M rows × 97 columns twice would not fit in 7 GB.

- [ ] **Step 1: Write the failing test**

```python
# tests/panel/test_validate.py
import polars as pl
import pytest

from src.panel.validate import verify


def _ref(**cols) -> pl.DataFrame:
    """Reference frames arrive all-String, exactly as read from R's write.csv."""
    return pl.DataFrame(cols)


def test_identical_frames_pass():
    act = pl.DataFrame({"k": ["a", "b"], "v": [1.0, 2.0]})
    ref = _ref(k=["a", "b"], v=["1.0", "2.0"])
    rep = verify(act, ref, key=["k"], name="t")
    assert rep.passed()
    assert rep.n_diff_columns() == 0


def test_reports_column_and_row_counts_of_a_difference():
    act = pl.DataFrame({"k": ["a", "b", "c"], "v": [1.0, 9.0, 3.0]})
    ref = _ref(k=["a", "b", "c"], v=["1.0", "2.0", "3.0"])
    rep = verify(act, ref, key=["k"], name="t")
    assert not rep.passed()
    col = rep.column("v")
    assert col.n_diff == 1
    assert col.n_compared == 3
    assert col.examples[0] == {"k": "b", "reference": "2.0", "actual": "9.0"}


def test_float_comparison_uses_relative_tolerance():
    act = pl.DataFrame({"k": ["a"], "v": [1.0 + 1e-12]})
    ref = _ref(k=["a"], v=["1.0"])
    assert verify(act, ref, key=["k"], name="t").passed()
    act2 = pl.DataFrame({"k": ["a"], "v": [1.001]})
    assert not verify(act2, ref, key=["k"], name="t").passed()


def test_na_mismatches_are_counted_separately_from_value_mismatches():
    act = pl.DataFrame({"k": ["a", "b"], "v": [None, 2.0]})
    ref = _ref(k=["a", "b"], v=["1.0", "NA"])
    col = verify(act, ref, key=["k"], name="t").column("v")
    assert col.n_na_only_act == 1
    assert col.n_na_only_ref == 1
    assert col.n_diff == 2


def test_string_nulls_are_compared_in_r_serialized_form():
    # A null on our side and the token NA in the reference must match, because
    # write.csv cannot distinguish them.
    act = pl.DataFrame({"k": ["a", "b"], "s": [None, "MIT"]})
    ref = _ref(k=["a", "b"], s=["NA", "MIT"])
    rep = verify(act, ref, key=["k"], name="t")
    assert rep.passed()
    assert rep.column("s").n_ambiguous_na == 1


def test_boolean_columns_are_parsed_not_string_compared():
    act = pl.DataFrame({"k": ["a", "b"], "b": [True, False]})
    ref = _ref(k=["a", "b"], b=["TRUE", "FALSE"])
    assert verify(act, ref, key=["k"], name="t").passed()


def test_missing_and_extra_keys_are_reported_and_excluded():
    act = pl.DataFrame({"k": ["a", "c"], "v": [1.0, 3.0]})
    ref = _ref(k=["a", "b"], v=["1.0", "2.0"])
    rep = verify(act, ref, key=["k"], name="t")
    assert rep.keys_only_ref == 1
    assert rep.keys_only_act == 1
    assert rep.column("v").n_compared == 1  # only the shared key "a"
    assert not rep.passed()


def test_expected_diff_columns_do_not_fail_the_report():
    act = pl.DataFrame({"k": ["a"], "TotalRaised_Est": [9.0]})
    ref = _ref(k=["a"], TotalRaised_Est=["1.0"])
    rep = verify(act, ref, key=["k"], name="t", expected_diff={"TotalRaised_Est"})
    assert rep.passed()
    assert rep.column("TotalRaised_Est").expected is True
    assert rep.column("TotalRaised_Est").n_diff == 1


def test_duplicate_keys_are_rejected():
    act = pl.DataFrame({"k": ["a", "a"], "v": [1.0, 2.0]})
    ref = _ref(k=["a", "a"], v=["1.0", "2.0"])
    with pytest.raises(ValueError, match="duplicate keys"):
        verify(act, ref, key=["k"], name="t")


def test_assert_clean_raises_only_on_failure():
    act = pl.DataFrame({"k": ["a"], "v": [1.0]})
    verify(act, _ref(k=["a"], v=["1.0"]), key=["k"], name="t").assert_clean()
    with pytest.raises(AssertionError, match="verification failed"):
        verify(act, _ref(k=["a"], v=["2.0"]), key=["k"], name="t").assert_clean()


def test_render_lists_worst_column_first():
    act = pl.DataFrame({"k": ["a", "b"], "few": [1.0, 2.0], "many": [9.0, 9.0]})
    ref = _ref(k=["a", "b"], few=["1.0", "5.0"], many=["1.0", "2.0"])
    text = verify(act, ref, key=["k"], name="t").render()
    assert text.index("many") < text.index("few")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/panel/test_validate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.panel.validate'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/panel/validate.py
"""Compare a stage's output against the R reference, column by column.

The report answers exactly two questions: which columns differ, and on how
many rows. Everything else is supporting detail.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import polars as pl

_TRUE = {"TRUE", "T", "True", "true"}
_FALSE = {"FALSE", "F", "False", "false"}


@dataclass
class ColumnDiff:
    column: str
    dtype_ref: str
    dtype_act: str
    n_compared: int
    n_diff: int
    pct_diff: float
    n_na_only_ref: int
    n_na_only_act: int
    max_abs_diff: float | None
    n_ambiguous_na: int
    expected: bool
    examples: list[dict] = field(default_factory=list)


@dataclass
class VerificationReport:
    name: str
    n_rows_ref: int
    n_rows_act: int
    keys_only_ref: int
    keys_only_act: int
    cols_only_ref: list[str]
    cols_only_act: list[str]
    columns: list[ColumnDiff]

    def column(self, name: str) -> ColumnDiff:
        for c in self.columns:
            if c.column == name:
                return c
        raise KeyError(name)

    def n_diff_columns(self) -> int:
        return sum(1 for c in self.columns if c.n_diff and not c.expected)

    def passed(self) -> bool:
        return (
            self.n_diff_columns() == 0
            and self.keys_only_ref == 0
            and self.keys_only_act == 0
            and not self.cols_only_ref
            and not self.cols_only_act
        )

    def assert_clean(self) -> None:
        if not self.passed():
            raise AssertionError(
                f"verification failed for {self.name}\n" + self.render()
            )

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str))

    def render(self) -> str:
        bad = sorted(
            (c for c in self.columns if c.n_diff or c.n_na_only_ref or c.n_na_only_act),
            key=lambda c: c.n_diff,
            reverse=True,
        )
        clean = len(self.columns) - len(bad)
        out = [
            f"=== {self.name}: {'PASS' if self.passed() else 'FAIL'} ===",
            f"righe   riferimento={self.n_rows_ref:,}  ottenute={self.n_rows_act:,}",
            f"chiavi  solo R={self.keys_only_ref:,}  solo PY={self.keys_only_act:,}",
            f"colonne solo R={self.cols_only_ref}  solo PY={self.cols_only_act}",
            f"colonne divergenti={self.n_diff_columns()} su {len(self.columns)} "
            f"({clean} coincidono al 100%)",
        ]
        if bad:
            out.append("")
            out.append(
                f"{'colonna':<32}{'tipo R':<10}{'tipo PY':<10}"
                f"{'confr.':>10}{'diverse':>10}{'%':>8}"
                f"{'NA solo R':>11}{'NA solo PY':>12}{'max diff':>14}  nota"
            )
            for c in bad:
                note = "ATTESA" if c.expected else ""
                md = "" if c.max_abs_diff is None else f"{c.max_abs_diff:.6g}"
                out.append(
                    f"{c.column:<32}{c.dtype_ref:<10}{c.dtype_act:<10}"
                    f"{c.n_compared:>10,}{c.n_diff:>10,}{c.pct_diff:>7.2f}%"
                    f"{c.n_na_only_ref:>11,}{c.n_na_only_act:>12,}{md:>14}  {note}"
                )
            out.append("")
            for c in bad:
                if not c.examples:
                    continue
                out.append(f"-- {c.column}: prime {len(c.examples)} righe divergenti")
                for ex in c.examples:
                    out.append(f"   {ex}")
        return "\n".join(out)


def _align(actual: pl.DataFrame, reference: pl.DataFrame, key: list[str]):
    """Restrict both frames to the shared keys and sort them identically."""
    for name, df in (("actual", actual), ("reference", reference)):
        if df.select(key).is_duplicated().any():
            raise ValueError(f"duplicate keys in {name} on {key}")
    ref_keys = reference.select(key).with_columns(
        pl.col(k).cast(pl.String) for k in key
    )
    act_keys = actual.select(key).with_columns(pl.col(k).cast(pl.String) for k in key)
    only_ref = ref_keys.join(act_keys, on=key, how="anti").height
    only_act = act_keys.join(ref_keys, on=key, how="anti").height
    shared = ref_keys.join(act_keys, on=key, how="semi")
    a = (
        actual.with_columns(pl.col(k).cast(pl.String).alias(f"__k{i}") for i, k in enumerate(key))
        .join(shared.rename({k: f"__k{i}" for i, k in enumerate(key)}),
              on=[f"__k{i}" for i in range(len(key))], how="semi")
        .sort([f"__k{i}" for i in range(len(key))])
    )
    r = (
        reference.with_columns(pl.col(k).cast(pl.String).alias(f"__k{i}") for i, k in enumerate(key))
        .join(shared.rename({k: f"__k{i}" for i, k in enumerate(key)}),
              on=[f"__k{i}" for i in range(len(key))], how="semi")
        .sort([f"__k{i}" for i in range(len(key))])
    )
    return a, r, only_ref, only_act


def _to_bool(col: pl.Expr) -> pl.Expr:
    return (
        pl.when(col.is_in(list(_TRUE))).then(True)
        .when(col.is_in(list(_FALSE))).then(False)
        .otherwise(None)
    )


def verify(
    actual: pl.DataFrame,
    reference: pl.DataFrame,
    key: list[str],
    name: str,
    *,
    expected_diff: frozenset[str] | set[str] = frozenset(),
    rtol: float = 1e-9,
    n_examples: int = 10,
) -> VerificationReport:
    """Compare `actual` against the R `reference` (all-String) on `key`."""
    a, r, only_ref, only_act = _align(actual, reference, key)
    kcols = [f"__k{i}" for i in range(len(key))]

    shared_cols = [c for c in actual.columns if c in reference.columns and c not in key]
    diffs: list[ColumnDiff] = []

    for c in shared_cols:
        dtype = actual.schema[c]
        ref_raw = r.get_column(c)
        act_raw = a.get_column(c)
        n = a.height
        ambiguous = 0
        max_abs = None

        if dtype == pl.String:
            # Compare in R-serialized form: null <-> the token "NA".
            ref_v = ref_raw
            act_v = act_raw.fill_null("NA")
            ambiguous = int((ref_v == "NA").sum())
            is_diff = ref_v != act_v
            na_only_ref = 0
            na_only_act = 0
        elif dtype == pl.Boolean:
            ref_v = pl.select(_to_bool(pl.lit(ref_raw))).to_series()
            act_v = act_raw
            na_only_ref = int((ref_v.is_null() & act_v.is_not_null()).sum())
            na_only_act = int((act_v.is_null() & ref_v.is_not_null()).sum())
            is_diff = (ref_v != act_v).fill_null(False) | (ref_v.is_null() != act_v.is_null())
        else:
            ref_v = ref_raw.cast(pl.Float64, strict=False)
            act_v = act_raw.cast(pl.Float64, strict=False)
            na_only_ref = int((ref_v.is_null() & act_v.is_not_null()).sum())
            na_only_act = int((act_v.is_null() & ref_v.is_not_null()).sum())
            delta = (ref_v - act_v).abs()
            scale = pl.select(
                pl.max_horizontal(pl.lit(ref_v).abs(), pl.lit(act_v).abs())
            ).to_series()
            over_tol = (delta > (scale * rtol)).fill_null(False)
            is_diff = over_tol | (ref_v.is_null() != act_v.is_null())
            if delta.drop_nulls().len():
                max_abs = float(delta.max())

        n_diff = int(is_diff.sum())
        examples: list[dict] = []
        if n_diff:
            idx = [i for i, flag in enumerate(is_diff.to_list()) if flag][:n_examples]
            keyvals = a.select(kcols).to_dicts()
            for i in idx:
                ex = {k: keyvals[i][f"__k{j}"] for j, k in enumerate(key)}
                ex["reference"] = ref_raw[i]
                ex["actual"] = act_raw[i]
                examples.append(ex)

        diffs.append(
            ColumnDiff(
                column=c,
                dtype_ref="str",
                dtype_act=str(dtype),
                n_compared=n,
                n_diff=n_diff,
                pct_diff=(100.0 * n_diff / n) if n else 0.0,
                n_na_only_ref=na_only_ref,
                n_na_only_act=na_only_act,
                max_abs_diff=max_abs,
                n_ambiguous_na=ambiguous,
                expected=c in expected_diff,
                examples=examples,
            )
        )

    return VerificationReport(
        name=name,
        n_rows_ref=reference.height,
        n_rows_act=actual.height,
        keys_only_ref=only_ref,
        keys_only_act=only_act,
        cols_only_ref=[c for c in reference.columns if c not in actual.columns],
        cols_only_act=[c for c in actual.columns if c not in reference.columns],
        columns=diffs,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/panel/test_validate.py -v`
Expected: 11 passed

Two likely stumbles: `pl.lit(series)` inside `pl.select` needs the series to
carry a name, so use `series.alias(...)` if polars complains; and
`is_duplicated()` on a multi-column selection needs `.is_duplicated()` on the
whole frame, which is what the code does.

- [ ] **Step 5: Commit**

```bash
git add src/panel/validate.py tests/panel/test_validate.py
git commit -m "feat(panel): add per-column verification engine"
```

---

### Task 5: Checkpoint registry and CLI

**Files:**
- Modify: `src/panel/validate.py`
- Test: `tests/panel/test_checkpoints.py`

**Interfaces:**
- Consumes: `verify`, `PanelConfig`.
- Produces:
  - `CHECKPOINTS: dict[str, Checkpoint]` with `Checkpoint(name, stage, interim, reference, key, expect_rows)`.
  - `load_reference(cfg, filename) -> pl.DataFrame` — all-String, nothing nulled.
  - `COLUMN_FINALISED_AT_STAGE: dict[str, int]`
  - `run_checkpoint(cfg, letter) -> VerificationReport`
  - `run_partial(cfg, stage: int, actual: pl.DataFrame) -> VerificationReport`
  - `python -m src.panel.validate --checkpoint C`

**Background.** Stages 1 and 4 have no reference file of their own, but many
`db_master_2` columns are already final before the end — `EmployeeCount` and
`N_News` never change after stage 3. `COLUMN_FINALISED_AT_STAGE` records the
stage after which each column is frozen, so a partial check can classify every
column as *verified*, *expected to differ* (not final yet) or **regression**
(final at an earlier stage but now diverging). The map starts empty and each
stage task adds its own columns — that is why it is a plain dict, editable
without touching the engine.

- [ ] **Step 1: Write the failing test**

```python
# tests/panel/test_checkpoints.py
import polars as pl
import pytest

from src.panel.config import PanelConfig
from src.panel.validate import (
    CHECKPOINTS,
    COLUMN_FINALISED_AT_STAGE,
    classify_partial,
    load_reference,
)


def test_four_checkpoints_are_registered_with_the_expected_shapes():
    assert set(CHECKPOINTS) == {"A", "B", "C", "D"}
    assert CHECKPOINTS["A"].reference == "db3.csv"
    assert CHECKPOINTS["A"].key == ["CompanyID", "PersonID"]
    assert CHECKPOINTS["A"].expect_rows == 534_851
    assert CHECKPOINTS["B"].reference == "db_master_1.csv"
    assert CHECKPOINTS["B"].key == ["CompanyID"]
    assert CHECKPOINTS["B"].expect_rows == 116_920
    assert CHECKPOINTS["C"].reference == "db_master_2.csv"
    assert CHECKPOINTS["C"].key == ["CompanyID", "Year_Delta"]
    assert CHECKPOINTS["C"].expect_rows == 1_001_625
    assert CHECKPOINTS["D"].reference == "db_selected.csv"
    assert CHECKPOINTS["D"].expect_rows == 1_001_625


def test_db_master_csv_is_not_a_reference():
    assert all(cp.reference != "db_master.csv" for cp in CHECKPOINTS.values())


def test_load_reference_nulls_nothing_and_keeps_strings():
    cfg = PanelConfig()
    df = load_reference(cfg, "db_master_1.csv").head(50)
    assert set(df.schema.values()) == {pl.String}
    assert df.null_count().sum_horizontal().item() == 0


def test_classify_partial_separates_regressions_from_expected_differences():
    finalised = {"a": 1, "b": 3}
    diff_cols = {"a", "b", "c"}
    verified, expected, regressions = classify_partial(
        stage=2, diff_columns=diff_cols, all_columns={"a", "b", "c", "d"},
        finalised=finalised,
    )
    assert regressions == {"a"}       # final at stage 1, differs at stage 2
    assert expected == {"b", "c"}     # not final yet (stage 3, or unknown)
    assert verified == {"d"}          # matches


def test_column_finalised_map_only_names_real_columns():
    cfg = PanelConfig()
    ref_cols = set(load_reference(cfg, "db_master_2.csv").head(1).columns)
    unknown = set(COLUMN_FINALISED_AT_STAGE) - ref_cols
    assert not unknown, f"unknown columns in the map: {sorted(unknown)}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/panel/test_checkpoints.py -v`
Expected: FAIL with `ImportError: cannot import name 'CHECKPOINTS'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/panel/validate.py`. Add `from src.panel.config import PanelConfig`
to the imports at the top of the module — Task 4 did not need it, this task does:

```python
@dataclass(frozen=True)
class Checkpoint:
    name: str
    stage: int
    interim: str
    reference: str
    key: list[str]
    expect_rows: int


CHECKPOINTS: dict[str, Checkpoint] = {
    "A": Checkpoint("A", 2, "db3.parquet", "db3.csv",
                    ["CompanyID", "PersonID"], 534_851),
    "B": Checkpoint("B", 3, "db_master_1.parquet", "db_master_1.csv",
                    ["CompanyID"], 116_920),
    "C": Checkpoint("C", 5, "db_master_2.parquet", "db_master_2.csv",
                    ["CompanyID", "Year_Delta"], 1_001_625),
    "D": Checkpoint("D", 5, "db_selected.parquet", "db_selected.csv",
                    ["CompanyID", "Year_Delta"], 1_001_625),
}

#: Stage after which each db_master_2 column stops changing. Each stage task
#: appends its own columns; see the plan's stage tasks.
COLUMN_FINALISED_AT_STAGE: dict[str, int] = {}

#: Columns the RandomForest imputation makes non-reproducible. They are
#: expected to differ under the `r_legacy` strategy and only then.
RF_DEPENDENT_COLUMNS: frozenset[str] = frozenset(
    {
        "TotalRaised_Est",
        "TotalRaised_Est_NA",
        "TotalRaised_Est_any",
        "TotalRaised_Est_cum",
        "TotalRaised_Est_any_cum",
        "TotalRaised_Est_NA_cum",
    }
)


def load_reference(cfg: PanelConfig, filename: str) -> pl.DataFrame:
    """Read an R reference CSV as pure text, nulling nothing.

    `null_values=[]` is deliberate: it keeps the token `NA` visible so string
    columns can be compared in R-serialized form (see `verify`).
    """
    return pl.read_csv(
        cfg.reference(filename),
        infer_schema_length=0,
        null_values=[],
        quote_char='"',
    )


def classify_partial(
    stage: int,
    diff_columns: set[str],
    all_columns: set[str],
    finalised: dict[str, int],
) -> tuple[set[str], set[str], set[str]]:
    """Split columns into (verified, expected-to-differ, regressions)."""
    regressions = {c for c in diff_columns if finalised.get(c, stage + 1) <= stage}
    expected = diff_columns - regressions
    verified = all_columns - diff_columns
    return verified, expected, regressions


def run_checkpoint(cfg: PanelConfig, letter: str) -> VerificationReport:
    cp = CHECKPOINTS[letter]
    actual = pl.read_parquet(cfg.interim(cp.interim))
    reference = load_reference(cfg, cp.reference)
    expected = (
        RF_DEPENDENT_COLUMNS
        if (cfg.imputation_strategy == "r_legacy" and cp.stage >= 4)
        else frozenset()
    )
    report = verify(
        actual, reference, key=cp.key, name=f"checkpoint {letter} ({cp.reference})",
        expected_diff=expected, rtol=cfg.rtol, n_examples=cfg.n_examples,
    )
    report.to_json(cfg.interim_dir / "reports" / f"checkpoint_{letter}.json")
    return report


def run_partial(cfg: PanelConfig, stage: int, actual: pl.DataFrame) -> VerificationReport:
    """Compare an intermediate db_master_2 against the final reference."""
    reference = load_reference(cfg, "db_master_2.csv")
    report = verify(
        actual, reference, key=["CompanyID", "Year_Delta"],
        name=f"verifica parziale stadio {stage}",
        rtol=cfg.rtol, n_examples=cfg.n_examples,
    )
    diff_cols = {c.column for c in report.columns if c.n_diff}
    all_cols = {c.column for c in report.columns}
    _, expected, regressions = classify_partial(
        stage, diff_cols, all_cols, COLUMN_FINALISED_AT_STAGE
    )
    for c in report.columns:
        c.expected = c.column in expected
    report.to_json(cfg.interim_dir / "reports" / f"partial_stage{stage}.json")
    if regressions:
        print(f"REGRESSIONI allo stadio {stage}: {sorted(regressions)}")
    return report


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Verifica un checkpoint del panel.")
    parser.add_argument("--checkpoint", required=True, choices=sorted(CHECKPOINTS))
    parser.add_argument("--strategy", default="r_legacy")
    args = parser.parse_args()
    cfg = PanelConfig(imputation_strategy=args.strategy)
    print(run_checkpoint(cfg, args.checkpoint).render())


if __name__ == "__main__":
    _main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/panel/test_checkpoints.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/panel/validate.py tests/panel/test_checkpoints.py
git commit -m "feat(panel): add checkpoint registry, partial checks and CLI"
```

---

### Task 6: Stage 1 — company and affiliates

**Files:**
- Create: `src/panel/stage1_company.py`
- Modify: `src/panel/validate.py` (add stage-1 columns to `COLUMN_FINALISED_AT_STAGE`)
- Test: `tests/panel/test_stage1.py`

**Interfaces:**
- Consumes: `PanelConfig`, `read_raw`, `to_num`, `as_na`, `parse_date_r`, `R_NA`, `R_NA_NAN`.
- Produces: `run(cfg: PanelConfig) -> None`, writing three parquets:
  `db1.parquet` (the Company frame **before** the `YearFounded > 1999` filter,
  columns `CompanyID` and `YearFounded` only), `db_master_1_v1.parquet` and
  `db_master_2_skeleton.parquet`.

**Why `db1.parquet` exists.** `1_Arrange_DB.R:448` joins `YearFounded` onto the
team table from `db1` — the *unfiltered* company frame — while `db_master_1` has
by then been cut to `YearFounded > 1999`. Task 7 needs the unfiltered version:
using the filtered one would drop the people of older companies and change
`db3`. Persist it here rather than re-reading the 184 MB `Company.csv` twice.

**R source:** `1_Arrange_DB.R:37-238`.

**Translation notes.**
- `db1`: `Company` table, the 44 columns of lines 47-56, NA tokens `R_NA_NAN`.
- Booleans `Website_d`, `Linkedin`, `Facebook`, `Twitter` are
  `nchar(x) > 0` **after** the NA replacement, so a nulled URL gives null,
  not `False`. Reproduce with `pl.col(c).str.len_chars() > 0`.
- `db2`: `CompanyAffiliateRelation`, NA tokens `R_NA` (**no** `"NaN"` — line
  113 differs from line 43, and the difference is real).
- `db2_summary` (lines 117-128) then a left join, with the eight affiliate
  columns filled with 0 where missing (lines 136-139).
- The panel skeleton is `seq(YearFounded, MaxYear)` per company (lines
  144-172). Build it with `pl.int_ranges(start, end + 1)` then `.explode()`.
  **`MaxYear` can be smaller than `YearFounded`**, and R's `seq` then counts
  *down*, producing negative `Delta` — 245 rows over 106 companies. Under
  `fix_negative_delta=False` reproduce it by generating the descending range;
  under `True`, emit only the single founding year.
- Five joins on `(CompanyID, year)` bring the time-varying columns in
  (lines 202-228).
- Both frames end filtered to `YearFounded > 1999` (lines 236-238).

- [ ] **Step 1: Write the failing test**

```python
# tests/panel/test_stage1.py
import polars as pl

from src.panel.config import PanelConfig
from src.panel.stage1_company import expand_years


def test_expand_years_produces_one_row_per_year_inclusive():
    df = pl.DataFrame({"CompanyID": ["a"], "YearFounded": [2010], "MaxYear": [2013]})
    out = expand_years(df, fix_negative_delta=False)
    assert out["Year_Delta"].to_list() == [2010, 2011, 2012, 2013]
    assert out["Delta"].to_list() == [0, 1, 2, 3]


def test_expand_years_counts_down_when_maxyear_precedes_founding():
    # R's seq(2010, 2008) walks backwards and yields negative Delta.
    df = pl.DataFrame({"CompanyID": ["a"], "YearFounded": [2010], "MaxYear": [2008]})
    out = expand_years(df, fix_negative_delta=False)
    assert out["Year_Delta"].to_list() == [2010, 2009, 2008]
    assert out["Delta"].to_list() == [0, -1, -2]


def test_expand_years_fix_flag_keeps_only_the_founding_year():
    df = pl.DataFrame({"CompanyID": ["a"], "YearFounded": [2010], "MaxYear": [2008]})
    out = expand_years(df, fix_negative_delta=True)
    assert out["Year_Delta"].to_list() == [2010]
    assert out["Delta"].to_list() == [0]


def test_stage1_outputs_have_the_expected_shape():
    """Integration: run stage 1 and check the invariants the R code guarantees."""
    cfg = PanelConfig()
    m1 = pl.read_parquet(cfg.interim("db_master_1_v1.parquet"))
    skel = pl.read_parquet(cfg.interim("db_master_2_skeleton.parquet"))
    assert m1.height == 116_920
    assert m1["CompanyID"].n_unique() == 116_920
    assert m1["YearFounded"].min() > 1999
    assert skel["YearFounded"].min() > 1999
    # R's descending seq() must have produced some negative Delta. The exact
    # 245 rows / 106 companies were measured on the FINAL db_master_2.csv, so
    # they are asserted at checkpoint C, not here: stage 2b's full_join can
    # still change the row count.
    assert skel.filter(pl.col("Delta") < 0).height > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/panel/test_stage1.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.panel.stage1_company'`

- [ ] **Step 3: Write the implementation**

Create `src/panel/stage1_company.py` translating `1_Arrange_DB.R:37-238`.
`expand_years` is the one piece worth writing out here, because the
descending-range behaviour is easy to lose:

```python
def expand_years(df: pl.DataFrame, *, fix_negative_delta: bool) -> pl.DataFrame:
    """Expand each company to one row per year, reproducing R's ``seq``.

    ``seq(from, to)`` counts *down* when ``to < from``, which is how 245 rows
    across 106 companies end up with a negative ``Delta``.
    """
    if fix_negative_delta:
        years = pl.when(pl.col("MaxYear") < pl.col("YearFounded")) \
            .then(pl.int_ranges(pl.col("YearFounded"), pl.col("YearFounded") + 1)) \
            .otherwise(pl.int_ranges(pl.col("YearFounded"), pl.col("MaxYear") + 1))
    else:
        years = pl.when(pl.col("MaxYear") < pl.col("YearFounded")) \
            .then(pl.int_ranges(pl.col("YearFounded"), pl.col("MaxYear") - 1, step=-1)) \
            .otherwise(pl.int_ranges(pl.col("YearFounded"), pl.col("MaxYear") + 1))
    return (
        df.with_columns(years.alias("Year_Delta"))
        .explode("Year_Delta")
        .with_columns((pl.col("Year_Delta") - pl.col("YearFounded")).alias("Delta"))
    )
```

The rest of the module is a direct transcription of the R lines listed in the
translation notes above. Follow the R source statement by statement; do not
reorder operations, because several depend on the previous statement's output.

Then record the stage-1 columns in `COLUMN_FINALISED_AT_STAGE`:

```python
COLUMN_FINALISED_AT_STAGE.update(
    {"CompanyID": 1, "YearFounded": 1, "Year_Delta": 1, "Delta": 1}
)
```

- [ ] **Step 4: Run stage 1 and the tests**

```bash
.venv/bin/python -c "
from src.panel.config import PanelConfig
from src.panel import stage1_company
stage1_company.run(PanelConfig())
"
.venv/bin/python -m pytest tests/panel/test_stage1.py -v
```
Expected: 4 passed. `db_master_1_v1.parquet` must have 116,920 rows.

- [ ] **Step 5: Commit**

```bash
git add src/panel/stage1_company.py src/panel/validate.py tests/panel/test_stage1.py
git commit -m "feat(panel): add stage 1 (company and affiliates)"
```

---

### Task 7: Stage 2a — person-level table, CHECKPOINT A

**Files:**
- Create: `src/panel/stage2_team.py`
- Test: `tests/panel/test_stage2_db3.py`

**Interfaces:**
- Consumes: `db_master_1_v1.parquet`, the primitives from Tasks 2/2b.
- Produces: `build_db3(cfg) -> pl.DataFrame` and `db3.parquet` (534,851 rows).

**R source:** `1_Arrange_DB.R:240-524`.

**Translation notes.**
- Dedup block (lines 248-278): select rows duplicated on
  `(CompanyID, PersonID)`, then filter `db3` by **`PersonID`** membership —
  not by the pair, which is bug B9 — aggregate with `coalesce_first_last` on
  all 12 non-key columns, concatenate the aggregate **in front of** `db3`,
  and keep the first row per `(CompanyID, PersonID)`.
- `Person` join (lines 292-301): 16 attributes plus `PersonID`.
- Experience indices (lines 304-315): `RolesCount_Total` sums the eight
  columns from `CurrentPositionsCount` to `NumberOfAffiliatedFunds`
  inclusive; the three `_z` columns use `scale_r(log(x + 1))` computed
  **globally over all of db3**, not per group.
- Education (lines 326-398): the `case_when` chains are short-circuit — the
  first match wins, so preserve the order exactly. `Is_Other` compares against
  `"Other/Unknown"`, which `Field` never produces (bug B2): under
  `fix_is_other_label=False` keep the impossible comparison, under `True`
  compare against `"Other"`.
- Name overrides (lines 402-409), `PersonPositionRelation` join (lines
  411-421), `IsFounder` (lines 423-428).
- `ownership_out` join (lines 430-444) leaves `Is_Out` **null** for companies
  that are not out of business (bug B10).
- `StartDate`/`EndDate` imputation (lines 446-505). `PermanenzaMedia` is a
  single global number (bug B5): under `fix_permanenza_media_per_company=False`
  compute one mean over all rows and round it; under `True`, per company.
- `EndDate < StartDate -> StartDate`, then `DeltaStart`/`DeltaEnd`, then
  `DeltaStart = 0` for founders (lines 509-524).

- [ ] **Step 1: Write the failing test**

```python
# tests/panel/test_stage2_db3.py
import polars as pl

from src.panel.config import PanelConfig
from src.panel.validate import CHECKPOINTS, load_reference, verify


def test_checkpoint_a_matches_the_r_reference():
    cfg = PanelConfig()
    cp = CHECKPOINTS["A"]
    actual = pl.read_parquet(cfg.interim(cp.interim))
    reference = load_reference(cfg, cp.reference)
    report = verify(actual, reference, key=cp.key, name="checkpoint A")
    print(report.render())
    assert actual.height == cp.expect_rows
    report.assert_clean()


def test_is_out_is_null_for_companies_still_in_business():
    """Bug B10: the left join leaves Is_Out null, not False."""
    cfg = PanelConfig()
    db3 = pl.read_parquet(cfg.interim("db3.parquet"))
    assert db3["Is_Out"].null_count() > 0


def test_is_other_is_never_true():
    """Bug B2: Field never takes the value the comparison looks for."""
    cfg = PanelConfig()
    db3 = pl.read_parquet(cfg.interim("db3.parquet"))
    assert db3.filter(pl.col("Is_Other")).height == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/panel/test_stage2_db3.py -v`
Expected: FAIL with `ModuleNotFoundError` / missing `db3.parquet`

- [ ] **Step 3: Write the implementation**

Transcribe `1_Arrange_DB.R:240-524` into `build_db3(cfg)` following the notes
above. `Person.csv` is 956 MB: read it with `read_raw` projecting only the 17
needed columns.

- [ ] **Step 4: Run stage 2a and verify checkpoint A**

```bash
.venv/bin/python -c "
from src.panel.config import PanelConfig
from src.panel import stage2_team
stage2_team.run_db3(PanelConfig())
"
.venv/bin/python -m src.panel.validate --checkpoint A
.venv/bin/python -m pytest tests/panel/test_stage2_db3.py -v
```
Expected: `checkpoint A: PASS`, 534,851 rows, 3 passed.

If a column fails, the report names it and shows the first 10 divergent rows
with their `(CompanyID, PersonID)`. Re-read the corresponding R lines for that
column before changing anything.

- [ ] **Step 5: Commit**

```bash
git add src/panel/stage2_team.py tests/panel/test_stage2_db3.py
git commit -m "feat(panel): add stage 2a (person-level table), checkpoint A passes"
```

---

### Task 8: Stage 2b — team panel

**Files:**
- Modify: `src/panel/stage2_team.py`, `src/panel/validate.py`
- Test: `tests/panel/test_stage2_panel.py`

**Interfaces:**
- Consumes: `db3.parquet`, `db_master_2_skeleton.parquet`.
- Produces: `run_panel(cfg) -> None`, writing `db_master_2_team.parquet`.

**R source:** `1_Arrange_DB.R:527-664`.

**Translation notes.**
- `db3_years` (lines 531-544): per company, `seq(min(DeltaStart), max(DeltaEnd))`.
- `db3_person_years` (lines 549-555): per person, `seq(DeltaStart, DeltaEnd)`,
  dropping rows with a null `DeltaStart`.
- `db3_expanded` = left join of the two on `(CompanyID, Years)`, then
  `filter(YearFounded > 2000)` — bug B1. Under
  `fix_founding_year_threshold=True` use `> 1999` to match stage 1.
- Aggregation (lines 617-644) by `(CompanyID, Years)`. `Total_People` is
  `.N`, which counts the empty join row for a year with no people; that
  would give `Total_People = 1` with every person field null. It never
  happens in this data — assert it rather than flagging it.
- `Institute = paste(unique(Institute), collapse = "; ")` includes nulls as
  the literal text `"NA"` (bug B7).
- Line 650 applies the NA loop with `R_NA_INF`, converting the `-Inf` from
  `max()` over all-null groups and the `NaN` from `mean()` into real nulls.
- `full_join` with the skeleton on `(CompanyID, Delta = Years)`, then
  `fill(YearFounded)` both directions within company and
  `Year_Delta = YearFounded + Delta` where missing (lines 654-664).

- [ ] **Step 1: Write the failing test**

```python
# tests/panel/test_stage2_panel.py
import polars as pl

from src.panel.config import PanelConfig
from src.panel.validate import run_partial


def test_no_phantom_team_rows():
    """A year with no people would give Total_People == 1 and null fields.
    The spec measured zero occurrences; assert it stays that way."""
    cfg = PanelConfig()
    df = pl.read_parquet(cfg.interim("db_master_2_team.parquet"))
    phantom = df.filter(
        (pl.col("Total_People") == 1)
        & pl.col("RolesCount_Max").is_null()
        & pl.col("Highest_Degree_Max").is_null()
        & (pl.col("Percent_Females") == 0)
    )
    assert phantom.height == 0


def test_year_2000_cohort_has_no_team_data():
    """Bug B1: the team panel filters YearFounded > 2000 while the skeleton
    used > 1999, so the 2000 cohort keeps 29,603 rows with no team at all."""
    cfg = PanelConfig()
    df = pl.read_parquet(cfg.interim("db_master_2_team.parquet"))
    cohort = df.filter(pl.col("YearFounded") == 2000)
    assert cohort.height == 29_603
    assert cohort.filter(pl.col("Total_People").is_not_null()).height == 0


def test_partial_verification_reports_only_expected_differences():
    cfg = PanelConfig()
    df = pl.read_parquet(cfg.interim("db_master_2_team.parquet"))
    report = run_partial(cfg, stage=2, actual=df)
    print(report.render())
    team_cols = ["Total_People", "Percent_Females", "Total_Founders",
                 "Highest_Degree_Max", "WorkExp_Idx_Mean", "Institute"]
    for c in team_cols:
        assert report.column(c).n_diff == 0, f"{c} should already be final"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/panel/test_stage2_panel.py -v`
Expected: FAIL — `db_master_2_team.parquet` does not exist

- [ ] **Step 3: Write the implementation**

Transcribe `1_Arrange_DB.R:527-664`. Then register the columns this stage
freezes:

```python
COLUMN_FINALISED_AT_STAGE.update(
    {c: 2 for c in [
        "Total_People", "Percent_Females", "Is_Eco", "Is_Eng", "Is_NS", "Is_Hum",
        "Is_SS", "Is_Med", "Is_Other", "Is_Law", "Is_IT", "Avg_Earliest_Year",
        "Highest_Degree_Max", "Highest_Degree_Mean", "Institute", "RolesCount_Max",
        "RolesCount_Mean", "Positions", "BoardSeats", "OtherRoles",
        "WorkExp_Idx_Max", "WorkExp_Idx_Mean", "Total_Founders",
    ]}
)
```

- [ ] **Step 4: Run stage 2b and the tests**

```bash
.venv/bin/python -c "
from src.panel.config import PanelConfig
from src.panel import stage2_team
stage2_team.run_panel(PanelConfig())
"
.venv/bin/python -m pytest tests/panel/test_stage2_panel.py -v
```
Expected: 3 passed, and the partial report shows the 23 team columns at zero
differences.

- [ ] **Step 5: Commit**

```bash
git add src/panel/stage2_team.py src/panel/validate.py tests/panel/test_stage2_panel.py
git commit -m "feat(panel): add stage 2b (team panel)"
```

---

### Task 9: Derive the country→Europe mapping from the data

**Files:**
- Create: `src/panel/data/europe.csv`
- Create: `scripts/derive_europe_mapping.py`
- Test: `tests/panel/test_europe.py`

**Interfaces:**
- Consumes: `CompanySimilarRelation.csv`, `db_master_1.csv` (for `N_Europe`).
- Produces: `src/panel/data/europe.csv` with columns `country,is_europe`, and
  `load_europe() -> dict[str, bool]` in `src/panel/io.py`.

**Background.** R calls `countrycode(..., destination = "continent")`. Guessing
which of the 209 country names that package calls "Europe" is unreliable —
Cyprus, Russia, Turkey, Georgia, Armenia, Azerbaijan, Kazakhstan, Greenland,
Kosovo and the Faroe Islands all sit on classification boundaries. Depending on
the library at runtime is also wrong: a future release would change the panel.

So derive the set instead. `N_Europe` in `db_master_1.csv` is
`sum(SimilarCompanyIsEurope)` over each company's rows, so with the raw counts
per country per company we can **solve** for membership: any company whose
similar companies all come from a single country `c`, with count `k`, gives
`is_europe(c) = (N_Europe == k)`. That determines almost every country in one
pass; the rest fall out by subtracting already-determined countries from
companies with two distinct countries. Freeze the result in the repo.

- [ ] **Step 1: Write the failing test**

```python
# tests/panel/test_europe.py
from src.panel.io import load_europe


def test_mapping_covers_every_country_in_the_source():
    mapping = load_europe()
    assert len(mapping) == 209


def test_unambiguous_cases_are_classified_correctly():
    m = load_europe()
    for c in ["Italy", "France", "Germany", "Spain", "Poland", "Sweden"]:
        assert m[c] is True, c
    for c in ["United States", "China", "Brazil", "Australia", "Nigeria"]:
        assert m[c] is False, c


def test_every_country_has_a_boolean_verdict():
    assert all(isinstance(v, bool) for v in load_europe().values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/panel/test_europe.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_europe'`

- [ ] **Step 3: Write the derivation script**

```python
# scripts/derive_europe_mapping.py
"""Solve the country->Europe mapping from N_Europe in db_master_1.csv.

R used countrycode(); rather than guess how that package classifies the
boundary cases, we recover the exact set the R run used and freeze it.
"""

from __future__ import annotations

import csv
from pathlib import Path

import polars as pl

RAW = Path("data/raw/pitchbook/CompanySimilarRelation.csv")
REF = Path("data/reference/db_master_1.csv")
OUT = Path("src/panel/data/europe.csv")


def main() -> None:
    counts = (
        pl.scan_csv(RAW, infer_schema_length=0, null_values=[])
        .select(["CompanyID", "SimilarCompanyHQCountry"])
        .filter(pl.col("SimilarCompanyHQCountry").is_not_null()
                & (pl.col("SimilarCompanyHQCountry") != ""))
        .group_by(["CompanyID", "SimilarCompanyHQCountry"])
        .len()
        .collect(engine="streaming")
    )
    target = (
        pl.read_csv(REF, infer_schema_length=0, null_values=[])
        .select(["CompanyID", "N_Europe"])
        .with_columns(pl.col("N_Europe").cast(pl.Int64, strict=False))
        .drop_nulls()
    )
    countries = sorted(counts["SimilarCompanyHQCountry"].unique().to_list())
    known: dict[str, bool] = {}

    per_company = (
        counts.join(target, on="CompanyID", how="inner")
        .group_by("CompanyID")
        .agg(
            pl.col("SimilarCompanyHQCountry").alias("cs"),
            pl.col("len").alias("ns"),
            pl.col("N_Europe").first().alias("target"),
        )
    )
    rows = per_company.to_dicts()

    for _ in range(len(countries) + 2):          # iterate to a fixed point
        progress = False
        for r in rows:
            unknown = [(c, n) for c, n in zip(r["cs"], r["ns"]) if c not in known]
            if len(unknown) != 1:
                continue
            resolved = sum(n for c, n in zip(r["cs"], r["ns"]) if known.get(c))
            c, n = unknown[0]
            remainder = r["target"] - resolved
            if remainder in (0, n):
                known[c] = remainder == n
                progress = True
        if not progress:
            break

    missing = [c for c in countries if c not in known]
    if missing:
        raise SystemExit(f"undetermined countries: {missing}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["country", "is_europe"])
        for c in countries:
            w.writerow([c, "true" if known[c] else "false"])
    n_eu = sum(known.values())
    print(f"scritto {OUT}: {len(countries)} paesi, {n_eu} in Europa")


if __name__ == "__main__":
    main()
```

Add to `src/panel/io.py`:

```python
def load_europe() -> dict[str, bool]:
    """Frozen country -> is-Europe mapping, recovered from the R run.

    Derived once by scripts/derive_europe_mapping.py so that the panel never
    depends on a third-party classification that could change.
    """
    path = Path(__file__).parent / "data" / "europe.csv"
    df = pl.read_csv(path)
    return {r["country"]: r["is_europe"] == "true" for r in df.to_dicts()}
```

- [ ] **Step 4: Run the derivation and the tests**

```bash
.venv/bin/python scripts/derive_europe_mapping.py
.venv/bin/python -m pytest tests/panel/test_europe.py -v
```
Expected: the script prints 209 countries and exits 0; 3 passed.

If the script reports undetermined countries, they appear too rarely as a
sole country. Extend the loop to handle two unknowns whose counts are equal
by trying both assignments and keeping the one consistent across all
companies, or classify those few by hand and record the reasoning in a
comment — the stage-3 checkpoint on `N_Europe` will catch a wrong choice.

- [ ] **Step 5: Commit**

```bash
git add src/panel/data/europe.csv scripts/derive_europe_mapping.py src/panel/io.py tests/panel/test_europe.py
git commit -m "feat(panel): derive and freeze the country-to-Europe mapping"
```

---

### Task 10: Stage 3 — competitors, CHECKPOINT B

**Files:**
- Create: `src/panel/stage3_relations.py`
- Test: `tests/panel/test_stage3_competitors.py`

**Interfaces:**
- Consumes: `db_master_1_v1.parquet`, `load_europe`.
- Produces: `run_competitors(cfg) -> None`, writing `db_master_1.parquet`.

**R source:** `1_Arrange_DB.R:681-708`.

**Translation notes.** `CompanySimilarRelation.csv` is 836 MB — scan lazily,
project `CompanyID`, `SimilarCompanyID`, `SimilarityScore`, `IsCompetitor`,
`SimilarCompanyHQCountry`, and aggregate in streaming.

- `mean(SimilarityScore)` and `sum(IsCompetitor == "Yes")` are **without**
  `na.rm`, so a single null makes the whole aggregate null. Measured: this
  never fires, but reproduce it faithfully.
- `Same_Country = any(a[score > 90] == b[score > 90])` without `na.rm` — null
  when no comparison is true and at least one is null. Measured: 1,563
  companies (bug B4).
- `N_Europe` sums over **all** rows; `N_Outside_Europe` only over rows with
  `SimilarityScore > 90` (bug B8).

- [ ] **Step 1: Write the failing test**

```python
# tests/panel/test_stage3_competitors.py
import polars as pl

from src.panel.config import PanelConfig
from src.panel.validate import CHECKPOINTS, load_reference, verify


def test_checkpoint_b_matches_the_r_reference():
    cfg = PanelConfig()
    cp = CHECKPOINTS["B"]
    actual = pl.read_parquet(cfg.interim(cp.interim))
    reference = load_reference(cfg, cp.reference)
    report = verify(actual, reference, key=cp.key, name="checkpoint B")
    print(report.render())
    assert actual.height == cp.expect_rows
    report.assert_clean()


def test_same_country_is_null_for_the_measured_number_of_companies():
    """Bug B4: any() without na.rm. The spec measured 1,563 companies."""
    cfg = PanelConfig()
    m1 = pl.read_parquet(cfg.interim("db_master_1.parquet"))
    with_comp = m1.filter(pl.col("SimilarityScoreMax").is_not_null())
    assert with_comp["Same_Country"].null_count() == 1_563


def test_competitor_aggregates_are_never_null_where_competitors_exist():
    cfg = PanelConfig()
    m1 = pl.read_parquet(cfg.interim("db_master_1.parquet"))
    with_comp = m1.filter(pl.col("SimilarityScoreMax").is_not_null())
    assert with_comp.height == 116_674
    for c in ["SimilarityScoreMean", "N_Competitors", "N_Europe", "N_Outside_Europe"]:
        assert with_comp[c].null_count() == 0, c
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/panel/test_stage3_competitors.py -v`
Expected: FAIL — `db_master_1.parquet` does not exist

- [ ] **Step 3: Write the implementation**

Transcribe `1_Arrange_DB.R:681-708` into `run_competitors(cfg)`.

- [ ] **Step 4: Run and verify checkpoint B**

```bash
.venv/bin/python -c "
from src.panel.config import PanelConfig
from src.panel import stage3_relations
stage3_relations.run_competitors(PanelConfig())
"
.venv/bin/python -m src.panel.validate --checkpoint B
.venv/bin/python -m pytest tests/panel/test_stage3_competitors.py -v
```
Expected: `checkpoint B: PASS`, 116,920 rows, 3 passed. A mismatch on
`N_Europe` means the Task 9 mapping is wrong — the report's examples name the
companies, whose country lists point straight at the offending country.

- [ ] **Step 5: Commit**

```bash
git add src/panel/stage3_relations.py tests/panel/test_stage3_competitors.py
git commit -m "feat(panel): add stage 3 competitors, checkpoint B passes"
```

---

### Task 11: Stage 3b — employees, financials, news

**Files:**
- Modify: `src/panel/stage3_relations.py`, `src/panel/validate.py`
- Test: `tests/panel/test_stage3_panel.py`

**Interfaces:**
- Consumes: `db_master_2_team.parquet`.
- Produces: `run_panel(cfg) -> None`, writing `db_master_2_relations.parquet`.

**R source:** `1_Arrange_DB.R:710-813`.

**Translation notes.**
- Employees (lines 711-732): parse `Date`, sort `(CompanyID, Year, desc(Date))`
  and keep the first row per `(CompanyID, Year)` — the latest reading of the
  year. Join on `(CompanyID, Year_Delta = Year)`.
- Financials (lines 736-774): same dedup on `PeriodEndDate`; the values only
  **fill** cells already null in `db_master_2` (coalesce, not overwrite).
- News (lines 776-813): count per `(CompanyID, year)`, then `N_News = 0`
  wherever the join produced a null.

- [ ] **Step 1: Write the failing test**

```python
# tests/panel/test_stage3_panel.py
import polars as pl

from src.panel.config import PanelConfig
from src.panel.validate import run_partial


def test_news_count_is_never_null():
    cfg = PanelConfig()
    df = pl.read_parquet(cfg.interim("db_master_2_relations.parquet"))
    assert df["N_News"].null_count() == 0


def test_employee_financial_and_news_columns_are_final_after_stage_3():
    cfg = PanelConfig()
    df = pl.read_parquet(cfg.interim("db_master_2_relations.parquet"))
    report = run_partial(cfg, stage=3, actual=df)
    print(report.render())
    for c in ["EmployeeCount", "N_News", "Revenue", "GrossProfit", "NetIncome",
              "EnterpriseValue", "EBITDA", "EBIT"]:
        assert report.column(c).n_diff == 0, f"{c} should be final after stage 3"


def test_no_regression_in_team_columns():
    cfg = PanelConfig()
    df = pl.read_parquet(cfg.interim("db_master_2_relations.parquet"))
    report = run_partial(cfg, stage=3, actual=df)
    for c in ["Total_People", "Percent_Females", "Total_Founders", "Institute"]:
        assert report.column(c).n_diff == 0, f"{c} regressed at stage 3"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/panel/test_stage3_panel.py -v`
Expected: FAIL — `db_master_2_relations.parquet` does not exist

- [ ] **Step 3: Write the implementation**

Transcribe `1_Arrange_DB.R:710-813`, then register:

```python
COLUMN_FINALISED_AT_STAGE.update(
    {c: 3 for c in ["EmployeeCount", "N_News", "Revenue", "GrossProfit",
                    "NetIncome", "EnterpriseValue", "EBITDA", "EBIT"]}
)
```

- [ ] **Step 4: Run and test**

```bash
.venv/bin/python -c "
from src.panel.config import PanelConfig
from src.panel import stage3_relations
stage3_relations.run_panel(PanelConfig())
"
.venv/bin/python -m pytest tests/panel/test_stage3_panel.py -v
```
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/panel/stage3_relations.py src/panel/validate.py tests/panel/test_stage3_panel.py
git commit -m "feat(panel): add stage 3b (employees, financials, news)"
```

---

### Task 12: Imputation strategies

**Files:**
- Create: `src/panel/imputation.py`
- Test: `tests/panel/test_imputation.py`

**Interfaces:**
- Consumes: `PanelConfig`, `quantile_type7`.
- Produces: `impute_total_invested(deals: pl.DataFrame, cfg) -> pl.DataFrame`
  adding two columns — `TotalInvestedCapital_Est` (Float64) and
  `TotalInvestedCapital_WasImputed` (Boolean, true only on the rows the model
  actually filled); `PREDICTOR_VARS: list[str]`.

**R source:** `1_Arrange_DB.R:1037-1119`.

**Background.** The R model runs `randomForest(ntree = 50)` with **no
`set.seed`**, so it is not reproducible even by re-running the R. It is also a
leakage vector — trained across all years, fitted before any train/test split,
with `DealTypeGrouped` (which determines the target) among its predictors.
Only `TotalRaised_Est` reaches the models, at `src/preprocessing.py:475`.

Three strategies:
- `r_legacy` — same logic with `RandomForestRegressor(n_estimators=50,
  random_state=cfg.seed)`: same predictors, same p95 winsorisation, same
  cap at the per-`DealTypeGrouped` third quartile. Makes the pipeline
  standalone and measures how far an equivalent re-run lands from R's.
- `r_injected` — **at deal level this is a no-op**: it returns the frame with
  `TotalInvestedCapital_Est` equal to `TotalInvestedCapital` and
  `TotalInvestedCapital_WasImputed` all `False`. The injection cannot happen
  here because the R reference only exists aggregated: stage 4 overwrites the
  three `TotalRaised_Est*` columns from `db_master_2.csv` *after* building
  `deals_panel`. This isolates the RF as the only known divergence so
  everything downstream verifies exactly.
- `leakage_free` — raises `NotImplementedError`. Phase 2.

- [ ] **Step 1: Write the failing test**

```python
# tests/panel/test_imputation.py
import polars as pl
import pytest

from src.panel.config import PanelConfig
from src.panel.imputation import PREDICTOR_VARS, impute_total_invested


def _deals() -> pl.DataFrame:
    n = 200
    return pl.DataFrame(
        {
            "DealID": [f"d{i}" for i in range(n)],
            "TotalInvestedCapital": [None if i % 5 == 0 else float(i) for i in range(n)],
            "TotalInvestedCapital_Est": [None if i % 5 == 0 else float(i) for i in range(n)],
            "UndisclosedAmountFlag": [1] * n,
            "DealTypeGrouped": ["Seed Round"] * n,
            "PrimaryIndustrySector": ["Information Technology"] * n,
            "Age": [i % 10 for i in range(n)],
            "MeanTotalActivePortfolio": [float(i % 7) for i in range(n)],
            "MeanTotalInvestments": [float(i % 11) for i in range(n)],
            "MeanMedianRoundAmount": [float(i % 13) for i in range(n)],
            "TotalInvestors": [i % 4 for i in range(n)],
            "has_Angel": [i % 2 == 0 for i in range(n)],
            "has_Corporate": [i % 3 == 0 for i in range(n)],
            "has_VentureCapital": [True] * n,
            "has_Accelerator": [False] * n,
            "has_PublicInvestor": [False] * n,
            "LeadInvestorCount": [i % 2 for i in range(n)],
        }
    )


def test_predictor_list_matches_the_r_script():
    assert PREDICTOR_VARS == [
        "DealTypeGrouped", "MeanTotalActivePortfolio", "MeanTotalInvestments",
        "MeanMedianRoundAmount", "TotalInvestors", "has_Angel", "has_Corporate",
        "has_VentureCapital", "has_Accelerator", "has_PublicInvestor",
        "LeadInvestorCount", "PrimaryIndustrySector", "Age",
    ]


def test_r_legacy_fills_only_the_eligible_missing_rows():
    cfg = PanelConfig(imputation_strategy="r_legacy")
    out = impute_total_invested(_deals(), cfg)
    assert out["TotalInvestedCapital_Est"].null_count() == 0
    # Rows that already had a value must be untouched.
    src = _deals()
    keep = src.filter(pl.col("TotalInvestedCapital").is_not_null())["DealID"]
    a = out.filter(pl.col("DealID").is_in(keep)).sort("DealID")
    b = src.filter(pl.col("DealID").is_in(keep)).sort("DealID")
    assert a["TotalInvestedCapital_Est"].to_list() == b["TotalInvestedCapital_Est"].to_list()


def test_was_imputed_flag_marks_exactly_the_filled_rows():
    cfg = PanelConfig(imputation_strategy="r_legacy")
    src = _deals()
    out = impute_total_invested(src, cfg)
    expected = src["TotalInvestedCapital_Est"].is_null()
    assert out["TotalInvestedCapital_WasImputed"].to_list() == expected.to_list()


def test_r_legacy_is_deterministic_for_a_fixed_seed():
    cfg = PanelConfig(imputation_strategy="r_legacy")
    a = impute_total_invested(_deals(), cfg)["TotalInvestedCapital_Est"].to_list()
    b = impute_total_invested(_deals(), cfg)["TotalInvestedCapital_Est"].to_list()
    assert a == b


def test_imputed_values_are_capped_at_the_group_third_quartile():
    cfg = PanelConfig(imputation_strategy="r_legacy")
    src = _deals()
    out = impute_total_invested(src, cfg)
    from src.panel.rutils import quantile_type7

    q3 = quantile_type7(src["TotalInvestedCapital"].to_list(), 0.75)
    imputed = out.filter(src["TotalInvestedCapital"].is_null())
    assert imputed["TotalInvestedCapital_Est"].max() <= q3 + 1e-9


def test_leakage_free_is_not_implemented_yet():
    cfg = PanelConfig(imputation_strategy="leakage_free")
    with pytest.raises(NotImplementedError, match="phase 2"):
        impute_total_invested(_deals(), cfg)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/panel/test_imputation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.panel.imputation'`

- [ ] **Step 3: Write the implementation**

Translate `1_Arrange_DB.R:1037-1119`: build the complete-cases training set,
drop `Zero_Invested` rows from it, winsorise the four `winsor_vars` at the
`quantile_type7` p95, one-hot the two categorical predictors, fit
`RandomForestRegressor(n_estimators=50, random_state=cfg.seed)`, predict on the
rows where `TotalInvestedCapital_Est` is null **and** `UndisclosedAmountFlag == 1`
**and** all predictors are present, then cap each prediction at its group's
third quartile. `r_injected` skips the model entirely and returns the frame unchanged with
`TotalInvestedCapital_WasImputed` all `False` — stage 4 does the injection on
the aggregates. `leakage_free` raises
`NotImplementedError("leakage_free imputation is phase 2")`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/panel/test_imputation.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/panel/imputation.py tests/panel/test_imputation.py
git commit -m "feat(panel): add pluggable TotalInvestedCapital imputation"
```

---

### Task 13: Stage 4 — deals and investors

**Files:**
- Create: `src/panel/stage4_deals.py`
- Modify: `src/panel/validate.py`
- Test: `tests/panel/test_stage4.py`

**Interfaces:**
- Consumes: `db_master_1.parquet`, `db_master_2_relations.parquet`, `impute_total_invested`.
- Produces: `run(cfg) -> None`, writing `deals_panel.parquet` and `db_master_2_deals.parquet`.

**R source:** `1_Arrange_DB.R:826-1251`.

**Translation notes.**
- `db22_s` (lines 849-904): join `DealInvestorRelation` with 12 `Investor`
  columns, map `PrimaryInvestorType` to seven `InvestorCategory` values
  (lines 861-871), aggregate by `DealID`. Most aggregates are gated on
  `any(InvestorStatus == "New Investor")`; the `_Lead` block filters on
  `IsLeadInvestor == "Yes"` while keeping the same gate. Reproduce the
  asymmetry exactly.
- `Deal` date repair (lines 909-976): three conditional fills, then the
  midpoint fill `ceiling((Year_Prev + Year_Next) / 2)` using lag/lead ordered
  by `(CompanyID, DealNo)`.
- `filter(YearFounded > 2000)` — bug B1, same flag as stage 2b.
- `Year_Delta = pmax(year(DealDate), YearFounded)`.
- `Zero_Invested` rule (lines 994-1019): `DealType`s with more than 90%
  missing `TotalInvestedCapital` get 0; those with fewer than 200 occurrences
  and under 90% missing become `"Other"`. These thresholds come from global
  statistics — record that in a comment; it is the same family of problem as
  the RF, and phase 2 revisits it.
- Imputation via `impute_total_invested`. Under `r_injected` that call is a
  no-op; after `deals_panel` is built, overwrite its `TotalRaised_Est`,
  `TotalRaised_Est_NA` and `TotalRaised_Est_any` columns with the values joined
  from `db_master_2.csv` on `(CompanyID, Year_Delta)`. Under `r_legacy` leave
  the computed values alone.
- Deal-type flags (lines 1125-1151) and `deals_panel` (lines 1153-1209).
  The six `TotalRaised` variants have three distinct missing semantics:
  `sum(na.rm = TRUE)`; null if **all** are null; null if **any** is null.
- Join into `db_master_2` plus `TR_D` (lines 1222-1251), then the `R_NA_INF`
  NA loop.

- [ ] **Step 1: Write the failing test**

```python
# tests/panel/test_stage4.py
import polars as pl

from src.panel.config import PanelConfig
from src.panel.validate import run_partial


def test_zero_invested_rule_selects_the_measured_deal_types():
    """The >90%-missing rule must select exactly these 21 deal types.

    Measured during design over the 285,336 deals of companies founded after
    2000. A change here means the rule or the upstream filter drifted.
    """
    from src.panel.stage4_deals import zero_invested_deal_types

    cfg = PanelConfig()
    assert sorted(zero_invested_deal_types(cfg)) == [
        "Bankruptcy: Admin/Reorg", "Bankruptcy: Liquidation", "Buyout/LBO",
        "Corporate Asset Purchase", "Corporate Licensing", "Debt Repayment",
        "GP Stakes", "Investor Buyout by Management", "Joint Venture",
        "Merger of Equals", "Merger/Acquisition", "Out of Business",
        "Platform Creation", "Product Crowdfunding",
        "Sale-Lease back facility", "Secondary Transaction - Open Market",
        "Secondary Transaction - Private", "Spin-Off", "Undetermined",
        "University Spin-Out", "Working Capital",
    ]


def test_total_raised_variants_have_distinct_missing_semantics():
    cfg = PanelConfig()
    dp = pl.read_parquet(cfg.interim("deals_panel.parquet"))
    # "any null -> null" must be at least as missing as "all null -> null",
    # which in turn is at least as missing as the na.rm sum.
    assert dp["TotalRaised_any"].null_count() >= dp["TotalRaised_NA"].null_count()
    assert dp["TotalRaised_NA"].null_count() >= dp["TotalRaised"].null_count()


def test_deal_columns_are_final_after_stage_4():
    cfg = PanelConfig(imputation_strategy="r_injected")
    df = pl.read_parquet(cfg.interim("db_master_2_deals.parquet"))
    report = run_partial(cfg, stage=4, actual=df)
    print(report.render())
    for c in ["N_Deal", "N_VCround", "TotalRaised", "TotalRaised_NA",
              "TotalRaised_any", "TR_D", "InvestorOwnership", "PremoneyValuation"]:
        assert report.column(c).n_diff == 0, f"{c} should be final after stage 4"


def test_no_regression_in_earlier_stage_columns():
    cfg = PanelConfig(imputation_strategy="r_injected")
    df = pl.read_parquet(cfg.interim("db_master_2_deals.parquet"))
    report = run_partial(cfg, stage=4, actual=df)
    for c in ["Total_People", "EmployeeCount", "N_News", "Revenue"]:
        assert report.column(c).n_diff == 0, f"{c} regressed at stage 4"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/panel/test_stage4.py -v`
Expected: FAIL — `deals_panel.parquet` does not exist

- [ ] **Step 3: Write the implementation**

Transcribe `1_Arrange_DB.R:826-1251`, then register:

```python
COLUMN_FINALISED_AT_STAGE.update(
    {c: 4 for c in ["N_Deal", "N_VCround", "TotalRaised", "TotalRaised_NA",
                    "TotalRaised_any", "TR_D", "DealType", "InvestorOwnership",
                    "PremoneyValuation", "PreferredVerticals", "CEO_ID",
                    "Other_Deal"]}
)
```

Note `N_Deal` and `N_VCround` are *replaced* by their cumulative versions in
stage 5, so they are final at 4 only for the partial check that runs before
stage 5 — the stage-5 checkpoint compares the cumulative values. Record that
in a comment next to the entry so the next reader is not surprised.

- [ ] **Step 4: Run and test**

```bash
.venv/bin/python -c "
from src.panel.config import PanelConfig
from src.panel import stage4_deals
stage4_deals.run(PanelConfig(imputation_strategy='r_injected'))
"
.venv/bin/python -m pytest tests/panel/test_stage4.py -v
```
Expected: 4 passed

- [ ] **Step 5: Report the true imputation count**

The spec's 48,000 was an upper bound: it lacked the `complete.cases` filter on
the predictors, four of which only exist for deals with registered investors.
Now that the predictors exist, print the real number. Have `run(cfg)` persist
the deal-level frame as `deals_imputed.parquet` (it carries
`TotalInvestedCapital_WasImputed` from Task 12), then:

```bash
.venv/bin/python -c "
import polars as pl
from src.panel.config import PanelConfig
cfg = PanelConfig(imputation_strategy='r_legacy')
d = pl.read_parquet(cfg.interim('deals_imputed.parquet'))
n = d.filter(pl.col('TotalInvestedCapital_WasImputed')).height
print(f'deal imputati dalla RF: {n:,} su {d.height:,} ({100*n/d.height:.1f}%)')
"
```

Record the result in the `stage4_deals` module docstring and carry it into the
notebook — it is the number the phase-2 decision on the imputation rests on.

- [ ] **Step 6: Commit**

```bash
git add src/panel/stage4_deals.py src/panel/validate.py tests/panel/test_stage4.py
git commit -m "feat(panel): add stage 4 (deals and investors)"
```

---

### Task 14: Stage 5 — finalisation, CHECKPOINTS C and D

**Files:**
- Create: `src/panel/stage5_final.py`
- Test: `tests/panel/test_stage5.py`

**Interfaces:**
- Consumes: `db_master_2_deals.parquet`, `db3.parquet`, `db_master_1.parquet`.
- Produces: `run(cfg) -> None`, writing `db_master_2.parquet`,
  `db_selected.parquet`, `db_final.parquet`.

**R source:** all of `2_Arrange_Final.R`.

**Translation notes.**
- Lines 23-78: 24 flags null→`False`, then `cumany` per company ordered by
  `Year_Delta`.
- Lines 80-92: `GrowthStage`, a short-circuit cascade of seven conditions —
  the order is binding.
- Lines 94-101: where `TR_D == 1`, the six `TotalRaised` variants go to 0.
- Lines 103-119: the cumulative block. `NewInvestors` is computed **before**
  `TotalInvestors` is overwritten by its own cumulative sum, inside the same
  `mutate`; dplyr evaluates sequentially, so keep that order. `cumsum`
  propagates nulls, so `TotalRaised_NA_cum` and `TotalRaised_any_cum` go null
  from the first null onward.
- Lines 122-145: `weighted_cumulative` over four variables weighted by
  `NewInvestors`.
- Lines 148-167: forward-fill `CEO_ID`, then join 18 attributes from `db3` on
  `(CompanyID, PersonID)` with a `_CEO` suffix.
- Lines 202-203: `Age = Delta`, then select `vars_selected`.
- Lines 206-231: `StageBlock` (`stage_block`, bug B3), `YearsInStage`
  (`rle_sequence`), `GrowthNextStage` / `TimeNextStage` (`next_different`).
- Lines 237-242: `db_final` = `db_selected` left-joined with 22
  `db_master_1` columns.

- [ ] **Step 1: Write the failing test**

```python
# tests/panel/test_stage5.py
import polars as pl

from src.panel.config import PanelConfig
from src.panel.validate import CHECKPOINTS, load_reference, run_checkpoint, verify


def test_checkpoint_c_matches_the_r_reference():
    cfg = PanelConfig(imputation_strategy="r_injected")
    report = run_checkpoint(cfg, "C")
    print(report.render())
    report.assert_clean()


def test_checkpoint_d_matches_the_r_reference():
    cfg = PanelConfig(imputation_strategy="r_injected")
    report = run_checkpoint(cfg, "D")
    print(report.render())
    report.assert_clean()


def test_stageblock_goes_null_for_a_company_with_a_missing_stage():
    """Bug B3, verified against company 100026-46 in the R output."""
    cfg = PanelConfig()
    sel = pl.read_parquet(cfg.interim("db_selected.parquet"))
    c = sel.filter(pl.col("CompanyID") == "100026-46").sort("Age")
    assert c["StageBlock"].null_count() == c.height


def test_years_in_stage_restarts_after_a_null():
    cfg = PanelConfig()
    sel = pl.read_parquet(cfg.interim("db_selected.parquet"))
    c = sel.filter(pl.col("CompanyID") == "100026-46").sort("Age")
    assert c["YearsInStage"].to_list()[:3] == [1, 1, 1]


def test_db_final_join_does_not_duplicate_rows():
    cfg = PanelConfig()
    sel = pl.read_parquet(cfg.interim("db_selected.parquet"))
    fin = pl.read_parquet(cfg.interim("db_final.parquet"))
    assert fin.height == sel.height == 1_001_625


def test_r_legacy_differs_only_in_the_rf_dependent_columns():
    cfg = PanelConfig(imputation_strategy="r_legacy")
    cp = CHECKPOINTS["D"]
    actual = pl.read_parquet(cfg.interim(cp.interim))
    report = verify(actual, load_reference(cfg, cp.reference), key=cp.key,
                    name="D (r_legacy)")
    differing = {c.column for c in report.columns if c.n_diff}
    assert differing <= {"TotalRaised_Est", "TotalRaised_Est_cum",
                         "TotalRaised_Est_any", "TotalRaised_Est_any_cum",
                         "TotalRaised_Est_NA", "TotalRaised_Est_NA_cum"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/panel/test_stage5.py -v`
Expected: FAIL — `db_master_2.parquet` does not exist

- [ ] **Step 3: Write the implementation**

Transcribe `2_Arrange_Final.R:23-242`.

- [ ] **Step 4: Run and verify both checkpoints**

```bash
.venv/bin/python -c "
from src.panel.config import PanelConfig
from src.panel import stage5_final
stage5_final.run(PanelConfig(imputation_strategy='r_injected'))
"
.venv/bin/python -m src.panel.validate --checkpoint C --strategy r_injected
.venv/bin/python -m src.panel.validate --checkpoint D --strategy r_injected
.venv/bin/python -m pytest tests/panel/test_stage5.py -v
```
Expected: both checkpoints `PASS` at 1,001,625 rows; 6 passed.

- [ ] **Step 5: Measure the RandomForest's arbitrariness**

Re-run stage 4 and 5 with `r_legacy`, then report how far `TotalRaised_Est`
moves relative to R. This number belongs in the notebook and is an input to
the phase-2 decision:

```bash
.venv/bin/python -c "
from src.panel.config import PanelConfig
from src.panel import stage4_deals, stage5_final
from src.panel.validate import run_checkpoint
cfg = PanelConfig(imputation_strategy='r_legacy')
stage4_deals.run(cfg); stage5_final.run(cfg)
r = run_checkpoint(cfg, 'D')
c = r.column('TotalRaised_Est')
print(f'righe divergenti: {c.n_diff:,} / {c.n_compared:,} ({c.pct_diff:.2f}%)')
print(f'massima differenza assoluta: {c.max_abs_diff:,.0f}')
"
```

- [ ] **Step 6: Commit**

```bash
git add src/panel/stage5_final.py tests/panel/test_stage5.py
git commit -m "feat(panel): add stage 5, checkpoints C and D pass"
```

---

### Task 15: Narrative notebook

**Files:**
- Create: `notebooks/build_panel.ipynb`
- Modify: `README.md`

**Interfaces:**
- Consumes: every stage module and `validate`.
- Produces: the runnable, documented pipeline.

**Requirements.** For every stage the notebook must carry a markdown cell
covering the three things the user asked for, then a code cell that runs the
stage, then a cell that prints the verification report:

1. **What the cell does**, with the corresponding R line range.
2. **Which bugs from the registry are active at that point**, what they do and
   why they are bugs.
3. **The measured impact on the rest of the pipeline** — the numbers from the
   spec's registry, not estimates.

The opening cell prints `cfg.active_fixes()` so a reader always knows which
behaviour they are looking at.

- [ ] **Step 1: Write the notebook skeleton with one section per stage**

Sections: setup and configuration; stage 1; stage 2a + checkpoint A;
stage 2b; stage 3 + checkpoint B; stage 3b; stage 4; stage 5 + checkpoints C
and D; the RandomForest arbitrariness measurement; a closing summary table of
all ten bugs with their measured impact.

- [ ] **Step 2: Run the notebook top to bottom**

```bash
.venv/bin/python -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=7200 \
  --output build_panel_executed.ipynb notebooks/build_panel.ipynb
```
Expected: no exception; all four checkpoint cells print `PASS`.

- [ ] **Step 3: Measure and record the one unknown**

`fix_permanenza_media_per_company` (B5) is the only bug whose impact the spec
could not measure. Run stage 2 both ways and report the difference in the team
columns:

```bash
.venv/bin/python - <<'EOF'
from pathlib import Path

import polars as pl

from src.panel import stage1_company, stage2_team
from src.panel.config import PanelConfig

for fix in (False, True):
    cfg = PanelConfig(
        fix_permanenza_media_per_company=fix,
        interim_dir=Path(f"data/interim_b5_{fix}"),
    )
    stage1_company.run(cfg)
    stage2_team.run_db3(cfg)
    stage2_team.run_panel(cfg)
    df = pl.read_parquet(cfg.interim("db_master_2_team.parquet"))
    print(
        f"fix={fix}  righe_con_team={df.filter(pl.col('Total_People').is_not_null()).height:,}"
        f"  somma_Total_People={df['Total_People'].sum():,}"
        f"  somma_Total_Founders={df['Total_Founders'].sum():,}"
    )
EOF
```

Add the result to the notebook's closing table and to the spec's registry
row for B5, replacing "DA MISURARE" with the number.

- [ ] **Step 4: Update the README**

Add a "Panel construction" section to `README.md` documenting: that the panel
is built by `src/panel/`, how to run it, where the verification reports land,
and that `data/raw/panel.csv.gz` additionally carries the post-processing that
is still to be ported (year-by-year competitor columns, the `*Group` columns,
and the row filter that takes 1,001,625 rows down to 882,324).

- [ ] **Step 5: Commit**

```bash
git add notebooks/build_panel.ipynb README.md docs/superpowers/specs/
git commit -m "docs(panel): add narrative notebook and measure bug B5"
```

---

### Task 16: Stage 6 — stage grouping and truncation, CHECKPOINT E

**Files:**
- Create: `src/panel/stage6_panel.py`
- Modify: `src/panel/validate.py`
- Test: `tests/panel/test_stage6.py`

**Interfaces:**
- Consumes: `db_final.parquet`.
- Produces: `run(cfg) -> None`, writing `db_master_panel.parquet`.

**Source:** no R and no Python script exists for this step — the code that
produced `db_master_panel.csv.gz` was lost. The four rules below were
reverse-engineered from the file itself and each was verified against it at
zero divergence over all 882.324 rows. **They are the specification.**

**The rules, in order.** Sort by `(CompanyID, Year_Delta)` first; every rule
depends on row order.

- **R1 — `GrowthStageGroup`.** A plain remap of `GrowthStage`, null-preserving:
  `Preseed`/`Seed`/`EarlyVC` → `Early`; `LaterVC_or_Other` → `Later`;
  `Out` → `Out`; `Exit_M&A`/`Exit_Public` → `Exit`.
  *Verified: 0 divergent rows.*

- **R2 — `GrowthNextStageGroup` and `TimeNextStageGroup`, computed on the
  UNTRUNCATED sequence.** This is the rule a naive implementation gets wrong:
  they must be computed *before* R3 removes the terminal rows, otherwise `Out`
  and `Exit` could never appear as a future stage. For row *i* of a company,
  scan forward for the first *j > i* whose group is non-null and different
  from row *i*'s group.
  - If row *i*'s own group is **null**, there is never a match — this
    reproduces R's `NA != x` semantics, the same mechanism as
    `GrowthNextStage` in `2_Arrange_Final.R:216-231`.
  - Match found → `GrowthNextStageGroup` is that group, `TimeNextStageGroup`
    is `j - i`.
  - No match → `GrowthNextStageGroup` is the literal string `"Stay"` and
    `TimeNextStageGroup` is the distance in rows to the company's **last
    untruncated row**, `n - 1 - i`. Note `"Stay"` is a real value in
    `db_master_panel.csv.gz`; the competitors notebook replaces it with the
    current group, and that replacement belongs to stage 7, not here.

  *Verified: 0 divergent rows on both columns (one row with a null
  `Year_Delta` is unjoinable and excluded).*

- **R3 — truncation.** Drop every row from the company's first row whose
  group is `Out` or `Exit` onwards, that row included. This also removes
  non-terminal rows that happen to follow a terminal one — 5.365 of them, and
  they are not an anomaly to work around.
  *Verified: 882.324/882.324 rows and 116.327/116.327 companies, exactly.*

- **R4 — `YearsInStage` recomputed on the group.** `rle_sequence` (Task 2b)
  applied to `GrowthStageGroup` on the truncated frame, R's `rle` semantics
  with each null opening its own run. This **overwrites** the `YearsInStage`
  that stage 5 computed on the ungrouped `GrowthStage`.
  *Verified: 0 divergent rows.*

**`StageBlock` is knowingly not reproduced.** The value in
`db_master_panel.csv.gz` matches neither a recomputation on `GrowthStage`
(81.954 rows off) nor one on `GrowthStageGroup` (36.408 off) nor the value
carried over from `db_selected`. The column is dead — nothing downstream
reads it, `preprocessing.py` included. Carry stage 5's value through, add a
comment saying so, and register `StageBlock` as an expected difference at
checkpoint E. Do not spend time on it.

- [ ] **Step 1: Write the failing test**

`tests/panel/test_stage6.py`, on a small hand-built frame, must pin:
one company whose stage goes `null, Preseed, Seed, LaterVC_or_Other, Out,
Seed` — asserting the `Out` row and the `Seed` row after it are both gone,
that the `Preseed` row's `GrowthNextStageGroup` is `Later` (not `Early`,
because `Seed` maps to the same group), that the `LaterVC_or_Other` row's is
`Out` with time 1, that the leading null row gets `"Stay"` with time 5
(distance to the last untruncated row), and that `YearsInStage` reads
`1, 1, 2, 1` on the four surviving rows.

- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement `src/panel/stage6_panel.py`**
- [ ] **Step 4: Run test to verify it passes**

- [ ] **Step 5: Register checkpoint E and run it**

```bash
.venv/bin/python -m src.panel.validate --checkpoint E
```
Expected: `PASS`, with `StageBlock` reported as an expected difference, the
six `_Est` columns as expected-missing and `TR_D` as expected-extra.

- [ ] **Step 6: Commit**

```bash
git add src/panel/stage6_panel.py src/panel/validate.py tests/panel/test_stage6.py
git commit -m "feat(panel): add stage 6, stage grouping and truncation"
```

---

### Task 17: Stage 7 — competitor timing, CHECKPOINT F

**Files:**
- Create: `src/panel/stage7_competitors.py`
- Modify: `src/panel/validate.py`
- Test: `tests/panel/test_stage7.py`

**Interfaces:**
- Consumes: `db_master_panel.parquet`, `Company.csv`, `CompanySimilarRelation.csv`.
- Produces: `run(cfg) -> None`, writing `panel.parquet` and
  `data/interim/panel.csv.gz`.

**Source:** `notebook_temporizzazione_competitors.ipynb`, cells 4 and 6. The
R leaves the competitor columns static; this stage replaces them with
year-by-year ones. **Never write to `data/raw/panel.csv.gz`** — that file is
the reference for this checkpoint.

**Translation notes.**
- `company_life` is built from **all** 134.355 companies in `Company.csv`, not
  only those with `YearFounded > 1999`: a competitor may be older than the
  panel's cohorts. Keep only rows with both `YearFounded` and `MaxYear`
  non-null.
- **`MaxYear` must use `rutils.parse_date_r`, not the notebook's parser.** The
  notebook reads dates as `%m/%d/%Y` with an `%m/%d/%y` fallback; chrono maps
  a two-digit `25`-`69` to 2025-2069 while R's `cutoff_2000 = 24` maps it to
  1925-1969. Since `MaxYear` is the end of a competitor's activity window, the
  two conventions disagree about whether a competitor is alive. Use the R
  primitive so the whole pipeline parses dates one way. Also use
  `FiscalDate` day 30, as stage 1 does, not the notebook's 28 — it cannot
  change a year, but there is no reason to keep two spellings.
- The relation is **directed** and stays directed: `CompanyID` declares
  `SimilarCompanyID` as similar; the reverse pair is not added. The notebook
  says so explicitly.
- Active-competitor pairs: `IsCompetitor == "Yes"`, `CompanyID` in the panel,
  `SimilarCompanyID` in `company_life`, and
  `YF_comp <= Year_Delta <= MY_comp`.
- Per `(CompanyID, Year_Delta)`: `N_Competitors` = distinct
  `SimilarCompanyID`; `Same_Country` = count of pairs whose two `HQCountry`
  agree, nulls dropped rather than counted false; `SimilarityScoreMean` = mean
  `SimilarityScore` over **all** similar companies active that year, not only
  the competitors. Null → 0 on all three.
- The static `_All` trio is the same aggregation without the year filter.
- Drop `N_Europe` and `N_Outside_Europe`.
- Replace `GrowthNextStageGroup == "Stay"` with the current
  `GrowthStageGroup`.
- Remap `CompanyID` to consecutive integers from 1, ordered by the string id.
  **This is the last operation of the pipeline**, because it destroys every
  join back to the reference files.

**Three semantic changes to record in the module docstring**, since the
columns keep their R names while changing meaning: `Same_Country` goes from
boolean to count; `SimilarityScoreMean` is filled with 0 where no similar
company was active, which is the *minimum* of the scale, not a neutral value;
and `N_Competitors_All` is not the R's `N_Competitors` — it is restricted to
competitors that have a usable life window. A fourth, methodological, belongs
in the paper rather than the code: `MaxYear` is the last year with *data*, not
the year of death, so the timing over-weights competitors that PitchBook
covers well.

- [ ] **Step 1: Write the failing test**

`tests/panel/test_stage7.py` on hand-built frames: a competitor whose window
covers only part of the panel years must be counted in those years and not
outside them; a pair with one null `HQCountry` must not increment
`Same_Country`; `SimilarityScoreMean` must average non-competitor similar
companies too; a `"Stay"` row must take the current group.

- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement `src/panel/stage7_competitors.py`**
- [ ] **Step 4: Run test to verify it passes**

- [ ] **Step 5: Register checkpoint F and run it**

```bash
.venv/bin/python -m src.panel.validate --checkpoint F
```
Expected: `PASS` against `data/raw/panel.csv.gz`, with the same three expected
exclusions as checkpoint E.

- [ ] **Step 6: Commit**

```bash
git add src/panel/stage7_competitors.py src/panel/validate.py tests/panel/test_stage7.py
git commit -m "feat(panel): add stage 7, year-by-year competitor columns"
```

---

## Verification of the whole plan

Run once at the end:

```bash
.venv/bin/python -m pytest tests/ -v
for c in A B C D E F; do .venv/bin/python -m src.panel.validate --checkpoint $c; done
```

Expected: every test passes, and all six checkpoints report `PASS`.
