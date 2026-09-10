"""Compare a stage's output against the R reference, column by column.

The report answers exactly two questions: which columns differ, and on how
many rows. Everything else is supporting detail.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import polars as pl

from src.panel.config import PanelConfig

_BOOL_TOKENS = {
    "TRUE": True,
    "T": True,
    "True": True,
    "true": True,
    "FALSE": False,
    "F": False,
    "False": False,
    "false": False,
}


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
    #: Rows dropped because the key itself is null, and so unalignable.
    keys_null_ref: int
    keys_null_act: int
    cols_only_ref: list[str]
    cols_only_act: list[str]
    columns: list[ColumnDiff]
    #: Columns declared absent on purpose, so they do not fail the report.
    expected_missing: list[str] = field(default_factory=list)
    #: Columns we add on purpose and the reference cannot have.
    expected_extra: list[str] = field(default_factory=list)

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
            and self.keys_null_ref == self.keys_null_act
            and not set(self.cols_only_ref) - set(self.expected_missing)
            and not set(self.cols_only_act) - set(self.expected_extra)
        )

    def assert_clean(self) -> None:
        if not self.passed():
            raise AssertionError(f"verification failed for {self.name}\n" + self.render())

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
        unexpected_ref = sorted(set(self.cols_only_ref) - set(self.expected_missing))
        unexpected_act = sorted(set(self.cols_only_act) - set(self.expected_extra))
        out = [
            f"=== {self.name}: {'PASS' if self.passed() else 'FAIL'} ===",
            f"righe   riferimento={self.n_rows_ref:,}  ottenute={self.n_rows_act:,}",
            f"chiavi  solo R={self.keys_only_ref:,}  solo PY={self.keys_only_act:,}"
            f"  nulle R={self.keys_null_ref:,} PY={self.keys_null_act:,}",
            f"colonne solo R={unexpected_ref}  solo PY={unexpected_act}",
            f"colonne assenti attese={sorted(self.expected_missing)}  "
            f"in piu' attese={sorted(self.expected_extra)}",
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


def _norm_key(df: pl.DataFrame, col: str, alias: str) -> pl.Expr:
    """Normalise one key column so both sides join on the same representation.

    The reference CSVs were exported with a float dtype for `Year_Delta`, so a
    key reads `"2013.0"` there and `2013` here; comparing the two as text
    matches nothing and every row looks unpaired. Numeric-looking keys are
    therefore normalised through Float64 and back. `CompanyID` ("100020-70")
    and `PersonID` ("58322-80P") do not cast and stay text.
    """
    s = df.get_column(col)
    as_float = s.cast(pl.Float64, strict=False)
    if as_float.null_count() == s.null_count():
        return pl.col(col).cast(pl.Float64, strict=False).cast(pl.String).alias(alias)
    return pl.col(col).cast(pl.String).alias(alias)


def _align(actual: pl.DataFrame, reference: pl.DataFrame, key: list[str]):
    """Restrict both frames to the shared keys and sort them identically.

    Rows whose key is itself null cannot be aligned with anything; they are
    dropped and counted separately, because they are a property of the data,
    not a difference between the two sides.
    """
    for name, df in (("actual", actual), ("reference", reference)):
        if df.select(key).is_duplicated().any():
            raise ValueError(f"duplicate keys in {name} on {key}")
    kcols = [f"__k{i}" for i in range(len(key))]
    a = actual.with_columns([_norm_key(actual, k, kc) for k, kc in zip(key, kcols, strict=True)])
    r = reference.with_columns(
        [_norm_key(reference, k, kc) for k, kc in zip(key, kcols, strict=True)]
    )
    has_key = pl.all_horizontal([pl.col(kc).is_not_null() for kc in kcols])
    null_act, null_ref = a.filter(~has_key).height, r.filter(~has_key).height
    a, r = a.filter(has_key), r.filter(has_key)
    only_ref = r.select(kcols).join(a.select(kcols), on=kcols, how="anti").height
    only_act = a.select(kcols).join(r.select(kcols), on=kcols, how="anti").height
    shared = r.select(kcols).join(a.select(kcols), on=kcols, how="semi")
    a = a.join(shared, on=kcols, how="semi").sort(kcols)
    r = r.join(shared, on=kcols, how="semi").sort(kcols)
    return a, r, only_ref, only_act, null_ref, null_act


def verify(
    actual: pl.DataFrame,
    reference: pl.DataFrame,
    key: list[str],
    name: str,
    *,
    expected_diff: frozenset[str] | set[str] = frozenset(),
    expected_missing: frozenset[str] | set[str] = frozenset(),
    expected_extra: frozenset[str] | set[str] = frozenset(),
    na_token: str | None = None,
    na_collapsed_columns: frozenset[str] | set[str] = frozenset(),
    rtol: float = 1e-9,
    n_examples: int = 10,
) -> VerificationReport:
    """Compare `actual` against the all-String `reference` on `key`.

    `expected_diff` names columns allowed to differ, `expected_missing` columns
    the reference has and we deliberately do not produce, `expected_extra`
    columns we add and the reference cannot have. None of the three makes the
    report pass silently: they are printed in the header either way.

    `na_collapsed_columns` names string columns where the export turned a cell
    holding the literal text "NA" into a null. `Institute` is built by R's
    `paste()`, which renders a missing institute as that text (bug B7): the
    reference keeps it inside longer strings ("NA; University of Oxford; ...")
    but has no cell equal to "NA" alone, so a whole-cell "NA" here must match a
    null there. Only that exact case is excused; every other value still has to
    agree.

    `na_token` says how the reference spells a missing value. `None` means the
    file has real empty fields, which polars reads as nulls — the case for
    db3, db_master_1, db_master_2, db_selected and panel.csv.gz. `"NA"` means
    R's `write.csv`, where a missing value and the literal string "NA" are the
    same six bytes and cannot be told apart; only `db_master_panel.csv.gz` is
    written that way, and there the ambiguous cells are counted and reported.
    """
    if na_token is not None:
        # In an R write.csv reference a missing key is the token, not an empty
        # field; left as text it defeats the numeric-key normalisation below
        # and every single row looks unpaired.
        reference = reference.with_columns(
            pl.when(pl.col(k) == na_token).then(None).otherwise(pl.col(k)).alias(k) for k in key
        )
    a, r, only_ref, only_act, null_ref, null_act = _align(actual, reference, key)
    kcols = [f"__k{i}" for i in range(len(key))]
    keyvals = a.select(kcols).to_dicts()

    shared_cols = [c for c in actual.columns if c in reference.columns and c not in key]
    diffs: list[ColumnDiff] = []

    for c in shared_cols:
        dtype = actual.schema[c]
        ref_raw = r.get_column(c)
        act_raw = a.get_column(c)
        n = a.height
        ambiguous = 0
        max_abs = None

        # A Date has no meaningful numeric comparison against "2025-07-30";
        # render it the way the reference spells it and compare as text. When
        # the reference is a parquet the date is a Date on both sides, so both
        # get rendered.
        if dtype in (pl.Date, pl.Datetime):
            act_raw = act_raw.dt.to_string("%Y-%m-%d")
            if ref_raw.dtype in (pl.Date, pl.Datetime):
                ref_raw = ref_raw.dt.to_string("%Y-%m-%d")
            dtype = pl.String

        if dtype == pl.String:
            if c in na_collapsed_columns:
                act_raw = pl.select(
                    pl.when(pl.lit(act_raw) == "NA").then(None).otherwise(pl.lit(act_raw))
                ).to_series()
            if na_token is None:
                ref_v, act_v = ref_raw, act_raw
                na_only_ref = int((ref_v.is_null() & act_v.is_not_null()).sum())
                na_only_act = int((act_v.is_null() & ref_v.is_not_null()).sum())
                is_diff = ref_v.ne_missing(act_v)
            else:
                # R's write.csv: null and the literal string are the same text.
                ref_v = ref_raw.fill_null(na_token)
                act_v = act_raw.fill_null(na_token)
                ambiguous = int((ref_v == na_token).sum())
                is_diff = ref_v.ne_missing(act_v)
                na_only_ref = 0
                na_only_act = 0
        elif dtype == pl.Boolean:
            # An exported reference spells a boolean as text; a parquet one is
            # already Boolean and must not go through the token table.
            ref_v = (
                ref_raw
                if ref_raw.dtype == pl.Boolean
                else ref_raw.replace_strict(_BOOL_TOKENS, default=None, return_dtype=pl.Boolean)
            )
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
            scale = pl.DataFrame({"r": ref_v.abs(), "a": act_v.abs()}).select(
                pl.max_horizontal("r", "a")
            )[:, 0]
            over_tol = (delta > (scale * rtol)).fill_null(False)
            is_diff = over_tol | (ref_v.is_null() != act_v.is_null())
            if delta.drop_nulls().len():
                max_abs = float(delta.max())

        n_diff = int(is_diff.sum())
        examples: list[dict] = []
        if n_diff:
            idx = [i for i, flag in enumerate(is_diff.to_list()) if flag][:n_examples]
            for i in idx:
                ex = {k: keyvals[i][f"__k{j}"] for j, k in enumerate(key)}
                # Both sides as text: the reference is always String and the
                # actual is not, and a report that mixes the two is unreadable.
                ex["reference"] = None if ref_raw[i] is None else str(ref_raw[i])
                ex["actual"] = None if act_raw[i] is None else str(act_raw[i])
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
        keys_null_ref=null_ref,
        keys_null_act=null_act,
        cols_only_ref=[c for c in reference.columns if c not in actual.columns],
        cols_only_act=[c for c in actual.columns if c not in reference.columns],
        columns=diffs,
        expected_missing=sorted(expected_missing),
        expected_extra=sorted(expected_extra),
    )


@dataclass(frozen=True)
class Checkpoint:
    """One stage output and the exported file it must reproduce."""

    name: str
    stage: int
    interim: str
    reference: str
    key: list[str]
    expect_rows: int
    #: True when the reference lives under the repo root rather than ref_dir.
    reference_at_root: bool = False
    #: How the file spells a missing value; see `verify`.
    na_token: str | None = None
    expected_missing: frozenset[str] = frozenset()
    expected_extra: frozenset[str] = frozenset()
    expected_diff: frozenset[str] = frozenset()
    na_collapsed_columns: frozenset[str] = frozenset({"Institute"})


#: The six columns fed by the RandomForest imputation of TotalInvestedCapital.
#: They exist in every reference from db_master_2 onwards and are deliberately
#: not produced here — see the amendment in the design spec.
EST_COLUMNS: frozenset[str] = frozenset(
    {
        "TotalRaised_Est",
        "TotalRaised_Est_NA",
        "TotalRaised_Est_any",
        "TotalRaised_Est_cum",
        "TotalRaised_Est_any_cum",
        "TotalRaised_Est_NA_cum",
    }
)

#: The six competitor columns of `data/raw/panel.csv.gz`. They cannot be
#: reproduced from the extraction in `data/raw/pitchbook/`, and the cause is
#: known: that panel's competitor columns were computed from a different
#: download of CompanySimilarRelation.csv.
#:
#: The evidence is not circumstantial. Checkpoint B reproduces
#: SimilarityScoreMean, SimilarityScoreMax, N_Competitors, Same_Country,
#: N_Europe and N_Outside_Europe in db_master_1.csv exactly from the extraction
#: we have, so that file is the one the R ran on. Yet 655,869 rows over 84,138
#: companies carry the *same* competitor count as the reference panel and a
#: different mean similarity, in both directions — same companies, different
#: scores. On company 100063-00 the reference mean of 97.315 is not the mean of
#: any subset of that company's ten similarity scores in our file.
#:
#: Similarity scores are model output and PitchBook recomputes them between
#: downloads. Nothing here can be fixed by changing the code; matching those
#: numbers would mean fitting to a dataset we do not have.
COMPETITOR_VINTAGE_COLUMNS: frozenset[str] = frozenset(
    {
        "N_Competitors",
        "Same_Country",
        "SimilarityScoreMean",
        "N_Competitors_All",
        "Same_Country_All",
        "SimilarityScoreMean_All",
    }
)

#: Computed by the R at 1_Arrange_DB.R:1246 and then dropped by vars_selected.
#: We keep it, so from db_selected onwards it is a column the references lack.
KEPT_EXTRA: frozenset[str] = frozenset({"TR_D"})

CHECKPOINTS: dict[str, Checkpoint] = {
    "A": Checkpoint("A", 2, "db3.parquet", "db3.csv", ["CompanyID", "PersonID"], 534_851),
    "B": Checkpoint("B", 3, "db_master_1.parquet", "db_master_1.csv", ["CompanyID"], 116_920),
    "C": Checkpoint(
        "C",
        5,
        "db_master_2.parquet",
        "db_master_2.csv",
        ["CompanyID", "Year_Delta"],
        1_001_625,
        expected_missing=EST_COLUMNS,
    ),
    "D": Checkpoint(
        "D",
        5,
        "db_selected.parquet",
        "db_selected.csv",
        ["CompanyID", "Year_Delta"],
        1_001_625,
        expected_missing=EST_COLUMNS,
        expected_extra=KEPT_EXTRA,
    ),
    "E": Checkpoint(
        "E",
        6,
        "db_master_panel.parquet",
        "db_master_panel.csv.gz",
        ["CompanyID", "Year_Delta"],
        882_324,
        na_token="NA",  # the only reference written by R's write.csv
        expected_missing=EST_COLUMNS,
        expected_extra=KEPT_EXTRA,
        expected_diff=frozenset({"StageBlock"}),
    ),
    "F": Checkpoint(
        "F",
        7,
        "panel.parquet",
        "data/raw/panel.csv.gz",
        ["CompanyID", "Year_Delta"],
        882_324,
        reference_at_root=True,
        expected_missing=EST_COLUMNS,
        expected_extra=KEPT_EXTRA,
        expected_diff=frozenset({"StageBlock"}) | COMPETITOR_VINTAGE_COLUMNS,
    ),
}

#: Stage after which each db_master_2 column stops changing. Each stage task
#: appends its own columns; see the plan's stage tasks.
COLUMN_FINALISED_AT_STAGE: dict[str, int] = {}


def load_reference(cfg: PanelConfig, filename: str, *, at_root: bool = False) -> pl.DataFrame:
    """Read an exported reference CSV as pure text.

    `null_values=[]` keeps the literal token `NA` visible, which matters only
    for `db_master_panel.csv.gz`; empty fields still read as nulls, which is
    how every other reference spells a missing value.
    """
    path = Path(filename) if at_root else cfg.reference(filename)
    return pl.read_csv(path, infer_schema_length=0, null_values=[], quote_char='"')


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
    reference = load_reference(cfg, cp.reference, at_root=cp.reference_at_root)
    report = verify(
        actual,
        reference,
        key=cp.key,
        name=f"checkpoint {letter} ({cp.reference})",
        expected_diff=cp.expected_diff,
        expected_missing=cp.expected_missing,
        expected_extra=cp.expected_extra,
        na_token=cp.na_token,
        na_collapsed_columns=cp.na_collapsed_columns,
        rtol=cfg.rtol,
        n_examples=cfg.n_examples,
    )
    report.to_json(cfg.interim_dir / "reports" / f"checkpoint_{letter}.json")
    return report


def run_partial(cfg: PanelConfig, stage: int, actual: pl.DataFrame) -> VerificationReport:
    """Compare an intermediate db_master_2 against the final reference."""
    reference = load_reference(cfg, "db_master_2.csv")
    report = verify(
        actual,
        reference,
        key=["CompanyID", "Year_Delta"],
        name=f"verifica parziale stadio {stage}",
        expected_missing=EST_COLUMNS,
        na_collapsed_columns=frozenset({"Institute"}),
        rtol=cfg.rtol,
        n_examples=cfg.n_examples,
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


#: Natural key of each interim parquet, for the baseline comparison.
BASELINE_KEYS: dict[str, list[str]] = {
    "db1": ["CompanyID"],
    "db_master_1_v1": ["CompanyID"],
    "db_master_1": ["CompanyID"],
    "db_master_2_skeleton": ["CompanyID", "Year_Delta"],
    "db3": ["CompanyID", "PersonID"],
    "db_master_2_team": ["CompanyID", "Year_Delta"],
    "db_master_2_relations": ["CompanyID", "Year_Delta"],
    "deals_panel": ["CompanyID", "Year_Delta"],
    "db_master_2_deals": ["CompanyID", "Year_Delta"],
    "db_master_2": ["CompanyID", "Year_Delta"],
    "db_selected": ["CompanyID", "Year_Delta"],
    "db_final": ["CompanyID", "Year_Delta"],
    "db_master_panel": ["CompanyID", "Year_Delta"],
    "panel": ["CompanyID", "Year_Delta"],
}

#: Where the pre-migration outputs are frozen. Not data/interim, which gets
#: wiped by a from-scratch run.
BASELINE_DIR = Path("data/baseline")


def run_baseline(cfg: PanelConfig, name: str) -> VerificationReport:
    """Compare a stage's current output against the frozen pre-migration one.

    Stronger than the checkpoints, and complementary to them: it covers
    **every** column, including the ones a checkpoint declares and therefore
    never compares — the six `_Est`, `TR_D`, `StageBlock` and the six
    competitor columns. Both sides are typed parquet, so nothing is compared
    as text and nothing is excused.
    """
    actual = pl.read_parquet(cfg.interim(f"{name}.parquet"))
    baseline = pl.read_parquet(BASELINE_DIR / f"{name}.parquet")
    report = verify(
        actual,
        baseline,
        key=BASELINE_KEYS[name],
        name=f"{name} contro la base congelata",
        rtol=cfg.rtol,
        n_examples=cfg.n_examples,
    )
    report.to_json(cfg.interim_dir / "reports" / f"baseline_{name}.json")
    return report


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Verifica un checkpoint del panel.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", choices=sorted(CHECKPOINTS))
    group.add_argument("--baseline", choices=sorted(BASELINE_KEYS))
    args = parser.parse_args()
    cfg = PanelConfig()
    report = (
        run_checkpoint(cfg, args.checkpoint)
        if args.checkpoint
        else run_baseline(cfg, args.baseline)
    )
    print(report.render())
    return 0 if report.passed() else 1


if __name__ == "__main__":
    raise SystemExit(_main())
