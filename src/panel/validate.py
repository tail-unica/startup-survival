"""Compare a stage's output against the R reference, column by column.

The report answers exactly two questions: which columns differ, and on how
many rows. Everything else is supporting detail.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import polars as pl

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
            f"chiavi  solo R={self.keys_only_ref:,}  solo PY={self.keys_only_act:,}",
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


def _align(actual: pl.DataFrame, reference: pl.DataFrame, key: list[str]):
    """Restrict both frames to the shared keys and sort them identically."""
    for name, df in (("actual", actual), ("reference", reference)):
        if df.select(key).is_duplicated().any():
            raise ValueError(f"duplicate keys in {name} on {key}")
    kcols = [f"__k{i}" for i in range(len(key))]
    a = actual.with_columns(
        pl.col(k).cast(pl.String).alias(kc) for k, kc in zip(key, kcols, strict=True)
    )
    r = reference.with_columns(
        pl.col(k).cast(pl.String).alias(kc) for k, kc in zip(key, kcols, strict=True)
    )
    only_ref = r.select(kcols).join(a.select(kcols), on=kcols, how="anti").height
    only_act = a.select(kcols).join(r.select(kcols), on=kcols, how="anti").height
    shared = r.select(kcols).join(a.select(kcols), on=kcols, how="semi")
    a = a.join(shared, on=kcols, how="semi").sort(kcols)
    r = r.join(shared, on=kcols, how="semi").sort(kcols)
    return a, r, only_ref, only_act


def verify(
    actual: pl.DataFrame,
    reference: pl.DataFrame,
    key: list[str],
    name: str,
    *,
    expected_diff: frozenset[str] | set[str] = frozenset(),
    expected_missing: frozenset[str] | set[str] = frozenset(),
    expected_extra: frozenset[str] | set[str] = frozenset(),
    rtol: float = 1e-9,
    n_examples: int = 10,
) -> VerificationReport:
    """Compare `actual` against the R `reference` (all-String) on `key`.

    `expected_diff` names columns allowed to differ, `expected_missing` columns
    the reference has and we deliberately do not produce, `expected_extra`
    columns we add and the reference cannot have. None of the three makes the
    report pass silently: they are printed in the header either way.
    """
    a, r, only_ref, only_act = _align(actual, reference, key)
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

        if dtype == pl.String:
            # Compare in R-serialized form: null <-> the token "NA".
            ref_v = ref_raw.fill_null("NA")
            act_v = act_raw.fill_null("NA")
            ambiguous = int((ref_v == "NA").sum())
            is_diff = ref_v != act_v
            na_only_ref = 0
            na_only_act = 0
        elif dtype == pl.Boolean:
            ref_v = ref_raw.replace_strict(_BOOL_TOKENS, default=None, return_dtype=pl.Boolean)
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
        cols_only_ref=[c for c in reference.columns if c not in actual.columns],
        cols_only_act=[c for c in actual.columns if c not in reference.columns],
        columns=diffs,
        expected_missing=sorted(expected_missing),
        expected_extra=sorted(expected_extra),
    )
