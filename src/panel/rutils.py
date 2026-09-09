"""Primitives that reproduce R/dplyr semantics in polars.

Each function here exists because a naive polars translation of the R code
would differ in a way that is silent and hard to spot. Read the docstrings
before changing anything.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
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
            exprs.append(pl.when(col.is_in(list(tok))).then(None).otherwise(col).alias(name))
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
                exprs.append(pl.when(bad.fill_null(False)).then(None).otherwise(col).alias(name))
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
    is_new = ((pl.col(value_col) != prev) | pl.col(value_col).is_null() | prev.is_null()).fill_null(
        True
    )
    return is_new.cast(pl.Int64).cum_sum().over(group_cols)


def rle_sequence(
    df: pl.DataFrame, value_col: str, group_cols: list[str], out_col: str
) -> pl.DataFrame:
    """``sequence(rle(x)$lengths)`` — position within the current run, 1-based."""
    return (
        df.with_columns(_run_id(value_col, group_cols).alias("__run"))
        .with_columns((pl.int_range(pl.len()).over([*group_cols, "__run"]) + 1).alias(out_col))
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
    is_first = pl.int_range(pl.len()).over(group_cols) == 0
    lagged = pl.when(is_first).then(first).otherwise(prev)
    if fix:
        changed = (
            (lagged != pl.col(value_col)) | (lagged.is_null() != pl.col(value_col).is_null())
        ).fill_null(False)
        return df.with_columns(changed.cast(pl.Int64).cum_sum().over(group_cols).alias(out_col))
    # R: a null comparison makes cumsum null from there to the end of the group.
    changed = lagged != pl.col(value_col)  # null when either side is null
    poisoned = changed.is_null().cast(pl.Int8).cum_max().over(group_cols).cast(pl.Boolean)
    running = changed.fill_null(False).cast(pl.Int64).cum_sum().over(group_cols)
    return df.with_columns(pl.when(poisoned).then(None).otherwise(running).alias(out_col))


def weighted_cumulative(value_col: str, weight_col: str, group_cols: list[str]) -> pl.Expr:
    """Cumulative weighted mean over rows where value and weight are present
    and the weight is positive. Algebraically identical to the R loop, O(n)."""
    valid = (
        pl.col(value_col).is_not_null()
        & pl.col(weight_col).is_not_null()
        & (pl.col(weight_col) > 0)
    )
    num = (
        (pl.when(valid).then(pl.col(value_col) * pl.col(weight_col)).otherwise(0.0))
        .cum_sum()
        .over(group_cols)
    )
    den = (pl.when(valid).then(pl.col(weight_col)).otherwise(0.0)).cum_sum().over(group_cols)
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
        ordered.select([*group_cols, "__i", value_col])
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
        pl.when(pl.col(value_col).is_null())
        .then(None)
        .otherwise(pl.col(stage_col))
        .alias(stage_col),
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


def r_if_else(cond: pl.Expr, then, otherwise) -> pl.Expr:
    """``dplyr::if_else``, which returns NA when the condition itself is NA.

    ``pl.when`` treats a null predicate as false and takes the else branch;
    R does not, and the difference is not cosmetic. At
    `1_Arrange_DB.R:452` the condition is
    ``is.na(StartDate) | year(StartDate) < YearFounded``: for a company with
    no YearFounded the comparison is NA, so R **erases** a perfectly good
    StartDate. That is 5,357 rows of db3.

    ``case_when`` needs no such helper: there an NA condition simply fails to
    match and the row falls through, which is what ``pl.when`` already does.
    """
    return pl.when(cond.is_null()).then(None).when(cond).then(then).otherwise(otherwise)
