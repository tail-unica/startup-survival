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
