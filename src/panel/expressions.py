"""Expressions with the missing-value semantics the panel needs.

polars is deliberate about missing values, and so is this pipeline, but they do
not always agree: here a null in a cumulative sum has to invalidate the rest of
the group, a null condition has to produce a null result rather than silently
taking the else branch, and a sequence has to stay inclusive. Each function
states which behaviour it guarantees; the difference matters, and it is silent.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

#: Tokens that mean "missing" in the raw CSVs. Applied on read, before anything
#: else: without this pass a textual placeholder becomes a category of its own.
MISSING_TOKENS: tuple[str, ...] = ("", "NA", "N/A", "NULL", "NaN")
#: The same tokens plus the infinities, applied *after* aggregation so that the
#: ``-Inf`` of ``max()`` and the ``NaN`` of ``mean()`` over an all-missing group
#: become null. They are artefacts of the computation, not values that were read.
MISSING_TOKENS_WITH_INF: tuple[str, ...] = MISSING_TOKENS + ("-Inf", "Inf")


def nullify(df: pl.DataFrame, tokens: Sequence[str]) -> pl.DataFrame:
    """Turn every occurrence of ``tokens`` into null, over every column.

    On String columns this is a membership test. On Float columns the numeric
    values are matched against their textual form, so ``NaN``, ``Inf`` and
    ``-Inf`` match the corresponding tokens. Integer and Boolean columns can
    never match a token and are left alone.

    :param df: Frame to clean.
    :param tokens: Token set: :data:`MISSING_TOKENS` on read, or
        :data:`MISSING_TOKENS_WITH_INF` after an aggregation.
    :return: The frame with those values replaced by null.
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


def parse_date(col: pl.Expr) -> pl.Expr:
    """Parse a date, choosing the format by the length of the string.

    Ten characters are ``%m/%d/%Y``; eight are month/day/year with a two-digit
    year resolved at a cutoff of 24 (00–24 in the 2000s, 25–99 in the 1900s);
    any other length is null, including a six-character date such as ``1/5/24``.
    On the current extraction every date has ten characters, so the second
    branch is latent rather than active.

    :param col: String expression holding the date.
    :return: Date expression, null where the string is unparseable.
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


def cumulative_any(col: pl.Expr) -> pl.Expr:
    """Cumulative "any": once true, true for the rest of the group.

    It is what turns an event ("the company closed a seed round this year") into
    a state ("the company has already closed a seed round"), and it makes the
    growth stages monotone. Callers apply ``.over(group)`` themselves.

    :param col: Boolean expression; nulls count as false.
    :return: Boolean expression, cumulative in row order.
    """
    return col.fill_null(False).cast(pl.Int8).cum_max().cast(pl.Boolean)


def weighted_cumulative(value_col: str, weight_col: str, group_cols: list[str]) -> pl.Expr:
    """Cumulative weighted mean, over the rows with a value and a positive weight.

    Computed as ``cumsum(value * weight) / cumsum(weight)``, which is
    algebraically the running weighted mean and costs one pass instead of one
    per row.

    :param value_col: Column being averaged.
    :param weight_col: Column of weights; a null or non-positive weight excludes
        the row from both sums.
    :param group_cols: Grouping columns, typically ``["CompanyID"]``.
    :return: Float expression, null until the first valid row of the group.
    """
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
    """Add the first later value different from the current one, and its distance.

    Both results are null where no such value exists, and where the current
    value is itself null. This is what gives the panel its target: the growth
    stage a company moves to next, and in how many years.

    The search is a self-join inside each group, so it is quadratic in the group
    size. It runs on a three-column projection and the results are attached back
    by row index: joining the whole frame would materialise every column once
    per candidate pair, which on the panel is tens of millions of wide rows.

    :param df: Frame sorted by group and time.
    :param value_col: Column whose next different value is wanted.
    :param group_cols: Grouping columns, typically ``["CompanyID"]``.
    :param stage_col: Name of the output column holding that value.
    :param time_col: Name of the output column holding the distance in rows.
    :return: The frame with the two columns added.
    """
    slim = df.select([*group_cols, value_col]).with_row_index("__i")
    later = slim.filter(pl.col(value_col).is_not_null()).select(
        [*group_cols, pl.col("__i").alias("__j"), pl.col(value_col).alias("__cand")]
    )
    joined = (
        slim.join(later, on=group_cols, how="left")
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
            pl.col("__cand").first().alias("__stage"),
            pl.col("__j").first().alias("__jj"),
        )
    )
    out = (
        slim.join(picked, on="__i", how="left")
        .with_columns((pl.col("__jj") - pl.col("__i")).cast(pl.Int64).alias("__time"))
        .sort("__i")
    )
    own_null = pl.col(value_col).is_null()
    out = out.with_columns(
        pl.when(own_null).then(None).otherwise(pl.col("__stage")).alias("__stage"),
        pl.when(own_null).then(None).otherwise(pl.col("__time")).alias("__time"),
    )
    return df.with_columns(
        out.get_column("__stage").alias(stage_col),
        out.get_column("__time").alias(time_col),
    )


def last_non_null(name: str) -> pl.Expr:
    """Take the last non-null value of a group, in row order.

    :param name: Column to read.
    :return: Aggregation expression.
    """
    return pl.col(name).drop_nulls().last()


def if_else_null(cond: pl.Expr, then, otherwise) -> pl.Expr:
    """An if/else whose result is null when the condition itself is null.

    ``pl.when`` treats a null predicate as false and takes the else branch,
    which hides the difference between "the condition is false" and "we cannot
    tell". Where a condition compares two columns and one of them is missing,
    the answer is unknown, and an unknown answer must not silently pick a
    branch: this helper returns null instead.

    A chain of pattern tests needs no such helper: there a null condition simply
    fails to match and the row falls through to the default.

    :param cond: Boolean expression, possibly null.
    :param then: Value or expression for a true condition.
    :param otherwise: Value or expression for a false condition.
    :return: The three-valued conditional.
    """
    return pl.when(cond.is_null()).then(None).when(cond).then(then).otherwise(otherwise)


def cum_sum_null(col: pl.Expr) -> pl.Expr:
    """Cumulative sum in which a null poisons the rest of the group.

    On ``1, null, 3`` this gives ``1, null, null``, while ``cum_sum`` leaves the
    null in place and keeps accumulating, giving ``1, null, 4``: a total that
    claims to be complete while one of its terms is unknown. Callers apply
    ``.over(group)`` themselves.

    :param col: Numeric expression.
    :return: Cumulative sum, null from the first null onward.
    """
    poisoned = col.is_null().cast(pl.Int8).cum_max().cast(pl.Boolean)
    return pl.when(poisoned).then(None).otherwise(col.fill_null(0).cum_sum())


def seq_inclusive(from_: pl.Expr, to: pl.Expr) -> pl.Expr:
    """An inclusive sequence that counts **backwards** when ``to < from_``.

    From 2010 to 2008 the sequence is ``2010, 2009, 2008``, not an empty one.
    The pipeline expands a company into one row per year with this, so a dirty
    date leaving the last year before the founding year would produce years
    before the company existed: callers cap the upper bound to prevent it.

    :param from_: First value, inclusive.
    :param to: Last value, inclusive.
    :return: List expression; the caller explodes it.
    """
    return (
        pl.when(to < from_)
        .then(pl.int_ranges(from_, to - 1, step=-1))
        .otherwise(pl.int_ranges(from_, to + 1))
    )


def first_match(rules: list[tuple[str, str]], col: pl.Expr, otherwise) -> pl.Expr:
    """Build a short-circuiting chain of case-insensitive pattern tests.

    The first rule that matches wins and the rest are never evaluated, so the
    order of the rules is part of the logic: ``MD`` appears in the doctorate
    rule and ``Master`` in the next one, so an ``MD`` never reaches
    ``Master's``. A null input matches nothing and falls through to
    ``otherwise``.

    Rules are matched case-insensitively.

    :param rules: Pairs of regex pattern and value to assign, in priority order.
    :param col: String expression to test.
    :param otherwise: Value or expression for an input matching no rule.
    :return: The conditional expression.
    """
    expr = pl.when(col.str.contains(f"(?i){rules[0][0]}").fill_null(False)).then(
        pl.lit(rules[0][1])
    )
    for pattern, value in rules[1:]:
        expr = expr.when(col.str.contains(f"(?i){pattern}").fill_null(False)).then(pl.lit(value))
    return expr.otherwise(otherwise)
