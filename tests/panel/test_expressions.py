"""The behaviour the aggregation expressions guarantee.

Each test states one guarantee, on three or four hand-written rows: what a
missing value does, where a group ends, which rule wins.
"""

import polars as pl

from src.panel.expressions import (
    cum_sum_null,
    cumulative_any,
    first_match,
    if_else_null,
    last_non_null,
    next_different,
    seq_inclusive,
    weighted_cumulative,
)


def _stages() -> pl.DataFrame:
    """A company with a leading null stage, then two runs of two years."""
    return pl.DataFrame(
        {
            "CompanyID": ["c"] * 6,
            "Age": [0, 1, 2, 3, 4, 5],
            "GrowthStage": [None, "Seed", "EarlyVC", "EarlyVC", "LaterVC", "LaterVC"],
        }
    )


def test_cumulative_any_is_monotone_within_group():
    df = pl.DataFrame({"g": ["a"] * 4, "v": [False, True, False, False]})
    out = df.with_columns(cumulative_any(pl.col("v")).over("g").alias("c"))
    assert out["c"].to_list() == [False, True, True, True]


def test_cumulative_any_restarts_on_the_next_group():
    df = pl.DataFrame({"g": ["a", "a", "b", "b"], "v": [True, False, False, False]})
    out = df.with_columns(cumulative_any(pl.col("v")).over("g").alias("c"))
    assert out["c"].to_list() == [True, True, False, False]


def test_cumulative_any_reads_a_null_as_false():
    df = pl.DataFrame({"g": ["a"] * 3, "v": [None, None, True]})
    out = df.with_columns(cumulative_any(pl.col("v")).over("g").alias("c"))
    assert out["c"].to_list() == [False, False, True]


def test_next_different_skips_nulls_and_reports_the_distance():
    out = next_different(_stages(), "GrowthStage", ["CompanyID"], "Next", "Time")
    assert out["Next"].to_list() == [None, "EarlyVC", "LaterVC", "LaterVC", None, None]
    assert out["Time"].to_list() == [None, 1, 2, 1, None, None]


def test_next_different_is_null_where_the_current_value_is_null():
    # A row without a stage has no "next different" stage either: the target of
    # the panel is undefined there, and must not borrow the following row's.
    out = next_different(_stages(), "GrowthStage", ["CompanyID"], "Next", "Time")
    assert out["Next"][0] is None and out["Time"][0] is None


def test_next_different_never_crosses_into_another_company():
    df = pl.DataFrame(
        {
            "CompanyID": ["a", "a", "b", "b"],
            "GrowthStage": ["Early", "Early", "Later", "Exit"],
        }
    )
    out = next_different(df, "GrowthStage", ["CompanyID"], "Next", "Time")
    # Company a never changes stage, so its rows have no next stage, even though
    # the very next row of the frame belongs to b and carries a different one.
    assert out["Next"].to_list() == [None, None, "Exit", None]


def test_weighted_cumulative_averages_only_valid_rows():
    df = pl.DataFrame(
        {
            "g": ["a"] * 4,
            "x": [10.0, None, 20.0, 40.0],
            "w": [0.0, 5.0, 1.0, 3.0],
        }
    )
    out = df.with_columns(weighted_cumulative("x", "w", ["g"]).alias("c"))
    # row 0: zero weight, no valid entry yet -> null
    # row 1: no value, still nothing -> null
    # row 2: (20*1)/1 = 20
    # row 3: (20*1 + 40*3)/4 = 35
    assert out["c"].to_list() == [None, None, 20.0, 35.0]


def test_weighted_cumulative_equals_the_plain_mean_when_weights_are_equal():
    df = pl.DataFrame({"g": ["a"] * 3, "x": [1.0, 2.0, 6.0], "w": [2.0, 2.0, 2.0]})
    out = df.with_columns(weighted_cumulative("x", "w", ["g"]).alias("c"))
    assert out["c"].to_list() == [1.0, 1.5, 3.0]


def test_last_non_null_takes_it_in_row_order():
    df = pl.DataFrame({"g": ["a"] * 4, "v": [None, "first", "last", None]})
    out = df.group_by("g", maintain_order=True).agg(last_non_null("v").alias("tail"))
    assert out["tail"].to_list() == ["last"]


def test_if_else_null_returns_null_when_the_condition_is_null():
    df = pl.DataFrame({"c": [True, False, None], "a": [1, 1, 1], "b": [2, 2, 2]})
    out = df.select(if_else_null(pl.col("c"), pl.col("a"), pl.col("b")).alias("r"))
    # pl.when would take the else branch on the third row, turning "we cannot
    # tell" into "false".
    assert out["r"].to_list() == [1, 2, None]


def test_cum_sum_null_invalidates_the_rest_of_the_group():
    df = pl.DataFrame({"g": ["a"] * 4, "x": [1.0, None, 3.0, 4.0]})
    out = df.with_columns(cum_sum_null(pl.col("x")).over("g").alias("c"))
    assert out["c"].to_list() == [1.0, None, None, None]


def test_cum_sum_null_matches_cum_sum_when_nothing_is_null():
    df = pl.DataFrame({"g": ["a"] * 3, "x": [1.0, 2.0, 3.0]})
    out = df.with_columns(cum_sum_null(pl.col("x")).over("g").alias("c"))
    assert out["c"].to_list() == [1.0, 3.0, 6.0]


def test_seq_inclusive_counts_down_when_the_end_precedes_the_start():
    df = pl.DataFrame({"da": [2010, 2010, 2010], "a": [2013, 2010, 2008]})
    out = df.select(seq_inclusive(pl.col("da"), pl.col("a")).alias("s"))["s"].to_list()
    assert out == [
        [2010, 2011, 2012, 2013],  # ascending, both ends included
        [2010],  # a single year
        [2010, 2009, 2008],  # counts backwards: this is why callers cap the end
    ]


def test_first_match_short_circuits():
    # "MD" is in the first rule and "Master" in the second, so an "MD" never
    # reaches Master's. Swapping the rules changes the classification.
    rules = [("PhD|MD", "PhD/Doctorate"), ("MBA|Master", "Master's")]
    df = pl.DataFrame({"d": ["MD", "MBA", "Master of Science", None, "Geologia"]})
    out = df.select(first_match(rules, pl.col("d"), pl.lit("Other")).alias("v"))["v"].to_list()
    assert out == ["PhD/Doctorate", "Master's", "Master's", "Other", "Other"]


def test_first_match_is_case_insensitive():
    df = pl.DataFrame({"d": ["phd in physics", "mba"]})
    out = df.select(
        first_match(
            [("PhD", "PhD/Doctorate"), ("MBA", "Master's")], pl.col("d"), pl.lit("Other")
        ).alias("v")
    )["v"].to_list()
    assert out == ["PhD/Doctorate", "Master's"]
