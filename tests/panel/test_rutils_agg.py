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
    # Row 0's own stage is null: in R `which(future != NA)[1]` is NA, so both
    # outputs are null there. Checked against company 100026-46 in
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


def test_r_if_else_returns_null_when_the_condition_is_null():
    from src.panel.rutils import r_if_else

    df = pl.DataFrame({"c": [True, False, None], "a": [1, 1, 1], "b": [2, 2, 2]})
    out = df.select(r_if_else(pl.col("c"), pl.col("a"), pl.col("b")).alias("r"))
    assert out["r"].to_list() == [1, 2, None]
