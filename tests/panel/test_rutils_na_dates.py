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
    assert out == [
        dt.date(2020, 5, 13),
        dt.date(2024, 5, 13),
        dt.date(1925, 5, 13),
        dt.date(1999, 5, 13),
    ]


def test_parse_date_r_eight_character_four_digit_year():
    df = pl.DataFrame({"d": ["1/5/2024"]})
    assert df.select(parse_date_r(pl.col("d")))["d"].to_list() == [dt.date(2024, 1, 5)]


def test_parse_date_r_rejects_other_lengths():
    df = pl.DataFrame({"d": ["1/5/24", "2020-05-13", "", None]})
    assert df.select(parse_date_r(pl.col("d")))["d"].to_list() == [None, None, None, None]
