import datetime as dt

import polars as pl

from src.panel.expressions import MISSING_TOKENS, MISSING_TOKENS_WITH_INF, nullify, parse_date


def test_nullify_nulls_every_missing_token():
    df = pl.DataFrame({"a": ["x", "", "NA", "N/A", "NULL", "NaN"]})
    assert nullify(df, MISSING_TOKENS)["a"].to_list() == ["x", None, None, None, None, None]


def test_nullify_leaves_a_value_that_is_not_a_token():
    # "N/D" is not in the set: a token list is a decision, not a heuristic.
    df = pl.DataFrame({"a": ["N/D", "-", "0"]})
    assert nullify(df, MISSING_TOKENS_WITH_INF)["a"].to_list() == ["N/D", "-", "0"]


def test_nullify_matches_the_textual_form_of_a_float():
    df = pl.DataFrame({"a": [1.0, float("nan"), float("inf"), float("-inf"), None]})
    # A float is matched against its textual form, so "NaN"/"Inf"/"-Inf" match.
    expected = [1.0, None, float("inf"), float("-inf"), None]
    assert nullify(df, MISSING_TOKENS)["a"].to_list() == expected
    out = nullify(df, MISSING_TOKENS_WITH_INF)["a"].to_list()
    assert out == [1.0, None, None, None, None]


def test_nullify_leaves_integers_untouched():
    df = pl.DataFrame({"a": [0, 1, None]})
    assert nullify(df, MISSING_TOKENS_WITH_INF)["a"].to_list() == [0, 1, None]


def test_parse_date_ten_character_format():
    df = pl.DataFrame({"d": ["05/13/2020", "12/01/1999"]})
    out = df.select(parse_date(pl.col("d")))["d"].to_list()
    assert out == [dt.date(2020, 5, 13), dt.date(1999, 12, 1)]


def test_parse_date_eight_character_two_digit_year_uses_cutoff_24():
    df = pl.DataFrame({"d": ["05/13/20", "05/13/24", "05/13/25", "05/13/99"]})
    out = df.select(parse_date(pl.col("d")))["d"].to_list()
    assert out == [
        dt.date(2020, 5, 13),
        dt.date(2024, 5, 13),
        dt.date(1925, 5, 13),
        dt.date(1999, 5, 13),
    ]


def test_parse_date_eight_character_four_digit_year():
    df = pl.DataFrame({"d": ["1/5/2024"]})
    assert df.select(parse_date(pl.col("d")))["d"].to_list() == [dt.date(2024, 1, 5)]


def test_parse_date_rejects_other_lengths():
    df = pl.DataFrame({"d": ["1/5/24", "2020-05-13", "", None]})
    assert df.select(parse_date(pl.col("d")))["d"].to_list() == [None, None, None, None]
