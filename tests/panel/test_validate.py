import polars as pl
import pytest

from src.panel.validate import verify


def _ref(**cols) -> pl.DataFrame:
    """Reference frames arrive all-String, as read from the exported CSVs."""
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
    # Only db_master_panel.csv.gz is written by R's write.csv: there a null and
    # the literal string "NA" are the same six bytes and must both match a null
    # on our side. Opt in with na_token.
    act = pl.DataFrame({"k": ["a", "b"], "s": [None, "MIT"]})
    ref = _ref(k=["a", "b"], s=["NA", "MIT"])
    rep = verify(act, ref, key=["k"], name="t", na_token="NA")
    assert rep.passed()
    assert rep.column("s").n_ambiguous_na == 1


def test_string_nulls_match_real_nulls_by_default():
    # Every other reference has empty fields, which polars reads as nulls.
    act = pl.DataFrame({"k": ["a", "b"], "s": [None, "MIT"]})
    assert verify(act, _ref(k=["a", "b"], s=[None, "MIT"]), key=["k"], name="t").passed()
    rep = verify(act, _ref(k=["a", "b"], s=["MIT", "MIT"]), key=["k"], name="t")
    assert rep.column("s").n_na_only_act == 1
    assert not rep.passed()


def test_float_formatted_keys_align_with_integer_keys():
    # db_selected.csv writes Year_Delta as "2013.0"; ours is the integer 2013.
    act = pl.DataFrame({"CompanyID": ["c"], "Year_Delta": [2013], "v": [1.0]})
    ref = _ref(CompanyID=["c"], Year_Delta=["2013.0"], v=["1.0"])
    rep = verify(act, ref, key=["CompanyID", "Year_Delta"], name="t")
    assert rep.keys_only_ref == 0
    assert rep.passed()


def test_null_keys_are_excluded_and_counted_not_treated_as_a_difference():
    act = pl.DataFrame({"k": ["a", None], "v": [1.0, 2.0]})
    ref = _ref(k=["a", None], v=["1.0", "2.0"])
    rep = verify(act, ref, key=["k"], name="t")
    assert rep.keys_null_ref == 1
    assert rep.keys_null_act == 1
    assert rep.column("v").n_compared == 1
    assert rep.passed()


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
    act = pl.DataFrame({"k": ["a"], "TotalRaised": [9.0]})
    ref = _ref(k=["a"], TotalRaised=["1.0"])
    rep = verify(act, ref, key=["k"], name="t", expected_diff={"TotalRaised"})
    assert rep.passed()
    assert rep.column("TotalRaised").expected is True
    assert rep.column("TotalRaised").n_diff == 1


def test_expected_missing_columns_do_not_fail_the_report():
    # The six _Est columns exist in the R reference and are not produced here.
    act = pl.DataFrame({"k": ["a"], "v": [1.0]})
    ref = _ref(k=["a"], v=["1.0"], TotalRaised_Est=["3.0"])
    rep = verify(act, ref, key=["k"], name="t", expected_missing={"TotalRaised_Est"})
    assert rep.passed()
    assert rep.cols_only_ref == ["TotalRaised_Est"]
    assert verify(act, ref, key=["k"], name="t").passed() is False


def test_expected_extra_columns_do_not_fail_the_report():
    # TR_D is kept by us and absent from db_selected.csv onwards.
    act = pl.DataFrame({"k": ["a"], "v": [1.0], "TR_D": [1]})
    ref = _ref(k=["a"], v=["1.0"])
    rep = verify(act, ref, key=["k"], name="t", expected_extra={"TR_D"})
    assert rep.passed()
    assert rep.cols_only_act == ["TR_D"]
    assert verify(act, ref, key=["k"], name="t").passed() is False


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


def test_date_columns_are_compared_as_iso_text():
    import datetime as dt

    act = pl.DataFrame({"k": ["a", "b"], "d": [dt.date(2025, 7, 30), None]})
    ref = _ref(k=["a", "b"], d=["2025-07-30", None])
    assert verify(act, ref, key=["k"], name="t").passed()
    bad = _ref(k=["a", "b"], d=["2025-07-31", None])
    assert not verify(act, bad, key=["k"], name="t").passed()


def test_na_collapsed_columns_excuse_only_a_whole_cell_na():
    # R's paste() writes a missing institute as the text "NA"; the reference
    # export collapsed a cell that was only that into a null, but kept it
    # inside longer strings.
    act = pl.DataFrame({"k": ["a", "b", "c"], "Institute": ["NA", "NA; MIT", "MIT"]})
    ref = _ref(k=["a", "b", "c"], Institute=[None, "NA; MIT", "MIT"])
    assert verify(act, ref, key=["k"], name="t", na_collapsed_columns={"Institute"}).passed()
    wrong = _ref(k=["a", "b", "c"], Institute=[None, None, "MIT"])
    assert not verify(act, wrong, key=["k"], name="t", na_collapsed_columns={"Institute"}).passed()


def test_na_token_is_stripped_from_keys_before_aligning():
    # An R write.csv reference spells a missing key as "NA"; left as text it
    # makes the key column non-numeric and nothing pairs up at all.
    act = pl.DataFrame({"k": ["c", "c"], "Year_Delta": [2013, None], "v": [1.0, 2.0]})
    ref = _ref(k=["c", "c"], Year_Delta=["2013", "NA"], v=["1.0", "2.0"])
    rep = verify(act, ref, key=["k", "Year_Delta"], name="t", na_token="NA")
    assert rep.keys_only_ref == 0
    assert rep.keys_null_ref == 1
    assert rep.keys_null_act == 1
    assert rep.passed()
