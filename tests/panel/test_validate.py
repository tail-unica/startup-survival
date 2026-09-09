import polars as pl
import pytest

from src.panel.validate import verify


def _ref(**cols) -> pl.DataFrame:
    """Reference frames arrive all-String, exactly as read from R's write.csv."""
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
    # A null on our side and the token NA in the reference must match, because
    # write.csv cannot distinguish them.
    act = pl.DataFrame({"k": ["a", "b"], "s": [None, "MIT"]})
    ref = _ref(k=["a", "b"], s=["NA", "MIT"])
    rep = verify(act, ref, key=["k"], name="t")
    assert rep.passed()
    assert rep.column("s").n_ambiguous_na == 1


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
