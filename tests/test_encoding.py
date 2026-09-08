import numpy as np
import pandas as pd
import pytest

from src.encoding import apply_frequency_encoding, fit_frequency_encoding

OTHERS = "Others"


def _series(counts):
    """Builds a categorical Series from a {category: n_rows} mapping."""
    values = []
    for label, n in counts.items():
        values.extend([label] * n)
    return pd.Series(values, name="HQCountry")


# --- fit_frequency_encoding -------------------------------------------------


def test_frequent_category_maps_to_its_share_of_the_training_rows():
    train = _series({"USA": 60, "GBR": 40})

    enc = fit_frequency_encoding(train, min_frequency=0.05)

    assert enc["USA"] == pytest.approx(0.60)
    assert enc["GBR"] == pytest.approx(0.40)


def test_categories_below_the_threshold_are_pooled_into_others():
    train = _series({"USA": 90, "ITA": 4, "ESP": 3, "PRT": 3})

    enc = fit_frequency_encoding(train, min_frequency=0.05)

    assert "ITA" not in enc and "ESP" not in enc
    assert enc[OTHERS] == pytest.approx(0.10)


def test_a_category_exactly_at_the_threshold_is_kept():
    """The rule is 'below the threshold collapses', so 5% survives at 0.05."""
    train = _series({"USA": 95, "ITA": 5})

    enc = fit_frequency_encoding(train, min_frequency=0.05)

    assert enc["ITA"] == pytest.approx(0.05)
    assert enc[OTHERS] == pytest.approx(0.0)


def test_the_threshold_is_configurable():
    train = _series({"USA": 90, "ITA": 6, "ESP": 4})

    lenient = fit_frequency_encoding(train, min_frequency=0.05)
    strict = fit_frequency_encoding(train, min_frequency=0.20)

    assert "ITA" in lenient
    assert "ITA" not in strict
    assert strict[OTHERS] == pytest.approx(0.10)


def test_shares_sum_to_one():
    train = _series({"USA": 50, "GBR": 30, "ITA": 12, "ESP": 5, "PRT": 3})

    enc = fit_frequency_encoding(train, min_frequency=0.05)

    assert sum(enc.values()) == pytest.approx(1.0)


def test_missing_training_values_land_in_others():
    """Nulls are part of the sample, so they count towards the Others share."""
    train = pd.Series(["USA"] * 90 + [None] * 10)

    enc = fit_frequency_encoding(train, min_frequency=0.05)

    assert enc["USA"] == pytest.approx(0.90)
    assert enc[OTHERS] == pytest.approx(0.10)


def test_others_label_is_configurable():
    train = _series({"USA": 96, "ITA": 4})

    enc = fit_frequency_encoding(train, min_frequency=0.05, other_label="RARE")

    assert enc["RARE"] == pytest.approx(0.04)
    assert OTHERS not in enc


# --- apply_frequency_encoding -----------------------------------------------


def test_apply_replaces_each_category_with_its_training_share():
    train = _series({"USA": 60, "GBR": 40})
    enc = fit_frequency_encoding(train, min_frequency=0.05)

    out = apply_frequency_encoding(pd.Series(["GBR", "USA", "GBR"]), enc)

    np.testing.assert_allclose(out.to_numpy(), [0.40, 0.60, 0.40])


def test_apply_uses_training_shares_not_the_frequencies_of_the_set_it_transforms():
    """This is the leakage the change exists to remove: a test row must never
    be encoded with a frequency computed on the test set."""
    train = _series({"USA": 90, "GBR": 10})
    enc = fit_frequency_encoding(train, min_frequency=0.05)

    # in this test set the proportions are inverted
    test = _series({"USA": 10, "GBR": 90})
    out = apply_frequency_encoding(test, enc)

    assert set(np.round(out.to_numpy(), 6)) == {0.9, 0.1}
    assert out[out == 0.9].size == 10  # the 10 USA rows, not the 90 GBR ones


def test_a_category_never_seen_in_training_becomes_others():
    train = _series({"USA": 90, "ITA": 4, "ESP": 3, "PRT": 3})
    enc = fit_frequency_encoding(train, min_frequency=0.05)

    out = apply_frequency_encoding(pd.Series(["JPN"]), enc)

    assert out.iloc[0] == pytest.approx(enc[OTHERS])


def test_a_rare_training_category_is_encoded_as_others():
    train = _series({"USA": 90, "ITA": 4, "ESP": 3, "PRT": 3})
    enc = fit_frequency_encoding(train, min_frequency=0.05)

    out = apply_frequency_encoding(pd.Series(["ITA", "ESP"]), enc)

    np.testing.assert_allclose(out.to_numpy(), [0.10, 0.10])


def test_missing_values_become_others():
    train = _series({"USA": 90, "ITA": 4, "ESP": 3, "PRT": 3})
    enc = fit_frequency_encoding(train, min_frequency=0.05)

    out = apply_frequency_encoding(pd.Series([None, np.nan]), enc)

    np.testing.assert_allclose(out.to_numpy(), [0.10, 0.10])


def test_unseen_category_is_zero_when_nothing_was_collapsed():
    """With an empty Others bucket the encoder must not emit NaN: a category
    absent from training has, by construction, a training share of zero."""
    train = _series({"USA": 50, "GBR": 50})
    enc = fit_frequency_encoding(train, min_frequency=0.05)

    out = apply_frequency_encoding(pd.Series(["JPN", None]), enc)

    np.testing.assert_allclose(out.to_numpy(), [0.0, 0.0])


def test_apply_preserves_the_index():
    train = _series({"USA": 60, "GBR": 40})
    enc = fit_frequency_encoding(train, min_frequency=0.05)
    s = pd.Series(["USA", "GBR"], index=[17, 3])

    out = apply_frequency_encoding(s, enc)

    assert list(out.index) == [17, 3]


def test_apply_returns_floats_even_for_an_all_missing_column():
    train = _series({"USA": 96, "ITA": 4})
    enc = fit_frequency_encoding(train, min_frequency=0.05)

    out = apply_frequency_encoding(pd.Series([None, None]), enc)

    assert out.dtype == float
