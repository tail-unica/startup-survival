import numpy as np
import pandas as pd
import pytest

from src.utils import (
    clear_split_cache,
    compare_metrics,
    get_split,
    prepare_splits,
    summarize_metrics,
)

# --- prepare_splits / get_split ---------------------------------------------


@pytest.fixture
def toy_frame():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(200, 5)), columns=list("abcde"))
    X.iloc[::17, 2] = np.nan  # a few holes for the imputer to fill
    y = pd.Series((rng.random(200) < 0.3).astype(int), name="Target")
    return X, y


def test_scaler_is_fitted_on_train_only(toy_frame):
    """The held-out sets must not influence the scaler: that is the leakage the paper is about."""
    X, y = toy_frame
    s = prepare_splits(X, y, seed=1, test_size=0.4)

    # RobustScaler centres on the training median, so the training set is the
    # only one guaranteed to come out with a ~zero median.
    np.testing.assert_allclose(
        np.median(s["X_train_scaled"], axis=0), np.zeros(X.shape[1]), atol=1e-9
    )


def test_get_split_keeps_tags_and_seeds_apart(toy_frame, tmp_path):
    X, y = toy_frame

    get_split(X, y, seed=1, test_size=0.4, cache_dir=tmp_path, tag="controlled")
    get_split(X, y, seed=2, test_size=0.4, cache_dir=tmp_path, tag="controlled")
    get_split(X, y, seed=1, test_size=0.4, cache_dir=tmp_path, tag="leakboth")

    assert len(list(tmp_path.glob("*.joblib"))) == 3


# --- compare_metrics: mean ± std across seeds -------------------------------


def _runs(auc_values, f1=0.5):
    return [
        {
            "AUC": a,
            "F1": f1,
            "accuracy": 0.7,
            "precision": 0.6,
            "recall": 0.6,
            "accuracy_train": 0.9,
            "seed": i,
        }
        for i, a in enumerate(auc_values, start=1)
    ]


def test_single_seed_gives_zero_std_instead_of_nan():
    """ddof=1 is undefined for one run; it must degrade to 0.0, not NaN."""
    store = {
        ("rf", "controlled"): _runs([0.80]),
        ("rf", "leakboth"): _runs([0.85]),
    }

    df = compare_metrics(store, "controlled", "leakboth")

    assert df.set_index("Model")["AUC"]["rf"] == "0.800 ± 0.000"


def test_compare_metrics_latex_output_uses_pm_notation():
    store = {
        ("rf", "controlled"): _runs([0.80, 0.82]),
        ("rf", "leakboth"): _runs([0.90, 0.90]),
    }

    df = compare_metrics(store, "controlled", "leakboth", latex=True)
    cell = df.set_index("Model")["AUC"]["rf"]

    assert "\\pm" in cell and cell.startswith("$")


def test_models_missing_from_one_experiment_are_skipped():
    store = {
        ("rf", "controlled"): _runs([0.80]),
        ("rf", "leakboth"): _runs([0.85]),
        ("svm", "controlled"): _runs([0.70]),  # no leakboth counterpart
    }

    df = compare_metrics(store, "controlled", "leakboth")

    assert list(df["Model"]) == ["rf", "rf w/ leakage"]


def test_std_is_the_sample_one_not_the_population_one():
    """Values chosen so ddof=0 and ddof=1 differ in the second decimal."""
    store = {
        ("rf", "controlled"): _runs([0.10, 0.30, 0.50, 0.70, 0.90]),
        ("rf", "leakboth"): _runs([0.50]),
    }

    df = compare_metrics(store, "controlled", "leakboth")
    cell = df.set_index("Model")["AUC"]["rf"]

    # population std = 0.2828; sample std = 0.3162
    assert cell == "0.500 ± 0.316", cell


def test_default_precision_resolves_a_std_two_decimals_would_round_away():
    """A std below 0.005 renders as '± 0.00' at two decimals, a bar that reads
    as zero variance rather than small variance. Three decimals keeps it."""
    store = {
        ("rf", "controlled"): _runs([0.780, 0.781, 0.782, 0.783, 0.784]),  # sample std 0.0016
        ("rf", "leakboth"): _runs([0.70]),
    }

    default = compare_metrics(store, "controlled", "leakboth")
    coarse = compare_metrics(store, "controlled", "leakboth", decimals=2)

    assert default.set_index("Model")["AUC"]["rf"] == "0.782 ± 0.002"
    assert coarse.set_index("Model")["AUC"]["rf"] == "0.78 ± 0.00"


def test_row_label_names_the_isolated_leak():
    """The 2x2 leakage tables reuse the layout with a different second row."""
    store = {
        ("rf", "controlled"): _runs([0.80]),
        ("rf", "leakfeat"): _runs([0.90]),
    }

    df = compare_metrics(store, "controlled", "leakfeat", label_b="leaked features")

    assert list(df["Model"]) == ["rf", "rf leaked features"]


# --- summarize_metrics: one experiment, mean ± std across seeds --------------


def test_seed_count_distinguishes_one_run_from_a_flat_metric():
    """'± 0.000' alone cannot say whether it is one run or five identical ones."""
    store = {
        ("rf", "controlled"): _runs([0.80]),
        ("lr", "controlled"): _runs([0.80, 0.80, 0.80]),
    }

    df = summarize_metrics(store, "controlled").set_index("Model")

    assert df["AUC"]["rf"] == df["AUC"]["lr"] == "0.800 ± 0.000"
    assert df["Seeds"]["rf"] == 1
    assert df["Seeds"]["lr"] == 3


def test_summarize_metrics_latex_output_uses_pm_notation():
    store = {("rf", "controlled"): _runs([0.80, 0.82, 0.84])}

    df = summarize_metrics(store, "controlled", latex=True)

    assert df.set_index("Model")["AUC"]["rf"] == "$0.820 \\pm 0.020$"


def test_unknown_tag_raises_instead_of_returning_an_empty_table():
    store = {("rf", "controlled"): _runs([0.80])}

    with pytest.raises(KeyError, match="noteam"):
        summarize_metrics(store, "noteam")


def test_two_targets_stratified_on_the_same_labels_share_the_split(toy_frame):
    """The settings compare the same firms: a different target must not move a row."""
    X, y = toy_frame
    other = 1 - y  # a target that disagrees everywhere
    a = prepare_splits(X, y, seed=1, test_size=0.4, stratify=y)
    b = prepare_splits(X, other, seed=1, test_size=0.4, stratify=y)
    for part in ("X_train", "X_val", "X_test"):
        assert a[part].index.equals(b[part].index)


def test_the_split_cache_tells_the_strata_apart(toy_frame, tmp_path):
    X, y = toy_frame
    get_split(X, y, seed=1, test_size=0.4, cache_dir=tmp_path, tag="leakboth")
    get_split(X, y, seed=1, test_size=0.4, cache_dir=tmp_path, tag="leakboth", stratify=1 - y)
    assert len(list(tmp_path.glob("*.joblib"))) == 2


# --- prepare_splits: frequency encoding fitted on the training split ---------


@pytest.fixture
def toy_categorical_frame():
    """A frame carrying a raw categorical column, as the processed CSV now does."""
    rng = np.random.default_rng(11)
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    countries = ["USA"] * 120 + ["GBR"] * 50 + ["ITA"] * 22 + ["JPN"] * 5 + ["ESP"] * 3
    rng.shuffle(countries)
    X["HQCountry"] = countries
    y = pd.Series((rng.random(n) < 0.3).astype(int))
    return X, y


def test_held_out_rows_are_encoded_only_with_training_values(toy_categorical_frame):
    """The leakage check: no value may appear in val/test that the training
    split did not produce."""
    X, y = toy_categorical_frame

    s = prepare_splits(
        X, y, seed=1, test_size=0.4, categorical_columns=["HQCountry"], min_frequency=0.05
    )

    training_values = set(np.round(list(s["encodings"]["HQCountry"].values()), 12))
    for key in ("X_val", "X_test"):
        assert set(np.round(s[key]["HQCountry"].to_numpy(), 12)) <= training_values


def test_encoding_happens_before_imputation(toy_categorical_frame):
    """A categorical column must never reach the KNN imputer as a string."""
    X, y = toy_categorical_frame
    X.loc[X.index[:10], "HQCountry"] = None

    s = prepare_splits(
        X, y, seed=1, test_size=0.4, categorical_columns=["HQCountry"], min_frequency=0.05
    )

    for key in ("X_train_imp", "X_val_imp", "X_test_imp"):
        assert np.isfinite(s[key]).all()


def test_columns_absent_from_the_frame_are_ignored(toy_categorical_frame):
    """noteam/nocompetitors drop features, so one config must serve them all."""
    X, y = toy_categorical_frame

    s = prepare_splits(
        X,
        y,
        seed=1,
        test_size=0.4,
        categorical_columns=["HQCountry", "NotHere"],
        min_frequency=0.05,
    )

    assert set(s["encodings"]) == {"HQCountry"}


# --- get_split: cache invalidation and regeneration --------------------------


@pytest.mark.parametrize(
    ("changed", "edit_the_data"),
    [
        ({"min_frequency": 0.30}, False),
        ({"test_size": 0.2}, False),
        ({"n_neighbors": 3}, False),
        ({"other_label": "Rest"}, False),
        ({}, True),
    ],
    ids=["min_frequency", "test_size", "n_neighbors", "other_label", "data"],
)
def test_the_cache_key_covers_everything_that_changes_a_split(
    toy_categorical_frame, tmp_path, changed, edit_the_data
):
    """A split cached before any of these changed must not be served after.

    The fingerprint mixes the columns and the contents of X with every parameter
    that shapes the split, precisely because none of them is in the file name,
    only the tag and the seed are. One of them left out of the payload would go
    on serving the old split, with no error and no sign in the results.
    """
    X, y = toy_categorical_frame
    kwargs = dict(
        seed=1,
        test_size=0.4,
        cache_dir=tmp_path,
        tag="controlled",
        categorical_columns=["HQCountry"],
        min_frequency=0.05,
    )

    get_split(X, y, **kwargs)
    if edit_the_data:
        X = X.copy()
        X.iloc[0, 0] += 1000.0
    get_split(X, y, **{**kwargs, **changed})

    assert len(list(tmp_path.glob("*.joblib"))) == 2


def test_force_regenerates_a_cached_split(toy_categorical_frame, tmp_path):
    """The notebook needs a one-flag way to rebuild tmp/splits without deleting
    files by hand."""
    import joblib

    X, y = toy_categorical_frame
    kwargs = dict(
        seed=1,
        test_size=0.4,
        cache_dir=tmp_path,
        tag="controlled",
        categorical_columns=["HQCountry"],
    )

    get_split(X, y, **kwargs)
    cache_file = next(tmp_path.glob("*.joblib"))
    joblib.dump({"stale": True}, cache_file)

    fresh = get_split(X, y, force=True, **kwargs)

    assert "X_train" in fresh
    assert "stale" not in joblib.load(cache_file)


def test_without_force_the_cached_split_is_reused(toy_categorical_frame, tmp_path):
    import joblib

    X, y = toy_categorical_frame
    kwargs = dict(
        seed=1,
        test_size=0.4,
        cache_dir=tmp_path,
        tag="controlled",
        categorical_columns=["HQCountry"],
    )

    get_split(X, y, **kwargs)
    cache_file = next(tmp_path.glob("*.joblib"))
    joblib.dump({"stale": True}, cache_file)

    assert get_split(X, y, **kwargs) == {"stale": True}


def test_clear_split_cache_removes_only_the_requested_tag(tmp_path):
    import joblib

    for name in (
        "split_controlled_seed1_abc.joblib",
        "split_controlled_seed2_abc.joblib",
        "split_leakboth_seed1_abc.joblib",
    ):
        joblib.dump({"x": 1}, tmp_path / name)

    removed = clear_split_cache(cache_dir=tmp_path, tag="controlled")

    assert removed == 2
    assert [p.name for p in tmp_path.glob("*.joblib")] == ["split_leakboth_seed1_abc.joblib"]


def test_clear_split_cache_without_a_tag_removes_everything(tmp_path):
    import joblib

    joblib.dump({"x": 1}, tmp_path / "split_controlled_seed1_abc.joblib")
    joblib.dump({"x": 1}, tmp_path / "split_leakboth_seed1_abc.joblib")

    removed = clear_split_cache(cache_dir=tmp_path)

    assert removed == 2
    assert list(tmp_path.glob("*.joblib")) == []


def test_clearing_a_missing_cache_directory_is_not_an_error(tmp_path):
    assert clear_split_cache(cache_dir=tmp_path / "nope") == 0
