import numpy as np
import pandas as pd
import pytest

from src.utils import (
    THRESHOLD_METRICS,
    clear_split_cache,
    compare_metrics,
    f1_optimal_threshold,
    get_split,
    load_stores,
    prepare_splits,
    save_predictions,
    save_stores,
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


# --- threshold tuned on validation ------------------------------------------


def test_f1_optimal_threshold_picks_the_cut_that_separates_the_classes():
    """0.5 would call every row negative; the tuned cut recovers all positives."""
    labels = np.array([0, 0, 0, 0, 1, 1])
    probs = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40])

    assert f1_optimal_threshold(labels, probs) == pytest.approx(0.30)


def test_tuned_metrics_stay_out_of_the_main_table_and_fill_the_threshold_one():
    runs = [
        {
            **r,
            "F1_tuned": 0.7,
            "precision_tuned": 0.6,
            "recall_tuned": 0.8,
            "accuracy_tuned": 0.75,
            "threshold": 0.3,
        }
        for r in _runs([0.80, 0.82])
    ]
    store = {("rf", "controlled"): runs}

    main = summarize_metrics(store, "controlled")
    tuned = summarize_metrics(store, "controlled", metric_order=THRESHOLD_METRICS)

    tuned_keys = {"F1_tuned", "precision_tuned", "recall_tuned", "accuracy_tuned", "threshold"}
    assert not tuned_keys & set(main.columns)
    assert list(tuned.columns) == ["Model", "Seeds", *THRESHOLD_METRICS]
    assert tuned["threshold"][0] == "0.300 ± 0.000"


def test_saved_predictions_keep_labels_and_dataset_rows_aligned(tmp_path):
    """A later analysis must be able to recompute a metric and find the firm of each row."""
    y_val = pd.Series([0, 1], index=[7, 3])
    y_test = pd.Series([1, 0, 0], index=[5, 1, 9])
    split = {"y_val": y_val, "y_test": y_test}
    config = {"predictions": {"dir": str(tmp_path)}}

    path = save_predictions(split, [0.2, 0.9], [0.8, 0.1, 0.3], "rf", "controlled", 2, config)
    saved = np.load(path)

    assert path.name == "predictions_rf_controlled_seed2.npz"
    np.testing.assert_array_equal(saved["y_test"], [1, 0, 0])
    np.testing.assert_array_equal(saved["row_test"], [5, 1, 9])
    np.testing.assert_allclose(saved["prob_val"], [0.2, 0.9])
    np.testing.assert_array_equal(saved["row_val"], [7, 3])


def test_stores_saved_per_experiment_merge_back(tmp_path):
    """Each experiment writes only its own keys, and loading brings all of them back."""
    config = {"stores": {"dir": str(tmp_path)}}
    metrics = {("rf", "controlled"): _runs([0.8]), ("rf", "leakboth"): _runs([0.9])}
    shap_values = {("rf", "controlled"): {"shap_values": np.zeros((2, 3))}}

    save_stores(metrics, shap_values, "controlled", config)
    save_stores(metrics, shap_values, "leakboth", config)
    loaded_metrics, loaded_shap = load_stores(config)

    assert loaded_metrics == metrics
    assert set(loaded_shap) == {("rf", "controlled")}


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


# --- prepare_splits: imputer and scaler fitted per group ---------------------


@pytest.fixture
def toy_grouped_frame():
    """Two groups whose column ``b`` sits on two levels, and noise everywhere else,
    so a global KNN would draw neighbours from both groups."""
    rng = np.random.default_rng(3)
    n = 400
    g = np.repeat([0, 1], n // 2)
    X = pd.DataFrame({"g": g, "a": rng.normal(scale=10, size=n), "b": 50.0 * g})
    X.loc[X.index[::7], "b"] = np.nan
    y = pd.Series((rng.random(n) < 0.3).astype(int))
    return X, y


def test_the_grouped_imputer_draws_neighbours_from_the_row_s_own_group(toy_grouped_frame):
    X, y = toy_grouped_frame
    s = prepare_splits(X, y, seed=1, test_size=0.4, impute_by="g")
    for raw, imputed in (
        ("X_train", "X_train_imp"),
        ("X_val", "X_val_imp"),
        ("X_test", "X_test_imp"),
    ):
        b = s[imputed][:, X.columns.get_loc("b")]
        np.testing.assert_array_equal(b, 50.0 * s[raw]["g"].to_numpy())


def test_the_grouped_scaler_centres_every_group_on_its_training_rows(toy_grouped_frame):
    X, y = toy_grouped_frame
    s = prepare_splits(X, y, seed=1, test_size=0.4, scale_by="g")
    reference = prepare_splits(X, y, seed=1, test_size=0.4)
    a = X.columns.get_loc("a")
    for group in (0, 1):
        rows = s["X_train"]["g"].to_numpy() == group
        assert abs(np.median(s["X_train_scaled"][rows, a])) < 1e-9
    # The grouping column goes through the global scaler, or it would be zero everywhere.
    g = X.columns.get_loc("g")
    for key in ("X_train_scaled", "X_val_scaled", "X_test_scaled"):
        np.testing.assert_array_equal(s[key][:, g], reference[key][:, g])


def test_a_group_the_training_split_lacks_goes_through_the_global_fit(toy_grouped_frame):
    X, y = toy_grouped_frame
    X = X.astype({"g": float})
    X.loc[X.index[:5], "g"] = np.nan  # rows with no group at all
    s = prepare_splits(X, y, seed=1, test_size=0.4, impute_by="g", scale_by="g")
    for key in ("X_train_scaled", "X_val_scaled", "X_test_scaled"):
        assert not np.isnan(np.delete(s[key], X.columns.get_loc("g"), axis=1)).any()


def test_a_frequency_encoded_column_is_grouped_as_the_encoding_pools_it(toy_categorical_frame):
    """JPN and ESP fall under the threshold: they form one group, not two tiny ones."""
    from src.utils import _group_labels

    X, y = toy_categorical_frame
    encodings = {"HQCountry": {"USA": 0.6, "GBR": 0.25, "ITA": 0.11, "Others": 0.04}}
    labels = _group_labels(X, "HQCountry", encodings, "Others")
    assert set(labels) == {"USA", "GBR", "ITA", "Others"}


def test_grouped_variants_cross_settings_groups_and_steps():
    from src.experiments import grouped_variants

    config = {
        "grouped_preprocessing": {
            "settings": ["controlled"],
            "groups": {"cohort": "YearFounded"},
            "fit": ["impute", "scale", "both"],
        },
        "sweep_settings": {"parameters": {"model_type": {"values": ["lgb", "lr", "svm"]}}},
    }
    variants = {v["tag"]: v for v in grouped_variants(config)}
    assert set(variants) == {
        "controlled_impute-cohort",
        "controlled_scale-cohort",
        "controlled_both-cohort",
    }
    assert variants["controlled_both-cohort"]["split"] == {
        "impute_by": "YearFounded",
        "scale_by": "YearFounded",
    }

    def models(tag):
        return variants[tag]["config"]["sweep_settings"]["parameters"]["model_type"]["values"]

    assert models("controlled_scale-cohort") == ["lr", "svm"]
    assert models("controlled_impute-cohort") == ["lgb", "lr", "svm"]
    # The caller's config is left alone.
    assert config["sweep_settings"]["parameters"]["model_type"]["values"] == ["lgb", "lr", "svm"]


# --- get_split: cache invalidation and regeneration --------------------------


@pytest.mark.parametrize(
    ("changed", "edit_the_data"),
    [
        ({"min_frequency": 0.30}, False),
        ({"test_size": 0.2}, False),
        ({"n_neighbors": 3}, False),
        ({"other_label": "Rest"}, False),
        ({"impute_by": "HQCountry"}, False),
        ({"scale_by": "HQCountry"}, False),
        ({}, True),
    ],
    ids=[
        "min_frequency",
        "test_size",
        "n_neighbors",
        "other_label",
        "impute_by",
        "scale_by",
        "data",
    ],
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
