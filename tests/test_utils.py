import numpy as np
import pandas as pd
import pytest

from src.utils import (
    clear_split_cache,
    compare_metrics,
    compute_permutation_shap,
    get_split,
    plot_correlation_heatmap,
    prepare_splits,
    summarize_metrics,
)

# --- compute_permutation_shap ----------------------------------------------


class _LinearProbaModel:
    """Minimal predict_proba stand-in: p(pos) is a fixed linear map of the features."""

    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)
        self.n_calls = 0

    def predict_proba(self, X):
        self.n_calls += len(X)
        logit = np.asarray(X, dtype=float) @ self.weights
        p = 1.0 / (1.0 + np.exp(-logit))
        return np.column_stack([1.0 - p, p])


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    background = rng.normal(size=(40, 4))
    to_explain = rng.normal(size=(12, 4))
    return background, to_explain


def test_returns_one_shap_value_per_cell_explained(toy_data):
    background, to_explain = toy_data
    model = _LinearProbaModel([2.0, 0.0, -1.0, 0.0])

    values = compute_permutation_shap(
        model, background, to_explain, n_explain=5, n_background=10, random_state=12
    )

    assert values.shape == (5, 4)


def test_ranks_features_by_true_influence(toy_data):
    background, to_explain = toy_data
    # feature 0 dominates, feature 1 is ignored by the model
    model = _LinearProbaModel([3.0, 0.0, -1.0, 0.0])

    values = compute_permutation_shap(
        model, background, to_explain, n_explain=8, n_background=20, random_state=12
    )
    importance = np.abs(values).mean(axis=0)

    assert importance.argmax() == 0
    assert importance[1] == pytest.approx(0.0, abs=1e-9)
    assert importance[3] == pytest.approx(0.0, abs=1e-9)


def test_is_deterministic_for_a_fixed_seed(toy_data):
    background, to_explain = toy_data
    model = _LinearProbaModel([1.0, -2.0, 0.5, 0.0])

    kwargs = dict(n_explain=6, n_background=15, random_state=12)
    first = compute_permutation_shap(model, background, to_explain, **kwargs)
    second = compute_permutation_shap(model, background, to_explain, **kwargs)

    np.testing.assert_allclose(first, second)


def test_caps_work_at_the_requested_budget(toy_data):
    background, to_explain = toy_data
    model = _LinearProbaModel([1.0, 1.0, 1.0, 1.0])

    compute_permutation_shap(
        model, background, to_explain, n_explain=3, n_background=5, random_state=12
    )

    # One permutation round is 2*n_features+1 masks per explained row, each
    # evaluated over the background set; shap adds a base-value pass on top, so
    # allow a factor of 2. This still fails loudly if max_evals ever falls back
    # to shap's "auto" (500*n_features), which would be ~30k calls here.
    budget = 3 * (2 * 4 + 1) * 5
    assert model.n_calls <= 2 * budget


def test_requesting_more_rows_than_available_is_clamped(toy_data):
    background, to_explain = toy_data
    model = _LinearProbaModel([1.0, 0.0, 0.0, 0.0])

    values = compute_permutation_shap(
        model, background, to_explain, n_explain=999, n_background=999, random_state=12
    )

    assert values.shape == (len(to_explain), 4)


# --- prepare_splits / get_split ---------------------------------------------


@pytest.fixture
def toy_frame():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(200, 5)), columns=list("abcde"))
    X.iloc[::17, 2] = np.nan  # a few holes for the imputer to fill
    y = pd.Series((rng.random(200) < 0.3).astype(int), name="Target")
    return X, y


def test_split_sizes_follow_the_configured_ratio(toy_frame):
    X, y = toy_frame
    s = prepare_splits(X, y, seed=1, test_size=0.4)

    assert len(s["y_train"]) == 120
    assert len(s["y_val"]) == 40
    assert len(s["y_test"]) == 40


def test_different_seeds_produce_different_splits(toy_frame):
    X, y = toy_frame
    a = prepare_splits(X, y, seed=1, test_size=0.4)
    b = prepare_splits(X, y, seed=2, test_size=0.4)

    assert set(a["X_train"].index) != set(b["X_train"].index)


def test_same_seed_reproduces_the_same_split(toy_frame):
    X, y = toy_frame
    a = prepare_splits(X, y, seed=3, test_size=0.4)
    b = prepare_splits(X, y, seed=3, test_size=0.4)

    assert list(a["X_test"].index) == list(b["X_test"].index)
    np.testing.assert_allclose(a["X_test_scaled"], b["X_test_scaled"])


def test_splits_are_disjoint(toy_frame):
    X, y = toy_frame
    s = prepare_splits(X, y, seed=1, test_size=0.4)

    tr, va, te = (set(s[k].index) for k in ("X_train", "X_val", "X_test"))
    assert tr & va == set() and tr & te == set() and va & te == set()
    assert len(tr | va | te) == len(X)


def test_imputation_fills_every_hole(toy_frame):
    X, y = toy_frame
    s = prepare_splits(X, y, seed=1, test_size=0.4)

    for key in ("X_train_imp", "X_val_imp", "X_test_imp"):
        assert np.isfinite(s[key]).all()


def test_scaler_is_fitted_on_train_only(toy_frame):
    """The held-out sets must not influence the scaler: that is the leakage the paper is about."""
    X, y = toy_frame
    s = prepare_splits(X, y, seed=1, test_size=0.4)

    # RobustScaler centres on the training median, so the training set is the
    # only one guaranteed to come out with a ~zero median.
    np.testing.assert_allclose(
        np.median(s["X_train_scaled"], axis=0), np.zeros(X.shape[1]), atol=1e-9
    )


def test_get_split_caches_to_disk_and_reuses_it(toy_frame, tmp_path):
    X, y = toy_frame

    first = get_split(X, y, seed=1, test_size=0.4, cache_dir=tmp_path, tag="window")
    cached = list(tmp_path.glob("*.joblib"))
    assert len(cached) == 1

    second = get_split(X, y, seed=1, test_size=0.4, cache_dir=tmp_path, tag="window")
    np.testing.assert_allclose(first["X_test_scaled"], second["X_test_scaled"])


def test_get_split_keeps_tags_and_seeds_apart(toy_frame, tmp_path):
    X, y = toy_frame

    get_split(X, y, seed=1, test_size=0.4, cache_dir=tmp_path, tag="window")
    get_split(X, y, seed=2, test_size=0.4, cache_dir=tmp_path, tag="window")
    get_split(X, y, seed=1, test_size=0.4, cache_dir=tmp_path, tag="nowindow")

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


def test_reports_mean_and_std_across_seeds():
    store = {
        ("rf", "window"): _runs([0.80, 0.82, 0.84]),
        ("rf", "nowindow"): _runs([0.90, 0.90, 0.90]),
    }

    df = compare_metrics(store, "window", "nowindow")
    auc = df.set_index("Model")["AUC"]

    assert auc["rf"] == "0.820 ± 0.020"
    assert auc["rf w/o window"].startswith("0.900 ± 0.000")


def test_percent_change_is_computed_on_the_means():
    store = {
        ("rf", "window"): _runs([0.40, 0.60]),  # mean 0.50
        ("rf", "nowindow"): _runs([0.70, 0.80]),  # mean 0.75
    }

    df = compare_metrics(store, "window", "nowindow")
    cell = df.set_index("Model")["AUC"]["rf w/o window"]

    assert "(+50.0%)" in cell


def test_single_seed_gives_zero_std_instead_of_nan():
    """ddof=1 is undefined for one run; it must degrade to 0.0, not NaN."""
    store = {
        ("rf", "window"): _runs([0.80]),
        ("rf", "nowindow"): _runs([0.85]),
    }

    df = compare_metrics(store, "window", "nowindow")

    assert df.set_index("Model")["AUC"]["rf"] == "0.800 ± 0.000"


def test_compare_metrics_latex_output_uses_pm_notation():
    store = {
        ("rf", "window"): _runs([0.80, 0.82]),
        ("rf", "nowindow"): _runs([0.90, 0.90]),
    }

    df = compare_metrics(store, "window", "nowindow", latex=True)
    cell = df.set_index("Model")["AUC"]["rf"]

    assert "\\pm" in cell and cell.startswith("$")


def test_models_missing_from_one_experiment_are_skipped():
    store = {
        ("rf", "window"): _runs([0.80]),
        ("rf", "nowindow"): _runs([0.85]),
        ("svm", "window"): _runs([0.70]),  # no nowindow counterpart
    }

    df = compare_metrics(store, "window", "nowindow")

    assert list(df["Model"]) == ["rf", "rf w/o window"]


def test_row_order_follows_the_paper():
    store = {}
    for m in ("tabpfn", "svm", "lgb", "lr"):
        store[(m, "window")] = _runs([0.80])
        store[(m, "nowindow")] = _runs([0.85])

    df = compare_metrics(store, "window", "nowindow")
    models = [m for m in df["Model"] if "w/o" not in m]

    assert models == ["lr", "lgb", "svm", "tabpfn"]


def test_std_is_the_sample_one_not_the_population_one():
    """Values chosen so ddof=0 and ddof=1 differ in the second decimal."""
    store = {
        ("rf", "window"): _runs([0.10, 0.30, 0.50, 0.70, 0.90]),
        ("rf", "nowindow"): _runs([0.50]),
    }

    df = compare_metrics(store, "window", "nowindow")
    cell = df.set_index("Model")["AUC"]["rf"]

    # population std = 0.2828; sample std = 0.3162
    assert cell == "0.500 ± 0.316", cell


def test_default_precision_resolves_a_std_two_decimals_would_round_away():
    """A std below 0.005 renders as '± 0.00' at two decimals — a bar that reads
    as zero variance rather than small variance. Three decimals keeps it."""
    store = {
        ("rf", "window"): _runs([0.780, 0.781, 0.782, 0.783, 0.784]),  # sample std 0.0016
        ("rf", "nowindow"): _runs([0.70]),
    }

    default = compare_metrics(store, "window", "nowindow")
    coarse = compare_metrics(store, "window", "nowindow", decimals=2)

    assert default.set_index("Model")["AUC"]["rf"] == "0.782 ± 0.002"
    assert coarse.set_index("Model")["AUC"]["rf"] == "0.78 ± 0.00"


def test_measured_auc_spread_is_reported_at_three_decimals():
    """The five AUCs actually measured for the decision tree on this dataset."""
    store = {
        ("dt", "window"): _runs([0.7834, 0.7850, 0.7721, 0.7780, 0.7832]),
        ("dt", "nowindow"): _runs([0.7834, 0.7850, 0.7721, 0.7780, 0.7832]),
    }

    df = compare_metrics(store, "window", "nowindow")

    assert df.set_index("Model")["AUC"]["dt"] == "0.780 ± 0.005"


def test_row_label_names_the_isolated_leak():
    """The 2x2 leakage tables reuse the layout with a different second row."""
    store = {
        ("rf", "window"): _runs([0.80]),
        ("rf", "leakfeat"): _runs([0.90]),
    }

    df = compare_metrics(store, "window", "leakfeat", label_b="leaked features")

    assert list(df["Model"]) == ["rf", "rf leaked features"]


# --- summarize_metrics: one experiment, mean ± std across seeds --------------


def test_reports_one_row_per_model_with_mean_and_std():
    store = {
        ("rf", "window"): _runs([0.80, 0.82, 0.84]),
        ("lr", "window"): _runs([0.70, 0.72]),
    }

    df = summarize_metrics(store, "window")

    assert list(df["Model"]) == ["lr", "rf"]  # paper order
    assert df.set_index("Model")["AUC"]["rf"] == "0.820 ± 0.020"


def test_only_the_requested_tag_is_reported():
    store = {
        ("rf", "window"): _runs([0.80]),
        ("rf", "nowindow"): _runs([0.90]),
        ("lr", "nowindow"): _runs([0.60]),
    }

    df = summarize_metrics(store, "nowindow")

    assert list(df["Model"]) == ["lr", "rf"]
    assert df.set_index("Model")["AUC"]["rf"] == "0.900 ± 0.000"


def test_seed_count_distinguishes_one_run_from_a_flat_metric():
    """'± 0.000' alone cannot say whether it is one run or five identical ones."""
    store = {
        ("rf", "window"): _runs([0.80]),
        ("lr", "window"): _runs([0.80, 0.80, 0.80]),
    }

    df = summarize_metrics(store, "window").set_index("Model")

    assert df["AUC"]["rf"] == df["AUC"]["lr"] == "0.800 ± 0.000"
    assert df["Seeds"]["rf"] == 1
    assert df["Seeds"]["lr"] == 3


def test_columns_follow_the_paper_order():
    store = {("rf", "window"): _runs([0.80, 0.82])}

    df = summarize_metrics(store, "window")

    assert list(df.columns) == [
        "Model",
        "Seeds",
        "AUC",
        "F1",
        "precision",
        "recall",
        "accuracy",
        "accuracy_train",
    ]


def test_tuning_metrics_are_not_reported():
    """F1_val and F1_train drive the sweep; the result table reports test metrics."""
    runs = _runs([0.80, 0.82])
    for r in runs:
        r["F1_val"] = 0.44
        r["F1_train"] = 0.99

    df = summarize_metrics({("rf", "window"): runs}, "window")

    assert "F1_val" not in df.columns
    assert "F1_train" not in df.columns


def test_summarize_metrics_latex_output_uses_pm_notation():
    store = {("rf", "window"): _runs([0.80, 0.82, 0.84])}

    df = summarize_metrics(store, "window", latex=True)

    assert df.set_index("Model")["AUC"]["rf"] == "$0.820 \\pm 0.020$"


def test_unknown_tag_raises_instead_of_returning_an_empty_table():
    store = {("rf", "window"): _runs([0.80])}

    with pytest.raises(KeyError, match="noteam"):
        summarize_metrics(store, "noteam")


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


def test_categorical_columns_come_out_numeric(toy_categorical_frame):
    X, y = toy_categorical_frame

    s = prepare_splits(
        X, y, seed=1, test_size=0.4, categorical_columns=["HQCountry"], min_frequency=0.05
    )

    for key in ("X_train", "X_val", "X_test"):
        assert np.issubdtype(s[key]["HQCountry"].dtype, np.floating)


def test_split_exposes_the_fitted_encodings(toy_categorical_frame):
    X, y = toy_categorical_frame

    s = prepare_splits(
        X, y, seed=1, test_size=0.4, categorical_columns=["HQCountry"], min_frequency=0.05
    )

    assert set(s["encodings"]) == {"HQCountry"}
    assert sum(s["encodings"]["HQCountry"].values()) == pytest.approx(1.0)


def test_encoded_values_are_the_training_shares(toy_categorical_frame):
    """Each kept category must appear in the training column as its own share,
    on exactly as many rows as it holds."""
    X, y = toy_categorical_frame

    s = prepare_splits(
        X, y, seed=1, test_size=0.4, categorical_columns=["HQCountry"], min_frequency=0.05
    )

    enc = s["encodings"]["HQCountry"]
    col = s["X_train"]["HQCountry"]
    n_train = len(col)
    for category, share in enc.items():
        if category == "Others":
            continue
        assert (col == share).sum() == pytest.approx(share * n_train, abs=1e-6)


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


def test_min_frequency_controls_how_much_is_pooled(toy_categorical_frame):
    X, y = toy_categorical_frame

    lenient = prepare_splits(
        X, y, seed=1, test_size=0.4, categorical_columns=["HQCountry"], min_frequency=0.05
    )
    strict = prepare_splits(
        X, y, seed=1, test_size=0.4, categorical_columns=["HQCountry"], min_frequency=0.30
    )

    assert len(strict["encodings"]["HQCountry"]) < len(lenient["encodings"]["HQCountry"])
    assert strict["encodings"]["HQCountry"]["Others"] > lenient["encodings"]["HQCountry"]["Others"]


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


def test_no_categorical_columns_leaves_the_frame_untouched(toy_frame):
    X, y = toy_frame

    s = prepare_splits(X, y, seed=1, test_size=0.4)

    assert s["encodings"] == {}


# --- get_split: cache invalidation and regeneration --------------------------


def test_cache_key_separates_different_thresholds(toy_categorical_frame, tmp_path):
    """A split cached before the threshold changed must not be served after."""
    X, y = toy_categorical_frame
    kwargs = dict(
        seed=1, test_size=0.4, cache_dir=tmp_path, tag="window", categorical_columns=["HQCountry"]
    )

    a = get_split(X, y, min_frequency=0.05, **kwargs)
    b = get_split(X, y, min_frequency=0.30, **kwargs)

    assert len(list(tmp_path.glob("*.joblib"))) == 2
    assert a["encodings"]["HQCountry"] != b["encodings"]["HQCountry"]


def test_force_regenerates_a_cached_split(toy_categorical_frame, tmp_path):
    """The notebook needs a one-flag way to rebuild tmp/splits without deleting
    files by hand."""
    import joblib

    X, y = toy_categorical_frame
    kwargs = dict(
        seed=1, test_size=0.4, cache_dir=tmp_path, tag="window", categorical_columns=["HQCountry"]
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
        seed=1, test_size=0.4, cache_dir=tmp_path, tag="window", categorical_columns=["HQCountry"]
    )

    get_split(X, y, **kwargs)
    cache_file = next(tmp_path.glob("*.joblib"))
    joblib.dump({"stale": True}, cache_file)

    assert get_split(X, y, **kwargs) == {"stale": True}


def test_clear_split_cache_removes_only_the_requested_tag(tmp_path):
    import joblib

    for name in (
        "split_window_seed1_abc.joblib",
        "split_window_seed2_abc.joblib",
        "split_nowindow_seed1_abc.joblib",
    ):
        joblib.dump({"x": 1}, tmp_path / name)

    removed = clear_split_cache(cache_dir=tmp_path, tag="window")

    assert removed == 2
    assert [p.name for p in tmp_path.glob("*.joblib")] == ["split_nowindow_seed1_abc.joblib"]


def test_clear_split_cache_without_a_tag_removes_everything(tmp_path):
    import joblib

    joblib.dump({"x": 1}, tmp_path / "split_window_seed1_abc.joblib")
    joblib.dump({"x": 1}, tmp_path / "split_nowindow_seed1_abc.joblib")

    removed = clear_split_cache(cache_dir=tmp_path)

    assert removed == 2
    assert list(tmp_path.glob("*.joblib")) == []


def test_clearing_a_missing_cache_directory_is_not_an_error(tmp_path):
    assert clear_split_cache(cache_dir=tmp_path / "nope") == 0


# --- plot_correlation_heatmap ------------------------------------------------


def test_correlation_heatmap_ignores_raw_categorical_columns():
    """The processed CSV now carries HQCountry as a string; pandas .corr()
    raises on it, and the notebook draws this heatmap before the split."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    df = pd.DataFrame(
        {
            "CompanyID": [1, 2, 3, 4],
            "Target": [0, 1, 0, 1],
            "Age": [1.0, 2.0, 3.0, 4.0],
            "Total_People": [4.0, 3.0, 2.0, 1.0],
            "HQCountry": ["USA", "ITA", "USA", "GBR"],
        }
    )

    corr = plot_correlation_heatmap(df)
    plt.close("all")

    assert list(corr.columns) == ["Age", "Total_People"]
    assert corr.loc["Age", "Total_People"] == pytest.approx(-1.0)


def test_cache_key_changes_when_the_data_changes(toy_frame, tmp_path):
    """Same columns, same row count, different values: the stale split from the
    previous dataset must not be served."""
    X, y = toy_frame
    kwargs = dict(seed=1, test_size=0.4, cache_dir=tmp_path, tag="window")

    get_split(X, y, **kwargs)
    X_edited = X.copy()
    X_edited.iloc[0, 0] = X_edited.iloc[0, 0] + 1000.0
    get_split(X_edited, y, **kwargs)

    assert len(list(tmp_path.glob("*.joblib"))) == 2
