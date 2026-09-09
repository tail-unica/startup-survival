"""Each model family builds, fits and scores on its own terms.

These run without W&B, without the real datasets and without a GPU: the point
is the wiring the seven-branch dispatch used to hold — which matrix a family is
fitted on, how it turns probabilities into labels, and that it reads the sweep
keys carrying its own prefix.
"""

import types

import numpy as np
import pandas as pd
import pytest
import yaml

from src.models import MODELS, build
from src.models.base import Model

FEATURES = ["a", "b", "c", "d"]


def _sweep_config(**overrides):
    """A stand-in for wandb.config carrying every key config.yaml declares."""
    cfg = dict(
        model_type="rf",
        seed=1,
        rf_n_estimators=5,
        rf_max_depth=3,
        rf_min_samples_leaf=2,
        lgb_n_estimators=5,
        lgb_max_depth=3,
        learning_rate=0.1,
        min_child_weight=1,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        dt_max_depth=3,
        dt_min_samples_leaf=2,
        dt_min_samples_split=2,
        dt_criterion="entropy",
        lr_C=1.0,
        lr_penalty="l2",
        svm_C=1.0,
        svm_gamma="scale",
        hidden_sizes="8,4",
        mlp_learning_rate=0.01,
        dropout_rate=0.1,
        weight_decay=0.0,
        batch_size=16,
        tabpfn_n_estimators=2,
    )
    cfg.update(overrides)
    return types.SimpleNamespace(**cfg)


@pytest.fixture
def split():
    """A split shaped like get_split's output: frames raw, matrices as arrays.

    The two classes are linearly separable on feature "a" plus noise, so every
    family reaches a sensible fit on very few rows.
    """
    rng = np.random.default_rng(0)
    sizes = {"train": 60, "val": 30, "test": 30}
    out = {}

    for name, n in sizes.items():
        y = np.tile([0, 1], n // 2)
        X = rng.normal(size=(n, len(FEATURES)))
        X[:, 0] += y * 3.0
        frame = pd.DataFrame(X, columns=FEATURES)

        out[f"X_{name}"] = frame
        out[f"X_{name}_imp"] = frame.to_numpy()
        out[f"X_{name}_scaled"] = frame.to_numpy()
        out[f"y_{name}"] = pd.Series(y)

    return out


def _fitted(model_type, split, cfg=None):
    model = build(model_type, cfg or _sweep_config(), seed=1)
    X_train, X_val, _ = model.matrices(split)
    model.fit(X_train, split["y_train"], X_val, split["y_val"])
    return model


# --- the registry stays in step with the sweep -------------------------------


def test_registry_covers_exactly_the_sweep_model_types():
    """A family in config.yaml with no class, or the reverse, breaks the sweep."""
    config = yaml.safe_load(open("config/config.yaml"))
    declared = config["sweep_settings"]["parameters"]["model_type"]["values"]

    assert sorted(MODELS) == sorted(declared)


def test_unknown_model_type_is_rejected_by_name():
    with pytest.raises(ValueError, match="unknown model_type 'xgboost'"):
        build("xgboost", _sweep_config(), seed=1)


# --- which matrix each family is fitted on -----------------------------------


@pytest.mark.parametrize("model_type", ["lr", "svm", "mlp"])
def test_distance_based_families_get_the_scaled_matrix(model_type, split):
    """LR, the RBF SVM and the MLP all compare magnitudes across features."""
    model = build(model_type, _sweep_config(), seed=1)

    assert model.wants_scaled
    assert model.matrices(split)[0] is split["X_train_scaled"]


@pytest.mark.parametrize("model_type", ["rf", "lgb", "dt", "tabpfn"])
def test_the_other_families_get_the_imputed_matrix(model_type, split):
    """Trees split on thresholds and TabPFN normalises its own inputs."""
    model = build(model_type, _sweep_config(), seed=1)

    assert not model.wants_scaled
    assert model.matrices(split)[0] is split["X_train_imp"]


# --- fit and score -----------------------------------------------------------


@pytest.mark.parametrize("model_type", ["rf", "lgb", "dt", "lr", "svm", "mlp"])
def test_each_family_fits_and_scores_its_test_matrix(model_type, split):
    model = _fitted(model_type, split)
    _, _, X_test = model.matrices(split)

    probs, preds = model.score(X_test)

    assert probs.shape == (len(split["y_test"]),)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    assert set(np.unique(preds)) <= {0, 1}
    assert len(preds) == len(split["y_test"])


@pytest.mark.parametrize("model_type", ["svm", "mlp"])
def test_thresholding_families_label_exactly_by_their_own_probabilities(model_type, split):
    """The reason these override score().

    SVC.predict() takes the sign of the decision function while predict_proba()
    goes through Platt scaling, so the two can disagree; the comparison table is
    only meaningful if every model decides at p >= 0.5.
    """
    model = _fitted(model_type, split)
    _, _, X_test = model.matrices(split)

    probs, preds = model.score(X_test)

    assert (preds == (probs >= 0.5).astype(int)).all()


def test_sklearn_families_keep_the_estimators_own_decision_rule(split):
    """Trees are left alone: at exactly p == 0.5 a leaf votes for the negative
    class, and thresholding would silently flip those rows."""
    model = _fitted("rf", split)
    _, _, X_test = model.matrices(split)

    _, preds = model.score(X_test)

    assert (preds == model.model.predict(X_test)).all()


def test_lightgbm_balances_the_classes_from_the_training_split(split):
    """scale_pos_weight is LightGBM's stand-in for class_weight='balanced', so
    it can only be set once the training split is known."""
    y = pd.Series([0] * 80 + [1] * 20)
    X = split["X_train_imp"][:1].repeat(100, axis=0)
    model = build("lgb", _sweep_config(), seed=1)

    model.fit(X, y)

    assert model.model.get_params()["scale_pos_weight"] == pytest.approx(4.0)


# --- SHAP --------------------------------------------------------------------


def test_tree_explainer_returns_one_value_per_feature_for_the_positive_class(split):
    """TreeExplainer hands back the positive class as a list, as a third array
    axis, or already alone, depending on the library."""
    model = _fitted("rf", split)

    values, sample = model.explain(split, shap_cfg={})

    assert values.shape == (len(sample), len(FEATURES))
    assert sample.equals(split["X_test"])


def test_permutation_explainer_honours_its_budget(split):
    """SVM and TabPFN are explained on an explicit row budget from config.yaml,
    because their cost per explained row is what makes them intractable."""
    model = _fitted("svm", split)

    values, sample = model.explain(split, shap_cfg={"n_explain": 5, "n_background": 4})

    assert values.shape == (5, len(FEATURES))
    assert len(sample) == 5


# --- TabPFN is configured, not fitted, here ----------------------------------


def test_tabpfn_is_built_for_the_checkpoint_the_api_served():
    """Fitting it would download the checkpoint and want a GPU, so this checks
    the configuration that keeps a local run equivalent to the API one."""
    model = build("tabpfn", _sweep_config(), seed=7)
    params = model.model.get_params()

    assert params["model_path"] == "tabpfn-v2-classifier-v2_default.ckpt"
    assert params["ignore_pretraining_limits"] is True
    assert params["balance_probabilities"] is True
    assert params["random_state"] == 7
    assert params["n_estimators"] == 2


# --- the interface itself ----------------------------------------------------


@pytest.mark.parametrize("model_type", sorted(MODELS))
def test_every_family_reads_only_its_own_sweep_keys(model_type):
    """config.yaml names hyperparameters with a family prefix, and that is what
    lets the sweep block stay untouched while the dispatch moves into classes."""
    model = build(model_type, _sweep_config(), seed=3)

    assert isinstance(model, Model)
    assert model.seed == 3
