"""The explainers explain the predictions the models actually make."""

import numpy as np
import pandas as pd
import shap
from sklearn.tree import DecisionTreeClassifier

from src.models.sklearn_models import DecisionTreeModel
from src.utils import prepare_splits


def test_tree_shap_values_add_up_to_the_scored_probability():
    """A row with a missing value is scored imputed, so it has to be explained imputed."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=list("abc"))
    y = pd.Series(((X["a"] + X["b"]) > 0).astype(int))
    X.loc[rng.random(300) < 0.3, "a"] = np.nan
    split = prepare_splits(X, y, seed=1, test_size=0.4)

    model = DecisionTreeModel(DecisionTreeClassifier(max_depth=4, random_state=0), seed=1)
    model.fit(split["X_train_imp"], split["y_train"])
    values, sample = model.explain(split, {})

    base = np.ravel(shap.TreeExplainer(model.model).expected_value)[-1]
    scored = model.predict_proba(split["X_test_imp"][: len(sample)])
    np.testing.assert_allclose(values.sum(axis=1) + base, scored, atol=1e-6)
    assert list(sample.columns) == list("abc")
