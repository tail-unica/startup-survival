"""The contract between the three stages of the pipeline.

The panel is built by ``src/panel/``, the datasets by ``src/preprocessing.py``, and
the experiments read the datasets. What holds them together is a list of column
names and a handful of settings, and nothing enforces it at runtime: a column
renamed in one place would surface as an error in the middle of a sweep, or worse as
a silently missing feature.

Everything checked here is read from the code and the configuration, never from the
text of a notebook cell.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.experiments import SETTINGS, split_strata
from src.panel.config import PanelRules
from src.panel.pipeline import panel_name
from src.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
RULES = PanelRules.from_config(CONFIG)


def test_the_panel_carries_every_feature_the_models_read():
    missing = [c for c in FEATURE_COLUMNS if c not in RULES.columns]
    assert not missing, f"the panel does not produce: {missing}"


def test_the_panel_carries_the_columns_the_target_is_built_from():
    missing = [c for c in TARGET_COLUMNS if c not in RULES.columns]
    assert not missing, f"the panel does not produce: {missing}"


def test_the_panel_has_no_duplicate_column():
    assert len(RULES.columns) == len(set(RULES.columns))


def test_every_panel_column_reaches_the_datasets():
    unused = set(RULES.columns) - set(FEATURE_COLUMNS) - set(TARGET_COLUMNS)
    assert not unused, f"the panel carries columns no dataset reads: {sorted(unused)}"


def test_the_ablations_name_columns_the_datasets_carry():
    # The ablation lists are applied to the processed dataset, so a name the
    # missing-value threshold has dropped would raise mid-experiment. Two panel
    # columns are renamed on the way there, and those are the two exceptions.
    renamed = {"HasTop50Institute", "Gender_CEO_Female"}
    available = set(FEATURE_COLUMNS) | renamed
    for name, columns in CONFIG["ablations"].items():
        unknown = [c for c in columns if c not in available]
        assert not unknown, f"ablation {name} names columns that do not exist: {unknown}"


def test_the_six_experiments_read_the_two_datasets_alone():
    assert {"dataset_controlled", "dataset_leakboth"} <= set(CONFIG["paths"])
    assert set(SETTINGS) == {
        "controlled",
        "leakboth",
        "noteam",
        "nocompetitors",
        "leaklabel",
        "leakfeat",
    }


def test_the_two_panels_have_distinct_names():
    # One panel per switch configuration, and the two live side by side, so neither
    # a rebuild nor a demo can silently overwrite the other.
    assert panel_name(timed=True) != panel_name(timed=False)
    for timed, key in ((True, "panel_timed"), (False, "panel_snapshot")):
        assert Path(CONFIG["paths"][key]).name == f"{panel_name(timed=timed)}.csv.gz"


def test_the_sample_threshold_is_older_than_the_first_decision_year():
    # The panel keeps the firms founded from `first_year` on, and the datasets keep
    # the decision years from `first_decision_year` on: a founding threshold later
    # than the decision one would silently empty the sample.
    assert int(CONFIG["first_year"]) <= int(CONFIG["first_decision_year"])


def test_the_horizon_is_a_positive_number_of_years():
    # T is the prediction horizon: a firm has grown if it reaches the next stage
    # within T years of its StartingAge. At zero or below there is no window to
    # reach anything in, and the target would be empty rather than wrong.
    assert int(CONFIG["T"]) > 0


def test_the_missing_value_policy_is_declared_in_one_place():
    policy = CONFIG["preprocessing"]
    assert 0 < float(policy["missing_threshold"]) < 1
    assert int(policy["max_missing_per_row"]) >= 1


def test_the_split_is_stratified_on_the_controlled_target_only_for_its_own_firms(tmp_path):
    path = tmp_path / "controlled.csv"
    pd.DataFrame({"CompanyID": [1, 2, 3], "Target": [0, 1, 0]}).to_csv(path, index=False)
    config = {"paths": {"dataset_controlled": str(path)}}

    strata = split_strata(pd.DataFrame({"CompanyID": [1, 2, 3], "Target": [1, 1, 1]}), config)
    assert strata.tolist() == [0, 1, 0]
    with pytest.raises(ValueError):
        split_strata(pd.DataFrame({"CompanyID": [2, 1, 3]}), config)
