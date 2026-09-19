"""The six experiments: which dataset each one reads, and how it is assembled.

Two of them are the datasets as they come out of the preprocessing. Two are
ablations, the same bias-controlled dataset with a family of features removed. The
last two are the controls that take the 2x2 apart: the features of one setting
against the target of the other, swapped on ``CompanyID``, which is legitimate
because both datasets carry the same firms and the same columns.

The notebook and the command line call the same functions here, so an experiment
cannot mean one thing in one place and something else in the other.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

#: What each setting is, in one line.
SETTINGS: dict[str, str] = {
    "window": "bias-controlled: attributes of the row's own year, features at the starting age",
    "nowindow": "look-ahead: attributes as of the extraction, features over the whole life",
    "noteam": "window without the team features",
    "nocompetitors": "window without the competitor features",
    "leaklabel": "bias-free features, leaked target",
    "leakfeat": "leaked features, bias-free target",
}


def load_setting(setting: str, config: dict[str, Any]) -> pd.DataFrame:
    """Assemble the dataset of one experiment.

    :param setting: One of :data:`SETTINGS`.
    :param config: The parsed ``config.yaml``.
    :return: The dataset, with ``CompanyID`` and ``Target`` among its columns.
    :raises ValueError: If the setting is not one of the six.
    """
    if setting not in SETTINGS:
        raise ValueError(f"unknown setting {setting!r}; expected one of {sorted(SETTINGS)}")
    paths, ablations = config["paths"], config["ablations"]
    window = paths["dataset_window"]
    nowindow = paths["dataset_nowindow"]

    if setting == "window":
        return pd.read_csv(window)
    if setting == "nowindow":
        return pd.read_csv(nowindow)
    if setting in ("noteam", "nocompetitors"):
        dataset = pd.read_csv(window)
        return dataset.drop(columns=ablations[setting])

    # The two controls: same firms, same columns, the other definition of the
    # target. The base rate travels with the label, and no model here tunes its
    # decision threshold, so AUC is the metric to read across this axis.
    swapped = setting == "leaklabel"
    features_from, labels_from = (window, nowindow) if swapped else (nowindow, window)
    features = pd.read_csv(features_from).drop(columns="Target")
    labels = pd.read_csv(labels_from)[["CompanyID", "Target"]]
    return features.merge(labels, on="CompanyID")


def grid_runs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the sweep declaration into the list of runs it stands for.

    Only the parameters that declare more than one value are crossed — in the
    published sweep, the model family and the seed — and every other parameter is
    the same in every run. It is what the sweep agent would do, made explicit so
    that the runs can also happen without one.

    :param config: The parsed ``config.yaml``.
    :return: One dict of parameters per run.
    """
    declared = config["sweep_settings"]["parameters"]
    fixed = {k: v["values"][0] for k, v in declared.items() if len(v.get("values", [])) == 1}
    crossed = {k: v["values"] for k, v in declared.items() if len(v.get("values", [])) > 1}
    runs: list[dict[str, Any]] = [dict(fixed)]
    for name, values in crossed.items():
        runs = [{**run, name: value} for value in values for run in runs]
    return runs
