"""The six experiments: which dataset each one reads, and how it is assembled.

Two of them are the datasets as they come out of the preprocessing. Two are
ablations, the same bias-controlled dataset with a family of features removed. The
last two are the controls that take the 2x2 apart: the features of one setting
against the target of the other, swapped on ``CompanyID``.

The notebook and the command line call the same functions here, so an experiment
cannot mean one thing in one place and something else in the other.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

#: What each setting is, in one line.
SETTINGS: dict[str, str] = {
    "controlled": "bias-controlled: attributes of the row's own year, features at the starting age",
    "leakboth": "look-ahead: attributes as of the extraction, features over the whole life",
    "noteam": "controlled without the team features",
    "nocompetitors": "controlled without the competitor features",
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
    controlled = paths["dataset_controlled"]
    leakboth = paths["dataset_leakboth"]

    if setting == "controlled":
        return pd.read_csv(controlled)
    if setting == "leakboth":
        return pd.read_csv(leakboth)
    if setting in ("noteam", "nocompetitors"):
        dataset = pd.read_csv(controlled)
        return dataset.drop(columns=ablations[setting])

    # The two controls: same firms, same columns, the other definition of the target.

    swapped = setting == "leaklabel"
    features_from, labels_from = (controlled, leakboth) if swapped else (leakboth, controlled)
    features = pd.read_csv(features_from).drop(columns="Target")
    labels = pd.read_csv(labels_from)[["CompanyID", "Target"]]
    return features.merge(labels, on="CompanyID")


def split_strata(dataset: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    """The labels every experiment stratifies its split on: the controlled target.

    Stratifying each setting on its own target would draw different firms into the
    test set whenever the two targets disagree, even with the same seed and the same
    rows. The comparisons between settings, and the paired tests on their SHAP
    values, need the same firms in the same sets, so every setting stratifies on the
    target of ``controlled``.

    :param dataset: The dataset of one setting, from :func:`load_setting`.
    :param config: The parsed ``config.yaml``.
    :return: The controlled target, row for row with ``dataset``.
    :raises ValueError: If the dataset does not carry the firms of ``controlled`` in its order.
    """
    controlled = pd.read_csv(config["paths"]["dataset_controlled"], usecols=["CompanyID", "Target"])
    if not dataset["CompanyID"].reset_index(drop=True).equals(controlled["CompanyID"]):
        raise ValueError(
            "the dataset does not carry the firms of the controlled one in the same order: "
            "the split cannot be shared"
        )
    return controlled["Target"]


def grid_runs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the sweep declaration into the list of runs it stands for.

    Only the parameters that declare more than one value are crossed and every other parameter is
    the same in every run.

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


def grouped_variants(config: dict[str, Any]) -> list[dict[str, Any]]:
    """The runs of the grouped-preprocessing check, from ``grouped_preprocessing`` in the config.

    One variant per setting, grouping column and fitted step: the imputer fitted per
    group (``impute``), the scaler (``scale``), or both. A variant reads the dataset of
    its setting and draws its splits, so its reference is the setting itself. The
    ``scale`` variants keep only the families fitted on scaled data: the others never
    see the scaler, and would repeat the reference runs.

    :param config: The parsed ``config.yaml``.
    :return: One dict per variant: ``tag`` (``<setting>_<fit>-<group>``), ``setting``,
        the ``split`` options for ``get_split`` and the ``config`` to run it with.
    :raises ValueError: If ``fit`` names a step other than impute, scale or both.
    """
    from src.models import MODELS

    grouped = config["grouped_preprocessing"]
    unknown = set(grouped["fit"]) - {"impute", "scale", "both"}
    if unknown:
        raise ValueError(f"unknown grouped step(s) {sorted(unknown)}; expected impute, scale, both")
    models = config["sweep_settings"]["parameters"]["model_type"]["values"]
    scaled = [m for m in models if MODELS[m].wants_scaled]
    variants = []
    for setting in grouped["settings"]:
        for group, column in grouped["groups"].items():
            for fit in grouped["fit"]:
                sweep = config["sweep_settings"]
                parameters = {
                    **sweep["parameters"],
                    "model_type": {"values": scaled if fit == "scale" else models},
                }
                variants.append(
                    {
                        "tag": f"{setting}_{fit}-{group}",
                        "setting": setting,
                        "split": {
                            "impute_by": column if fit in ("impute", "both") else None,
                            "scale_by": column if fit in ("scale", "both") else None,
                        },
                        "config": {**config, "sweep_settings": {**sweep, "parameters": parameters}},
                    }
                )
    return variants
