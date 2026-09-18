"""
Rebuilds the two processed datasets from the two panels.

The notebook keeps these steps in commented-out cells because they are slow and
run once; this is the same pipeline as a script, so a change to preprocessing
can be replayed without editing the notebook.

The two datasets differ in everything the paper calls look-ahead bias, and each
one reads the panel built for it:

- ``dataset_window`` comes from the **timed** panel, and every feature is read at
  the age the firm entered its first early stage;
- ``dataset_nowindow`` comes from the **snapshot** panel, where the attributes are
  the ones declared at extraction time, and every feature is cumulated over the
  firm's whole observed life.

The order matters and is not an implementation detail: dataset_nowindow is built
against dataset_window and then reduced to its rows and columns, because every
cross-experiment comparison in the paper assumes the two carry the same firms and
the same features.

    python scripts/build_datasets.py
"""

import sys
from pathlib import Path

import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import (  # noqa: E402
    build_full_history_dataset,
    build_windowed_dataset,
    preprocess_dataset,
)


def main():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "config" / "config.yaml").read_text())
    paths = config["paths"]

    ranking = str(root / paths["raw_university_ranking"])

    print("reading the timed panel ...")
    timed = pl.read_csv(root / paths["panel_timed"], null_values=["NA"])
    print("reading the snapshot panel ...")
    snapshot = pl.read_csv(root / paths["panel_snapshot"], null_values=["NA"])

    # The firms are the same in both panels, and so is their numbering: the
    # switches change values, never rows. Swapping the target between the two
    # datasets on CompanyID depends on it, so it is checked rather than assumed.
    keys = ["CompanyID", "Age"]
    if not timed.select(keys).equals(snapshot.select(keys)):
        raise ValueError(
            "the two panels do not carry the same firm-years: rebuild them from "
            "the same extraction, flipping only the switches"
        )

    print("building the time-window dataset ...")
    window = preprocess_dataset(
        build_windowed_dataset(timed, int(config["time_window"]), int(config["last_year"])),
        ranking,
    )
    out_window = root / paths["dataset_window"]
    window.write_csv(out_window)
    print(f"  {out_window}: {window.shape[0]} rows x {window.shape[1]} columns")

    print("building the no-window dataset ...")
    nowindow = preprocess_dataset(
        build_full_history_dataset(snapshot, window),
        ranking,
        flag_no_time_window=True,
    )
    # Same firms, same columns, same order as the window dataset: the comparisons
    # of the paper are between the same companies described in two ways.
    nowindow = nowindow.filter(pl.col("CompanyID").is_in(window["CompanyID"].implode()))
    nowindow = nowindow.select(window.columns)
    out_nowindow = root / paths["dataset_nowindow"]
    nowindow.write_csv(out_nowindow)
    print(f"  {out_nowindow}: {nowindow.shape[0]} rows x {nowindow.shape[1]} columns")


if __name__ == "__main__":
    main()
