"""
Rebuilds the two processed datasets from the raw panel.

The notebook keeps these steps in commented-out cells because they are slow and
run once; this is the same pipeline as a script, so a change to preprocessing
can be replayed without editing the notebook.

The order matters and is not an implementation detail: dataset_nowindow is
built against dataset_window and then reduced to its columns, because every
cross-experiment comparison in the paper assumes the two carry the same firms
and the same features.

    python scripts/build_datasets.py
"""

import sys
from pathlib import Path

import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import (getCompleteDatasetWithoutTimeWindow,  # noqa: E402
                               getCompleteDatasetWithTimeWindow,
                               preprocessDataset)


def main():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "config" / "config.yaml").read_text())
    paths = config["paths"]

    print("reading the raw panel ...")
    panel = pl.read_csv(root / paths["raw_dataset"], null_values=["NA"])
    ranking = str(root / paths["raw_university_ranking"])

    print("building the time-window dataset ...")
    window = preprocessDataset(
        getCompleteDatasetWithTimeWindow(panel, int(config["time_window"]),
                                         int(config["last_year"])),
        ranking,
    )
    out_window = root / paths["dataset_window"]
    window.write_csv(out_window)
    print(f"  {out_window}: {window.shape[0]} rows x {window.shape[1]} columns")

    print("building the no-window dataset ...")
    nowindow = preprocessDataset(
        getCompleteDatasetWithoutTimeWindow(panel, window),
        ranking,
        flag_no_time_window=True,
    )
    # same columns, in the same order, as the window dataset
    nowindow = nowindow.select(window.columns)
    out_nowindow = root / paths["dataset_nowindow"]
    nowindow.write_csv(out_nowindow)
    print(f"  {out_nowindow}: {nowindow.shape[0]} rows x {nowindow.shape[1]} columns")


if __name__ == "__main__":
    main()
