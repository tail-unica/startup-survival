"""The pipeline from the command line, one stage at a time or all of them.

The notebooks walk through the same functions with the explanation of each step
beside them; this is the same pipeline without the narration, for who only wants
to run it. There is no second implementation: every subcommand calls what the
notebooks call.

    python scripts/pipeline.py tables
    python scripts/pipeline.py panel --both
    python scripts/pipeline.py datasets
    python scripts/pipeline.py experiments --setting controlled
    python scripts/pipeline.py all --example

Run ``--help`` on any subcommand for its options. Nothing here needs a Weights &
Biases account: the experiments log there only with ``--wandb``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.panel import checks  # noqa: E402
from src.panel.config import PanelConfig, PanelRules  # noqa: E402
from src.panel.pipeline import build_panel, panel_name  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

#: Where each extraction lives. The example writes everything under
#: ``data/example/``, the released datasets included: a demo run must not be able
#: to overwrite the files the paper reports on.
EXTRACTIONS = {
    "pitchbook": {
        "raw": "data/raw/pitchbook",
        "interim": "data/interim",
        "processed": "data/processed",
    },
    "example": {
        "raw": "data/example/pitchbook",
        "interim": "data/example/interim",
        "processed": "data/example/processed",
    },
}


def load_config() -> dict[str, Any]:
    """Read ``config/config.yaml``.

    :return: The parsed configuration.
    """
    return yaml.safe_load((ROOT / "config" / "config.yaml").read_text())


def paths_for(args: argparse.Namespace) -> tuple[PanelConfig, str]:
    """The paths of the chosen extraction.

    :param args: Parsed arguments; ``--example`` and the two path overrides are
        read.
    :return: The pipeline paths and the name of the extraction.
    """
    name = "example" if args.example else "pitchbook"
    where = EXTRACTIONS[name]
    return (
        PanelConfig(
            raw_dir=args.raw_dir or ROOT / where["raw"],
            interim_dir=args.interim_dir or ROOT / where["interim"],
        ),
        name,
    )


def with_dataset_paths(config: dict[str, Any], extraction: str) -> dict[str, Any]:
    """The configuration, with the processed datasets pointed at that extraction.

    :param config: The parsed configuration.
    :param extraction: ``"pitchbook"`` or ``"example"``.
    :return: A copy whose ``dataset_controlled`` and ``dataset_leakboth`` live in the
        directory of that extraction.
    """
    where = EXTRACTIONS[extraction]["processed"]
    paths = dict(config["paths"])
    for name in ("dataset_controlled", "dataset_leakboth"):
        paths[name] = str(ROOT / where / Path(config["paths"][name]).name)
    return {**config, "paths": paths}


def cmd_tables(args: argparse.Namespace) -> int:
    """Write the synthetic extraction the example runs read.

    :param args: Parsed arguments.
    :return: Exit code.
    """
    from scripts.make_example_data import main as write_tables

    return write_tables()


def cmd_panel(args: argparse.Namespace) -> int:
    """Build one panel, or both.

    :param args: Parsed arguments; ``--timed``, ``--snapshot`` and ``--both``
        choose the configuration.
    :return: Exit code.
    """
    config = load_config()
    rules = PanelRules.from_config(config)
    cfg, extraction = paths_for(args)

    wanted = [True, False] if args.both else [not args.snapshot]
    for timed in wanted:
        label = "timed" if timed else "snapshot"
        print(f"\n-- {label} panel, from {cfg.raw_dir} --")
        panel, report = build_panel(
            cfg,
            rules,
            min_founding_year=int(config["first_year"]),
            timed=timed,
            write=True,
        )
        print(report)
        checks.verify(panel, rules)
        print(f"written: {cfg.interim(panel_name(timed=timed) + '.csv.gz')}")
        if extraction == "example":
            # The released copy of the example panel: uncompressed, beside the
            # extraction it comes from, so it can be opened without running
            # anything. tests/test_example_pipeline.py checks it is current.
            released = ROOT / "data" / "example" / f"{panel_name(timed=timed)}.csv"
            panel.write_csv(released)
            print(f"released copy: {released}")
    return 0


def cmd_datasets(args: argparse.Namespace) -> int:
    """Build the two processed datasets from the two panels.

    :param args: Parsed arguments; ``--missing-threshold`` and
        ``--max-missing-per-row`` override the configuration for this run.
    :return: Exit code.
    """
    from src.preprocessing import build_processed_datasets

    cfg, extraction = paths_for(args)
    config = with_dataset_paths(load_config(), extraction)
    pre = config["preprocessing"]
    ranking = str(ROOT / config["paths"]["raw_university_ranking"])

    print(f"reading the two panels from {cfg.interim_dir} ...")
    timed = pl.read_csv(cfg.interim("panel_timed.csv.gz"), null_values=["NA"])
    snapshot = pl.read_csv(cfg.interim("panel_snapshot.csv.gz"), null_values=["NA"])

    controlled, leakboth = build_processed_datasets(
        timed,
        snapshot,
        ranking,
        T=int(config["T"]),
        last_year=int(config["last_year"]),
        first_decision_year=int(config["first_decision_year"]),
        max_starting_age=int(config["max_starting_age"]),
        missing_threshold=float(pre["missing_threshold"])
        if args.missing_threshold is None
        else args.missing_threshold,
        max_missing_per_row=int(pre["max_missing_per_row"])
        if args.max_missing_per_row is None
        else args.max_missing_per_row,
    )
    for name, dataset in (("dataset_controlled", controlled), ("dataset_leakboth", leakboth)):
        out = ROOT / config["paths"][name]
        out.parent.mkdir(parents=True, exist_ok=True)
        dataset.write_csv(out)
        share = dataset["Target"].mean()
        print(f"  {out}: {dataset.height} rows x {dataset.width} columns, prevalence {share:.3f}")
    return 0


def cmd_experiments(args: argparse.Namespace) -> int:
    """Train the seven model families on one experiment, or on all of them.

    :param args: Parsed arguments; ``--setting``, ``--wandb`` and ``--runs`` are
        read.
    :return: Exit code.
    """
    from src.experiments import SETTINGS, load_setting, split_strata
    from src.training import make_train, run_grid
    from src.utils import summarize_metrics

    _, extraction = paths_for(args)
    config = with_dataset_paths(load_config(), extraction)
    settings = list(SETTINGS) if args.setting == "all" else [args.setting]
    frequency = config["frequency_encoding"]
    split_kwargs = {
        "test_size": config["test_size"],
        "cache_dir": str(ROOT / "tmp" / "splits"),
        "categorical_columns": frequency["columns"],
        "min_frequency": frequency["min_frequency"],
        "other_label": frequency["other_label"],
    }
    metrics_store: dict = {}
    shap_store: dict = {}

    for setting in settings:
        dataset = load_setting(setting, config)
        X = dataset.drop(["CompanyID", "Target"], axis=1)
        y = dataset["Target"]
        setting_split = {**split_kwargs, "stratify": split_strata(dataset, config)}
        print(f"\n── {setting}: {SETTINGS[setting]}")
        print(f"   {len(dataset)} firms, prevalence {y.mean():.3f}")

        if args.wandb:
            import os

            import wandb
            from dotenv import load_dotenv

            load_dotenv()
            sweep_id = wandb.sweep(
                config["sweep_settings"],
                entity=os.getenv("entity"),
                project=os.getenv("project"),
            )
            train = make_train(setting, X, y, config, setting_split, metrics_store, shap_store)
            wandb.agent(sweep_id, function=train, count=args.runs)
        else:
            run_grid(setting, X, y, config, setting_split, metrics_store, shap_store)

    print("\n── results ──")
    for setting in settings:
        print(f"\n{setting}")
        print(summarize_metrics(metrics_store, setting))
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    """Run every stage in order: tables when needed, panels, datasets, experiments.

    :param args: Parsed arguments.
    :return: Exit code.
    """
    if args.example:
        cmd_tables(args)
    args.both, args.snapshot = True, False
    cmd_panel(args)
    cmd_datasets(args)
    if not args.skip_experiments:
        args.setting = args.setting or "controlled"
        cmd_experiments(args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Declare the command line.

    :return: The parser.
    """
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Build the panels, the datasets and the experiment results.",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--example",
        action="store_true",
        help="read the synthetic extraction in data/example/pitchbook instead of the "
        "PitchBook one, and write beside it; the only route that needs no licensed data",
    )
    common.add_argument(
        "--raw-dir", type=Path, default=None, help="override the directory of the raw tables"
    )
    common.add_argument(
        "--interim-dir", type=Path, default=None, help="override the directory of the outputs"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_tables = sub.add_parser(
        "tables", parents=[common], help="write the synthetic extraction (14 CSV files)"
    )
    p_tables.set_defaults(func=cmd_tables)

    p_panel = sub.add_parser("panel", parents=[common], help="build the panel from the raw tables")
    group = p_panel.add_mutually_exclusive_group()
    group.add_argument(
        "--timed",
        action="store_true",
        help="attributes of each row's own year (the default): writes panel_timed.parquet/.csv.gz",
    )
    group.add_argument(
        "--snapshot",
        action="store_true",
        help="attributes as declared at extraction time, the panel that carries the "
        "look-ahead: writes panel_snapshot.parquet/.csv.gz",
    )
    group.add_argument("--both", action="store_true", help="build both, one after the other")
    p_panel.set_defaults(func=cmd_panel)

    p_datasets = sub.add_parser(
        "datasets", parents=[common], help="build the two processed datasets from the two panels"
    )
    p_datasets.add_argument(
        "--missing-threshold",
        type=float,
        default=None,
        help="share of missing values past which a feature is dropped (config: 0.5)",
    )
    p_datasets.add_argument(
        "--max-missing-per-row",
        type=int,
        default=None,
        help="a row with this many missing values or more is dropped (config: 6)",
    )
    p_datasets.set_defaults(func=cmd_datasets)

    p_exp = sub.add_parser(
        "experiments", parents=[common], help="train the seven model families on one experiment"
    )
    p_exp.add_argument(
        "--setting",
        default="controlled",
        choices=[
            "controlled",
            "leakboth",
            "noteam",
            "nocompetitors",
            "leaklabel",
            "leakfeat",
            "all",
        ],
        help="which experiment to run (default: controlled)",
    )
    p_exp.add_argument(
        "--wandb",
        action="store_true",
        help="log to Weights & Biases and let a sweep agent drive the runs; without it "
        "the same runs happen locally and the metrics are printed here",
    )
    p_exp.add_argument(
        "--runs", type=int, default=35, help="how many runs the sweep agent draws (default: 35)"
    )
    p_exp.set_defaults(func=cmd_experiments)

    p_all = sub.add_parser("all", parents=[common], help="every stage, in order")
    p_all.add_argument("--setting", default=None, help="experiment to run at the end")
    p_all.add_argument("--skip-experiments", action="store_true", help="stop after the datasets")
    p_all.add_argument("--wandb", action="store_true", help="log the experiments to W&B")
    p_all.add_argument("--runs", type=int, default=35)
    p_all.add_argument("--missing-threshold", type=float, default=None)
    p_all.add_argument("--max-missing-per-row", type=int, default=None)
    p_all.set_defaults(func=cmd_all)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse the arguments and run the chosen subcommand.

    :param argv: Arguments, defaulting to the process ones.
    :return: Exit code.
    """
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
