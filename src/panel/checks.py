"""What has to be true of a finished panel.

Two kinds of check, and the difference matters.

**Structural** ones hold of any extraction, because they are the rules the
pipeline enforces: no year precedes a company's founding, no company-year appears
twice, the columns are the declared schema, nothing survives past an exit, and the
cumulative columns never go backwards. A structural check that fails is a bug, so
it raises.

**Counts** belong to the data. They are compared against the expectation recorded
in ``config.yaml`` for that extraction, and a deviation means "something changed,
understand what before using the result", not "there is a bug".

One implementation, used by the notebook, by the command line and by the tests: a
check that lived in three places would end up meaning three things.
"""

from __future__ import annotations

import polars as pl

from src.panel.config import PanelRules
from src.panel.target import STAY


def measure(panel: pl.DataFrame) -> dict[str, int]:
    """Count what the expectations are written in terms of.

    :param panel: A finished panel.
    :return: Rows, companies, rows carrying team data and rows carrying a stage.
    """
    return {
        "rows": panel.height,
        "companies": panel["CompanyID"].n_unique(),
        "with_team": int(panel["Total_People"].is_not_null().sum()),
        "with_stage": int(panel["GrowthStageGroup"].is_not_null().sum()),
    }


def check_structure(panel: pl.DataFrame, rules: PanelRules) -> None:
    """Check what holds of any panel, whatever the extraction.

    :param panel: A finished panel.
    :param rules: Domain rules; the schema and the terminal groups are read.
    :raises ValueError: On the first violation, naming it.
    """
    if panel.columns != rules.columns:
        missing = [c for c in rules.columns if c not in panel.columns]
        extra = [c for c in panel.columns if c not in rules.columns]
        raise ValueError(
            f"the columns are not the declared schema: missing {missing}, unexpected {extra}"
        )

    before_founding = panel.filter(pl.col("Age") < 0).height
    if before_founding:
        raise ValueError(f"{before_founding} rows precede the founding year of their company")

    if panel.select("CompanyID", "Age").n_unique() != panel.height:
        raise ValueError("some company-year appears on more than one row")

    # The truncation drops the row of the exit as well, so no terminal stage can
    # survive in the panel; the outcome lives in the future stage instead.
    terminal = panel.filter(pl.col("GrowthStageGroup").is_in(rules.terminal_groups)).height
    if terminal:
        raise ValueError(f"{terminal} rows carry a terminal stage: the truncation did not run")

    left_over = panel.filter(pl.col("GrowthNextStageGroup") == STAY).height
    if left_over:
        raise ValueError(f"{left_over} rows still carry the {STAY!r} placeholder")

    # Both columns accumulate over the life of a company, so neither can decrease.
    for column in ("N_Deal", "TotalInvestors"):
        decreasing = (
            panel.sort("CompanyID", "Age")
            .with_columns(pl.col(column).diff().over("CompanyID").alias("_step"))
            .filter(pl.col("_step") < 0)
            .height
        )
        if decreasing:
            raise ValueError(f"{column} decreases on {decreasing} rows, and it is cumulative")


def check_counts(panel: pl.DataFrame, expected: dict[str, int]) -> pl.DataFrame:
    """Compare the counts of a panel with what that extraction should produce.

    :param panel: A finished panel.
    :param expected: The expectation, from ``config.yaml``.
    :return: One row per count, with the value, the expectation and whether they
        agree.
    """
    measured = measure(panel)
    return pl.DataFrame(
        {
            "count": list(measured),
            "value": [measured[k] for k in measured],
            "expected": [expected.get(k) for k in measured],
        }
    ).with_columns(pl.col("value").eq(pl.col("expected")).alias("ok"))


def verify(
    panel: pl.DataFrame, rules: PanelRules, expected: dict[str, int] | None = None
) -> pl.DataFrame:
    """Run both kinds of check.

    :param panel: A finished panel.
    :param rules: Domain rules.
    :param expected: The counts that extraction should produce, if known.
    :return: The report of the counts, empty when no expectation was given.
    :raises ValueError: If a structural check fails.
    """
    check_structure(panel, rules)
    if expected is None:
        return pl.DataFrame(schema={"count": pl.String, "value": pl.Int64, "expected": pl.Int64})
    return check_counts(panel, expected)
