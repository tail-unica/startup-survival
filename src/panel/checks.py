"""What has to be true of a finished panel.

Every check here holds of any extraction, because they are the rules the pipeline
enforces: no year precedes a company's founding, no company-year appears twice,
the columns are the declared schema, nothing survives past an exit, and the
cumulative columns never go backwards. A check that fails is a bug, so it raises.

How many rows an extraction produces is not among them: that is a property of the
data, and a number that moves whenever the data changes cannot tell a bug from a
new release of the source.

One implementation, used by the notebook, by the command line and by the tests: a
check that lived in three places would end up meaning three things.
"""

from __future__ import annotations

import polars as pl

from src.panel.config import PanelRules
from src.panel.target import STAY


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

    # Every one of these accumulates over the life of a company, so none of them can
    # decrease: the two counts by cumulative sum, the flags by cumulative "any", which
    # is a cumulative maximum over booleans. The weighted means are not here on
    # purpose: a mean is free to go down.
    flags = [
        *rules.deal_flags,
        *(f"has_{name}" for name in rules.investor_flags),
        *(f"has_{name}_Lead" for name in rules.investor_flags),
    ]
    cumulative = ["N_Deal", "TotalInvestors", *(f for f in flags if f in panel.columns)]
    # One sort for every column rather than one each: on the real panel this check
    # runs over eight hundred thousand rows.
    steps = panel.sort("CompanyID", "Age").select(
        pl.col(column).cast(pl.Int64).diff().over("CompanyID").min().alias(column)
        for column in cumulative
    )
    backwards = [column for column in cumulative if (steps[column][0] or 0) < 0]
    if backwards:
        raise ValueError(f"these columns decrease, and they are cumulative: {backwards}")


def verify(panel: pl.DataFrame, rules: PanelRules) -> None:
    """Run every check on a finished panel.

    :param panel: A finished panel.
    :param rules: Domain rules.
    :raises ValueError: If a check fails, naming the violation.
    """
    check_structure(panel, rules)
