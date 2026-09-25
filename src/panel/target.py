"""Phase 6: the stage groups, the future stage, and the truncation at the exit.

The seven growth stages collapse into four, the stage each company moves to next
is computed, and the panel is cut at the exit. The order of the last two is a
constraint, not a preference: computing the future stage after the truncation
would make an exit unreachable as a future stage, which is exactly what the column
is for.
"""

from __future__ import annotations

import polars as pl

from src.panel.config import PanelRules
from src.panel.expressions import next_different

#: The value the future-stage column carries while it means "this company does not
#: leave the group it is in". Phase 7 replaces it with the current group.
STAY = "Stay"


def group_stages(panel: pl.DataFrame, rules: PanelRules) -> pl.DataFrame:
    """Collapse the seven growth stages into four groups.

    A null stage stays null rather than becoming a category of its own.

    :param panel: Output of :func:`src.panel.stages.attach_registry`.
    :param rules: Domain rules; the stage groups are read.
    :return: The panel sorted by company and year, with ``GrowthStageGroup`` in
        place of the seven-valued stage.
    """
    return (
        panel.sort(["CompanyID", "Year_Delta"])
        .with_columns(
            pl.col("GrowthStage")
            .replace_strict(rules.stage_groups, default=None)
            .alias("GrowthStageGroup")
        )
        .drop("GrowthStage")
    )


def next_stage(panel: pl.DataFrame) -> pl.DataFrame:
    """Add the group each company moves to next, and in how many years.

    A company that never leaves its group carries a placeholder instead, and as a
    distance the number of years left in its panel. The two expressions sit in the
    same statement on purpose: the second reads the future stage *before* the first
    one fills it in.

    :param panel: Output of :func:`group_stages`.
    :return: The panel with ``GrowthNextStageGroup`` and ``TimeNextStageGroup``.
    """
    panel = next_different(
        panel, "GrowthStageGroup", ["CompanyID"], "GrowthNextStageGroup", "TimeNextStageGroup"
    )
    rows_left = pl.len().over("CompanyID") - pl.int_range(pl.len()).over("CompanyID") - 1
    return panel.with_columns(
        pl.col("GrowthNextStageGroup").fill_null(STAY).alias("GrowthNextStageGroup"),
        pl.when(pl.col("GrowthNextStageGroup").is_null())
        .then(rows_left)
        .otherwise(pl.col("TimeNextStageGroup"))
        .alias("TimeNextStageGroup"),
    )


def truncate_at_exit(panel: pl.DataFrame, rules: PanelRules) -> pl.DataFrame:
    """Cut each company's panel at its exit, the year of the exit included.

    Dropping that year too is not an oversight: it is the condition for the target
    to be correct. The outcome survives in the future stage, which was computed
    before the cut, so the last row left says "about to leave, in N years". Keeping
    the row of the exit would move the year the target is read at one step forward,
    onto that very row, where the future stage points at the rows *after* the exit,
    and hundreds of companies would come out as never having left.

    The few non-terminal rows that follow an exit go as well, companies given up
    for dead that raise another round, or acquired ones that keep raising. That is
    consistent with treating the exit as an absorbing state, and is worth declaring.

    :param panel: Output of :func:`next_stage`.
    :param rules: Domain rules; the terminal groups are read.
    :return: The truncated panel.
    """
    # cum_max over 0/1: one from the first terminal row onwards, so keeping the
    # zeros drops the row of the exit as well.
    reached_terminal = (
        pl.col("GrowthStageGroup")
        .is_in(rules.terminal_groups)
        .fill_null(False)
        .cast(pl.Int8)
        .cum_max()
        .over("CompanyID")
    )
    return panel.filter(reached_terminal == 0)
