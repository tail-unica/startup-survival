"""Stage 6 — stage grouping and truncation.

No R and no Python script exists for this step: the code that produced
`db_master_panel.csv.gz` was lost. The four rules below were reverse-engineered
from that file and each verified against it at zero divergence over all 882,324
rows, so they are the specification. See Task 16 of the implementation plan.
"""

from __future__ import annotations

import polars as pl

from src.panel.config import PanelConfig
from src.panel.rutils import next_different, rle_sequence

#: R1 — GrowthStage collapsed to four groups, null preserved.
STAGE_GROUP = {
    "Preseed": "Early",
    "Seed": "Early",
    "EarlyVC": "Early",
    "LaterVC_or_Other": "Later",
    "Out": "Out",
    "Exit_M&A": "Exit",
    "Exit_Public": "Exit",
}

TERMINAL_GROUPS = ["Out", "Exit"]

#: The value the producer writes when a company never reaches a different
#: group. The competitors notebook replaces it with the current group, and that
#: replacement belongs to stage 7, not here.
STAY = "Stay"


def run(cfg: PanelConfig) -> None:
    panel = pl.read_parquet(cfg.interim("db_final.parquet")).sort(["CompanyID", "Year_Delta"])

    # R1 --------------------------------------------------------------------
    panel = panel.with_columns(
        pl.col("GrowthStage").replace_strict(STAGE_GROUP, default=None).alias("GrowthStageGroup")
    )

    # R2 — computed on the UNTRUNCATED sequence. Doing it after the truncation
    # would make Out and Exit unreachable as a future stage, which is the whole
    # point of the column.
    panel = next_different(
        panel, "GrowthStageGroup", ["CompanyID"], "GrowthNextStageGroup", "TimeNextStageGroup"
    )
    rows_left = pl.len().over("CompanyID") - pl.int_range(pl.len()).over("CompanyID") - 1
    panel = panel.with_columns(
        pl.col("GrowthNextStageGroup").fill_null(STAY).alias("GrowthNextStageGroup"),
        # With no later different group the distance is to the company's last
        # untruncated row, not null.
        pl.when(pl.col("GrowthNextStageGroup").is_null())
        .then(rows_left)
        .otherwise(pl.col("TimeNextStageGroup"))
        .alias("TimeNextStageGroup"),
    )

    # R3 — drop everything from the first terminal row on, that row included.
    # This also removes the non-terminal rows that follow one: 5,365 of them,
    # and they are meant to go.
    reached_terminal = (
        pl.col("GrowthStageGroup")
        .is_in(TERMINAL_GROUPS)
        .fill_null(False)
        .cast(pl.Int8)
        .cum_max()
        .over("CompanyID")
    )
    panel = panel.filter(reached_terminal == 0)

    # R4 — YearsInStage is recomputed on the group, over the truncated frame,
    # replacing the value stage 5 computed on the ungrouped GrowthStage.
    panel = panel.drop("YearsInStage")
    panel = rle_sequence(panel, "GrowthStageGroup", ["CompanyID"], "YearsInStage")

    # StageBlock is carried through unchanged: the value in the reference
    # matches neither a recomputation on GrowthStage nor one on the group nor
    # db_selected's own, and nothing downstream reads it.
    panel.write_parquet(cfg.interim("db_master_panel.parquet"))
