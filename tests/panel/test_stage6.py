import polars as pl

from src.panel.config import PanelConfig
from src.panel.rutils import next_different, rle_sequence
from src.panel.stage6_panel import STAGE_GROUP, STAY, TERMINAL_GROUPS
from src.panel.validate import CHECKPOINTS, run_checkpoint


def _apply_rules(stages: list[str | None]) -> pl.DataFrame:
    """The four rules on a hand-built company, in the order stage 6 applies
    them: group, next-stage on the untruncated sequence, truncate, re-run
    YearsInStage on the group."""
    df = pl.DataFrame({"CompanyID": ["c"] * len(stages), "GrowthStage": stages}).with_columns(
        pl.col("GrowthStage").replace_strict(STAGE_GROUP, default=None).alias("G")
    )
    df = next_different(df, "G", ["CompanyID"], "Next", "Time")
    rows_left = pl.len().over("CompanyID") - pl.int_range(pl.len()).over("CompanyID") - 1
    df = df.with_columns(
        pl.col("Next").fill_null(STAY).alias("Next"),
        pl.when(pl.col("Next").is_null()).then(rows_left).otherwise(pl.col("Time")).alias("Time"),
    )
    reached = (
        pl.col("G")
        .is_in(TERMINAL_GROUPS)
        .fill_null(False)
        .cast(pl.Int8)
        .cum_max()
        .over("CompanyID")
    )
    df = df.filter(reached == 0)
    return rle_sequence(df, "G", ["CompanyID"], "YearsInStage")


def test_the_four_rules_on_one_company():
    out = _apply_rules([None, "Preseed", "Seed", "LaterVC_or_Other", "Out", "Seed"])
    # The Out row and the Seed row after it are both gone.
    assert out.height == 4
    assert out["G"].to_list() == [None, "Early", "Early", "Later"]
    # Preseed's next different group is Later, not Early: Seed maps to Early too.
    assert out["Next"].to_list() == [STAY, "Later", "Later", "Out"]
    assert out["Time"].to_list() == [5, 2, 1, 1]
    # YearsInStage runs on the group, so Preseed and Seed are one run.
    assert out["YearsInStage"].to_list() == [1, 1, 2, 1]


def test_checkpoint_e_matches_the_recovered_reference():
    cfg = PanelConfig()
    report = run_checkpoint(cfg, "E")
    print(report.render())
    panel = pl.read_parquet(cfg.interim("db_master_panel.parquet"))
    assert panel.height == CHECKPOINTS["E"].expect_rows
    assert panel["CompanyID"].n_unique() == 116_327
    report.assert_clean()


def test_no_terminal_stage_survives_the_truncation():
    panel = pl.read_parquet(PanelConfig().interim("db_master_panel.parquet"))
    assert panel.filter(pl.col("GrowthStageGroup").is_in(TERMINAL_GROUPS)).height == 0
    # ...but they are still reachable as a future stage, which is the point of
    # computing the next stage before truncating.
    assert panel.filter(pl.col("GrowthNextStageGroup").is_in(TERMINAL_GROUPS)).height > 0
