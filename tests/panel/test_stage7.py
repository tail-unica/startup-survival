import polars as pl

from src.panel.config import PanelConfig
from src.panel.validate import CHECKPOINTS, COMPETITOR_VINTAGE_COLUMNS, run_checkpoint


def _panel() -> pl.DataFrame:
    return pl.read_parquet(PanelConfig().interim("panel.parquet"))


def test_checkpoint_f_matches_apart_from_the_competitor_vintage():
    cfg = PanelConfig()
    report = run_checkpoint(cfg, "F")
    print(report.render())
    assert _panel().height == CHECKPOINTS["F"].expect_rows
    report.assert_clean()


def test_the_competitor_columns_are_the_only_unexplained_gap():
    """Everything but the six competitor columns and StageBlock reproduces the
    published panel exactly; those six come from a different download of
    CompanySimilarRelation.csv, which no change here can recover."""
    report = run_checkpoint(PanelConfig(), "F")
    differing = {c.column for c in report.columns if c.n_diff}
    assert differing == COMPETITOR_VINTAGE_COLUMNS | {"StageBlock"}


def test_stay_is_replaced_by_the_current_group():
    panel = _panel()
    assert panel.filter(pl.col("GrowthNextStageGroup") == "Stay").height == 0
    assert panel["GrowthNextStageGroup"].null_count() == panel["GrowthStageGroup"].null_count()


def test_the_dropped_columns_are_gone_and_company_ids_are_integers():
    panel = _panel()
    assert "N_Europe" not in panel.columns
    assert "N_Outside_Europe" not in panel.columns
    assert panel["CompanyID"].dtype in (pl.Int64, pl.UInt32, pl.Int32, pl.UInt64)
    assert panel["CompanyID"].min() == 1
    assert panel["CompanyID"].n_unique() == 116_327
