import polars as pl

from src.panel.config import PanelConfig
from src.panel.validate import CHECKPOINTS, run_checkpoint


def _report(letter: str):
    report = run_checkpoint(PanelConfig(), letter)
    print(report.render())
    return report


def test_checkpoint_c_matches_the_r_reference():
    report = _report("C")
    assert (
        pl.read_parquet(PanelConfig().interim("db_master_2.parquet")).height
        == CHECKPOINTS["C"].expect_rows
    )
    report.assert_clean()


def test_checkpoint_d_matches_the_r_reference():
    report = _report("D")
    assert (
        pl.read_parquet(PanelConfig().interim("db_selected.parquet")).height
        == CHECKPOINTS["D"].expect_rows
    )
    report.assert_clean()


def test_db_final_is_db_selected_plus_company_columns_without_duplicating_rows():
    """db_final has no reference of its own: it is verified by construction,
    so the only thing left to check is that the join did not fan out."""
    cfg = PanelConfig()
    selected = pl.read_parquet(cfg.interim("db_selected.parquet"))
    final = pl.read_parquet(cfg.interim("db_final.parquet"))
    assert final.height == selected.height
    assert set(selected.columns) <= set(final.columns)


def test_cumulative_flags_never_switch_back_off():
    """cumany latches: once a company has reached a stage it keeps the flag."""
    panel = pl.read_parquet(PanelConfig().interim("db_master_2.parquet")).sort(
        ["CompanyID", "Year_Delta"]
    )
    for flag in ("Is_Seed", "Is_EarlyVC", "Is_Out"):
        went_back = panel.filter(pl.col(flag).shift(1).over("CompanyID") & ~pl.col(flag))
        assert went_back.height == 0, flag


def test_total_raised_cumulative_stops_at_the_first_missing_value():
    """R's cumsum propagates NA to the end of the group; polars' does not, so
    this is the column where the two would silently disagree."""
    panel = pl.read_parquet(PanelConfig().interim("db_master_2.parquet")).sort(
        ["CompanyID", "Year_Delta"]
    )
    broken = panel.filter(
        pl.col("TotalRaised_any_cum").is_null().shift(1).over("CompanyID")
        & pl.col("TotalRaised_any_cum").is_not_null()
    )
    assert broken.height == 0
