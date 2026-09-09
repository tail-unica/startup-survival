import polars as pl

from src.panel.config import PanelConfig
from src.panel.stage4_deals import RF_SUSPENDED
from src.panel.validate import load_reference, verify

#: Columns that stop changing at stage 4; script 2 never writes to them again.
FINAL_AT_STAGE_4 = [
    "DealType",
    "InvestorOwnership",
    "PremoneyValuation",
    "PreferredVerticals",
    "TR_D",
]


def test_columns_final_at_stage_4_match_the_reference():
    cfg = PanelConfig()
    keep = ["CompanyID", "Year_Delta", *FINAL_AT_STAGE_4]
    act = pl.read_parquet(cfg.interim("db_master_2_deals.parquet")).select(keep)
    ref = load_reference(cfg, "db_master_2.csv").select(keep)
    report = verify(act, ref, key=["CompanyID", "Year_Delta"], name="stage 4", rtol=cfg.rtol)
    print(report.render())
    report.assert_clean()


def test_the_random_forest_columns_are_not_produced():
    panel = pl.read_parquet(PanelConfig().interim("db_master_2_deals.parquet"))
    assert not set(RF_SUSPENDED) & set(panel.columns)


def test_undated_deals_are_parked_in_a_null_year_and_never_reach_the_panel():
    """R's pmax has no na.rm: a deal that never got a date keeps Year_Delta NA,
    lands in a null group and falls out at the join. Ignoring the null instead
    would park those deals on the founding year and invent deals in 7,335
    company-years."""
    cfg = PanelConfig()
    deals = pl.read_parquet(cfg.interim("deals_panel.parquet"))
    undated = deals.filter(pl.col("Year_Delta").is_null())
    assert undated.height > 0, "the null-year group should exist"
    panel = pl.read_parquet(cfg.interim("db_master_2_deals.parquet"))
    # None of those companies gained a deal on their founding year from this.
    assert panel.filter(pl.col("Year_Delta").is_null() & pl.col("N_Deal").is_not_null()).height == 0
    # TR_D is 1 exactly where the company-year carries no deal at all.
    no_deal = panel.filter(pl.col("TR_D") == 1)
    assert no_deal["N_Deal"].null_count() == no_deal.height
