import polars as pl

from src.panel.config import PanelConfig
from src.panel.stage1_company import expand_years


def test_expand_years_produces_one_row_per_year_inclusive():
    df = pl.DataFrame({"CompanyID": ["a"], "YearFounded": [2010], "MaxYear": [2013]})
    out = expand_years(df, fix_negative_delta=False)
    assert out["Year_Delta"].to_list() == [2010, 2011, 2012, 2013]
    assert out["Delta"].to_list() == [0, 1, 2, 3]


def test_expand_years_counts_down_when_maxyear_precedes_founding():
    # R's seq(2010, 2008) walks backwards and yields negative Delta.
    df = pl.DataFrame({"CompanyID": ["a"], "YearFounded": [2010], "MaxYear": [2008]})
    out = expand_years(df, fix_negative_delta=False)
    assert out["Year_Delta"].to_list() == [2010, 2009, 2008]
    assert out["Delta"].to_list() == [0, -1, -2]


def test_expand_years_fix_flag_keeps_only_the_founding_year():
    df = pl.DataFrame({"CompanyID": ["a"], "YearFounded": [2010], "MaxYear": [2008]})
    out = expand_years(df, fix_negative_delta=True)
    assert out["Year_Delta"].to_list() == [2010]
    assert out["Delta"].to_list() == [0]


def test_stage1_outputs_have_the_expected_shape():
    """Integration: run stage 1 and check the invariants the R code guarantees."""
    cfg = PanelConfig()
    m1 = pl.read_parquet(cfg.interim("db_master_1_v1.parquet"))
    skel = pl.read_parquet(cfg.interim("db_master_2_skeleton.parquet"))
    assert m1.height == 116_920
    assert m1["CompanyID"].n_unique() == 116_920
    assert m1["YearFounded"].min() > 1999
    assert skel["YearFounded"].min() > 1999
    # R's descending seq() must have produced some negative Delta. The exact
    # 245 rows / 106 companies were measured on the FINAL db_master_2.csv, so
    # they are asserted at checkpoint C, not here: stage 2b's full_join can
    # still change the row count.
    assert skel.filter(pl.col("Delta") < 0).height > 0
