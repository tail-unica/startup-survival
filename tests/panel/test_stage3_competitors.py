import polars as pl

from src.panel.config import PanelConfig
from src.panel.validate import CHECKPOINTS, load_reference, verify


def test_checkpoint_b_matches_the_r_reference():
    cfg = PanelConfig()
    cp = CHECKPOINTS["B"]
    actual = pl.read_parquet(cfg.interim(cp.interim))
    reference = load_reference(cfg, cp.reference)
    report = verify(actual, reference, key=cp.key, name="checkpoint B", rtol=cfg.rtol)
    print(report.render())
    assert actual.height == cp.expect_rows
    report.assert_clean()


def test_same_country_is_null_where_nothing_matched_and_something_was_missing():
    """Bug B4: any() without na.rm returns NA rather than FALSE."""
    m1 = pl.read_parquet(PanelConfig().interim("db_master_1.parquet"))
    assert m1["Same_Country"].null_count() == 1_809


def test_europe_counts_are_asymmetric():
    """Bug B8: N_Europe spans every similar company, N_Outside_Europe only
    those scoring above 90, so the second can be smaller for no real reason."""
    m1 = pl.read_parquet(PanelConfig().interim("db_master_1.parquet"))
    both = m1.filter(pl.col("N_Europe").is_not_null() & pl.col("N_Outside_Europe").is_not_null())
    assert both.filter(pl.col("N_Europe") > pl.col("N_Outside_Europe")).height > 0
