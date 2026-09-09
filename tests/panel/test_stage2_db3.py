import polars as pl

from src.panel.config import PanelConfig
from src.panel.validate import CHECKPOINTS, load_reference, verify


def test_checkpoint_a_matches_the_r_reference():
    cfg = PanelConfig()
    cp = CHECKPOINTS["A"]
    actual = pl.read_parquet(cfg.interim(cp.interim))
    reference = load_reference(cfg, cp.reference)
    report = verify(actual, reference, key=cp.key, name="checkpoint A")
    print(report.render())
    assert actual.height == cp.expect_rows
    report.assert_clean()


def test_is_out_is_null_for_companies_still_in_business():
    """Bug B10: the left join leaves Is_Out null, not False."""
    cfg = PanelConfig()
    db3 = pl.read_parquet(cfg.interim("db3.parquet"))
    assert db3["Is_Out"].null_count() > 0


def test_is_other_is_never_true():
    """Bug B2: Field never takes the value the comparison looks for."""
    cfg = PanelConfig()
    db3 = pl.read_parquet(cfg.interim("db3.parquet"))
    assert db3.filter(pl.col("Is_Other")).height == 0
