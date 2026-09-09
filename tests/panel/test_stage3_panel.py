import polars as pl

from src.panel.config import PanelConfig
from src.panel.validate import run_partial


def _panel() -> pl.DataFrame:
    return pl.read_parquet(PanelConfig().interim("db_master_2_relations.parquet"))


def test_news_count_is_never_null():
    assert _panel()["N_News"].null_count() == 0


def test_partial_check_reports_no_regression():
    """Every column already final at stage 1, 2 or 3 must still match."""
    report = run_partial(PanelConfig(), 3, _panel())
    print(report.render())
    regressions = [c.column for c in report.columns if c.n_diff and not c.expected]
    assert not regressions, regressions


def test_financials_only_fill_empty_cells():
    """The financial table coalesces into db_master_2; it never overwrites a
    value the company table had already put there."""
    cfg = PanelConfig()
    before = pl.read_parquet(cfg.interim("db_master_2_team.parquet")).select(
        "CompanyID", "Year_Delta", "Revenue"
    )
    after = _panel().select("CompanyID", "Year_Delta", "Revenue")
    joined = before.join(after, on=["CompanyID", "Year_Delta"], suffix="_after")
    overwritten = joined.filter(
        pl.col("Revenue").is_not_null() & (pl.col("Revenue") != pl.col("Revenue_after"))
    )
    assert overwritten.height == 0
