import polars as pl

from src.panel.config import PanelConfig
from src.panel.stage2_team import TEAM_COLUMNS
from src.panel.validate import load_reference, verify


def _panel() -> pl.DataFrame:
    return pl.read_parquet(PanelConfig().interim("db_master_2_team.parquet"))


def test_team_columns_already_match_the_final_reference():
    """These columns are final after stage 2; nothing downstream rewrites them."""
    cfg = PanelConfig()
    act = _panel()
    keep = ["CompanyID", "Year_Delta", "YearFounded", "Delta", *TEAM_COLUMNS]
    ref = load_reference(cfg, "db_master_2.csv").select(keep)
    report = verify(
        act.select(keep),
        ref,
        key=["CompanyID", "Year_Delta"],
        name="team columns",
        na_collapsed_columns={"Institute"},
        rtol=cfg.rtol,
    )
    print(report.render())
    report.assert_clean()


def test_full_join_lands_on_the_checkpoint_row_count():
    assert _panel().height == 1_001_625


def test_no_phantom_team_row():
    """A company-year matching nobody would score Total_People = 1 with every
    person field null. The `YearFounded > 2000` filter removes those rows
    before the aggregation ever sees them, so none can exist."""
    panel = _panel()
    phantom = panel.filter(
        (pl.col("Total_People") == 1)
        & pl.col("Percent_Females").is_null()
        & pl.col("Total_Founders").is_null()
    )
    assert phantom.height == 0


def test_cohort_2000_has_no_team_data():
    """Bug B1: the team panel filters YearFounded > 2000 while the skeleton
    filters > 1999, so the whole 2000 cohort arrives with null team columns."""
    panel = _panel().filter(pl.col("YearFounded") == 2000)
    assert panel.height > 0
    assert panel["Total_People"].null_count() == panel.height
