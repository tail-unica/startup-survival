"""The whole panel pipeline, on the synthetic extraction.

``scripts/make_example_data.py`` writes fourteen invented tables, each company
built so that one branch of the pipeline fires. This module runs
:func:`src.panel.pipeline.build_panel` over them and states what the panel must say
about each company: not a stored file compared blindly, but the expected value of
the rules that matter.

It calls the same function the notebooks and the command line call, so nothing here
depends on the text of a notebook cell, and the whole module runs in about a second.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import yaml

from src.panel import checks
from src.panel.config import PanelConfig, PanelRules
from src.panel.pipeline import build_panel

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
RULES = PanelRules.from_config(CONFIG)
EXPECTED = CONFIG["panel"]["expected"]["example"]


def _build(tmp_path: Path, *, timed: bool) -> pl.DataFrame:
    """Run the pipeline over the synthetic tables.

    :param tmp_path: Directory for the per-phase outputs, so ``data/`` is untouched.
    :param timed: Whether the attributes are those of each row's own year.
    :return: The finished panel.
    """
    cfg = PanelConfig(raw_dir=ROOT / "data" / "example" / "pitchbook", interim_dir=tmp_path)
    panel, _ = build_panel(
        cfg, RULES, min_founding_year=int(CONFIG["first_year"]), timed=timed, write=False
    )
    return panel


@pytest.fixture(scope="module")
def panel(tmp_path_factory) -> pl.DataFrame:
    """The panel of the synthetic extraction, built once for the module."""
    return _build(tmp_path_factory.mktemp("timed"), timed=True)


@pytest.fixture(scope="module")
def snapshot_panel(tmp_path_factory) -> pl.DataFrame:
    """The same extraction with the timing off, for the columns that differ."""
    return _build(tmp_path_factory.mktemp("snapshot"), timed=False)


def _company(panel: pl.DataFrame, founding_year: int) -> pl.DataFrame:
    """The rows of the only company founded in that year, sorted by age.

    The pipeline renumbers the companies as its last step, so the identifiers of the
    fixture are gone by then; the founding years are distinct on purpose, and they
    are what the tests address a company by.

    :param panel: The panel.
    :param founding_year: Founding year of the wanted company.
    :return: Its rows.
    """
    rows = panel.filter(pl.col("YearFounded") == founding_year).sort("Age")
    how_many = rows["CompanyID"].n_unique()
    assert how_many == 1, f"companies founded in {founding_year}: {how_many}, not one"
    return rows


# ── the shape of the panel ─────────────────────────────────────────────────


def test_the_panel_has_one_row_per_company_year(panel):
    assert panel.height == EXPECTED["rows"]
    assert panel.select("CompanyID", "Age").n_unique() == panel.height


def test_the_panel_passes_its_own_checks(panel):
    report = checks.verify(panel, RULES, EXPECTED)
    assert report["ok"].all(), report


def test_the_company_founded_before_the_sample_threshold_is_absent(panel):
    # C6 is founded in 1999, and the sample starts in 2000.
    assert panel.filter(pl.col("YearFounded") == 1999).height == 0


def test_a_company_whose_data_predates_its_founding_keeps_year_zero_alone(panel):
    # C7 declares 2016 and carries a 2015 date: the panel cannot end before it
    # begins, so the company is left with its first year alone.
    assert _company(panel, 2016).select("Age").rows() == [(0,)]


# ── the dates of the rounds ────────────────────────────────────────────────


def test_an_undated_acquisition_takes_the_date_of_the_ownership_change(panel):
    # C2 is acquired in 2019 and its acquisition round carries no date: the panel
    # stops the year before, and the exit survives as the next stage.
    c2 = _company(panel, 2013)
    assert c2["Age"].max() == 5
    assert c2["GrowthNextStageGroup"].to_list()[-1] == "Exit"


def test_an_undated_bankruptcy_takes_the_date_of_the_ownership_change(panel):
    c3 = _company(panel, 2014)
    assert c3["Age"].max() == 3
    assert c3["GrowthNextStageGroup"].to_list()[-1] == "Out"


def test_an_undated_first_round_of_an_initial_type_goes_to_the_founding_year(panel):
    # C4's only round is a grant with no date: it lands on age zero, which is why
    # the company already has a stage in its first year.
    c4 = _company(panel, 2015)
    assert c4["GrowthStageGroup"].to_list()[0] == "Early"
    assert c4["N_Deal"].to_list()[0] == 1


def test_undated_rounds_between_two_dated_ones_are_spread_over_the_gap(panel):
    # C5 has rounds in 2012 and 2018 with two undated ones in between: they are
    # placed at 2014 and 2016, so the count grows one step at a time.
    c5 = _company(panel, 2011)
    assert c5.select("Age", "N_Deal").rows() == [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 2),
        (4, 2),
        (5, 3),
        (6, 3),
        (7, 4),
        (8, 4),
    ]


def test_the_stage_of_a_company_never_goes_backwards(panel):
    order = {"Early": 0, "Later": 1}
    for (_,), rows in panel.sort("Age").group_by("CompanyID", maintain_order=True):
        stages = [order[s] for s in rows["GrowthStageGroup"].drop_nulls()]
        assert stages == sorted(stages)


# ── the team ───────────────────────────────────────────────────────────────


def test_a_founder_counts_from_the_founding_year(panel):
    # P1 founded C1 and declares no start date.
    c1 = _company(panel, 2012)
    assert c1["Total_People"].to_list()[0] == 1
    assert c1["Total_Founders"].to_list()[0] == 1


def test_a_person_with_a_declared_start_does_not_count_before_it(panel):
    # P4 joins C1's board in 2015 (age 3) and P2 in 2016 (age 4).
    c1 = _company(panel, 2012)
    assert c1.select("Age", "Total_People").rows()[:5] == [(0, 1), (1, 1), (2, 1), (3, 2), (4, 3)]


def test_a_non_founder_without_a_start_date_is_never_counted(panel):
    # P3 is C1's chief financial officer with no start date: the team never reaches
    # four people, even though the raw table lists four.
    assert _company(panel, 2012)["Total_People"].max() == 3


def test_a_company_with_no_team_has_no_team_columns(panel):
    without = panel.filter(pl.col("Total_People").is_null())
    assert without.height == EXPECTED["rows"] - EXPECTED["with_team"]
    assert without["Percent_Females"].null_count() == without.height


def test_the_share_of_women_ignores_the_people_with_no_gender(panel):
    # C3's team is P8 (male) and P9 (no gender): the share is 0 over one person of
    # known gender, and not 0 over two.
    c3 = _company(panel, 2014)
    assert c3["Total_People"].max() == 2
    assert c3["Percent_Females"].drop_nulls().to_list() == [0.0] * 4


def test_the_education_of_a_person_changes_when_the_degree_is_awarded(panel):
    # P5 founded C2 with a bachelor's and earns a doctorate in 2017; C7 only spans
    # 2016, so there the same person still carries the bachelor's.
    assert _company(panel, 2016)["Highest_Degree_Mean"].to_list() == [3.0]


# ── the competitors ────────────────────────────────────────────────────────


def test_only_the_competitors_alive_that_year_are_counted(panel):
    # C1 declares three competitors: S1 alive throughout, S2 until 2016, S3 outside
    # this extraction. Timed, the count is two until 2016 and one after.
    c1 = _company(panel, 2012)
    assert c1.select("Age", "N_Competitors").rows() == [
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 2),
        (4, 2),
        (5, 1),
        (6, 1),
        (7, 1),
        (8, 1),
    ]


def test_a_similar_company_that_is_not_a_competitor_enters_the_mean_only(panel):
    # X1 is similar to C1 (score 40) without being a competitor, and it is alive
    # until 2012: in that year the mean covers S1, S2 and X1 while the count is two.
    c1 = _company(panel, 2012)
    assert c1["N_Similar"].to_list()[0] == 3
    assert c1["N_Competitors"].to_list()[0] == 2
    assert c1["SimilarityScoreMean"].to_list()[0] == pytest.approx((90 + 70 + 40) / 3)


def test_the_competitors_in_the_same_country_are_counted_apart(panel):
    # Of C1's two live competitors only S1 is Italian, like C1.
    assert _company(panel, 2012)["Same_Country"].to_list()[0] == 1


def test_a_company_whose_only_competitor_is_outside_the_extraction_counts_none(panel):
    # C8 declares S3 alone, and S3 has no life window in this extraction.
    c8 = _company(panel, 2010)
    assert c8["N_Competitors"].sum() == 0
    assert c8["SimilarityScoreMean"].sum() == 0.0
    # It does have a team, so what is missing is the counterpart, not the company.
    assert c8["Total_People"].is_not_null().all()


# ── the capital ────────────────────────────────────────────────────────────


def test_the_capital_of_a_year_without_rounds_is_zero(panel):
    c1 = _company(panel, 2012)
    assert c1["TotalRaised"].to_list()[0] == 0.0
    assert c1["UndisclosedAmountShare"].to_list()[0] == 0.0


def test_a_round_with_no_amount_declares_it(panel):
    # C1's 2018 round declares no amount: the capital of that year is zero and the
    # undisclosed share is one, which is what tells the two apart.
    row = _company(panel, 2012).filter(pl.col("Age") == 6)
    assert row["TotalRaised"].to_list() == [0.0]
    assert row["UndisclosedAmountShare"].to_list() == [1.0]


# ── the two configurations ─────────────────────────────────────────────────


def test_the_two_panels_carry_the_same_firm_years(panel, snapshot_panel):
    # The switches change values, never rows: the datasets swap the target between
    # the two panels on CompanyID, and that is only legitimate if this holds.
    keys = ["CompanyID", "Age"]
    assert panel.select(keys).equals(snapshot_panel.select(keys))


def test_the_snapshot_panel_counts_the_competitors_outside_the_extraction(panel, snapshot_panel):
    # With the timing off, C8's only competitor counts even though its life window
    # is unknown, and C1 keeps all three of its own in every year.
    assert _company(snapshot_panel, 2010)["N_Competitors"].unique().to_list() == [1]
    assert _company(snapshot_panel, 2012)["N_Competitors"].unique().to_list() == [3]
    assert _company(panel, 2012)["N_Competitors"].max() == 2


def test_the_snapshot_panel_freezes_the_education_of_a_person(panel, snapshot_panel):
    # P5 earns a doctorate in 2017, after C7's panel ends. Timed, C7 carries the
    # bachelor's; as a snapshot, it carries the doctorate she holds today.
    assert _company(panel, 2016)["Highest_Degree_Mean"].to_list() == [3.0]
    assert _company(snapshot_panel, 2016)["Highest_Degree_Mean"].to_list() == [5.0]
