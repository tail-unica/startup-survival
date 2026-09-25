"""The two expansions: which rows they create, and which ones they must not.

The decisions under test are the ones the panel depends on: a person is counted
only in the years their window covers, a person whose arrival is unknown is not
counted at all, and a competitor counts only while it is alive.
"""

import polars as pl

from src.panel.expansions import active_pairs, expand_team


def _team_windows(**columns) -> pl.DataFrame:
    """One row per (company, person), with the defaults the tests override."""
    base = {
        "CompanyID": ["c"],
        "PersonID": ["p"],
        "YearFounded": [2010],
        "DeltaStart": [0],
        "DeltaEnd": [2],
    }
    base.update(columns)
    return pl.DataFrame(base)


def test_a_person_appears_once_per_year_of_their_window():
    out = expand_team(_team_windows(), min_founding_year=2000)
    assert out["Years"].to_list() == [0, 1, 2]
    assert out["PersonID"].to_list() == ["p"] * 3


def test_a_person_is_absent_from_the_years_before_they_arrive():
    windows = _team_windows(
        CompanyID=["c", "c"],
        PersonID=["founder", "hire"],
        YearFounded=[2010, 2010],
        DeltaStart=[0, 2],
        DeltaEnd=[3, 3],
    )
    out = expand_team(windows, min_founding_year=2000).sort("Years", "PersonID")
    per_year = {
        year: sorted(g["PersonID"].drop_nulls().to_list())
        for year, g in out.group_by("Years", maintain_order=True)
    }
    assert per_year[(0,)] == ["founder"]
    assert per_year[(2,)] == ["founder", "hire"]


def test_a_person_without_an_arrival_year_is_never_counted():
    # A non-founder with no StartDate is not dated to the founding year, because
    # counting them from year zero would put people hired later into the team of
    # the first years.
    windows = _team_windows(
        CompanyID=["c", "c"],
        PersonID=["dated", "undated"],
        YearFounded=[2010, 2010],
        DeltaStart=[0, None],
        DeltaEnd=[2, 2],
    )
    out = expand_team(windows, min_founding_year=2000)
    assert out["PersonID"].drop_nulls().unique().to_list() == ["dated"]
    # The company-year grid still spans the years of the people who are dated.
    assert sorted(out["Years"].to_list()) == [0, 1, 2]


def test_a_company_whose_whole_team_is_undated_keeps_no_person_row():
    out = expand_team(_team_windows(DeltaStart=[None]), min_founding_year=2000)
    # The grid has nothing to span, and the filter on YearFounded (which
    # arrives from the person side) removes the empty row, so the company-year
    # is never counted as one person.
    assert out.height == 0


def test_a_person_with_no_departure_counts_for_their_arrival_year_alone():
    # The person-side grid falls back to the arrival year, so an unknown
    # departure does not stretch the presence to the end of the company's life.
    # A colleague provides the company-year grid; the departures are imputed
    # before this point, so the case is a guard rather than a live path.
    windows = _team_windows(
        CompanyID=["c", "c"],
        PersonID=["open ended", "colleague"],
        YearFounded=[2010, 2010],
        DeltaStart=[1, 0],
        DeltaEnd=[None, 3],
    )
    out = expand_team(windows, min_founding_year=2000)
    years = out.filter(pl.col("PersonID") == "open ended")["Years"].to_list()
    assert years == [1]


def test_the_founding_year_threshold_is_inclusive():
    windows = _team_windows(
        CompanyID=["old", "threshold"],
        PersonID=["p", "p"],
        YearFounded=[1999, 2000],
        DeltaStart=[0, 0],
        DeltaEnd=[0, 0],
    )
    out = expand_team(windows, min_founding_year=2000)
    assert out["CompanyID"].to_list() == ["threshold"]


def _pairs() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "CompanyID": ["c", "c", "c"],
            "SimilarCompanyID": ["alive", "dead", "future"],
            "YF": [2005, 2005, 2018],
            "MY": [2020, 2011, 2020],
            "SimilarityScore": [80.0, 70.0, 60.0],
        }
    )


def test_active_pairs_keeps_only_the_counterparts_alive_that_year():
    years = pl.DataFrame({"CompanyID": ["c"], "Year_Delta": [2015]})
    out = active_pairs(years, _pairs(), ["SimilarityScore"])
    assert sorted(out["SimilarCompanyID"].to_list()) == ["alive"]


def test_active_pairs_includes_both_ends_of_the_life_window():
    years = pl.DataFrame({"CompanyID": ["c", "c"], "Year_Delta": [2005, 2011]})
    out = active_pairs(years, _pairs(), ["SimilarityScore"])
    pairs = set(zip(out["Year_Delta"].to_list(), out["SimilarCompanyID"].to_list(), strict=True))
    assert (2005, "dead") in pairs and (2011, "dead") in pairs


def test_active_pairs_carries_the_extra_columns_the_aggregation_needs():
    years = pl.DataFrame({"CompanyID": ["c"], "Year_Delta": [2019]})
    out = active_pairs(years, _pairs(), ["SimilarityScore"])
    assert "SimilarityScore" in out.columns
    assert sorted(out["SimilarityScore"].to_list()) == [60.0, 80.0]


def test_active_pairs_never_matches_another_company():
    years = pl.DataFrame({"CompanyID": ["altra"], "Year_Delta": [2015]})
    assert active_pairs(years, _pairs(), ["SimilarityScore"]).height == 0
