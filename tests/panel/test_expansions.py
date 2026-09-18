"""The two expansions: which rows they create, and which ones they must not.

The decisions under test are the ones the panel depends on: a person is counted
only in the years their window covers, a person whose arrival is unknown is not
counted at all, and a competitor counts only while it is alive.
"""

import polars as pl

from src.panel.expansions import active_pairs, expand_team


def _db3(**colonne) -> pl.DataFrame:
    """One row per (company, person), with the defaults the tests override."""
    base = {
        "CompanyID": ["c"],
        "PersonID": ["p"],
        "YearFounded": [2010],
        "DeltaStart": [0],
        "DeltaEnd": [2],
    }
    base.update(colonne)
    return pl.DataFrame(base)


def test_a_person_appears_once_per_year_of_their_window():
    out = expand_team(_db3(), min_founding_year=2000)
    assert out["Years"].to_list() == [0, 1, 2]
    assert out["PersonID"].to_list() == ["p"] * 3


def test_a_person_is_absent_from_the_years_before_they_arrive():
    db3 = _db3(
        CompanyID=["c", "c"],
        PersonID=["fondatore", "assunto"],
        YearFounded=[2010, 2010],
        DeltaStart=[0, 2],
        DeltaEnd=[3, 3],
    )
    out = expand_team(db3, min_founding_year=2000).sort("Years", "PersonID")
    per_anno = {
        anno: sorted(g["PersonID"].drop_nulls().to_list())
        for anno, g in out.group_by("Years", maintain_order=True)
    }
    assert per_anno[(0,)] == ["fondatore"]
    assert per_anno[(2,)] == ["assunto", "fondatore"]


def test_a_person_without_an_arrival_year_is_never_counted():
    # This is the rule of block 2.11: a non-founder with no StartDate is not
    # dated to the founding year, because counting them from year zero would put
    # people hired later into the team of the first years.
    db3 = _db3(
        CompanyID=["c", "c"],
        PersonID=["datato", "non datato"],
        YearFounded=[2010, 2010],
        DeltaStart=[0, None],
        DeltaEnd=[2, 2],
    )
    out = expand_team(db3, min_founding_year=2000)
    assert out["PersonID"].drop_nulls().unique().to_list() == ["datato"]
    # The company-year grid still spans the years of the people who are dated.
    assert sorted(out["Years"].to_list()) == [0, 1, 2]


def test_a_company_whose_whole_team_is_undated_keeps_no_person_row():
    out = expand_team(_db3(DeltaStart=[None]), min_founding_year=2000)
    # The grid has nothing to span, and the filter on YearFounded — which
    # arrives from the person side — removes the empty row, so the company-year
    # is never counted as one person.
    assert out.height == 0


def test_a_person_with_no_departure_counts_for_their_arrival_year_alone():
    # The person-side grid falls back to the arrival year, so an unknown
    # departure does not stretch the presence to the end of the company's life.
    # A colleague provides the company-year grid; block 2.12 imputes every
    # departure before this point, so the case is a guard, not a live path.
    db3 = _db3(
        CompanyID=["c", "c"],
        PersonID=["senza fine", "collega"],
        YearFounded=[2010, 2010],
        DeltaStart=[1, 0],
        DeltaEnd=[None, 3],
    )
    out = expand_team(db3, min_founding_year=2000)
    anni = out.filter(pl.col("PersonID") == "senza fine")["Years"].to_list()
    assert anni == [1]


def test_the_founding_year_threshold_is_inclusive():
    db3 = _db3(
        CompanyID=["vecchia", "soglia"],
        PersonID=["p", "p"],
        YearFounded=[1999, 2000],
        DeltaStart=[0, 0],
        DeltaEnd=[0, 0],
    )
    out = expand_team(db3, min_founding_year=2000)
    assert out["CompanyID"].to_list() == ["soglia"]


def _pairs() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "CompanyID": ["c", "c", "c"],
            "SimilarCompanyID": ["viva", "morta", "futura"],
            "YF": [2005, 2005, 2018],
            "MY": [2020, 2011, 2020],
            "SimilarityScore": [80.0, 70.0, 60.0],
        }
    )


def test_active_pairs_keeps_only_the_counterparts_alive_that_year():
    anni = pl.DataFrame({"CompanyID": ["c"], "Year_Delta": [2015]})
    out = active_pairs(anni, _pairs(), ["SimilarityScore"])
    assert sorted(out["SimilarCompanyID"].to_list()) == ["viva"]


def test_active_pairs_includes_both_ends_of_the_life_window():
    anni = pl.DataFrame({"CompanyID": ["c", "c"], "Year_Delta": [2005, 2011]})
    out = active_pairs(anni, _pairs(), ["SimilarityScore"])
    coppie = set(zip(out["Year_Delta"].to_list(), out["SimilarCompanyID"].to_list(), strict=True))
    assert (2005, "morta") in coppie and (2011, "morta") in coppie


def test_active_pairs_carries_the_extra_columns_the_aggregation_needs():
    anni = pl.DataFrame({"CompanyID": ["c"], "Year_Delta": [2019]})
    out = active_pairs(anni, _pairs(), ["SimilarityScore"])
    assert "SimilarityScore" in out.columns
    assert sorted(out["SimilarityScore"].to_list()) == [60.0, 80.0]


def test_active_pairs_never_matches_another_company():
    anni = pl.DataFrame({"CompanyID": ["altra"], "Year_Delta": [2015]})
    assert active_pairs(anni, _pairs(), ["SimilarityScore"]).height == 0
