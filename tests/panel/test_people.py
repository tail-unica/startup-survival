"""Phase 2: merging the appointments, and the window each person spans."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
import yaml

from src.panel.config import PanelRules
from src.panel.people import (
    ceo_roles,
    company_years,
    education_by_year,
    merge_appointments,
    presence_window,
)

RULES = PanelRules.from_config(yaml.safe_load(open("config/config.yaml").read()))


def _board(rows: list[dict]) -> pl.DataFrame:
    """A raw board table with the columns the merge reads."""
    default = {
        "CompanyID": "A",
        "PersonID": "P1",
        "PersonName": "Anna",
        "FullTitle": "Board Member",
        "IsCurrent": "No",
        "StartDate": None,
        "EndDate": None,
        "LastUpdated": "01/01/2020",
    }
    schema = dict.fromkeys(default, pl.String)
    return pl.DataFrame([{**default, **r} for r in rows], schema=schema)


# the two merges


def test_the_same_appointment_recorded_twice_becomes_one_row():
    board = _board(
        [
            {"StartDate": "01/01/2015", "EndDate": "12/31/2018", "LastUpdated": "02/02/2019"},
            {"StartDate": "01/01/2015", "IsCurrent": "Yes", "LastUpdated": "05/05/2020"},
        ]
    )
    merged = merge_appointments(board)
    assert merged.height == 1
    # Still in office beats a leaving date.
    assert merged["IsCurrent"].to_list() == ["Yes"]
    assert merged["EndDate"].to_list() == [None]


def test_between_two_records_of_one_appointment_the_freshest_date_wins():
    board = _board(
        [
            {"StartDate": "01/01/2015", "EndDate": "12/31/2017", "LastUpdated": "01/01/2018"},
            {"StartDate": "01/01/2016", "EndDate": "12/31/2019", "LastUpdated": "01/01/2020"},
        ]
    )
    merged = merge_appointments(board)
    assert merged["StartDate"].to_list() == ["01/01/2016"]
    assert merged["EndDate"].to_list() == ["12/31/2019"]


def test_a_date_beats_a_missing_one_even_from_an_older_record():
    board = _board(
        [
            {"StartDate": "01/01/2015", "LastUpdated": "01/01/2018"},
            {"StartDate": None, "LastUpdated": "01/01/2020"},
        ]
    )
    assert merge_appointments(board)["StartDate"].to_list() == ["01/01/2015"]


def test_different_roles_of_one_person_merge_into_the_union_of_their_periods():
    board = _board(
        [
            {"FullTitle": "Founder", "StartDate": "01/01/2014", "EndDate": "06/30/2016"},
            {
                "FullTitle": "Chief Executive Officer",
                "StartDate": "07/01/2016",
                "EndDate": "11/10/2018",
            },
        ]
    )
    merged = merge_appointments(board)
    assert merged.height == 1
    assert merged["StartDate"].to_list() == ["01/01/2014"]
    assert merged["EndDate"].to_list() == ["11/10/2018"]
    # The titles are concatenated, so a founder stays a founder after the merge.
    assert "Founder" in merged["FullTitle"].to_list()[0]


def test_a_role_without_a_start_makes_the_merged_start_unknown():
    board = _board(
        [
            {"FullTitle": "Founder", "StartDate": None},
            {"FullTitle": "Chairman", "StartDate": "01/01/2016", "EndDate": "12/31/2018"},
        ]
    )
    merged = merge_appointments(board)
    assert merged["StartDate"].to_list() == [None]


def test_two_people_in_the_same_company_stay_two_rows():
    board = _board([{"PersonID": "P1"}, {"PersonID": "P2", "PersonName": "Bruno"}])
    assert merge_appointments(board).height == 2


# the chief-executive roles


def _years() -> pl.DataFrame:
    return pl.DataFrame({"CompanyID": ["A"], "YearFounded": [2015], "LastYear": [2020]})


def test_a_chief_executive_role_covers_from_its_start_to_its_end():
    board = merge_appointments(
        _board([{"FullTitle": "CEO", "StartDate": "01/01/2017", "EndDate": "12/31/2019"}])
    )
    roles = ceo_roles(board, _years(), RULES)
    assert roles.select("da", "a", "sv").rows() == [(2017, 2019, 2017)]


def test_a_role_with_no_start_is_bounded_by_the_founding_year():
    board = merge_appointments(_board([{"FullTitle": "Chief Executive Officer"}]))
    roles = ceo_roles(board, _years(), RULES)
    assert roles.select("da", "a").rows() == [(2015, 2020)]
    assert roles["sv"].to_list() == [None]


def test_a_role_is_never_counted_past_the_company_s_last_year():
    board = merge_appointments(
        _board([{"FullTitle": "CEO", "StartDate": "01/01/2017", "EndDate": "12/31/2030"}])
    )
    assert ceo_roles(board, _years(), RULES)["a"].to_list() == [2020]


def test_a_title_that_is_both_founder_and_chief_executive_is_flagged():
    board = merge_appointments(
        _board([{"FullTitle": "Co-Founder & CEO", "StartDate": "01/01/2015"}])
    )
    assert ceo_roles(board, _years(), RULES)["is_founder"].to_list() == [True]


def test_a_founder_s_associate_is_not_a_founder():
    board = merge_appointments(
        _board([{"FullTitle": "Founder's Associate, CEO", "StartDate": "01/01/2015"}])
    )
    assert ceo_roles(board, _years(), RULES)["is_founder"].to_list() == [False]


def test_a_role_outside_the_company_s_life_is_dropped():
    board = merge_appointments(_board([{"FullTitle": "CEO", "StartDate": "01/01/2025"}]))
    assert ceo_roles(board, _years(), RULES).height == 0


# the window each person spans


def _pairs(rows: list[dict]) -> pl.DataFrame:
    """The frame presence_window reads, with the dates already typed."""
    default = {
        "CompanyID": "A",
        "PersonID": "P1",
        "IsCurrent": "Yes",
        "StartDate": None,
        "EndDate": None,
        "IsFounder": False,
    }
    schema = {
        "CompanyID": pl.String,
        "PersonID": pl.String,
        "IsCurrent": pl.String,
        "StartDate": pl.Date,
        "EndDate": pl.Date,
        "IsFounder": pl.Boolean,
    }
    return pl.DataFrame([{**default, **r} for r in rows], schema=schema)


def test_a_founder_starts_at_year_zero_even_with_a_later_date():
    pairs = _pairs([{"IsFounder": True, "StartDate": dt.date(2018, 5, 1)}])
    out = presence_window(pairs, _years())
    assert out["DeltaStart"].to_list() == [0]


def test_a_non_founder_without_a_start_is_left_out_of_the_team():
    out = presence_window(_pairs([{"IsFounder": False}]), _years())
    # The row survives, because the chief-executive step reads it, but it spans no
    # year at all.
    assert out.height == 1
    assert out["DeltaStart"].to_list() == [None]


def test_a_start_before_the_founding_year_is_moved_to_it():
    pairs = _pairs([{"StartDate": dt.date(2013, 1, 1), "EndDate": dt.date(2018, 1, 1)}])
    out = presence_window(pairs, _years())
    assert out.select("DeltaStart", "DeltaEnd").rows() == [(0, 3)]


def test_a_missing_departure_becomes_the_company_s_last_year():
    pairs = _pairs([{"StartDate": dt.date(2017, 1, 1)}])
    out = presence_window(pairs, _years())
    assert out.select("DeltaStart", "DeltaEnd").rows() == [(2, 5)]


def test_a_departure_past_the_company_s_last_year_is_cut_back_to_it():
    pairs = _pairs([{"StartDate": dt.date(2017, 1, 1), "EndDate": dt.date(2030, 1, 1)}])
    assert presence_window(pairs, _years())["DeltaEnd"].to_list() == [5]


def test_a_role_starting_after_the_company_s_last_year_is_dropped():
    pairs = _pairs([{"StartDate": dt.date(2025, 1, 1)}])
    assert presence_window(pairs, _years()).height == 0


def test_an_end_before_the_start_is_straightened():
    pairs = _pairs([{"StartDate": dt.date(2018, 1, 1), "EndDate": dt.date(2016, 1, 1)}])
    out = presence_window(pairs, _years())
    assert out.select("DeltaStart", "DeltaEnd").rows() == [(3, 3)]


def test_company_years_reports_the_first_and_last_year_of_the_panel():
    skeleton = pl.DataFrame(
        {
            "CompanyID": ["A", "A", "B"],
            "YearFounded": [2015, 2015, 2018],
            "Year_Delta": [2015, 2017, 2019],
        }
    )
    assert company_years(skeleton).rows() == [("A", 2015, 2017), ("B", 2018, 2019)]


# education by year


def _studies(rows: list[dict]) -> pl.DataFrame:
    default = {
        "PersonID": "P1",
        "Degree": "MSc",
        "Major_Concentration": "Computer Science",
        "GraduatingYear": "2010",
        "Institute": "Politecnico",
        "DegreeLevel": "Master's",
        "Field": "IT and Computer Science",
    }
    schema = dict.fromkeys(default, pl.String)
    return pl.DataFrame([{**default, **r} for r in rows], schema=schema)


def _people() -> pl.DataFrame:
    return pl.DataFrame({"PersonID": ["P1"]})


def test_a_degree_counts_only_from_the_year_it_was_awarded():
    studies = _studies(
        [
            {"GraduatingYear": "2010", "DegreeLevel": "Bachelor's"},
            {"GraduatingYear": "2017", "DegreeLevel": "PhD/Doctorate"},
        ]
    )
    education = education_by_year(studies, _people(), RULES)
    assert education.select("year", "Highest_Degree").rows() == [(2010, 3), (2017, 5)]


def test_a_degree_with_no_year_counts_in_every_year():
    studies = _studies([{"GraduatingYear": None, "DegreeLevel": "Master's"}])
    education = education_by_year(studies, _people(), RULES)
    # Year zero means "from the beginning": the as-of join downstream picks it up
    # for every year of the panel.
    assert education["year"].to_list() == [0]
    assert education["Highest_Degree"].to_list() == [4]


def test_the_field_flags_stay_unknown_when_no_subject_is_classifiable():
    studies = _studies([{"Field": None, "Major_Concentration": None}])
    education = education_by_year(studies, _people(), RULES)
    assert education["Is_IT"].to_list() == [None]


def test_the_institutes_of_a_year_are_concatenated():
    studies = _studies(
        [
            {"GraduatingYear": "2010", "Institute": "Politecnico"},
            {"GraduatingYear": "2010", "Institute": "Bocconi"},
        ]
    )
    education = education_by_year(studies, _people(), RULES)
    assert education["Institute"].to_list() == ["Politecnico; Bocconi"]


def test_a_person_outside_the_panel_is_not_covered():
    education = education_by_year(_studies([{}]), pl.DataFrame({"PersonID": ["P2"]}), RULES)
    assert education.height == 0


def test_the_earliest_year_is_the_first_degree_already_earned():
    studies = _studies([{"GraduatingYear": "2012"}, {"GraduatingYear": "2016"}])
    education = education_by_year(studies, _people(), RULES)
    assert education.select("year", "Earliest_Year").rows() == [
        (2012, pytest.approx(2012.0)),
        (2016, pytest.approx(2012.0)),
    ]
