"""Phase 1: the shape of the panel.

Every fixture is written here, so what a failure points at is the rule, not an
extraction.
"""

import datetime as dt

import polars as pl
import pytest

from src.panel.companies import (
    COMPANY_COLUMNS,
    attach_ownership_status,
    build_skeleton,
    company_life,
    company_registry,
    read_companies,
)
from src.panel.config import PanelConfig


@pytest.fixture
def cfg(tmp_path):
    """A pipeline whose raw directory holds a three-company table."""
    header = ",".join(COMPANY_COLUMNS)
    (tmp_path / "Company.csv").write_text(
        f"{header}\n"
        # dates in every slot, the latest being the fiscal one
        "A,2015,Italy,Software,Privately Held,06/30/2020,,,,,TTM 2Q2021\n"
        # no date at all: it cannot enter the panel
        "B,2016,Spain,Health,Privately Held,,,,,,\n"
        # no founding year: same
        "C,,France,Energy,Out of Business,01/01/2019,,,,,\n"
    )
    return PanelConfig(raw_dir=tmp_path, interim_dir=tmp_path / "interim")


def test_dates_are_parsed_and_the_founding_year_becomes_a_number(cfg):
    companies = read_companies(cfg)
    assert companies["OwnershipStatusDate"].to_list()[0] == dt.date(2020, 6, 30)
    assert companies["YearFounded"].to_list() == [2015, 2016, None]


def test_the_fiscal_period_becomes_the_date_its_quarter_closes_on(cfg):
    companies = read_companies(cfg)
    assert companies["FiscalDate"].to_list()[0] == dt.date(2021, 6, 30)


def test_a_missing_fiscal_period_leaves_no_fiscal_date(cfg):
    companies = read_companies(cfg)
    assert companies["FiscalDate"].to_list()[1:] == [None, None]


def _companies(rows: list[dict]) -> pl.DataFrame:
    """A typed company frame with only the columns the life window reads."""
    default = {
        "CompanyID": "A",
        "YearFounded": 2015,
        "HQCountry": "Italy",
        "CompanyFinancingStatusDate": None,
        "BusinessStatusDate": None,
        "OwnershipStatusDate": None,
        "FirstFinancingDate": None,
        "LastKnownValuationDate": None,
        "FiscalDate": None,
    }
    schema = {
        "CompanyID": pl.String,
        "YearFounded": pl.Int64,
        "HQCountry": pl.String,
        "CompanyFinancingStatusDate": pl.Date,
        "BusinessStatusDate": pl.Date,
        "OwnershipStatusDate": pl.Date,
        "FirstFinancingDate": pl.Date,
        "LastKnownValuationDate": pl.Date,
        "FiscalDate": pl.Date,
    }
    return pl.DataFrame([{**default, **r} for r in rows], schema=schema)


def test_the_last_known_year_is_the_most_recent_of_the_dates():
    companies = _companies(
        [
            {
                "BusinessStatusDate": dt.date(2018, 5, 1),
                "FiscalDate": dt.date(2020, 3, 30),
                "FirstFinancingDate": dt.date(2016, 1, 1),
            }
        ]
    )
    assert company_life(companies)["MaxYear"].to_list() == [2020]


def test_a_company_without_dates_or_without_a_founding_year_is_left_out():
    companies = _companies(
        [
            {"CompanyID": "no date"},
            {"CompanyID": "no year", "YearFounded": None, "FiscalDate": dt.date(2020, 3, 30)},
        ]
    )
    assert company_life(companies).height == 0


def _life(**columns) -> pl.DataFrame:
    base = {"CompanyID": ["A"], "YearFounded": [2015], "MaxYear": [2018], "HQCountry": ["Italy"]}
    base.update(columns)
    return pl.DataFrame(base)


def test_one_row_per_year_from_the_founding_year_to_the_last_known_one():
    skeleton = build_skeleton(_life(), min_founding_year=2000)
    assert skeleton.select("Year_Delta", "Delta").rows() == [
        (2015, 0),
        (2016, 1),
        (2017, 2),
        (2018, 3),
    ]


def test_a_company_whose_last_year_precedes_its_founding_keeps_year_zero_alone():
    skeleton = build_skeleton(_life(MaxYear=[2013]), min_founding_year=2000)
    assert skeleton.select("Year_Delta", "Delta").rows() == [(2015, 0)]


def test_the_founding_year_threshold_is_inclusive():
    life = _life(
        CompanyID=["old", "edge"],
        YearFounded=[1999, 2000],
        MaxYear=[2001, 2001],
        HQCountry=["Italy", "Italy"],
    )
    skeleton = build_skeleton(life, min_founding_year=2000)
    assert skeleton["CompanyID"].unique().to_list() == ["edge"]


def test_the_ownership_status_lands_on_the_year_of_its_own_date():
    skeleton = build_skeleton(_life(), min_founding_year=2000)
    companies = _companies([{"OwnershipStatusDate": dt.date(2017, 8, 1)}]).with_columns(
        pl.lit("Acquired/Merged").alias("OwnershipStatus")
    )
    out = attach_ownership_status(skeleton, companies)
    assert out.select("Year_Delta", "OwnershipStatus").rows() == [
        (2015, None),
        (2016, None),
        (2017, "Acquired/Merged"),
        (2018, None),
    ]


def test_a_status_without_a_date_is_attached_to_no_year():
    skeleton = build_skeleton(_life(), min_founding_year=2000)
    companies = _companies([{}]).with_columns(pl.lit("Privately Held").alias("OwnershipStatus"))
    out = attach_ownership_status(skeleton, companies)
    assert out["OwnershipStatus"].null_count() == out.height


def test_the_registry_has_one_row_per_company_in_the_sample():
    companies = _companies(
        [{"CompanyID": "A", "YearFounded": 2015}, {"CompanyID": "B", "YearFounded": 1998}]
    ).with_columns(
        pl.lit("Software").alias("PrimaryIndustrySector"),
        pl.lit("Privately Held").alias("OwnershipStatus"),
    )
    registry = company_registry(companies, min_founding_year=2000)
    assert registry["CompanyID"].to_list() == ["A"]
    assert "HQCountry" in registry.columns
