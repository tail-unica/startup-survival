"""Phase 4: the dates of the rounds, and what a company-year says about them."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
import yaml

from src.panel.config import PanelRules
from src.panel.deals import (
    add_deal_flags,
    aggregate_by_company_year,
    aggregate_by_deal,
    companies_losing_rounds,
    place_deals_in_years,
    repair_deal_dates,
)

RULES = PanelRules.from_config(yaml.safe_load(open("config/config.yaml").read()))


def _rounds(rows: list[dict]) -> pl.DataFrame:
    """The frame the repair reads, with the dates already typed."""
    default = {
        "CompanyID": "A",
        "DealID": "D1",
        "DealNo": 1,
        "DealDate": None,
        "DealType": "Seed Round",
        "TotalInvestedCapital": 1.0,
        "CEOPBId": "P1",
        "YearFounded": 2015,
        "OwnershipStatus": "Privately Held",
        "OwnershipStatusDate": None,
    }
    schema = {
        "CompanyID": pl.String,
        "DealID": pl.String,
        "DealNo": pl.Int64,
        "DealDate": pl.Date,
        "DealType": pl.String,
        "TotalInvestedCapital": pl.Float64,
        "CEOPBId": pl.String,
        "YearFounded": pl.Int64,
        "OwnershipStatus": pl.String,
        "OwnershipStatusDate": pl.Date,
    }
    return pl.DataFrame([{**default, **r} for r in rows], schema=schema)


def _repair(rows: list[dict]) -> pl.DataFrame:
    return repair_deal_dates(_rounds(rows), RULES, min_founding_year=2000)


# ── the four steps ─────────────────────────────────────────────────────────


def test_a_bankruptcy_round_takes_the_date_of_the_closure():
    out = _repair(
        [
            {
                "DealType": "Bankruptcy: Liquidation",
                "OwnershipStatus": "Out of Business",
                "OwnershipStatusDate": dt.date(2018, 11, 10),
            }
        ]
    )
    assert out["DealDate"].to_list() == [dt.date(2018, 11, 10)]


def test_a_bankruptcy_round_of_a_living_company_keeps_no_date():
    out = _repair(
        [
            {
                "DealType": "Bankruptcy: Liquidation",
                "OwnershipStatus": "Privately Held",
                "OwnershipStatusDate": dt.date(2018, 11, 10),
            }
        ]
    )
    assert out["DealDate"].to_list() == [None]


def test_an_acquisition_takes_the_date_of_the_ownership_change():
    out = _repair(
        [
            {
                "DealType": "Merger/Acquisition",
                "OwnershipStatus": "Acquired/Merged",
                "OwnershipStatusDate": dt.date(2019, 3, 20),
            }
        ]
    )
    assert out["DealDate"].to_list() == [dt.date(2019, 3, 20)]


def test_a_first_round_of_an_initial_kind_goes_to_the_founding_year():
    out = _repair([{"DealType": "Grant", "DealNo": 1}])
    assert out["DealDate"].to_list() == [dt.date(2015, 1, 1)]


def test_a_later_round_of_an_initial_kind_does_not():
    out = _repair([{"DealType": "Grant", "DealNo": 3}])
    assert out["DealDate"].to_list() == [None]


def test_a_first_round_of_a_late_kind_does_not_either():
    out = _repair([{"DealType": "Later Stage VC", "DealNo": 1}])
    assert out["DealDate"].to_list() == [None]


def test_one_undated_round_between_two_dated_ones_lands_in_the_middle():
    out = _repair(
        [
            {
                "DealID": "D1",
                "DealNo": 1,
                "DealDate": dt.date(2012, 1, 1),
                "DealType": "Angel (individual)",
            },
            {"DealID": "D2", "DealNo": 2, "DealType": "Later Stage VC"},
            {
                "DealID": "D3",
                "DealNo": 3,
                "DealDate": dt.date(2018, 1, 1),
                "DealType": "Later Stage VC",
            },
        ]
    )
    assert out.sort("DealNo")["DealDate"].dt.year().to_list() == [2012, 2015, 2018]


def test_two_undated_rounds_in_a_row_are_spread_evenly():
    out = _repair(
        [
            {
                "DealID": "D1",
                "DealNo": 1,
                "DealDate": dt.date(2012, 1, 1),
                "DealType": "Angel (individual)",
            },
            {"DealID": "D2", "DealNo": 2, "DealType": "Later Stage VC"},
            {"DealID": "D3", "DealNo": 3, "DealType": "Later Stage VC"},
            {
                "DealID": "D4",
                "DealNo": 4,
                "DealDate": dt.date(2018, 1, 1),
                "DealType": "Later Stage VC",
            },
        ]
    )
    assert out.sort("DealNo")["DealDate"].dt.year().to_list() == [2012, 2014, 2016, 2018]


def test_an_undated_round_with_no_dated_round_after_it_keeps_no_date():
    out = _repair(
        [
            {
                "DealID": "D1",
                "DealNo": 1,
                "DealDate": dt.date(2012, 1, 1),
                "DealType": "Angel (individual)",
            },
            {"DealID": "D2", "DealNo": 2, "DealType": "Later Stage VC"},
        ]
    )
    assert out.sort("DealNo")["DealDate"].to_list() == [dt.date(2012, 1, 1), None]


def test_a_company_founded_before_the_threshold_leaves_with_its_rounds():
    out = repair_deal_dates(
        _rounds([{"YearFounded": 1999, "DealDate": dt.date(2005, 1, 1)}]),
        RULES,
        min_founding_year=2000,
    )
    assert out.height == 0


# ── placing a round in a year ──────────────────────────────────────────────


def _by_deal(rows: list[dict] | None = None) -> pl.DataFrame:
    """A per-round aggregate, empty by default."""
    return pl.DataFrame(
        rows or [],
        schema={
            "DealID": pl.String,
            "TotalInvestors": pl.Int64,
            "MeanTotalInvestments": pl.Float64,
            "MeanMedianRoundAmount": pl.Float64,
            "has_Accelerator": pl.Boolean,
            "has_Angel": pl.Boolean,
        },
    )


def test_a_round_dated_before_the_founding_year_is_moved_to_it():
    placed = place_deals_in_years(
        _rounds([{"DealDate": dt.date(2013, 6, 1)}]).drop("OwnershipStatus", "OwnershipStatusDate"),
        _by_deal(),
    )
    assert placed["Year_Delta"].to_list() == [2015]


def test_a_round_with_no_date_gets_no_year():
    placed = place_deals_in_years(
        _rounds([{}]).drop("OwnershipStatus", "OwnershipStatusDate"), _by_deal()
    )
    assert placed["Year_Delta"].to_list() == [None]


def test_the_companies_losing_rounds_are_recorded():
    rounds = _rounds(
        [
            {"DealID": "D1", "DealDate": dt.date(2016, 1, 1)},
            {"DealID": "D2", "DealType": "Later Stage VC"},
            {"CompanyID": "B", "DealID": "D3", "DealType": "Early Stage VC"},
        ]
    ).drop("OwnershipStatus", "OwnershipStatusDate")
    placed = place_deals_in_years(rounds, _by_deal())
    lost = companies_losing_rounds(placed, RULES)
    assert lost.rows() == [("A", 2, 1, 1, False), ("B", 1, 1, 1, True)]


# ── what a company-year says ───────────────────────────────────────────────


def _placed(rows: list[dict]) -> pl.DataFrame:
    """Rounds already placed in a year, with the per-round aggregates attached."""
    default = {
        "CompanyID": "A",
        "DealID": "D1",
        "Year_Delta": 2016,
        "DealType": "Seed Round",
        "TotalInvestedCapital": 1.0,
        "CEOPBId": "P1",
        "TotalInvestors": 1,
        "MeanTotalInvestments": 10.0,
        "MeanMedianRoundAmount": 2.0,
        **{f"has_{c}": False for c in RULES.investor_flags},
        **{f"has_{c}_Lead": False for c in RULES.investor_flags},
    }
    return pl.DataFrame([{**default, **r} for r in rows])


def test_the_capital_of_a_year_sums_the_declared_amounts():
    out = aggregate_by_company_year(
        add_deal_flags(
            _placed(
                [
                    {"DealID": "D1", "TotalInvestedCapital": 1.5},
                    {"DealID": "D2", "TotalInvestedCapital": 2.5},
                ]
            ),
            RULES,
        ),
        RULES,
    )
    assert out["TotalRaised"].to_list() == [4.0]
    assert out["UndisclosedAmountShare"].to_list() == [0.0]


def test_an_undeclared_amount_counts_as_zero_and_raises_the_share():
    out = aggregate_by_company_year(
        add_deal_flags(
            _placed(
                [
                    {"DealID": "D1", "TotalInvestedCapital": 2.0},
                    {"DealID": "D2", "TotalInvestedCapital": None},
                ]
            ),
            RULES,
        ),
        RULES,
    )
    assert out["TotalRaised"].to_list() == [2.0]
    assert out["UndisclosedAmountShare"].to_list() == [pytest.approx(0.5)]


def test_a_flag_is_on_when_one_round_of_the_year_turns_it_on():
    out = aggregate_by_company_year(
        add_deal_flags(
            _placed(
                [{"DealID": "D1", "DealType": "Seed Round"}, {"DealID": "D2", "DealType": "Grant"}]
            ),
            RULES,
        ),
        RULES,
    )
    assert out["Is_Seed"].to_list() == [True]
    assert out["Is_Grant"].to_list() == [True]
    assert out["Is_LaterVC"].to_list() == [False]


def test_the_accelerator_flag_also_reads_the_category_of_the_investors():
    # The round is a plain seed, but an accelerator put the money in.
    out = aggregate_by_company_year(
        add_deal_flags(_placed([{"has_Accelerator": True}]), RULES), RULES
    )
    assert out["Is_Accelerator"].to_list() == [True]


def test_the_chief_executive_of_the_year_is_the_last_one_declared():
    out = aggregate_by_company_year(
        add_deal_flags(
            _placed([{"DealID": "D1", "CEOPBId": "P1"}, {"DealID": "D2", "CEOPBId": None}]),
            RULES,
        ),
        RULES,
    )
    assert out["CEO_ID"].to_list() == ["P1"]


# ── the three-valued condition on the investors ────────────────────────────


def _participations(rows: list[dict]) -> pl.DataFrame:
    default = {
        "DealID": "D1",
        "InvestorID": "I1",
        "InvestorStatus": "New Investor",
        "IsLeadInvestor": "Yes",
        "PrimaryInvestorType": "Venture Capital",
        "TotalInvestments": 10.0,
        "MedianRoundAmount": 2.0,
        "InvestorCategory": "Venture Capital",
    }
    return pl.DataFrame([{**default, **r} for r in rows])


def test_the_mean_is_computed_over_the_new_investors_alone():
    out = aggregate_by_deal(
        _participations(
            [
                {"InvestorID": "I1", "InvestorStatus": "New Investor", "TotalInvestments": 10.0},
                {
                    "InvestorID": "I2",
                    "InvestorStatus": "Follow-on Investor",
                    "TotalInvestments": 100.0,
                },
            ]
        ),
        RULES,
    )
    assert out["TotalInvestors"].to_list() == [1]
    assert out["MeanTotalInvestments"].to_list() == [10.0]


def test_an_undeclared_status_makes_the_aggregates_unknown():
    # Not "no new investor": we cannot tell, and the aggregate says so.
    out = aggregate_by_deal(
        _participations([{"InvestorStatus": None}]),
        RULES,
    )
    assert out["MeanTotalInvestments"].to_list() == [None]
    assert out["has_VentureCapital"].to_list() == [None]


def test_the_lead_flags_read_the_leads_but_inherit_the_same_uncertainty():
    out = aggregate_by_deal(
        _participations(
            [
                {"InvestorID": "I1", "IsLeadInvestor": "Yes", "InvestorCategory": "Angel"},
                {"InvestorID": "I2", "IsLeadInvestor": "No", "InvestorCategory": "Venture Capital"},
            ]
        ),
        RULES,
    )
    assert out["has_Angel_Lead"].to_list() == [True]
    assert out["has_VentureCapital_Lead"].to_list() == [False]
