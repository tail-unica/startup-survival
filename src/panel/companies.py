"""Phase 1: from one row per company to one row per company-year.

This is where the shape of the panel is decided. A company enters with a founding
year and a last year in which anything at all is known about it, and comes out as
one row per year in between.

Eleven columns of ``Company.csv`` are read. Four of them, together with
``FiscalPeriod``, produce no panel column at all: they only feed the last year
with data.
"""

from __future__ import annotations

import polars as pl

from src.panel.config import PanelConfig
from src.panel.expressions import MISSING_TOKENS, nullify, parse_date, seq_inclusive
from src.panel.io import COMPANY_DATE_COLUMNS, read_raw

#: The projection of ``Company.csv`` this phase needs. The first four columns
#: reach the panel; the rest decide where each company's panel ends.
COMPANY_COLUMNS = [
    "CompanyID",
    "YearFounded",
    "HQCountry",
    "PrimaryIndustrySector",
    "OwnershipStatus",
    "OwnershipStatusDate",
    *[c for c in COMPANY_DATE_COLUMNS if c != "OwnershipStatusDate"],
    "FiscalPeriod",
]

#: Columns of the company registry, the phase's second output.
REGISTRY_COLUMNS = [
    "CompanyID",
    "YearFounded",
    "HQCountry",
    "PrimaryIndustrySector",
    "OwnershipStatus",
    "OwnershipStatusDate",
]


def read_companies(cfg: PanelConfig) -> pl.DataFrame:
    """Read the company table, with its dates parsed.

    The missing-value tokens are applied before anything else, because what
    counts as missing decides every step that follows. ``FiscalPeriod`` arrives as
    text in the form ``TTM 2Q2019`` and becomes a date: the quarter is turned into
    the month it closes on, on a conventional day.

    :param cfg: Pipeline paths.
    :return: One row per company, dates typed, plus the derived ``FiscalDate``.
    """
    companies = nullify(read_raw(cfg, "Company", COMPANY_COLUMNS), MISSING_TOKENS)
    companies = companies.with_columns(
        *[parse_date(pl.col(c)).alias(c) for c in COMPANY_DATE_COLUMNS],
        pl.col("YearFounded").cast(pl.Int64, strict=False),
    )
    quarter = pl.col("FiscalPeriod").str.extract(r"TTM (\d)Q\d{4}", 1).cast(pl.Int64, strict=False)
    fiscal_year = pl.col("FiscalPeriod").str.slice(-4).cast(pl.Int64, strict=False)
    return companies.with_columns(pl.date(fiscal_year, quarter * 3, 30).alias("FiscalDate"))


def company_life(companies: pl.DataFrame) -> pl.DataFrame:
    """The window in which each company is known to exist.

    ``MaxYear`` is the most recent of the six dates, that is the last year in
    which the source knows anything about that company. The panel of a company
    ends there: after it nothing is known, so not even its outcome could be read.
    A company without a founding year or without any date does not enter the
    panel at all.

    The same window is used twice, here and by the competitor phase, and is
    computed once so that the two cannot disagree.

    :param companies: Output of :func:`read_companies`.
    :return: One row per usable company, with its first and last known year.
    """
    all_dates = [*COMPANY_DATE_COLUMNS, "FiscalDate"]
    return (
        companies.with_columns(
            pl.max_horizontal([pl.col(c).dt.year() for c in all_dates]).alias("MaxYear")
        )
        .filter(pl.col("YearFounded").is_not_null() & pl.col("MaxYear").is_not_null())
        .select("CompanyID", "YearFounded", "MaxYear", "HQCountry")
    )


def build_skeleton(life: pl.DataFrame, *, min_founding_year: int) -> pl.DataFrame:
    """Expand each company into one row per year of its life.

    The upper bound is the later of the last known year and the founding year: a
    few records carry dates that precede the founding year, typically
    re-registrations, and without the bound they would produce years before the
    company existed. Those companies keep their first year alone.

    :param life: Output of :func:`company_life`.
    :param min_founding_year: Oldest founding year admitted, inclusive.
    :return: One row per company-year, with the calendar year and the age.
    """
    upper = pl.max_horizontal("MaxYear", "YearFounded")
    return (
        life.filter(pl.col("YearFounded") >= min_founding_year)
        .with_columns(seq_inclusive(pl.col("YearFounded"), upper).alias("Year_Delta"))
        .explode("Year_Delta", empty_as_null=True)
        .with_columns((pl.col("Year_Delta") - pl.col("YearFounded")).alias("Delta"))
        .select("CompanyID", "YearFounded", "Year_Delta", "Delta")
    )


def attach_ownership_status(skeleton: pl.DataFrame, companies: pl.DataFrame) -> pl.DataFrame:
    """Attach the ownership status to the single year of its own date.

    The status (privately held, acquired, out of business) is one row per
    company, that is the state as of the extraction, and its date says when that
    state was reached. It is therefore joined to that year only, and stays null on
    every other row of the company. The growth-stage cascade reads it later.

    :param skeleton: Output of :func:`build_skeleton`.
    :param companies: Output of :func:`read_companies`.
    :return: The skeleton with ``OwnershipStatus`` added.
    """
    status_year = companies.select(
        "CompanyID",
        pl.col("OwnershipStatusDate").dt.year().alias("_status_year"),
        "OwnershipStatus",
    ).drop_nulls("_status_year")
    return skeleton.join(
        status_year,
        left_on=["CompanyID", "Year_Delta"],
        right_on=["CompanyID", "_status_year"],
        how="left",
    )


def company_registry(companies: pl.DataFrame, *, min_founding_year: int) -> pl.DataFrame:
    """The company-level attributes the later phases need.

    Two of them are panel columns, the country and the sector, and they are
    attached at the end of the pipeline because they do not vary by year. The
    other two serve the repair of the deal dates.

    :param companies: Output of :func:`read_companies`.
    :param min_founding_year: Oldest founding year admitted, inclusive.
    :return: One row per company in the sample.
    """
    return companies.select(REGISTRY_COLUMNS).filter(pl.col("YearFounded") >= min_founding_year)
