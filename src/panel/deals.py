"""Phase 4: the funding rounds, and who put the money in.

The rounds decide the growth stage, and the growth stage is the target, so this
phase carries more decisions than any other: which round counts as which kind of
event, what to do with a round that carries no date, and how to describe the
investors behind it.

The capital raised is the sum of the amounts actually declared, and an unknown
amount is added as a zero. Since half the rows declare no amount, the share of
rounds with an undisclosed one travels beside it: without that column "raised
nothing" and "we do not know how much" would be the same zero.
"""

from __future__ import annotations

import polars as pl

from src.panel.config import PanelConfig, PanelRules
from src.panel.expressions import (
    MISSING_TOKENS,
    MISSING_TOKENS_WITH_INF,
    if_else_null,
    last_non_null,
    nullify,
    parse_date,
)
from src.panel.io import read_raw, to_num

#: The deal columns this phase reads.
DEAL_COLUMNS = [
    "CompanyID",
    "DealID",
    "DealNo",
    "DealDate",
    "DealType",
    "TotalInvestedCapital",
    "CEOPBId",
]

#: The two quantities read off an investor, and both describe the *investor* and
#: not the round: how many investments it has made, and the median size of the
#: rounds it takes part in. Together they say how large and how active whoever
#: puts the money in is.
INVESTOR_MEASURES = ["TotalInvestments", "MedianRoundAmount"]


def investor_participations(cfg: PanelConfig, rules: PanelRules, *, timed: bool) -> pl.DataFrame:
    """One row per participation of an investor in a round.

    With ``timed`` off the two measures are the ones the source declares, that is
    a snapshot at extraction time: the row of 2012 carries the number of
    investments that fund had made by the day of the download. With ``timed`` on
    they are rebuilt year by year from the dated rounds, and each participation
    takes the values its investor had in the year of its own round. The price of
    the second is that the extraction only holds the rounds of the companies in
    the sample, so the large funds come out smaller than they are.

    :param cfg: Pipeline paths.
    :param rules: Domain rules; the investor categories are read.
    :param timed: Whether to rebuild the measures year by year.
    :return: One row per (round, investor), with the investor's category.
    """
    columns = ["DealID", "InvestorID", "InvestorStatus", "IsLeadInvestor"]
    participations = nullify(read_raw(cfg, "DealInvestorRelation", columns), MISSING_TOKENS)
    investors = nullify(
        read_raw(cfg, "Investor", ["InvestorID", "PrimaryInvestorType", *INVESTOR_MEASURES]),
        MISSING_TOKENS,
    ).with_columns(to_num(c) for c in INVESTOR_MEASURES)
    participations = participations.join(investors, on="InvestorID", how="left").with_columns(
        pl.col("PrimaryInvestorType")
        .replace_strict(rules.investor_categories, default="Other")
        .alias("InvestorCategory")
    )
    if not timed:
        return participations

    dated = (
        nullify(read_raw(cfg, "Deal", ["DealID", "DealDate", "DealSize"]), MISSING_TOKENS)
        .with_columns(parse_date(pl.col("DealDate")).dt.year().alias("year"), to_num("DealSize"))
        .drop_nulls("year")
    )
    history = nullify(
        read_raw(cfg, "DealInvestorRelation", ["DealID", "InvestorID"]), MISSING_TOKENS
    ).join(dated.select("DealID", "year", "DealSize"), on="DealID", how="inner")
    # For each (investor, year in which it invested), the values cumulated up to
    # and including that year. The median size stands in for the declared measure,
    # which is the closest thing the extraction can rebuild.
    cumulated = (
        history.select("InvestorID", "year")
        .unique()
        .join(
            history.select("InvestorID", pl.col("year").alias("_deal_year"), "DealSize"),
            on="InvestorID",
        )
        .filter(pl.col("_deal_year") <= pl.col("year"))
        .group_by("InvestorID", "year")
        .agg(
            pl.len().cast(pl.Float64).alias("TotalInvestments"),
            pl.col("DealSize").median().alias("MedianRoundAmount"),
        )
    )
    base = participations.drop(*INVESTOR_MEASURES).join(
        dated.select("DealID", pl.col("year").alias("_deal_year")), on="DealID", how="left"
    )
    with_year = (
        base.drop_nulls("_deal_year")
        .sort("_deal_year")
        .join_asof(
            cumulated.sort("year"),
            left_on="_deal_year",
            right_on="year",
            by="InvestorID",
            strategy="backward",
        )
        .drop("year")
    )
    # A participation in a round with no date has no year to refer to.
    without_year = base.filter(pl.col("_deal_year").is_null()).with_columns(
        *[pl.lit(None, dtype=pl.Float64).alias(c) for c in INVESTOR_MEASURES]
    )
    return pl.concat([with_year, without_year], how="diagonal_relaxed").drop("_deal_year")


def aggregate_by_deal(participations: pl.DataFrame, rules: PanelRules) -> pl.DataFrame:
    """Describe each round by the investors that entered it.

    Almost every aggregate is conditional on "is there at least one new investor
    in this round?", and that question has three answers, not two: yes, no, and
    *we cannot tell* when a status is missing. Used as a condition, the uncertainty
    makes the whole aggregate null rather than answering no.

    Mind the asymmetry of the lead flags, which is deliberate: the filter selects
    the leads, while the condition that decides whether the answer is null stays
    the one about the new investors.

    :param participations: Output of :func:`investor_participations`.
    :param rules: Domain rules; the investor flags are read.
    :return: One row per round.
    """
    new = pl.col("InvestorStatus") == "New Investor"
    lead = pl.col("IsLeadInvestor") == "Yes"
    condition = (
        pl.when(new.fill_null(False).any())
        .then(True)
        .when(new.is_null().any())
        .then(None)
        .otherwise(False)
    )

    def mean_over_new(column: str) -> pl.Expr:
        """The mean over the new investors alone, null if the condition is."""
        return if_else_null(condition, pl.col(column).filter(new.fill_null(False)).mean(), None)

    def has_category(category: str, mask: pl.Expr) -> pl.Expr:
        """Is there an investor of that category among the selected ones?"""
        belongs = (
            pl.col("InvestorCategory").filter(mask.fill_null(False))
            == rules.investor_flags[category]
        )
        return if_else_null(condition, belongs.fill_null(False).any(), None)

    return participations.group_by("DealID").agg(
        # How many *new* investors. It becomes the weight of the cumulative
        # weighted means of the next phase.
        new.fill_null(False).sum().alias("TotalInvestors"),
        mean_over_new("TotalInvestments").alias("MeanTotalInvestments"),
        mean_over_new("MedianRoundAmount").alias("MeanMedianRoundAmount"),
        *[has_category(c, new).alias(f"has_{c}") for c in rules.investor_flags],
        *[has_category(c, lead).alias(f"has_{c}_Lead") for c in rules.investor_flags],
    )


def read_deals(cfg: PanelConfig, registry: pl.DataFrame) -> pl.DataFrame:
    """Read the rounds and attach what the repair of their dates will need.

    :param cfg: Pipeline paths.
    :param registry: Output of :func:`src.panel.companies.company_registry`.
    :return: One row per round, with the founding year and the ownership status.
    """
    deals = nullify(read_raw(cfg, "Deal", DEAL_COLUMNS), MISSING_TOKENS).with_columns(
        pl.col("DealNo").cast(pl.Int64, strict=False),
        parse_date(pl.col("DealDate")).alias("DealDate"),
        to_num("TotalInvestedCapital"),
    )
    return deals.join(
        registry.select("CompanyID", "YearFounded", "OwnershipStatus", "OwnershipStatusDate"),
        on="CompanyID",
        how="left",
    )


def repair_deal_dates(
    deals: pl.DataFrame, rules: PanelRules, *, min_founding_year: int
) -> pl.DataFrame:
    """Give a date to the rounds that carry none, in four steps.

    1. A bankruptcy round in a company that is out of business takes the date of
       the ownership change.
    2. An acquisition in a company that was acquired takes the same.
    3. The **first** round, if it is of an initial kind, goes to the founding year.
       It is a strong assumption, and it also decides who enters the sample, since
       it flattens those rounds onto age zero.
    4. Rounds with no date **between two dated ones** are spread evenly over the
       gap, in the order the round numbers give. The nearest dated round before and
       after is used, not merely the adjacent row, so the step also works when two
       or more undated rounds follow one another.

    The bounds of the fourth step are always real rounds: the founding year and the
    company's last year are not used, so a round with no dated round beside it
    keeps no date, and will leave the panel.

    The sample filter sits between the second step and the third, and that position
    is part of the rule: moving it would change which rounds get a date.

    :param deals: Output of :func:`read_deals`.
    :param rules: Domain rules; the ``date_repair`` block is read.
    :param min_founding_year: Oldest founding year admitted, inclusive.
    :return: One row per round of the sample, with the columns of the repair gone.
    """
    repair = rules.date_repair
    status_date = pl.col("OwnershipStatusDate")
    missing = pl.col("DealDate").is_null()

    deals = deals.with_columns(
        pl.when(
            missing
            & pl.col("DealType").is_in(repair["bankruptcy_types"])
            & (pl.col("OwnershipStatus") == repair["out_of_business_status"])
            & status_date.is_not_null()
        )
        .then(status_date)
        .otherwise(pl.col("DealDate"))
        .alias("DealDate")
    )
    deals = deals.with_columns(
        pl.when(
            missing
            & (pl.col("DealType") == repair["acquisition_type"])
            & pl.col("OwnershipStatus").is_in(repair["acquired_status"])
            & status_date.is_not_null()
        )
        .then(status_date)
        .otherwise(pl.col("DealDate"))
        .alias("DealDate")
    )

    deals = deals.filter(pl.col("YearFounded") >= min_founding_year)

    deals = deals.with_columns(
        pl.when(
            missing
            & pl.col("DealType").is_in(repair["first_round_types"])
            & (pl.col("DealNo") == 1)
            & pl.col("YearFounded").is_not_null()
        )
        .then(pl.date(pl.col("YearFounded"), 1, 1))
        .otherwise(pl.col("DealDate"))
        .alias("DealDate")
    )

    deals = deals.sort(["CompanyID", "DealNo"]).with_columns(
        pl.col("DealDate").dt.year().alias("_year")
    )
    undated = pl.col("_year").is_null()
    deals = deals.with_columns(
        # The nearest dated round before and after, skipping the undated ones.
        pl.col("_year").shift(1).forward_fill().over("CompanyID").alias("_before"),
        pl.col("_year").shift(-1).backward_fill().over("CompanyID").alias("_after"),
        # How many dated rounds precede this one: the undated rounds between the
        # same two dated ones share this value, hence the same interval.
        (~undated).cum_sum().over("CompanyID").alias("_gap"),
    )
    deals = deals.with_columns(
        undated.cum_sum().over(["CompanyID", "_gap"]).alias("_position"),
        undated.sum().over(["CompanyID", "_gap"]).alias("_how_many"),
    )
    # Evenly spaced inside the gap: L + (R - L) * i / (k + 1), rounded. With one
    # round in the gap it is the midpoint of the two bounds.
    step = (pl.col("_after") - pl.col("_before")) * pl.col("_position") / (pl.col("_how_many") + 1)
    estimate = (pl.col("_before") + step + 0.5).floor().cast(pl.Int64)
    return deals.with_columns(
        pl.when(undated & pl.col("_before").is_not_null() & pl.col("_after").is_not_null())
        .then(pl.date(estimate, 1, 1))
        .otherwise(pl.col("DealDate"))
        .alias("DealDate")
    ).drop(
        "_year",
        "_before",
        "_after",
        "_gap",
        "_position",
        "_how_many",
        "OwnershipStatus",
        "OwnershipStatusDate",
    )


def place_deals_in_years(deals: pl.DataFrame, by_deal: pl.DataFrame) -> pl.DataFrame:
    """Assign each round to a year of the panel, and attach its investors.

    A round dated **before** the founding year is moved to the founding year: the
    panel starts at age zero and there is no earlier row to land on, and those
    rounds are often real events that precede the legal incorporation.

    A round with no year keeps none. It therefore lands in a group with a null year
    and joins nothing, which is how those rounds leave the panel;
    :func:`companies_losing_rounds` records who loses them. Taking the maximum
    alone would ignore the null and park the round on the founding year, which is a
    different thing and not the right one either.

    :param deals: Output of :func:`repair_deal_dates`.
    :param by_deal: Output of :func:`aggregate_by_deal`.
    :return: One row per round, with ``Year_Delta`` and the investor aggregates.
    """
    deal_year = pl.col("DealDate").dt.year()
    deals = deals.with_columns(
        pl.when(deal_year.is_null())
        .then(None)
        .otherwise(pl.max_horizontal(deal_year, pl.col("YearFounded")))
        .alias("Year_Delta"),
    )
    return nullify(deals.join(by_deal, on="DealID", how="left"), MISSING_TOKENS)


def companies_losing_rounds(deals: pl.DataFrame, rules: PanelRules) -> pl.DataFrame:
    """Record which companies lose rounds, and how many.

    The rounds left without a year join no row of the panel and would disappear
    without a trace. The list is written beside the panel, which does not change by
    a row or a column, and serves the robustness check of the paper: the bias those
    rounds introduce goes one way only, since a round nobody dated cannot raise a
    growth stage.

    :param deals: Output of :func:`place_deals_in_years`.
    :param rules: Domain rules; the venture types are read.
    :return: One row per company that loses at least one round.
    """
    undated = pl.col("Year_Delta").is_null()
    return (
        deals.group_by("CompanyID")
        .agg(
            pl.len().alias("rounds"),
            undated.sum().alias("rounds_without_date"),
            (undated & pl.col("DealType").is_in(rules.venture_types))
            .sum()
            .alias("venture_rounds_without_date"),
        )
        .filter(pl.col("rounds_without_date") > 0)
        # loses_all: in the panel this company shows no funding at all, and with no
        # growth stage it does not even reach the dataset of the models.
        .with_columns((pl.col("rounds_without_date") == pl.col("rounds")).alias("loses_all"))
        .sort("CompanyID")
    )


def add_deal_flags(deals: pl.DataFrame, rules: PanelRules) -> pl.DataFrame:
    """Turn the type of each round into the fourteen flags.

    :param deals: Output of :func:`place_deals_in_years`.
    :param rules: Domain rules; the deal flags are read.
    :return: The rounds with one boolean column per flag.
    """
    # is_in over a null type would be null: fill_null makes it false.
    return deals.with_columns(
        *[
            pl.col("DealType").is_in(types).fill_null(False).alias(flag)
            for flag, types in rules.deal_flags.items()
        ]
    )


def aggregate_by_company_year(deals: pl.DataFrame, rules: PanelRules) -> pl.DataFrame:
    """Collapse the rounds of a company-year into the columns of the panel.

    A flag is on when at least one round of that year turns it on. Two of them —
    accelerator and angel — are an OR between the type of the round and the
    category of its investors, so that a round from an accelerator counts even when
    the type does not say so.

    :param deals: Output of :func:`add_deal_flags`.
    :param rules: Domain rules; the deal and investor flags are read.
    :return: One row per (company, year) with at least one round, plus one group
        with a null year holding the rounds that have none.
    """
    amount = pl.col("TotalInvestedCapital")

    def any_true(column: str) -> pl.Expr:
        """At least one true in the group; all-missing gives false, not null."""
        return pl.col(column).fill_null(False).any()

    both_sources = ("Is_Accelerator", "Is_Angel")
    return deals.group_by(["CompanyID", "Year_Delta"]).agg(
        pl.len().alias("N_Deal"),
        # An unknown amount counts as zero, which is what makes the column below
        # necessary rather than decorative.
        amount.fill_null(0.0).sum().alias("TotalRaised"),
        amount.is_null().mean().alias("UndisclosedAmountShare"),
        *[any_true(f).alias(f) for f in rules.deal_flags if f not in both_sources],
        *[(any_true(f) | any_true(f"has_{f.removeprefix('Is_')}")).alias(f) for f in both_sources],
        # New investors of the year: the weight of the weighted means of phase 5.
        pl.col("TotalInvestors").fill_null(0).sum().alias("TotalInvestors"),
        pl.col("MeanTotalInvestments").mean().alias("MeanTotalInvestments"),
        pl.col("MeanMedianRoundAmount").mean().alias("MeanMedianRoundAmount"),
        *[
            any_true(f"has_{c}").alias(f"has_{c}")
            for c in rules.investor_flags
            if f"Is_{c}" not in both_sources
        ],
        *[any_true(f"has_{c}_Lead").alias(f"has_{c}_Lead") for c in rules.investor_flags],
        # The last chief executive the rounds of that year declare, in row order.
        last_non_null("CEOPBId").alias("CEO_ID"),
    )


def attach_deals(panel: pl.DataFrame, by_company_year: pl.DataFrame) -> pl.DataFrame:
    """Graft the deal columns onto the panel.

    ``TR_D`` marks the company-years with no round at all. Reading ``TotalRaised``
    is enough to tell: inside the aggregate it is a sum that treats a missing
    amount as zero, so it is never null there, and becomes null only here, when the
    join finds nothing to attach.

    The tokens applied at the end include the infinities, which are the artefacts
    of aggregating an all-missing group, not values that were read.

    :param panel: Output of :func:`src.panel.team.join_skeleton`.
    :param by_company_year: Output of :func:`aggregate_by_company_year`.
    :return: The panel with the deal columns.
    """
    panel = panel.join(by_company_year, on=["CompanyID", "Year_Delta"], how="left").with_columns(
        pl.col("TotalRaised").is_null().cast(pl.Int64).alias("TR_D"),
    )
    return nullify(panel, MISSING_TOKENS_WITH_INF)
