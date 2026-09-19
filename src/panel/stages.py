"""Phase 5: growth stages, cumulative totals and the chief executive.

Three things happen here. The flags of the rounds become cumulative, which is what
turns an event into a state and makes the stages monotone. The growth stage is
derived from them by a cascade. And the attributes of the chief executive are
attached, which is the most delicate join of the pipeline, because who the chief
executive was in a given year is only declared by the rounds.
"""

from __future__ import annotations

import polars as pl

from src.panel.config import PanelRules
from src.panel.expressions import cum_sum_null, cumulative_any, weighted_cumulative
from src.panel.people import ROLE_GROUPS
from src.panel.team import PARAMETER_COLUMNS

#: The two investor measures that become cumulative weighted means.
WEIGHTED_MEASURES = ["MeanTotalInvestments", "MeanMedianRoundAmount"]

#: The statuses that make the cascade terminal on their own, by branch.
TERMINAL_STATUSES = {
    "Out": ["Out of Business"],
    "Exit_Public": ["Publicly Held", "In IPO Registration"],
    "Exit_M&A": ["Acquired/Merged", "Acquired/Merged (Operating Subsidiary)"],
}


def cumulate_flags(panel: pl.DataFrame, rules: PanelRules) -> pl.DataFrame:
    """Make the flags of the rounds cumulative within each company.

    Once a flag is on it stays on for every later year: that is how "closed a seed
    round this year" becomes "has already closed a seed round", and it is what
    makes the stages monotone, so that a company cannot move backwards. Most
    company-years hold no round at all, so without this the stage would be empty on
    most rows and the target would not stand up.

    The two flags that were already consumed by the accelerator and angel OR of
    phase 4 are not cumulated again: they have no work left.

    :param panel: Output of :func:`src.panel.deals.attach_deals`.
    :param rules: Domain rules; the deal and investor flags are read.
    :return: The panel sorted by company and year, with the flags cumulative.
    """
    consumed = {"has_Angel", "has_Accelerator"}
    flags = [
        *rules.deal_flags,
        *[f"has_{c}" for c in rules.investor_flags if f"has_{c}" not in consumed],
        *[f"has_{c}_Lead" for c in rules.investor_flags],
    ]
    panel = panel.sort(["CompanyID", "Year_Delta"])
    return panel.with_columns(cumulative_any(pl.col(c)).over("CompanyID").alias(c) for c in flags)


def growth_stage(panel: pl.DataFrame) -> pl.DataFrame:
    """Derive the growth stage of each company-year.

    A cascade of seven branches that short-circuits: the first one that matches
    wins, so their order is part of the definition. The first three are terminal —
    out of business, public, acquired — and in them the ownership status competes
    with the historical flags, because there are companies that closed and no round
    records it: without the status they would sit in the panel as if they were
    alive. The status only enters the year of its own date, which for a failure or
    an acquisition is the year of the event. The remaining branches order the
    growth stages from the most advanced to the most initial.

    Mind the three-valued logic: where the status is null, "null or true" is true
    and the flag decides, while "null or false" is null, the branch does not match
    and the next one is tried.

    :param panel: Output of :func:`cumulate_flags`.
    :return: The panel with ``GrowthStage``.
    """
    status = pl.col("OwnershipStatus")
    out, public, ma = pl.col("Is_Out"), pl.col("Is_Public_Exit"), pl.col("Is_MA")
    later, pe = pl.col("Is_LaterVC"), pl.col("Is_PE")
    early, seed, preseed = pl.col("Is_EarlyVC"), pl.col("Is_Seed"), pl.col("Is_Preseed")
    alive = ~ma & ~public & ~out
    return panel.with_columns(
        pl.when(status.is_in(TERMINAL_STATUSES["Out"]) | out)
        .then(pl.lit("Out"))
        .when(status.is_in(TERMINAL_STATUSES["Exit_Public"]) | public)
        .then(pl.lit("Exit_Public"))
        .when(status.is_in(TERMINAL_STATUSES["Exit_M&A"]) | ma)
        .then(pl.lit("Exit_M&A"))
        .when((later | pe) & alive)
        .then(pl.lit("LaterVC_or_Other"))
        .when(early & ~later & alive & ~pe)
        .then(pl.lit("EarlyVC"))
        .when(seed & ~early & ~later & alive & ~pe)
        .then(pl.lit("Seed"))
        .when(preseed & ~seed & ~early & ~later & alive & ~pe)
        .then(pl.lit("Preseed"))
        .otherwise(None)
        .alias("GrowthStage")
    )


def cumulative_totals(panel: pl.DataFrame) -> pl.DataFrame:
    """Accumulate the counts, and average the investor measures by weight.

    A year with no round raised nothing, and has no undisclosed amounts either: the
    indicator is zero, not null.

    The number of rounds and of investors accumulates. The two investor measures
    become cumulative means **weighted** by how many new investors entered in each
    year, so that a round with ten investors counts ten times one with a single
    investor.

    The new investors of the year are put aside before the column is overwritten by
    its own cumulative sum: they are the weight of those means, and polars
    evaluates every expression of one ``with_columns`` against the frame it started
    from.

    :param panel: Output of :func:`growth_stage`.
    :return: The panel with the cumulative columns.
    """
    no_round = pl.col("TR_D") == 1
    panel = panel.with_columns(
        pl.when(no_round).then(0.0).otherwise(pl.col("TotalRaised")).alias("TotalRaised"),
        pl.when(no_round)
        .then(0.0)
        .otherwise(pl.col("UndisclosedAmountShare"))
        .alias("UndisclosedAmountShare"),
    )
    panel = panel.with_columns(pl.col("TotalInvestors").fill_null(0).alias("NewInvestors"))
    panel = panel.with_columns(
        cum_sum_null(pl.col("N_Deal").fill_null(0)).over("CompanyID").alias("N_Deal"),
        cum_sum_null(pl.col("NewInvestors").fill_null(0)).over("CompanyID").alias("TotalInvestors"),
    )
    panel = panel.with_columns(
        weighted_cumulative(c, "NewInvestors", ["CompanyID"]).alias(f"{c}_cum")
        for c in WEIGHTED_MEASURES
    )
    return panel.drop(*WEIGHTED_MEASURES, "NewInvestors")


def resolve_ceo(panel: pl.DataFrame, ceo_roles: pl.DataFrame) -> tuple[pl.DataFrame, int, int]:
    """Decide who the chief executive is in each year, and say so.

    The rounds declare a chief executive, so the name is known only in the years
    that hold one; it is carried forward, and whoever was chief executive at the
    last round stays so until another one arrives. That leaves two problems, and
    this is where both are dealt with.

    **Before the first round there is no chief executive at all**, and those are
    exactly the early years the models read their features from. The gap is filled
    from the chief-executive roles of the board, but only where the attribution can
    be verified, because that title is a snapshot at extraction time and projecting
    it backwards blindly would say "was already chief executive" of whoever became
    one later. Every condition is necessary: a single such role covers the year, so
    there is nothing to arbitrate; the title is a founder's as well as a chief
    executive's, and for a founder "since they joined" is the founding year; the
    start date is declared, not imputed; it is the same person the first round
    confirms; and nobody else holds a chief-executive role that started earlier.

    **When a new chief executive arrives between two rounds**, carrying the previous
    one forward would attribute the wrong person until the next round. If a role of
    the board starts, with a declared date, after the last declaration of the
    rounds, and it is somebody else, that person wins.

    :param panel: Output of :func:`cumulative_totals`.
    :param ceo_roles: Output of :func:`src.panel.people.ceo_roles`.
    :return: The panel with ``CEO_ID`` resolved, how many rows were filled, and how
        many attributions were corrected.
    """
    panel = panel.sort("CompanyID", "Year_Delta").with_columns(
        pl.when(pl.col("CEO_ID").is_not_null())
        .then(pl.col("Year_Delta"))
        .otherwise(None)
        .fill_null(strategy="forward")
        .over("CompanyID")
        .alias("_declared_year"),
    )
    first = (
        panel.filter(pl.col("CEO_ID").is_not_null())
        .group_by("CompanyID")
        .agg(
            pl.col("Year_Delta").min().alias("_first_year"),
            pl.col("CEO_ID").sort_by("Year_Delta").first().alias("_first_ceo"),
        )
    )
    panel = panel.join(first, on="CompanyID", how="left").with_columns(
        pl.col("CEO_ID").fill_null(strategy="forward").over("CompanyID")
    )

    active = (
        panel.select("CompanyID", "Year_Delta")
        .join_where(
            ceo_roles,
            pl.col("CompanyID") == pl.col("CompanyID_right"),
            pl.col("Year_Delta") >= pl.col("da"),
            pl.col("Year_Delta") <= pl.col("a"),
        )
        .group_by("CompanyID", "Year_Delta")
        .agg(
            pl.col("PersonID").n_unique().alias("_n"),
            pl.col("PersonID").first().alias("_board"),
            pl.col("sv").first().alias("_sv"),
            pl.col("is_founder").first().alias("_founder"),
        )
        .filter(pl.col("_n") == 1)
    )
    # The counter-examples, where the data itself denies that the founder was always
    # in charge: somebody else holds a chief-executive role that started earlier.
    dated = ceo_roles.filter(pl.col("sv").is_not_null())
    excluded = (
        dated.join(dated, on="CompanyID")
        .filter(
            (pl.col("PersonID") != pl.col("PersonID_right")) & (pl.col("sv_right") < pl.col("sv"))
        )
        .select("CompanyID", pl.col("PersonID").alias("_board"))
        .unique()
        .with_columns(pl.lit(True).alias("_excluded"))
    )
    panel = panel.join(active, on=["CompanyID", "Year_Delta"], how="left").join(
        excluded, on=["CompanyID", "_board"], how="left"
    )

    fillable = (
        pl.col("CEO_ID").is_null()
        & pl.col("_first_year").is_not_null()
        & (pl.col("Year_Delta") < pl.col("_first_year"))
        & (pl.col("_n") == 1)
        & pl.col("_founder")
        & pl.col("_sv").is_not_null()
        & (pl.col("_board") == pl.col("_first_ceo"))
        & pl.col("_excluded").is_null()
    )
    to_correct = (
        pl.col("CEO_ID").is_not_null()
        & (pl.col("_n") == 1)
        & pl.col("_sv").is_not_null()
        & (pl.col("_sv") > pl.col("_declared_year"))
        & (pl.col("_board") != pl.col("CEO_ID"))
    )
    filled = panel.select(fillable.sum()).item()
    corrected = panel.select(to_correct.sum()).item()
    panel = panel.with_columns(
        pl.when(fillable)
        .then(pl.col("_board"))
        .when(to_correct)
        .then(pl.col("_board"))
        .otherwise(pl.col("CEO_ID"))
        .alias("CEO_ID")
    ).drop(
        "_n",
        "_board",
        "_sv",
        "_founder",
        "_excluded",
        "_first_year",
        "_first_ceo",
        "_declared_year",
    )
    return panel, filled, corrected


def ceo_attributes(
    panel: pl.DataFrame,
    pairs: pl.DataFrame,
    experience: pl.DataFrame,
    education: pl.DataFrame,
    parameters: pl.DataFrame,
    rules: PanelRules,
    *,
    timed: bool,
) -> pl.DataFrame:
    """Attach the three attributes of the chief executive.

    Gender, the experience index and the highest degree, all read for the year of
    the row and with the same parameters the team index uses, so that the two
    indices live on one scale. They stay null where the chief executive is not on
    the board of that same company, as every other attribute does.

    The few years of the panel that never appear in the team expansion take the
    parameters of the nearest year available, otherwise the index of the chief
    executive would be null there.

    :param panel: Output of :func:`resolve_ceo`.
    :param pairs: The person-company table, :func:`src.panel.people.presence_window`.
    :param experience: Output of :func:`src.panel.people.experience_events`.
    :param education: Output of :func:`src.panel.people.education_by_year`.
    :param parameters: Output of :func:`src.panel.team.experience_parameters`.
    :param rules: Domain rules; the name-degree flags are read.
    :param timed: Whether to read the attributes of the row's own year.
    :return: The panel with the three columns, and without ``CEO_ID``.
    :raises ValueError: If a year of the panel has no parameters.
    """
    phd, jd, md = rules.name_degree_patterns
    ceo = (
        pairs.select("CompanyID", "PersonID", "Gender", phd, jd, md)
        .rename({"Gender": "Gender_CEO"})
        .with_columns(
            pl.lit(True).alias("_on_the_board"),
            (pl.col(phd) | pl.col(jd) | pl.col(md)).alias("_is_doctor"),
        )
        .drop(phd, jd, md)
    )
    # The join is on the pair: the chief executive has to be on the board of that
    # same company.
    panel = panel.join(
        ceo, left_on=["CompanyID", "CEO_ID"], right_on=["CompanyID", "PersonID"], how="left"
    )

    education = education.select("PersonID", "year", "Highest_Degree").rename(
        {"year": "education_year"}
    )

    def last_row(table: pl.DataFrame, column: str) -> pl.DataFrame:
        """The last row of each person, their value at extraction time."""
        return (
            table.sort("PersonID", column).group_by("PersonID").agg(pl.all().exclude(column).last())
        )

    panel = (
        panel.with_row_index("_row")
        .sort("Year_Delta")
        .pipe(
            lambda d: (
                d.join_asof(
                    experience.sort("year"),
                    left_on="Year_Delta",
                    right_on="year",
                    by_left="CEO_ID",
                    by_right="PersonID",
                    strategy="backward",
                ).join_asof(
                    education.sort("education_year"),
                    left_on="Year_Delta",
                    right_on="education_year",
                    by_left="CEO_ID",
                    by_right="PersonID",
                    strategy="backward",
                )
                if timed
                else d.join(
                    last_row(experience, "year"),
                    left_on="CEO_ID",
                    right_on="PersonID",
                    how="left",
                ).join(
                    last_row(education, "education_year"),
                    left_on="CEO_ID",
                    right_on="PersonID",
                    how="left",
                )
            )
        )
        .sort("_row")
        .with_columns(pl.col(*ROLE_GROUPS).fill_null(0))
    )

    if timed:
        panel = (
            panel.sort("Year_Delta")
            .join(parameters.drop("_n"), left_on="Year_Delta", right_on="_year", how="left")
            .with_columns(pl.col(PARAMETER_COLUMNS).fill_null(strategy="forward"))
            .with_columns(pl.col(PARAMETER_COLUMNS).fill_null(strategy="backward"))
            .sort("_row")
        )
        if panel.select(pl.col(PARAMETER_COLUMNS[0]).is_null().sum()).item():
            raise ValueError("some years of the panel have no standardisation parameters")
        index = pl.mean_horizontal(
            [((pl.col(g) + 1).log() - pl.col(f"{g}_mean")) / pl.col(f"{g}_sd") for g in ROLE_GROUPS]
        )
    else:
        index = pl.mean_horizontal(
            [
                ((pl.col(g) + 1).log() - parameters[f"{g}_mean"][0]) / parameters[f"{g}_sd"][0]
                for g in ROLE_GROUPS
            ]
        )

    panel = panel.with_columns(
        pl.when(pl.col("_on_the_board")).then(index).alias("WorkExperienceIndex_CEO"),
        pl.when(pl.col("_on_the_board"))
        .then(pl.when(pl.col("_is_doctor")).then(5).otherwise(pl.col("Highest_Degree")))
        .alias("Highest_Degree_CEO"),
    )
    spent = [
        "_row",
        *ROLE_GROUPS,
        "Highest_Degree",
        "_on_the_board",
        "_is_doctor",
        "CEO_ID",
        *[c for c in ("year", "education_year") if c in panel.columns],
        *[c for c in PARAMETER_COLUMNS if c in panel.columns],
    ]
    return panel.drop(spent)


def attach_registry(panel: pl.DataFrame, registry: pl.DataFrame) -> pl.DataFrame:
    """Attach country and sector, and name the age of the company.

    Those two are panel columns that never varied by year, which is why they are
    joined here and not at the start. The registry has one row per company, so the
    join cannot multiply the rows; if it did the panel would be silently wrong,
    hence the check.

    :param panel: Output of :func:`ceo_attributes`.
    :param registry: Output of :func:`src.panel.companies.company_registry`.
    :return: The panel with ``HQCountry``, ``PrimaryIndustrySector`` and ``Age``.
    :raises ValueError: If the join multiplies the rows.
    """
    expected = panel.height
    panel = panel.join(
        registry.select("CompanyID", "HQCountry", "PrimaryIndustrySector"),
        on="CompanyID",
        how="left",
    )
    if panel.height != expected:
        raise ValueError("the join with the registry multiplied the rows")
    # The ownership status has done its work in the cascade.
    return panel.with_columns(pl.col("Delta").alias("Age")).drop("OwnershipStatus", "Delta", "TR_D")
