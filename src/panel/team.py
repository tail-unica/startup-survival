"""Phase 3: the team columns, year by year.

Each person is expanded over every year of their window, the attributes of that
year are attached, and the rows are collapsed back to one per (company, year).
Thirteen aggregates come out: how many people, the share of women, the eight
flags on fields of study, the mean graduation year, the mean degree, the
institutes, the mean experience index and how many founders.
"""

from __future__ import annotations

import polars as pl

from src.panel.config import PanelRules
from src.panel.expansions import expand_team
from src.panel.people import ROLE_GROUPS

#: Mean and standard deviation of each role group, the parameters the experience
#: index is standardised with.
PARAMETER_COLUMNS = [f"{g}_{s}" for g in ROLE_GROUPS for s in ("mean", "sd")]

__all__ = [
    "PARAMETER_COLUMNS",
    "aggregate_team",
    "attach_person_attributes",
    "expand_team",
    "experience_parameters",
    "join_skeleton",
]


def _log1p(group: str) -> pl.Expr:
    """The log of one plus a count, which is what gets standardised.

    :param group: Name of the role-group column.
    :return: The expression.
    """
    return (pl.col(group) + 1).log()


def _snapshot(table: pl.DataFrame, year_column: str) -> pl.DataFrame:
    """The last row of each person, that is their value at extraction time.

    :param table: A per-(person, year) table.
    :param year_column: Name of its year column.
    :return: One row per person.
    """
    return (
        table.sort("PersonID", year_column)
        .group_by("PersonID")
        .agg(pl.all().exclude(year_column).last())
    )


def attach_person_attributes(
    expanded: pl.DataFrame,
    experience: pl.DataFrame,
    education: pl.DataFrame,
    rules: PanelRules,
    *,
    timed: bool,
) -> pl.DataFrame:
    """Give every (company, year, person) row the attributes of that year.

    With ``timed`` on, two as-of joins take the person's most recent row that is
    not later than the year of the panel row: a degree earned in 2018 therefore
    does not raise the education of the row of 2010. With ``timed`` off, both joins
    take the person's last row instead, which is their value at extraction time —
    the snapshot the released panel used to carry.

    The degrees read off the name carry no date and apply in every year, so they
    override the table after the join, not before.

    :param expanded: Output of :func:`src.panel.expansions.expand_team`.
    :param experience: Output of :func:`src.panel.people.experience_events`.
    :param education: Output of :func:`src.panel.people.education_by_year`.
    :param rules: Domain rules; the names of the name-degree flags are read.
    :param timed: Whether to read the attributes of the row's own year.
    :return: The expanded frame with the counts and the education attached.
    """
    education = education.rename({"year": "education_year"})
    phd, jd, md = rules.name_degree_patterns
    return (
        expanded.with_row_index("_row")
        .with_columns((pl.col("YearFounded") + pl.col("Years")).alias("_year"))
        # The as-of joins need the frame sorted by year; _row puts the original
        # order back afterwards, because that order decides how the institutes are
        # concatenated in the aggregate.
        .sort("_year")
        .pipe(
            lambda d: (
                d.join_asof(
                    experience.sort("year"),
                    left_on="_year",
                    right_on="year",
                    by="PersonID",
                    strategy="backward",
                ).join_asof(
                    education.sort("education_year"),
                    left_on="_year",
                    right_on="education_year",
                    by="PersonID",
                    strategy="backward",
                )
                if timed
                else d.join(_snapshot(experience, "year"), on="PersonID", how="left").join(
                    _snapshot(education, "education_year"), on="PersonID", how="left"
                )
            )
        )
        .sort("_row")
        # No role started by that year means no roles, not unknown. No degree by
        # that year means no education, as for someone who declares none.
        .with_columns(pl.col(*ROLE_GROUPS).fill_null(0))
        .with_columns(
            pl.when(pl.col(phd) | pl.col(jd) | pl.col(md))
            .then(5)
            .otherwise(pl.col("Highest_Degree"))
            .alias("Highest_Degree"),
            pl.when(pl.col(jd)).then(True).otherwise(pl.col("Is_Law")).alias("Is_Law"),
            pl.when(pl.col(md)).then(True).otherwise(pl.col("Is_Med")).alias("Is_Med"),
        )
    )


def experience_parameters(expanded: pl.DataFrame, *, timed: bool) -> pl.DataFrame:
    """The mean and standard deviation the experience index is standardised with.

    With ``timed`` on they are computed **per year, over an expanding window**:
    for the row of a given year only the (person, year) pairs of that year or
    earlier take part. Otherwise the value of a row of 2005 would depend on the
    rows of 2020, and the scale of the early years — when few people are
    documented — would be the one of two decades later. The current year is inside
    the window: at 2005, 2005 is present, not future.

    With ``timed`` off there is no time axis to expand over, so the population is
    one row per (company, person) and the parameters are a single pair.

    The pairs are sorted before the aggregation: ``unique`` returns rows in an
    order that changes between runs, and the sum of a few million logs changes in
    its last digits with the order.

    :param expanded: Output of :func:`attach_person_attributes`.
    :param timed: Whether to standardise per year.
    :return: One row per year with the parameters and the size of the population,
        or a single row when ``timed`` is off.
    """
    if not timed:
        key = ["CompanyID", "PersonID"]
        return (
            expanded.unique(key)
            .sort(key)
            .select(
                *[_log1p(g).mean().alias(f"{g}_mean") for g in ROLE_GROUPS],
                *[_log1p(g).std(ddof=1).alias(f"{g}_sd") for g in ROLE_GROUPS],
            )
        )
    population = expanded.unique(["PersonID", "_year"]).sort(["PersonID", "_year"])
    # One pass per year rather than the sum-of-squares formula: a couple of dozen
    # passes over a few hundred thousand rows, and no cancellation error.
    return pl.concat(
        [
            population.filter(pl.col("_year") <= year).select(
                pl.lit(year, dtype=pl.Int64).alias("_year"),
                pl.len().alias("_n"),
                *[_log1p(g).mean().alias(f"{g}_mean") for g in ROLE_GROUPS],
                *[_log1p(g).std(ddof=1).alias(f"{g}_sd") for g in ROLE_GROUPS],
            )
            for year in sorted(population["_year"].unique().to_list())
        ]
    )


def experience_index(
    expanded: pl.DataFrame, parameters: pl.DataFrame, rules: PanelRules, *, timed: bool
) -> pl.DataFrame:
    """Add the experience index and drop what was only needed to compute it.

    The index is the mean of the three standardised counts. The counts are
    compressed by a logarithm first, because a handful of people hold hundreds of
    roles and without it they would set the scale for everyone.

    :param expanded: Output of :func:`attach_person_attributes`.
    :param parameters: Output of :func:`experience_parameters`.
    :param rules: Domain rules; the name-degree flags are dropped here.
    :param timed: Whether the parameters are per year.
    :return: One row per (company, year, person), with ``WorkExperienceIndex``.
    :raises ValueError: If a year of the panel has no parameters.
    """
    if timed:
        expanded = expanded.join(parameters.drop("_n"), on="_year", how="left")
        missing = expanded.select(pl.col(PARAMETER_COLUMNS[0]).is_null().sum()).item()
        if missing:
            raise ValueError(f"{missing} rows fall in a year with no parameters")
        index = pl.mean_horizontal(
            [(_log1p(g) - pl.col(f"{g}_mean")) / pl.col(f"{g}_sd") for g in ROLE_GROUPS]
        )
    else:
        index = pl.mean_horizontal(
            [
                (_log1p(g) - parameters[f"{g}_mean"][0]) / parameters[f"{g}_sd"][0]
                for g in ROLE_GROUPS
            ]
        )
    spent = [
        "_row",
        "_year",
        *ROLE_GROUPS,
        *rules.name_degree_patterns,
        *[c for c in ("year", "education_year") if c in expanded.columns],
        *[c for c in PARAMETER_COLUMNS if c in expanded.columns],
    ]
    return expanded.with_columns(index.alias("WorkExperienceIndex")).drop(spent)


def aggregate_team(expanded: pl.DataFrame, rules: PanelRules) -> pl.DataFrame:
    """Collapse the people of a company-year into the thirteen team columns.

    Two of the aggregates carry a decision. The denominator of the share of women
    is the people whose gender is known: keeping the others in it would lower the
    share precisely in the companies that are documented worst, and where nobody's
    gender is known the column stays empty. The institutes are concatenated
    dropping the missing ones, so the string never carries a placeholder.

    :param expanded: Output of :func:`experience_index`.
    :param rules: Domain rules; the field-of-study flags are read.
    :return: One row per (company, year).
    """
    institute = pl.col("Institute").drop_nulls()
    known_gender = pl.col("Gender").is_not_null().sum()
    team = expanded.group_by(["CompanyID", "Years"]).agg(
        pl.len().alias("Total_People"),
        pl.when(known_gender > 0)
        .then(pl.col("Gender").eq("Female").sum() / known_gender * 100)
        .alias("Percent_Females"),
        # fill_null(False) before any(): a group with nothing declared is false,
        # not unknown.
        *[pl.col(c).fill_null(False).any().alias(c) for c in rules.field_flags],
        pl.col("Earliest_Year").mean().alias("Avg_Earliest_Year"),
        pl.col("Highest_Degree").mean().alias("Highest_Degree_Mean"),
        institute.unique(maintain_order=True).str.join("; ").alias("Institute"),
        pl.col("WorkExperienceIndex").mean().alias("WorkExp_Idx_Mean"),
        # sum() over booleans counts the true ones and ignores the nulls.
        pl.col("IsFounder").sum().alias("Total_Founders"),
    )
    # With nobody carrying an institute the join of strings returns an empty one.
    return team.with_columns(
        pl.when(pl.col("Institute") == "")
        .then(None)
        .otherwise(pl.col("Institute"))
        .alias("Institute")
    )


def join_skeleton(skeleton: pl.DataFrame, team: pl.DataFrame) -> pl.DataFrame:
    """Attach the team columns to the skeleton.

    A left join, because the panel *is* the skeleton: a company-year covered only
    by the team would fall past the company's last year, where there is no target.
    After the cut of phase 2 that cannot happen, and the check makes sure of it at
    every run rather than trusting it.

    :param skeleton: Output of :func:`src.panel.companies.attach_ownership_status`.
    :param team: Output of :func:`aggregate_team`.
    :return: One row per company-year, with the team columns.
    :raises ValueError: If the team covers a company-year the skeleton does not.
    """
    team = team.rename({"Years": "Delta"})
    outside = team.join(skeleton, on=["CompanyID", "Delta"], how="anti").height
    if outside:
        raise ValueError(f"{outside} company-years of the team fall outside the skeleton")
    return skeleton.join(team, on=["CompanyID", "Delta"], how="left").sort(["CompanyID", "Delta"])
