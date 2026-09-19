"""Phase 2: one row per (company, person), with the years that person was there.

The board table has one row per *appointment*, so the same person appears more
than once in the same company when they held more than one role, or when the same
role was recorded twice. Team variables are per person, not per role, so the rows
are merged — and the two situations are merged by different rules, because in one
the duplication is noise and in the other it is information.

The phase also rebuilds two attributes of a person *by year*: how many roles they
had started, and which degrees they had already earned. The source exposes both
only as totals at extraction time, which cannot say what was true in 2012.
"""

from __future__ import annotations

import polars as pl

from src.panel.config import PanelConfig, PanelRules
from src.panel.expressions import MISSING_TOKENS, first_match, nullify, parse_date
from src.panel.io import read_raw, to_num

#: The board columns this phase reads. ``LastUpdated`` only arbitrates between two
#: records of the same appointment and does not survive the merge.
BOARD_COLUMNS = [
    "CompanyID",
    "PersonID",
    "PersonName",
    "FullTitle",
    "IsCurrent",
    "StartDate",
    "EndDate",
]
PAIR = ["CompanyID", "PersonID"]

#: The three groups of roles the experience index is built on, and the source
#: table each group is counted from.
ROLE_GROUPS = ["Positions", "Seats", "OtherRoles"]

#: Each counter of ``Person.csv``, the table that should reproduce it, and the
#: value of ``IsCurrent`` to count (``None`` counts every row).
ROLE_COUNTERS: dict[str, tuple[str, str | None]] = {
    "CurrentPositionsCount": ("PersonPositionRelation", "Yes"),
    "FormerPositionsCount": ("PersonPositionRelation", "No"),
    "CurrentBoardSeatsCount": ("PersonBoardSeatRelation", "Yes"),
    "FormerBoardSeatsCount": ("PersonBoardSeatRelation", "No"),
    "CurrentAdvisoryRolesCount": ("PersonAdvisoryRelation", "Yes"),
    "FormerAdvisoryRolesCount": ("PersonAdvisoryRelation", "No"),
    "AffiliatedDealsCount": ("PersonAffiliatedDealRelation", None),
    "NumberOfAffiliatedFunds": ("PersonAffiliatedFundRelation", None),
}

#: Which counters add up to which group.
COUNTERS_BY_GROUP: dict[str, list[str]] = {
    "Positions": ["CurrentPositionsCount", "FormerPositionsCount"],
    "Seats": ["CurrentBoardSeatsCount", "FormerBoardSeatsCount"],
    "OtherRoles": [
        "CurrentAdvisoryRolesCount",
        "FormerAdvisoryRolesCount",
        "AffiliatedDealsCount",
        "NumberOfAffiliatedFunds",
    ],
}

_year_of = lambda column: parse_date(pl.col(column)).dt.year()  # noqa: E731


def company_years(skeleton: pl.DataFrame) -> pl.DataFrame:
    """The first and last year of each company's panel.

    :param skeleton: Output of :func:`src.panel.companies.build_skeleton`.
    :return: One row per company, with ``YearFounded`` and ``LastYear``.
    """
    return (
        skeleton.group_by("CompanyID")
        .agg(pl.col("YearFounded").first(), pl.col("Year_Delta").max().alias("LastYear"))
        # group_by does not promise an order, and two runs of the pipeline should
        # produce the same rows in the same order.
        .sort("CompanyID")
    )


def read_board(cfg: PanelConfig, companies: pl.DataFrame) -> pl.DataFrame:
    """Read the board table, restricted to the companies of the panel.

    :param cfg: Pipeline paths.
    :param companies: Frame carrying the ``CompanyID`` of the panel.
    :return: One row per appointment, plus ``LastUpdated``.
    """
    board = nullify(
        read_raw(cfg, "CompanyBoardTeamRelation", [*BOARD_COLUMNS, "LastUpdated"]), MISSING_TOKENS
    )
    return board.join(companies.select("CompanyID").unique(), on="CompanyID", how="semi")


def merge_appointments(board: pl.DataFrame) -> pl.DataFrame:
    """Reduce the board table to one row per (company, person).

    Two merges, in this order.

    **The same appointment recorded twice** — same pair, same title. It is one
    fact written down twice, so the more reliable version wins: "still in office"
    beats a leaving date, a date beats a missing value, and when the two records
    were updated on the same day the wider interval wins.

    **Different roles of the same person** in the same company. What matters is
    which years the person was there, so the periods are merged by *union*: no
    start if any role has none, no end if any role is open, and the titles are
    concatenated so that a founder stays a founder. A gap between two roles is
    therefore covered; keeping the roles apart until the expansion would cost more
    than the handful of person-years it would save.

    :param board: Output of :func:`read_board`.
    :return: One row per pair, with the dates still as text.
    """
    same_appointment = ["CompanyID", "PersonID", "PersonName", "FullTitle"]
    updated = parse_date(pl.col("LastUpdated"))
    in_office = (pl.col("IsCurrent") == "Yes").any()
    several = pl.len() > 1

    def chosen_date(column: str, tie: str) -> pl.Expr:
        """The date of the most recently updated record that has one."""
        date = parse_date(pl.col(column))
        has_date = date.is_not_null()
        candidates = date.filter(has_date & (updated == updated.filter(has_date).max()))
        picked = candidates.min() if tie == "min" else candidates.max()
        return picked.dt.strftime("%m/%d/%Y")

    board = (
        board.with_row_index("_row")
        .group_by(same_appointment)
        .agg(
            pl.col("_row").min(),
            pl.when(several & in_office)
            .then(pl.lit("Yes"))
            .otherwise(pl.col("IsCurrent").drop_nulls().first())
            .alias("IsCurrent"),
            pl.when(several)
            .then(chosen_date("StartDate", "min"))
            .otherwise(pl.col("StartDate").first())
            .alias("StartDate"),
            pl.when(several & in_office)
            .then(pl.lit(None, dtype=pl.String))
            .when(several)
            .then(chosen_date("EndDate", "max"))
            .otherwise(pl.col("EndDate").first())
            .alias("EndDate"),
        )
        .sort("_row")
        .select(BOARD_COLUMNS)
    )

    start, end = parse_date(pl.col("StartDate")), parse_date(pl.col("EndDate"))
    merged = (
        board.with_row_index("_row")
        .group_by(PAIR)
        .agg(
            pl.col("_row").min(),
            pl.col("PersonName").first(),
            pl.when(pl.col("FullTitle").is_not_null().any())
            .then(pl.col("FullTitle").drop_nulls().unique(maintain_order=True).str.join(", "))
            .alias("FullTitle"),
            pl.when(several & in_office)
            .then(pl.lit("Yes"))
            .otherwise(pl.col("IsCurrent").drop_nulls().first())
            .alias("IsCurrent"),
            pl.when(several & start.is_null().any())
            .then(pl.lit(None, dtype=pl.String))
            .when(several)
            .then(start.min().dt.strftime("%m/%d/%Y"))
            .otherwise(pl.col("StartDate").first())
            .alias("StartDate"),
            pl.when(several & (in_office | end.is_null().any()))
            .then(pl.lit(None, dtype=pl.String))
            .when(several)
            .then(end.max().dt.strftime("%m/%d/%Y"))
            .otherwise(pl.col("EndDate").first())
            .alias("EndDate"),
        )
        .sort("_row")
        .select(BOARD_COLUMNS)
    )
    if merged.height != merged.select(PAIR).n_unique():
        raise ValueError("the merge left a pair on more than one row")
    return merged


def ceo_roles(board: pl.DataFrame, years: pl.DataFrame, rules: PanelRules) -> pl.DataFrame:
    """The chief-executive roles of the board, with the years each one covers.

    Built here because further on the dates become ages of the company and the
    title is dropped. Mind what these dates mean, because it decides everything
    the CEO phase can do with them: the start is when the person joined the
    company, not when they became chief executive.

    :param board: Output of :func:`merge_appointments`.
    :param years: Output of :func:`company_years`.
    :param rules: Domain rules; the chief-executive and founder patterns are read.
    :return: One row per role, with the first and last year it covers, the
        declared start year when there is one, and whether the title is a
        founder's as well.
    """
    title = pl.col("FullTitle").fill_null("")
    without_assistants = title.str.replace_all(f"(?i){rules.founder_assistant_pattern}", "")
    start_year = parse_date(pl.col("StartDate")).dt.year()
    end_year = parse_date(pl.col("EndDate")).dt.year()
    return (
        board.filter(title.str.contains(f"(?i){rules.ceo_pattern}"))
        .join(years, on="CompanyID", how="inner")
        .with_columns(
            start_year.alias("sv"),
            without_assistants.str.contains(f"(?i){rules.founder_pattern}").alias("is_founder"),
        )
        .with_columns(
            pl.max_horizontal(pl.coalesce("sv", "YearFounded"), pl.col("YearFounded")).alias("da"),
            pl.min_horizontal(pl.coalesce(end_year, pl.col("LastYear")), pl.col("LastYear")).alias(
                "a"
            ),
        )
        .filter(pl.col("da") <= pl.col("a"))
        .select("CompanyID", "PersonID", "da", "a", "sv", "is_founder")
    )


def person_attributes(board: pl.DataFrame, cfg: PanelConfig, rules: PanelRules) -> pl.DataFrame:
    """Attach gender, the degrees read off the name, and who is a founder.

    The sort order is not cosmetic: it decides the order in which the institutes
    are concatenated in the team aggregate further on.

    A degree in the name — "Ph.D", " JD", " MD" — carries no date, so it counts in
    every year. A founder is recognised from the title of the appointment or from
    the position level, whichever declares it.

    :param board: Output of :func:`merge_appointments`.
    :param cfg: Pipeline paths.
    :param rules: Domain rules.
    :return: One row per pair, with the dates typed and the text columns dropped.
    """
    pairs = board.with_columns(
        parse_date(pl.col(c)).alias(c) for c in ("StartDate", "EndDate")
    ).sort(["CompanyID", "StartDate"], nulls_last=True)

    gender = nullify(read_raw(cfg, "Person", ["PersonID", "Gender"]), MISSING_TOKENS)
    pairs = pairs.join(gender, on="PersonID", how="left")

    name = pl.col("PersonName")
    pairs = pairs.with_columns(
        *[
            name.str.contains(pattern, literal=" " in pattern).fill_null(False).alias(flag)
            for flag, pattern in rules.name_degree_patterns.items()
        ]
    )

    positions = nullify(
        read_raw(cfg, "PersonPositionRelation", ["PersonID", "EntityID", "PositionLevel"]),
        MISSING_TOKENS,
    ).unique(subset=["EntityID", "PersonID"], keep="first", maintain_order=True)
    pairs = pairs.join(
        positions, left_on=["CompanyID", "PersonID"], right_on=["EntityID", "PersonID"], how="left"
    )

    # Both fields rendered as the literal "NA" when missing, so that the string is
    # never null and the flag is always true or false, never unknown.
    qualification = pl.concat_str(
        [pl.col("FullTitle").fill_null("NA"), pl.col("PositionLevel").fill_null("NA")],
        separator="; ",
    ).str.replace_all(f"(?i){rules.founder_assistant_pattern}", "")
    return pairs.with_columns(
        qualification.str.contains(f"(?i){rules.founder_pattern}").alias("IsFounder")
    ).drop("PersonName", "FullTitle", "PositionLevel")


def presence_window(pairs: pl.DataFrame, years: pl.DataFrame) -> pl.DataFrame:
    """Turn the two dates into the window of company ages the person spans.

    Three rules, and they decide every team column.

    **Arrival.** A founder starts at year zero, even where a later start date
    exists: that is what being a founder means. A date before the founding year is
    moved to it, because a presence cannot begin before the company does. Someone
    who is *not* a founder and has no start date is not counted at all: starting
    them at the founding year would put people who arrived years later in the team
    of the early years, and those people are more numerous precisely in the
    companies that go on to grow. The row is kept, because the CEO phase reads it.

    **Departure.** A missing end date becomes the company's last year, whatever
    the declared status: counting a person one year too long costs less than
    losing them.

    **The cut.** No window goes past the company's last year: roles that start
    after it are dropped and later departures are cut back to it. Beyond that year
    nothing is known about the company, so those years could not carry a target.

    :param pairs: Output of :func:`person_attributes`.
    :param years: Output of :func:`company_years`.
    :return: One row per pair, with ``DeltaStart`` and ``DeltaEnd`` as ages.
    """
    pairs = pairs.join(years, on="CompanyID", how="left")
    founding_day = pl.date(pl.col("YearFounded"), 1, 1)
    starts_earlier = pl.col("StartDate").dt.year() < pl.col("YearFounded")
    pairs = pairs.with_columns(
        pl.when(pl.col("StartDate").is_null() & pl.col("IsFounder"))
        .then(founding_day)
        .when(starts_earlier.fill_null(False))
        .then(founding_day)
        .otherwise(pl.col("StartDate"))
        .alias("StartDate")
    )

    last_day = pl.date(pl.col("LastYear"), 12, 31)
    after_the_end = (pl.col("StartDate").dt.year() > pl.col("LastYear")).fill_null(False)
    pairs = (
        pairs.filter(~after_the_end)
        .with_columns(
            pl.min_horizontal(pl.col("EndDate").fill_null(last_day), last_day).alias("EndDate")
        )
        .drop("IsCurrent", "LastYear")
    )

    # An end before the start, from a dirty date or from an arrival moved to the
    # founding year, is straightened rather than dropped.
    return (
        pairs.with_columns(
            pl.when(pl.col("EndDate") < pl.col("StartDate"))
            .then(pl.col("StartDate"))
            .otherwise(pl.col("EndDate"))
            .alias("EndDate")
        )
        .with_columns(
            (pl.col("StartDate").dt.year() - pl.col("YearFounded")).alias("DeltaStart"),
            (pl.col("EndDate").dt.year() - pl.col("YearFounded")).alias("DeltaEnd"),
        )
        .with_columns(
            pl.when(pl.col("IsFounder")).then(0).otherwise(pl.col("DeltaStart")).alias("DeltaStart")
        )
        .drop("StartDate", "EndDate")
    )


def experience_events(cfg: PanelConfig, people: pl.DataFrame) -> pl.DataFrame:
    """How many roles each person had started, year by year.

    Every role of five detail tables becomes an event with a year, and the
    experience at year Y is the number of roles started by Y, finished or not: the
    end date plays no part. The year of a role is chosen in this order:

    1. the declared date;
    2. the founding year of the entity the role is held in — not the real date,
       but a role cannot start before the entity exists;
    3. the person's first known year, the earliest of those resolved above on
       their other roles. The entities that reach this point are companies outside
       the extraction, or entity-persons such as angels, and have no founding year
       to anchor to;
    4. year zero, meaning "counts in every year", for a person with no other dated
       role at all.

    A fund has no declared date: the affiliation cannot precede either the fund's
    vintage or the person's arrival at the firm that manages it, so the later of
    the two is used.

    :param cfg: Pipeline paths.
    :param people: Frame carrying the ``PersonID`` to cover.
    :return: One row per (person, year) in which something changed, with the
        cumulative number of roles of each group.
    """
    ids = people.select("PersonID").unique()

    def read(table: str, columns: list[str]) -> pl.DataFrame:
        return nullify(read_raw(cfg, table, columns), MISSING_TOKENS).join(
            ids, on="PersonID", how="semi"
        )

    # Founding years of every entity a role can be held in, companies and
    # investment firms alike. An identifier present in both tables is the same
    # entity, and the company table wins.
    founding = (
        pl.concat(
            [
                nullify(
                    read_raw(cfg, "Company", ["CompanyID", "YearFounded"]), MISSING_TOKENS
                ).select(
                    pl.col("CompanyID").alias("Entity"),
                    pl.col("YearFounded").cast(pl.Int64, strict=False).alias("Founding"),
                ),
                nullify(
                    read_raw(cfg, "Investor", ["InvestorID", "YearFounded"]), MISSING_TOKENS
                ).select(
                    pl.col("InvestorID").alias("Entity"),
                    pl.col("YearFounded").cast(pl.Int64, strict=False).alias("Founding"),
                ),
            ]
        )
        .drop_nulls()
        .unique("Entity", keep="first", maintain_order=True)
    )

    positions = read("PersonPositionRelation", ["PersonID", "EntityID", "StartDate"])
    seats = read("PersonBoardSeatRelation", ["PersonID", "CompanyID", "StartDate"])
    advisory = read("PersonAdvisoryRelation", ["PersonID", "EntityID", "StartDate"])
    deals = read("PersonAffiliatedDealRelation", ["PersonID", "CompanyID", "DealDate"])
    funds = read("PersonAffiliatedFundRelation", ["PersonID", "FundID", "InvestorID"])

    # The person's arrival at an investment firm is their oldest dated position
    # there.
    arrival = (
        positions.with_columns(_year_of("StartDate").alias("_arrival"))
        .group_by("PersonID", "EntityID")
        .agg(pl.col("_arrival").min())
    )
    dated_funds = (
        funds.join(
            nullify(read_raw(cfg, "Fund", ["FundID", "Vintage"]), MISSING_TOKENS).with_columns(
                pl.col("Vintage").cast(pl.Int64, strict=False)
            ),
            on="FundID",
            how="left",
        )
        .join(
            arrival,
            left_on=["PersonID", "InvestorID"],
            right_on=["PersonID", "EntityID"],
            how="left",
        )
        # max_horizontal ignores nulls: with only one of the two it takes that one.
        .with_columns(pl.max_horizontal("Vintage", "_arrival").alias("_affiliation"))
    )

    sources = {
        "positions": (positions, "EntityID", _year_of("StartDate"), "Positions"),
        "seats": (seats, "CompanyID", _year_of("StartDate"), "Seats"),
        "advisory": (advisory, "EntityID", _year_of("StartDate"), "OtherRoles"),
        "deals": (deals, "CompanyID", _year_of("DealDate"), "OtherRoles"),
        "funds": (dated_funds, "InvestorID", pl.col("_affiliation"), "OtherRoles"),
    }

    def events(name: str, rows: pl.DataFrame, entity: str, year: pl.Expr, group: str):
        """One event per role: declared date, else the entity's founding year."""
        return rows.join(founding, left_on=entity, right_on="Entity", how="left").select(
            "PersonID",
            pl.coalesce(year, pl.col("Founding"), pl.lit(0)).cast(pl.Int64).alias("year"),
            pl.lit(group).alias("group"),
            pl.lit(name).alias("source"),
            pl.when(year.is_not_null())
            .then(pl.lit("declared"))
            .when(pl.col("Founding").is_not_null())
            .then(pl.lit("founding"))
            .otherwise(pl.lit("always"))
            .alias("origin"),
        )

    all_events = pl.concat([events(name, *args) for name, args in sources.items()])

    # Step three of the cascade needs every event of the person, so it cannot run
    # inside the function above.
    first_year = (
        all_events.filter(pl.col("origin") != "always")
        .group_by("PersonID")
        .agg(pl.col("year").min().alias("_first"))
    )
    all_events = (
        all_events.join(first_year, on="PersonID", how="left")
        .with_columns(
            pl.when(pl.col("origin") != "always")
            .then(pl.col("origin"))
            .when(pl.col("_first").is_not_null())
            .then(pl.lit("first known year"))
            .otherwise(pl.lit("always"))
            .alias("origin"),
            pl.when(pl.col("origin") == "always")
            .then(pl.col("_first").fill_null(0))
            .otherwise(pl.col("year"))
            .cast(pl.Int64)
            .alias("year"),
        )
        .drop("_first")
    )

    # Every row read has to produce exactly one event: a join that duplicated or
    # lost rows would show up here.
    for name, (rows, *_) in sources.items():
        produced = all_events.filter(pl.col("source") == name).height
        if produced != rows.height:
            raise ValueError(f"{name}: {rows.height} rows read but {produced} events")

    return (
        all_events.group_by("PersonID", "year")
        .agg(*[(pl.col("group") == g).sum().cast(pl.Int64).alias(g) for g in ROLE_GROUPS])
        .sort("PersonID", "year")
        .with_columns(*[pl.col(g).cum_sum().over("PersonID") for g in ROLE_GROUPS])
    )


def check_experience_counts(cfg: PanelConfig, experience: pl.DataFrame) -> pl.DataFrame:
    """Compare the rebuilt counts with the counters the source declares.

    Two comparisons. Counter by counter, each one of ``Person.csv`` against the
    rows of its own detail table. And group by group, the sum of those counters
    against the last row of the rebuilt table, which by construction holds every
    role of any year: a difference there would be a defect of
    :func:`experience_events`, and raises.

    A counter left empty counts as zero, as it does upstream. Whatever difference
    survives is an inconsistency of the source, and the detail table is the more
    complete of the two.

    :param cfg: Pipeline paths.
    :param experience: Output of :func:`experience_events`.
    :return: One row per counter: how many people match, and how the rest split
        between an empty and a filled counter.
    :raises ValueError: If a group does not reproduce its detail tables.
    """
    last = (
        experience.sort("PersonID", "year")
        .group_by("PersonID")
        .agg(pl.col(*COUNTERS_BY_GROUP).last())
    )
    people = last.select("PersonID")

    tables: dict[str, pl.DataFrame] = {}
    for table, status in ROLE_COUNTERS.values():
        if table not in tables:
            columns = ["PersonID", "IsCurrent"] if status is not None else ["PersonID"]
            tables[table] = nullify(read_raw(cfg, table, columns), MISSING_TOKENS).join(
                people, on="PersonID", how="semi"
            )
    rows = people
    for counter, (table, status) in ROLE_COUNTERS.items():
        rows_of = tables[table]
        if status is not None:
            rows_of = rows_of.filter(pl.col("IsCurrent") == status)
        rows = rows.join(
            rows_of.group_by("PersonID").agg(pl.len().cast(pl.Int64).alias(f"rows_{counter}")),
            on="PersonID",
            how="left",
        )
    rows = rows.with_columns(pl.col("^rows_.*$").fill_null(0))

    declared = nullify(
        read_raw(cfg, "Person", ["PersonID", "FullName", *ROLE_COUNTERS]), MISSING_TOKENS
    ).with_columns(to_num(c) for c in ROLE_COUNTERS)
    comparison = (
        people.join(declared, on="PersonID", how="left")
        .join(rows, on="PersonID")
        .join(last, on="PersonID")
    )

    for group, counters in COUNTERS_BY_GROUP.items():
        wrong = comparison.filter(
            pl.sum_horizontal([f"rows_{c}" for c in counters]) != pl.col(group)
        ).height
        if wrong:
            raise ValueError(f"{group}: {wrong} people whose last row differs from their tables")

    report = []
    for counter in ROLE_COUNTERS:
        empty, n = pl.col(counter).is_null(), pl.col(f"rows_{counter}")
        differs = pl.col(counter).fill_null(0) != n
        report.append(
            comparison.select(
                pl.lit(counter).alias("counter"),
                (~differs).sum().alias("equal"),
                (empty & differs).sum().alias("differ, counter empty"),
                (~empty & differs).sum().alias("differ, counter filled"),
            )
        )
    return pl.concat(report)


def classify_studies(cfg: PanelConfig, rules: PanelRules) -> pl.DataFrame:
    """Read the education table and classify each degree and each subject.

    Where the subject is missing it is inferred from the name of the degree, and
    only for the three cases where the name says it outright. If it is still
    missing the field stays *null* rather than becoming "Other": the flags
    downstream distinguish "no classifiable subject" from "a subject of no
    particular area".

    :param cfg: Pipeline paths.
    :param rules: Domain rules; the degree and field patterns are read.
    :return: One row per degree, with ``DegreeLevel`` and ``Field``.
    """
    studies = nullify(
        read_raw(
            cfg,
            "PersonEducationRelation",
            ["PersonID", "Degree", "Major_Concentration", "GraduatingYear", "Institute"],
        ),
        MISSING_TOKENS,
    )
    studies = studies.with_columns(
        first_match(rules.degree_rules, pl.col("Degree"), pl.lit("Other")).alias("DegreeLevel")
    )
    degree, subject = pl.col("Degree"), pl.col("Major_Concentration")
    studies = studies.with_columns(
        pl.when(subject.is_null() & degree.str.contains("(?i)law").fill_null(False))
        .then(pl.lit("Law"))
        .when(subject.is_null() & degree.str.contains("(?i)MBA").fill_null(False))
        .then(pl.lit("Business"))
        .when(subject.is_null() & degree.str.contains("(?i)Medicine").fill_null(False))
        .then(pl.lit("Medicine"))
        .otherwise(subject)
        .alias("Major_Concentration")
    )
    return studies.with_columns(
        pl.when(pl.col("Major_Concentration").is_null())
        .then(None)
        .otherwise(first_match(rules.field_rules, pl.col("Major_Concentration"), pl.lit("Other")))
        .alias("Field")
    )


def _education_aggregate(rules: PanelRules) -> list[pl.Expr]:
    """The aggregation of a set of degrees into one education profile.

    Shared by the year-by-year table and by its check, so that the two cannot
    describe the same person differently.

    :param rules: Domain rules.
    :return: The aggregation expressions.
    """
    level = (
        pl.col("DegreeLevel")
        .replace_strict(
            {name: i + 1 for i, name in enumerate(rules.degree_hierarchy)}, default=None
        )
        .cast(pl.Int64)
    )
    # Guard: with no classifiable subject the flags stay null, not false. "We do
    # not know" and "not of that area" are different things.
    has_field = pl.col("Field").is_not_null().sum() > 0
    return [
        *[
            pl.when(has_field).then(pl.col("Field").eq(value).any()).otherwise(None).alias(flag)
            for flag, value in rules.field_flags.items()
        ],
        pl.col("GraduatingYear").cast(pl.Float64, strict=False).min().alias("Earliest_Year"),
        level.max().alias("Highest_Degree"),
        pl.when(pl.col("Institute").is_not_null().sum() > 0)
        .then(pl.col("Institute").drop_nulls().str.join("; "))
        .otherwise(None)
        .alias("Institute"),
    ]


def education_by_year(
    studies: pl.DataFrame, people: pl.DataFrame, rules: PanelRules
) -> pl.DataFrame:
    """The education of each person as of each year in which it changed.

    For a given year only the degrees earned by then count, otherwise a master's
    taken in 2018 would raise the education of the row of 2010. A degree with no
    graduating year counts in every year, and one dated in the future counts in
    none of the panel's.

    Between two of a person's own years nothing changes, so only the years in
    which something did are stored; the join by year downstream picks the most
    recent one that is not later than the row.

    :param studies: Output of :func:`classify_studies`.
    :param people: Frame carrying the ``PersonID`` to cover.
    :param rules: Domain rules.
    :return: One row per (person, year), with the education as of that year.
    """
    degrees = (
        studies.join(people.select("PersonID").unique(), on="PersonID", how="semi")
        # The row order decides the order of the institutes in the concatenation.
        .with_row_index("_order")
        .with_columns(
            pl.col("GraduatingYear").cast(pl.Int64, strict=False).fill_null(0).alias("degree_year")
        )
    )
    points = degrees.select("PersonID", pl.col("degree_year").alias("year")).unique()
    return (
        points.join(degrees, on="PersonID")
        .filter(pl.col("degree_year") <= pl.col("year"))
        .sort("_order")
        .group_by("PersonID", "year", maintain_order=True)
        .agg(_education_aggregate(rules))
        .sort("PersonID", "year")
    )


def check_education(
    studies: pl.DataFrame, education: pl.DataFrame, people: pl.DataFrame, rules: PanelRules
) -> int:
    """Check the last year of each person against all of their degrees at once.

    The last row of the year-by-year table holds, by construction, every degree of
    that person whatever its year. It must therefore match the aggregation of all
    of them, column by column: if it does not, building it by year lost or
    duplicated something.

    :param studies: Output of :func:`classify_studies`.
    :param education: Output of :func:`education_by_year`.
    :param people: Frame carrying the ``PersonID`` to cover.
    :param rules: Domain rules.
    :return: How many people have an education that changes over time.
    :raises ValueError: If the last row does not reproduce the whole aggregation.
    """
    degrees = studies.join(
        people.select("PersonID").unique(), on="PersonID", how="semi"
    ).with_row_index("_order")
    snapshot = (
        degrees.sort("_order")
        .group_by("PersonID", maintain_order=True)
        .agg(_education_aggregate(rules))
        .sort("PersonID")
    )
    last = (
        education.sort("PersonID", "year")
        .group_by("PersonID")
        .agg(pl.all().exclude("year").last())
        .sort("PersonID")
    )
    if not last.select(snapshot.columns).equals(snapshot):
        different = [
            c
            for c in snapshot.columns[1:]
            if snapshot.select("PersonID", c)
            .join(last.select("PersonID", c), on="PersonID", suffix="_last")
            .filter(pl.col(c).ne_missing(pl.col(f"{c}_last")))
            .height
        ]
        raise ValueError(f"the last year does not match the whole aggregation: {different}")
    return education.group_by("PersonID").agg(pl.len()).filter(pl.col("len") > 1).height
