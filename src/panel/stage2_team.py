"""Stage 2 — the person-company table and the team panel.

`build_db3` is a transcription of `1_Arrange_DB.R:240-524`; it produces the
534,851-row table that checkpoint A verifies.
"""

from __future__ import annotations

import polars as pl

from src.panel.config import PanelConfig
from src.panel.io import read_raw, to_num
from src.panel.rutils import (
    R_NA,
    R_NA_NAN,
    as_na,
    coalesce_first_last,
    parse_date_r,
    r_if_else,
    scale_r,
)
from src.panel.validate import COLUMN_FINALISED_AT_STAGE

#: `CompanyBoardTeamRelation`, all 14 columns: the dedup block rbinds an
#: aggregate of these onto the frame itself, so the sets must match.
BOARD_COLUMNS = [
    "CompanyID", "PersonID", "PersonName", "FullTitle", "IsOnBoard",
    "RepresentingID", "RepresentingName", "RoleOnBoard", "IsCurrent",
    "Location", "StartDate", "EndDate", "RowID", "LastUpdated",
]  # fmt: skip

#: The 16 `Person` attributes joined at `1_Arrange_DB.R:292-301`.
PERSON_COLUMNS = [
    "PersonID", "Gender", "Prefix", "City", "PostCode", "Country", "Biography",
    "University_Institution", "RolesCount", "CurrentPositionsCount",
    "FormerPositionsCount", "CurrentBoardSeatsCount", "FormerBoardSeatsCount",
    "CurrentAdvisoryRolesCount", "FormerAdvisoryRolesCount",
    "AffiliatedDealsCount", "NumberOfAffiliatedFunds",
]  # fmt: skip

#: `across(CurrentPositionsCount:NumberOfAffiliatedFunds)` — the eight count
#: columns, in the order the join leaves them in.
COUNT_COLUMNS = PERSON_COLUMNS[9:]

#: `1_Arrange_DB.R:381` — the factor levels behind Highest_Degree, whose value
#: is the 1-based index of the highest level reached.
DEGREE_HIERARCHY = ["Other", "Diploma/Certificate", "Bachelor's", "Master's", "PhD/Doctorate"]

#: `1_Arrange_DB.R:326-333`, in order: case_when short-circuits, first match wins.
DEGREE_LEVEL_RULES = [
    (r"PhD|Doctor|MD|PsyD|DPhil|DC|DDS|DPT|OD|JD", "PhD/Doctorate"),
    (r"MBA|LLM|Master|MSc|MPhil|MPA|MFA|MEng|MAcc|Graduate", "Master's"),
    (r"Bachelor|BSc|BEng|Laurea|BS|BFA|BCom|degree|Business Program|Undergrad", "Bachelor's"),
    (
        r"Certified|Certificat|A Levels|A-Levels|Diplom|DEA|DESS|Dipl\.-Ing|"
        r"Executive Development Program|Executive Education|Executive Education program|"
        r"Executive Program|First Legal State Exam|Legal Practice Course|Vordiplom",
        "Diploma/Certificate",
    ),
]

#: `1_Arrange_DB.R:350-374`, in order.
FIELD_RULES = [
    (
        r"business|management|bank|invest|financ|marketing|real estate|account|"
        r"Entrepreneur|commerce|econom|actuarial science|private equity",
        "Economics",
    ),
    (
        r"engineer|civil|electric|mechanic|electronic|Operations Research|material|"
        r"logistic|Engeneering",
        "Engineering",
    ),
    (
        r"statistic|machine learning|Natural Language|robot|technolog|comput|data|"
        r"informatic|Artificial Intelligence|information science|information systems|"
        r"data science|softwar",
        "IT and Computer Science",
    ),
    (r"law|tax|justice|forensic|legal|jurisprudence|intellectual property", "Law"),
    (r"medic|nursing|pharmac|health|immunolog|neuroscien|genetic|physio", "Health and Medicine"),
    (
        r"social science|strateg|sociology|psychology|anthropology|international relations|"
        r"polit|government|geography|polic|international|foreign service|social studies|"
        r"criminology|cognitive science|public affairs|urban planning|social work|"
        r"human resource|leadership|foreign",
        "Social Sciences",
    ),
    (
        r"natural|biolog|chemistry|physic|environmental science|math|geology|life science|"
        r"zoology|agriculture",
        "Natural Sciences",
    ),
    (
        r"humanit|literature|histor|philosoph|language|linguist|english|spanis|american|"
        r"religion|classic|french|theolog|cultural studies|europe|arts|design|music|"
        r"architecture|journalism|media|public relations|advertising|communic|education|"
        r"early childhood|special education|administration",
        "Humanities and Arts",
    ),
]

#: The nine Field flags, and the Field value each looks for.
FIELD_FLAGS = {
    "Is_Eco": "Economics",
    "Is_Eng": "Engineering",
    "Is_Med": "Health and Medicine",
    "Is_Hum": "Humanities and Arts",
    "Is_IT": "IT and Computer Science",
    "Is_Law": "Law",
    "Is_NS": "Natural Sciences",
    "Is_SS": "Social Sciences",
    # Bug B2: Field never takes this value, so Is_Other is False everywhere.
    "Is_Other": "Other/Unknown",
}

DB3_COLUMNS = [
    "CompanyID", "PersonID", "PersonName", "FullTitle", "IsOnBoard", "RoleOnBoard",
    "IsCurrent", "StartDate", "EndDate", "Gender", "Prefix", "City", "PostCode",
    "Country", "Biography", "University_Institution", "RolesCount",
    "CurrentPositionsCount", "FormerPositionsCount", "CurrentBoardSeatsCount",
    "FormerBoardSeatsCount", "CurrentAdvisoryRolesCount", "FormerAdvisoryRolesCount",
    "AffiliatedDealsCount", "NumberOfAffiliatedFunds", "RolesCount_Total", "Positions",
    "BoardSeats", "OtherRoles", "Positions_z", "BoardSeats_z", "OtherRoles_z",
    "WorkExperienceIndex", "Is_Eco", "Is_Eng", "Is_Med", "Is_Hum", "Is_IT", "Is_Law",
    "Is_NS", "Is_SS", "Is_Other", "Earliest_Year", "Highest_Degree", "Institute",
    "PositionLevel", "IsFounder", "OwnershipStatus", "OwnershipStatusDate", "Is_Out",
    "OutDate", "YearFounded", "DeltaStart", "DeltaEnd",
]  # fmt: skip


def _case_when(rules: list[tuple[str, str]], col: pl.Expr, otherwise) -> pl.Expr:
    """Short-circuit chain of case-insensitive `grepl` tests, first match wins."""
    expr = pl.when(col.str.contains(f"(?i){rules[0][0]}").fill_null(False)).then(
        pl.lit(rules[0][1])
    )
    for pattern, value in rules[1:]:
        expr = expr.when(col.str.contains(f"(?i){pattern}").fill_null(False)).then(pl.lit(value))
    return expr.otherwise(otherwise)


def _dedup_board(db3: pl.DataFrame, *, fix: bool) -> pl.DataFrame:
    """`1_Arrange_DB.R:248-278`.

    Two defects live here and both are reproduced when `fix` is False (B9):
    the duplicated rows are re-selected by `PersonID` alone rather than by the
    `(CompanyID, PersonID)` pair, and `coalesce(first(x), last(x))` cannot see
    a value that only exists in a middle row.
    """
    pair = ["CompanyID", "PersonID"]
    dup = db3.filter(db3.select(pair).is_duplicated())
    if fix:
        subset = db3.join(dup.select(pair).unique(), on=pair, how="semi")
    else:
        subset = db3.filter(pl.col("PersonID").is_in(dup["PersonID"]))
    others = [c for c in BOARD_COLUMNS if c not in pair]
    agg = (
        subset.sort(["PersonID", "CompanyID"], maintain_order=True)
        .group_by(pair, maintain_order=True)
        .agg([coalesce_first_last(c).alias(c) for c in others])
        .select(BOARD_COLUMNS)
    )
    # rbind puts the recomposed rows first, so distinct() keeps them.
    return pl.concat([agg, db3]).unique(subset=pair, keep="first", maintain_order=True)


def _education(cfg: PanelConfig) -> pl.DataFrame:
    """`1_Arrange_DB.R:317-398` — one row per PersonID."""
    db48 = as_na(
        read_raw(
            cfg,
            "PersonEducationRelation",
            ["PersonID", "Degree", "Major_Concentration", "GraduatingYear", "Institute"],
        ),
        R_NA,
    )
    degree = pl.col("Degree")
    major = pl.col("Major_Concentration")
    db48 = db48.with_columns(
        _case_when(DEGREE_LEVEL_RULES, degree, pl.lit("Other")).alias("DegreeLevel"),
        # A missing Major_Concentration is filled from the degree name first.
        pl.when(major.is_null() & degree.str.contains("(?i)law").fill_null(False))
        .then(pl.lit("Law"))
        .when(major.is_null() & degree.str.contains("(?i)MBA").fill_null(False))
        .then(pl.lit("Business"))
        .when(major.is_null() & degree.str.contains("(?i)Medicine").fill_null(False))
        .then(pl.lit("Medicine"))
        .otherwise(major)
        .alias("Major_Concentration"),
    )
    db48 = db48.with_columns(
        pl.when(pl.col("Major_Concentration").is_null())
        .then(None)
        .otherwise(_case_when(FIELD_RULES, pl.col("Major_Concentration"), pl.lit("Other")))
        .alias("Field")
    )

    degree_index = (
        pl.col("DegreeLevel")
        .replace_strict({name: i + 1 for i, name in enumerate(DEGREE_HIERARCHY)}, default=None)
        .cast(pl.Int64)
    )
    has_field = pl.col("Field").is_not_null().sum() > 0
    return db48.group_by("PersonID").agg(
        [
            pl.when(has_field).then(pl.col("Field").eq(value).any()).otherwise(None).alias(flag)
            for flag, value in FIELD_FLAGS.items()
        ]
        + [
            pl.col("GraduatingYear").cast(pl.Float64, strict=False).min().alias("Earliest_Year"),
            degree_index.max().alias("Highest_Degree"),
            pl.when(pl.col("Institute").is_not_null().sum() > 0)
            .then(pl.col("Institute").drop_nulls().str.join("; "))
            .otherwise(None)
            .alias("Institute"),
        ]
    )


def build_db3(cfg: PanelConfig) -> pl.DataFrame:
    """`1_Arrange_DB.R:240-524`."""
    raw = as_na(read_raw(cfg, "CompanyBoardTeamRelation", BOARD_COLUMNS), R_NA)
    db3 = (
        _dedup_board(raw, fix=cfg.fix_dup_coalesce)
        .select(
            "CompanyID",
            "PersonID",
            "PersonName",
            "FullTitle",
            "IsOnBoard",
            "RoleOnBoard",
            "IsCurrent",
            "StartDate",
            "EndDate",
        )  # fmt: skip
        .with_columns(parse_date_r(pl.col(c)).alias(c) for c in ("StartDate", "EndDate"))
        .sort(["CompanyID", "StartDate"], nulls_last=True)
    )

    # --- Person attributes and the experience indices, lines 292-315 --------
    person = as_na(read_raw(cfg, "Person", PERSON_COLUMNS), R_NA_NAN).with_columns(
        to_num(c) for c in [*COUNT_COLUMNS, "RolesCount"]
    )
    db3 = db3.join(person, on="PersonID", how="left")

    def _rowsum(cols: list[str]) -> pl.Expr:
        # rowSums(..., na.rm = TRUE): an all-missing row sums to 0, not null.
        return pl.sum_horizontal([pl.col(c).fill_null(0.0) for c in cols])

    db3 = db3.with_columns(
        _rowsum(COUNT_COLUMNS).alias("RolesCount_Total"),
        _rowsum(["CurrentPositionsCount", "FormerPositionsCount"]).alias("Positions"),
        _rowsum(["CurrentBoardSeatsCount", "FormerBoardSeatsCount"]).alias("BoardSeats"),
        _rowsum(
            [
                "CurrentAdvisoryRolesCount",
                "FormerAdvisoryRolesCount",
                "AffiliatedDealsCount",
                "NumberOfAffiliatedFunds",
            ]
        ).alias("OtherRoles"),
    ).with_columns(
        # scale() is global over the whole table, not per company.
        scale_r((pl.col("Positions") + 1).log()).alias("Positions_z"),
        scale_r((pl.col("BoardSeats") + 1).log()).alias("BoardSeats_z"),
        scale_r((pl.col("OtherRoles") + 1).log()).alias("OtherRoles_z"),
    )
    db3 = db3.with_columns(
        pl.mean_horizontal("Positions_z", "BoardSeats_z", "OtherRoles_z").alias(
            "WorkExperienceIndex"
        )
    )

    # --- education, then the PersonName overrides, lines 317-409 ------------
    db3 = db3.join(_education(cfg), on="PersonID", how="left")
    name = pl.col("PersonName")
    is_phd = name.str.contains(r"Ph\.?D").fill_null(False)
    is_jd = name.str.contains(" JD", literal=True).fill_null(False)
    is_md = name.str.contains(" MD", literal=True).fill_null(False)
    db3 = db3.with_columns(
        pl.when(is_phd | is_jd | is_md)
        .then(5)
        .otherwise(pl.col("Highest_Degree"))
        .alias("Highest_Degree"),
        pl.when(is_jd).then(True).otherwise(pl.col("Is_Law")).alias("Is_Law"),
        pl.when(is_md).then(True).otherwise(pl.col("Is_Med")).alias("Is_Med"),
    )

    # --- position level and IsFounder, lines 411-428 ------------------------
    db49 = as_na(
        read_raw(cfg, "PersonPositionRelation", ["PersonID", "EntityID", "PositionLevel"]),
        R_NA_NAN,
    ).unique(subset=["EntityID", "PersonID"], keep="first", maintain_order=True)
    db3 = db3.join(
        db49, left_on=["CompanyID", "PersonID"], right_on=["EntityID", "PersonID"], how="left"
    )
    # paste(x, y, sep = "; ") turns a missing value into the text "NA", so the
    # detection never yields null and IsFounder is always True or False.
    title = pl.concat_str(
        [pl.col("FullTitle").fill_null("NA"), pl.col("PositionLevel").fill_null("NA")],
        separator="; ",
    )
    db3 = db3.with_columns(title.str.contains("(?i)Found").alias("IsFounder"))

    # --- out-of-business window, lines 430-444 ------------------------------
    m1 = pl.read_parquet(cfg.interim("db_master_1_v1.parquet")).select(
        "CompanyID", "OwnershipStatus", "OwnershipStatusDate"
    )
    ownership_out = m1.filter(pl.col("OwnershipStatus") == "Out of Business").with_columns(
        pl.lit(True).alias("Is_Out"),
        pl.col("OwnershipStatusDate").alias("OutDate"),
    )
    # Bug B10: the left join leaves Is_Out null, not False, for every company
    # that is still in business, which neutralises one EndDate rule below.
    db3 = db3.join(ownership_out, on="CompanyID", how="left")

    # Line 448 takes YearFounded from the *unfiltered* company frame.
    db3 = db3.join(pl.read_parquet(cfg.interim("db1.parquet")), on="CompanyID", how="left")

    # --- StartDate / EndDate imputation, lines 450-505 ----------------------
    year_founded_start = pl.date(pl.col("YearFounded"), 1, 1)
    starts_before_founding = pl.col("StartDate").dt.year() < pl.col("YearFounded")
    db3 = db3.with_columns(
        # if_else, not when/otherwise: with no YearFounded the comparison is
        # null and R erases the StartDate instead of keeping it.
        r_if_else(
            pl.col("StartDate").is_null() | starts_before_founding,
            year_founded_start,
            pl.col("StartDate"),
        ).alias("StartDate")
    )
    end_2024 = pl.date(2024, 12, 31)
    db3 = db3.with_columns(
        # Is_Out == FALSE is null wherever Is_Out is null, so this rule fires
        # only for companies that are out of business (bug B10).
        pl.when(pl.col("EndDate").is_null() & (pl.col("IsCurrent") == "Yes") & ~pl.col("Is_Out"))
        .then(end_2024)
        .otherwise(pl.col("EndDate"))
        .alias("EndDate")
    ).with_columns(
        pl.when(pl.col("EndDate").is_null() & pl.col("Is_Out"))
        .then(pl.col("OutDate"))
        .otherwise(pl.col("EndDate"))
        .alias("EndDate")
    )

    duration = pl.col("EndDate").dt.year() - pl.col("StartDate").dt.year()
    if cfg.fix_permanenza_media_per_company:
        permanenza = duration.mean().round(0).over("CompanyID")
    else:
        # Bug B5: summarise() without group_by, so this is one global number.
        permanenza = pl.lit(
            db3.filter(pl.col("EndDate").is_not_null() & pl.col("StartDate").is_not_null())
            .select(duration.mean().round(0))
            .item()
        )
    db3 = db3.with_columns(permanenza.alias("_permanenza"))
    imputed_end = pl.min_horizontal(
        pl.date(pl.col("StartDate").dt.year() + pl.col("_permanenza").cast(pl.Int64), 1, 1),
        end_2024,
    )
    db3 = db3.with_columns(
        pl.when(
            pl.col("EndDate").is_null()
            & (pl.col("IsCurrent") == "No")
            & pl.col("StartDate").is_not_null()
            & pl.col("_permanenza").is_not_null()
        )
        .then(imputed_end)
        .when(pl.col("EndDate").is_null() & pl.col("StartDate").is_not_null())
        .then(end_2024)
        .otherwise(pl.col("EndDate"))
        .alias("EndDate")
    ).drop("_permanenza")

    # --- deltas, lines 509-524 ---------------------------------------------
    db3 = db3.with_columns(
        pl.when(pl.col("EndDate") < pl.col("StartDate"))
        .then(pl.col("StartDate"))
        .otherwise(pl.col("EndDate"))
        .alias("EndDate")
    ).with_columns(
        (pl.col("StartDate").dt.year() - pl.col("YearFounded")).alias("DeltaStart"),
        (pl.col("EndDate").dt.year() - pl.col("YearFounded")).alias("DeltaEnd"),
    )
    return db3.with_columns(
        pl.when(pl.col("IsFounder")).then(0).otherwise(pl.col("DeltaStart")).alias("DeltaStart")
    ).select(DB3_COLUMNS)


def run_db3(cfg: PanelConfig) -> None:
    build_db3(cfg).write_parquet(cfg.interim("db3.parquet"))


#: `1_Arrange_DB.R:617-644` — the team columns of db_master_2, all final after
#: this stage: nothing downstream writes to them again.
TEAM_COLUMNS = [
    "Total_People", "Percent_Females", "Is_Eco", "Is_Eng", "Is_NS", "Is_Hum",
    "Is_SS", "Is_Med", "Is_Other", "Is_Law", "Is_IT", "Avg_Earliest_Year",
    "Highest_Degree_Max", "Highest_Degree_Mean", "Institute", "RolesCount_Max",
    "RolesCount_Mean", "Positions", "BoardSeats", "OtherRoles", "WorkExp_Idx_Max",
    "WorkExp_Idx_Mean", "Total_Founders",
]  # fmt: skip

_ANY_FLAGS = [
    "Is_Eco", "Is_Eng", "Is_NS", "Is_Hum", "Is_SS", "Is_Med", "Is_Other", "Is_Law", "Is_IT",
]  # fmt: skip


def _seq_ranges(lo: pl.Expr, hi: pl.Expr) -> pl.Expr:
    """``seq(lo, hi)`` inclusive. Callers guarantee ``hi >= lo`` here: DeltaEnd
    is clamped to DeltaStart in stage 2a, so no descending case arises."""
    return pl.int_ranges(lo, hi + 1)


def run_panel(cfg: PanelConfig) -> None:
    """`1_Arrange_DB.R:527-664` — expand db3 to company-years and join the
    skeleton."""
    db3 = pl.read_parquet(cfg.interim("db3.parquet"))

    # One row per company-year spanned by anyone on its team (lines 531-544).
    company_years = (
        db3.group_by("CompanyID")
        .agg(
            pl.col("DeltaStart").min().alias("_lo"),
            pl.col("DeltaEnd").max().alias("_hi"),
        )
        .with_columns(
            pl.when(pl.col("_lo").is_null() | pl.col("_hi").is_null())
            .then(None)
            .otherwise(_seq_ranges(pl.col("_lo"), pl.col("_hi")))
            .alias("Years")
        )
        # unnest(keep_empty = TRUE): a company with no usable window keeps one
        # row with a null year.
        .explode("Years")
        .select("CompanyID", "Years")
    )

    # One row per person-year (lines 549-555).
    person_years = (
        db3.filter(pl.col("DeltaStart").is_not_null())
        .with_columns(pl.col("DeltaEnd").fill_null(pl.col("DeltaStart")))
        .with_columns(_seq_ranges(pl.col("DeltaStart"), pl.col("DeltaEnd")).alias("Years"))
        .explode("Years")
    )

    threshold = 1999 if cfg.fix_founding_year_threshold else 2000
    # YearFounded arrives from the person side, so a company-year that matched
    # nobody has a null YearFounded and this filter removes it. That is why the
    # phantom `Total_People = 1` row the aggregation could produce never exists.
    expanded = company_years.join(person_years, on=["CompanyID", "Years"], how="left").filter(
        pl.col("YearFounded") > threshold
    )

    institute = pl.col("Institute")
    if not cfg.fix_institute_na_literal:
        # Bug B7: paste() renders a missing institute as the text "NA".
        institute = institute.fill_null("NA")

    total = pl.len()
    team = expanded.group_by(["CompanyID", "Years"]).agg(
        total.alias("Total_People"),
        (pl.col("Gender").eq("Female").sum() / total * 100).alias("Percent_Females"),
        *[pl.col(c).fill_null(False).any().alias(c) for c in _ANY_FLAGS],
        pl.col("Earliest_Year").mean().alias("Avg_Earliest_Year"),
        pl.col("Highest_Degree").max().alias("Highest_Degree_Max"),
        pl.col("Highest_Degree").mean().alias("Highest_Degree_Mean"),
        institute.unique(maintain_order=True).str.join("; ").alias("Institute"),
        pl.col("RolesCount_Total").max().alias("RolesCount_Max"),
        pl.col("RolesCount_Total").mean().alias("RolesCount_Mean"),
        pl.col("Positions").mean().alias("Positions"),
        pl.col("BoardSeats").mean().alias("BoardSeats"),
        pl.col("OtherRoles").mean().alias("OtherRoles"),
        pl.col("WorkExperienceIndex").max().alias("WorkExp_Idx_Max"),
        pl.col("WorkExperienceIndex").mean().alias("WorkExp_Idx_Mean"),
        pl.col("IsFounder").sum().alias("Total_Founders"),
    )
    # Line 650's NA loop: polars already returns null where R returns -Inf from
    # max() and NaN from mean() over an all-missing group, so only the empty
    # Institute string needs turning into a null.
    team = team.with_columns(
        pl.when(pl.col("Institute") == "")
        .then(None)
        .otherwise(pl.col("Institute"))
        .alias("Institute")
    )

    skeleton = pl.read_parquet(cfg.interim("db_master_2_skeleton.parquet"))
    panel = (
        skeleton.join(
            team.rename({"Years": "Delta"}), on=["CompanyID", "Delta"], how="full", coalesce=True
        )
        .sort(["CompanyID", "Delta"])
        .with_columns(pl.col("YearFounded").fill_null(strategy="forward").over("CompanyID"))
        .with_columns(pl.col("YearFounded").fill_null(strategy="backward").over("CompanyID"))
        .with_columns(pl.col("Year_Delta").fill_null(pl.col("YearFounded") + pl.col("Delta")))
        .sort(["CompanyID", "Delta"])
    )
    panel.write_parquet(cfg.interim("db_master_2_team.parquet"))


COLUMN_FINALISED_AT_STAGE.update(dict.fromkeys(TEAM_COLUMNS, 2))
