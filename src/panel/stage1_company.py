"""Stage 1 — company cross-section and the panel skeleton.

Transcription of `1_Arrange_DB.R:37-238`. Produces three parquets:

- `db1.parquet`               CompanyID + YearFounded, **unfiltered**. Line 448
                              of the R joins YearFounded onto the team table
                              from `db1`, not from the already-filtered
                              `db_master_1`; stage 2 needs that version.
- `db_master_1_v1.parquet`    the company cross-section, YearFounded > 1999.
- `db_master_2_skeleton.parquet`  one row per company-year, with the
                              time-varying columns joined on by year.
"""

from __future__ import annotations

import polars as pl

from src.panel.config import PanelConfig
from src.panel.io import read_raw, to_num
from src.panel.rutils import R_NA, R_NA_NAN, as_na, parse_date_r
from src.panel.validate import COLUMN_FINALISED_AT_STAGE

#: `1_Arrange_DB.R:47-56`.
DB1_COLUMNS = [
    "CompanyID", "CompanyName", "CompanyLegalName", "Description", "Keywords",
    "HQCity", "HQCountry", "HQPostCode", "Website", "YearFounded",
    "CompanyFinancingStatus", "CompanyFinancingStatusDate", "BusinessStatus",
    "BusinessStatusDate", "OwnershipStatus", "OwnershipStatusDate",
    "ParentCompanyID", "PrimaryIndustrySector", "PrimaryIndustryGroup",
    "ActiveInvestors", "FormerInvestors", "FiscalPeriod", "Revenue",
    "GrossProfit", "NetIncome", "EnterpriseValue", "EBITDA", "EBIT",
    "FirstFinancingDate", "FirstFinancingSize", "FirstFinancingValuation",
    "FirstFinancingDealType", "FirstFinancingDebt", "FirstFinancingDebtSize",
    "LastKnownValuation", "LastKnownValuationDate", "FacebookProfileURL",
    "TwitterProfileURL", "LinkedInProfileURL",
]  # fmt: skip

DATE_COLUMNS = [
    "CompanyFinancingStatusDate",
    "BusinessStatusDate",
    "OwnershipStatusDate",
    "FirstFinancingDate",
    "LastKnownValuationDate",
]

FINANCIAL_COLUMNS = ["Revenue", "GrossProfit", "NetIncome", "EnterpriseValue", "EBITDA", "EBIT"]

#: `1_Arrange_DB.R:230-234` — the time-invariant company columns.
DB_MASTER_1_COLUMNS = [
    "CompanyID", "CompanyName", "CompanyLegalName", "YearFounded", "Description",
    "Keywords", "HQCity", "HQCountry", "HQPostCode", "OwnershipStatus",
    "OwnershipStatusDate", "ParentCompanyID", "PrimaryIndustrySector",
    "PrimaryIndustryGroup", "ActiveInvestors", "FormerInvestors", "Website",
    "Website_d", "Linkedin", "Facebook", "Twitter", "N_Affiliated", "Affiliated",
    "N_Parent", "N_Sister", "N_Subsidiary", "Has_Parent", "Has_Sister",
    "Has_Subsidiary",
]  # fmt: skip


def build_db2_summary(cfg: PanelConfig) -> pl.DataFrame:
    """`1_Arrange_DB.R:106-128` — affiliate counts per company.

    The NA token set here is R_NA, without "NaN": line 113 of the R differs
    from line 43, and the difference is reproduced rather than tidied away.
    """
    db2 = as_na(read_raw(cfg, "CompanyAffiliateRelation", ["CompanyID", "AffiliateType"]), R_NA)
    return db2.group_by("CompanyID").agg(
        pl.len().alias("N_Affiliated"),
        pl.col("AffiliateType").eq("Parent").sum().alias("N_Parent"),
        pl.col("AffiliateType").eq("Sister").sum().alias("N_Sister"),
        pl.col("AffiliateType").eq("Subsidiary").sum().alias("N_Subsidiary"),
    )


def expand_years(df: pl.DataFrame, *, fix_negative_delta: bool) -> pl.DataFrame:
    """Expand each company to one row per year, reproducing R's ``seq``.

    ``seq(from, to)`` counts *down* when ``to < from``, which is how a handful
    of rows end up with a negative ``Delta``.
    """
    descending = pl.col("MaxYear") < pl.col("YearFounded")
    ascending = pl.int_ranges(pl.col("YearFounded"), pl.col("MaxYear") + 1)
    if fix_negative_delta:
        years = (
            pl.when(descending)
            .then(pl.int_ranges(pl.col("YearFounded"), pl.col("YearFounded") + 1))
            .otherwise(ascending)
        )
    else:
        years = (
            pl.when(descending)
            .then(pl.int_ranges(pl.col("YearFounded"), pl.col("MaxYear") - 1, step=-1))
            .otherwise(ascending)
        )
    return (
        df.with_columns(years.alias("Year_Delta"))
        .explode("Year_Delta", empty_as_null=True)
        .with_columns((pl.col("Year_Delta") - pl.col("YearFounded")).alias("Delta"))
    )


def run(cfg: PanelConfig) -> None:
    df = as_na(read_raw(cfg, "Company", DB1_COLUMNS), R_NA_NAN)

    quarter = pl.col("FiscalPeriod").str.extract(r"TTM (\d)Q\d{4}", 1).cast(pl.Int64, strict=False)
    fiscal_year = pl.col("FiscalPeriod").str.slice(-4).cast(pl.Int64, strict=False)

    db1 = df.with_columns(
        # nchar(x) > 0 runs *after* the NA replacement, so a missing URL gives
        # null rather than False. The reference has only True and null here.
        pl.col("Website").str.len_chars().gt(0).alias("Website_d"),
        pl.col("LinkedInProfileURL").str.len_chars().gt(0).alias("Linkedin"),
        pl.col("FacebookProfileURL").str.len_chars().gt(0).alias("Facebook"),
        pl.col("TwitterProfileURL").str.len_chars().gt(0).alias("Twitter"),
        *[parse_date_r(pl.col(c)).alias(c) for c in DATE_COLUMNS],
        pl.col("YearFounded").cast(pl.Int64, strict=False),
        *[to_num(c) for c in [*FINANCIAL_COLUMNS, "LastKnownValuation"]],
    ).with_columns(
        # R builds the fiscal date as Year-Month-30; only quarters map to a
        # month, so an unparsable FiscalPeriod yields a null date.
        pl.date(fiscal_year, quarter * 3, 30).alias("FiscalDate")
    )

    # Line 448 of the R joins YearFounded from this unfiltered frame.
    db1.select("CompanyID", "YearFounded").write_parquet(cfg.interim("db1.parquet"))

    db2_summary = build_db2_summary(cfg)

    db_master_1 = (
        db1.join(db2_summary, on="CompanyID", how="left")
        .with_columns(pl.col("N_Affiliated", "N_Parent", "N_Sister", "N_Subsidiary").fill_null(0))
        .with_columns(
            pl.col("N_Affiliated").gt(0).alias("Affiliated"),
            pl.col("N_Parent").gt(0).cast(pl.Int64).alias("Has_Parent"),
            pl.col("N_Sister").gt(0).cast(pl.Int64).alias("Has_Sister"),
            pl.col("N_Subsidiary").gt(0).cast(pl.Int64).alias("Has_Subsidiary"),
        )
    )

    # --- panel skeleton, R lines 144-172 -----------------------------------
    year_cols = [*DATE_COLUMNS, "FiscalDate"]
    temp = (
        db_master_1.select(["CompanyID", "YearFounded", *year_cols])
        .with_columns(pl.max_horizontal([pl.col(c).dt.year() for c in year_cols]).alias("MaxYear"))
        .filter(pl.col("YearFounded").is_not_null() & pl.col("MaxYear").is_not_null())
        .select("CompanyID", "YearFounded", "MaxYear")
    )
    skeleton = expand_years(temp, fix_negative_delta=cfg.fix_negative_delta).select(
        "CompanyID", "YearFounded", "Year_Delta", "Delta"
    )

    # --- five joins on (CompanyID, year), R lines 174-228 -------------------
    years = db_master_1.with_columns(
        pl.col("CompanyFinancingStatusDate").dt.year().alias("_y_fin"),
        pl.col("BusinessStatusDate").dt.year().alias("_y_bus"),
        pl.col("OwnershipStatusDate").dt.year().alias("_y_own"),
        pl.col("LastKnownValuationDate").dt.year().alias("_y_val"),
        pl.col("FiscalDate").dt.year().alias("_y_fis"),
    )
    for year_col, cols in (
        ("_y_fin", ["CompanyFinancingStatus"]),
        ("_y_bus", ["BusinessStatus"]),
        ("_y_own", ["OwnershipStatus"]),
        ("_y_val", ["LastKnownValuation"]),
        ("_y_fis", FINANCIAL_COLUMNS),
    ):
        skeleton = skeleton.join(
            years.select(["CompanyID", year_col, *cols]).drop_nulls(year_col),
            left_on=["CompanyID", "Year_Delta"],
            right_on=["CompanyID", year_col],
            how="left",
        )

    db_master_1 = db_master_1.select(DB_MASTER_1_COLUMNS).filter(pl.col("YearFounded") > 1999)
    skeleton = skeleton.filter(pl.col("YearFounded") > 1999)

    db_master_1.write_parquet(cfg.interim("db_master_1_v1.parquet"))
    skeleton.write_parquet(cfg.interim("db_master_2_skeleton.parquet"))


COLUMN_FINALISED_AT_STAGE.update({"CompanyID": 1, "YearFounded": 1, "Year_Delta": 1, "Delta": 1})
