"""Stage 5 — finalisation.

Transcription of the whole of `2_Arrange_Final.R`: cumulative flags,
`GrowthStage`, the cumulative block, the CEO attributes and the column
selection that produces `db_selected` and `db_final`.
"""

from __future__ import annotations

import polars as pl

from src.panel.config import PanelConfig
from src.panel.rutils import (
    cumany,
    next_different,
    r_cum_sum,
    rle_sequence,
    stage_block,
    weighted_cumulative,
)

#: `2_Arrange_Final.R:23-78` — null becomes FALSE, then the flag latches on.
CUMULATIVE_FLAGS = [
    "Is_Preseed", "Is_Seed", "Is_EarlyVC", "Is_LaterVC", "Is_MA", "Is_Public_Exit",
    "Is_Out", "Is_Debt", "Is_PE", "Is_Grant", "Is_SpinOff", "Is_CrowdFunding",
    "Is_Accelerator", "Is_Angel", "has_Corporate", "has_VentureCapital",
    "has_PrivateEquity", "has_PublicInvestor", "has_Angel_Lead",
    "has_Corporate_Lead", "has_VentureCapital_Lead", "has_Accelerator_Lead",
    "has_PrivateEquity_Lead", "has_PublicInvestor_Lead",
]  # fmt: skip

#: The three surviving TotalRaised variants; the three `_Est` ones are not
#: produced, see stage4_deals.RF_SUSPENDED.
RAISED = ["TotalRaised", "TotalRaised_NA", "TotalRaised_any"]

#: `2_Arrange_Final.R:122-124` — cumulated as a mean weighted by NewInvestors.
WEIGHTED = [
    "MeanTotalActivePortfolio",
    "MeanTotalInvestments",
    "MeanMedianRoundAmount",
    "MeanMedianValuation",
]

#: `2_Arrange_Final.R:150-154` — db3 attributes carried over for the CEO.
CEO_VARS = [
    "Gender", "University_Institution", "RolesCount_Total", "Positions",
    "BoardSeats", "OtherRoles", "WorkExperienceIndex", "Is_Eco", "Is_Eng",
    "Is_Med", "Is_Hum", "Is_IT", "Is_Law", "Is_NS", "Is_SS", "Is_Other",
    "Earliest_Year", "Highest_Degree",
]  # fmt: skip

#: `2_Arrange_Final.R:176-198`, minus the six `_Est` columns and plus TR_D,
#: which the R computes and then drops here.
VARS_SELECTED = [
    "CompanyID", "Year_Delta", "YearFounded", "Age", "GrowthStage", "N_Deal",
    "N_VCround", "TotalRaised", "TotalRaised_NA", "TotalRaised_any",
    "TotalRaised_cum", "TotalRaised_NA_cum", "TotalRaised_any_cum", "TR_D",
    "CompanyFinancingStatus", "CompanyBusinessStatus", "OwnershipStatus",
    "Total_People", "Percent_Females", "Is_Eco", "Is_Eng", "Is_NS", "Is_Hum",
    "Is_SS", "Is_Med", "Is_Other", "Is_Law", "Is_IT", "Avg_Earliest_Year",
    "Highest_Degree_Max", "Highest_Degree_Mean", "Institute", "RolesCount_Max",
    "Positions", "BoardSeats", "OtherRoles", "WorkExp_Idx_Max",
    "WorkExp_Idx_Mean", "Total_Founders", "EmployeeCount", "N_News", "Is_Debt",
    "Is_Grant", "Is_SpinOff", "Is_CrowdFunding", "TotalInvestors",
    "MeanTotalActivePortfolio_cum", "MeanTotalInvestments_cum",
    "MeanMedianRoundAmount_cum", "MeanMedianValuation_cum", "Is_Accelerator",
    "Is_Angel", "has_Corporate", "has_VentureCapital", "has_PrivateEquity",
    "has_PublicInvestor", "LeadInvestors", "has_Angel_Lead",
    "has_Corporate_Lead", "has_VentureCapital_Lead", "has_Accelerator_Lead",
    "has_PrivateEquity_Lead", "has_PublicInvestor_Lead", "LastKnownValuation",
    "Revenue", "GrossProfit", "NetIncome", "EnterpriseValue", "EBITDA", "EBIT",
    *[f"{v}_CEO" for v in CEO_VARS],
]  # fmt: skip

#: `2_Arrange_Final.R:236-239` — the 22 company columns joined into db_final.
DB_FINAL_COMPANY_COLUMNS = [
    "CompanyID", "CompanyName", "Description", "Keywords", "HQCity", "HQCountry",
    "PrimaryIndustryGroup", "PrimaryIndustrySector", "Website_d", "Linkedin",
    "Facebook", "Twitter", "N_Affiliated", "Has_Parent", "Has_Sister",
    "Has_Subsidiary", "SimilarityScoreMean", "SimilarityScoreMax",
    "N_Competitors", "Same_Country", "N_Europe", "N_Outside_Europe",
]  # fmt: skip

OUT_OF_BUSINESS = "Out of Business"
PUBLIC = ["Publicly Held", "In IPO Registration"]
ACQUIRED = ["Acquired/Merged", "Acquired/Merged (Operating Subsidiary)"]


def _growth_stage() -> pl.Expr:
    """`2_Arrange_Final.R:80-92` — seven conditions, first match wins."""
    own = pl.col("OwnershipStatus")
    out, public, ma = pl.col("Is_Out"), pl.col("Is_Public_Exit"), pl.col("Is_MA")
    later, pe = pl.col("Is_LaterVC"), pl.col("Is_PE")
    early, seed, preseed = pl.col("Is_EarlyVC"), pl.col("Is_Seed"), pl.col("Is_Preseed")
    terminal = ~ma & ~public & ~out
    return (
        pl.when((own == OUT_OF_BUSINESS) | out)
        .then(pl.lit("Out"))
        .when(own.is_in(PUBLIC) | public)
        .then(pl.lit("Exit_Public"))
        .when(own.is_in(ACQUIRED) | ma)
        .then(pl.lit("Exit_M&A"))
        .when((later | pe) & terminal)
        .then(pl.lit("LaterVC_or_Other"))
        .when(early & ~later & terminal & ~pe)
        .then(pl.lit("EarlyVC"))
        .when(seed & ~early & ~later & terminal & ~pe)
        .then(pl.lit("Seed"))
        .when(preseed & ~seed & ~early & ~later & terminal & ~pe)
        .then(pl.lit("Preseed"))
        .otherwise(None)
    )


def build_db_master_2(cfg: PanelConfig) -> pl.DataFrame:
    """`2_Arrange_Final.R:23-167`."""
    panel = pl.read_parquet(cfg.interim("db_master_2_deals.parquet")).sort(
        ["CompanyID", "Year_Delta"]
    )

    panel = panel.with_columns(
        cumany(pl.col(c)).over("CompanyID").alias(c) for c in CUMULATIVE_FLAGS
    )
    panel = panel.with_columns(_growth_stage().alias("GrowthStage"))

    # Where the company-year has no deal at all, every raised variant is zero.
    panel = panel.with_columns(
        pl.when(pl.col("TR_D") == 1).then(0.0).otherwise(pl.col(c)).alias(c) for c in RAISED
    )

    # dplyr evaluates a mutate in order: NewInvestors is read off TotalInvestors
    # before TotalInvestors is replaced by its own cumulative sum.
    panel = panel.with_columns(pl.col("TotalInvestors").fill_null(0).alias("NewInvestors"))
    panel = panel.with_columns(
        r_cum_sum(pl.col("N_Deal").fill_null(0)).over("CompanyID").alias("N_Deal"),
        r_cum_sum(pl.col("N_VCround").fill_null(0)).over("CompanyID").alias("N_VCround"),
        *[r_cum_sum(pl.col(c)).over("CompanyID").alias(f"{c}_cum") for c in RAISED],
        r_cum_sum(pl.col("NewInvestors").fill_null(0)).over("CompanyID").alias("TotalInvestors"),
        r_cum_sum(pl.col("LeadInvestorCount").fill_null(0))
        .over("CompanyID")
        .alias("LeadInvestors"),
    )
    panel = panel.with_columns(
        weighted_cumulative(c, "NewInvestors", ["CompanyID"]).alias(f"{c}_cum") for c in WEIGHTED
    )

    # CEO_ID carries forward until the company names a different one.
    panel = panel.with_columns(pl.col("CEO_ID").fill_null(strategy="forward").over("CompanyID"))
    db3 = (
        pl.read_parquet(cfg.interim("db3.parquet"))
        .select("CompanyID", "PersonID", *CEO_VARS)
        .rename({v: f"{v}_CEO" for v in CEO_VARS})
    )
    return panel.join(
        db3, left_on=["CompanyID", "CEO_ID"], right_on=["CompanyID", "PersonID"], how="left"
    )


def build_db_selected(cfg: PanelConfig, db_master_2: pl.DataFrame) -> pl.DataFrame:
    """`2_Arrange_Final.R:200-232`."""
    df = db_master_2.with_columns(pl.col("Delta").alias("Age")).select(VARS_SELECTED)
    df = df.sort(["CompanyID", "Age"])
    df = stage_block(df, "GrowthStage", ["CompanyID"], "StageBlock", fix=cfg.fix_stageblock_na)
    df = rle_sequence(df, "GrowthStage", ["CompanyID"], "YearsInStage")
    return next_different(df, "GrowthStage", ["CompanyID"], "GrowthNextStage", "TimeNextStage")


def run(cfg: PanelConfig) -> None:
    db_master_2 = build_db_master_2(cfg)
    db_master_2.write_parquet(cfg.interim("db_master_2.parquet"))

    db_selected = build_db_selected(cfg, db_master_2)
    db_selected.write_parquet(cfg.interim("db_selected.parquet"))

    company = pl.read_parquet(cfg.interim("db_master_1.parquet")).select(DB_FINAL_COMPANY_COLUMNS)
    db_final = db_selected.join(company, on="CompanyID", how="left")
    if db_final.height != db_selected.height:
        raise ValueError(
            f"db_final join duplicated rows: {db_selected.height} -> {db_final.height}"
        )
    db_final.write_parquet(cfg.interim("db_final.parquet"))
