"""Stage 4 — deals and investors.

Transcription of `1_Arrange_DB.R:826-1251`, minus the RandomForest imputation:
see `RF_SUSPENDED` below for what it created and where.
"""

from __future__ import annotations

import polars as pl

from src.panel.config import PanelConfig
from src.panel.io import read_raw, to_num
from src.panel.rutils import R_NA_INF, R_NA_NAN, as_na, parse_date_r, tail_na_omit
from src.panel.validate import COLUMN_FINALISED_AT_STAGE

# ---------------------------------------------------------------------------
# THE RANDOMFOREST IMPUTATION IS NOT PORTED
#
# `1_Arrange_DB.R:1037-1119` fits `randomForest(TotalInvestedCapital ~ .,
# ntree = 50)` with **no set.seed** and uses it to fill missing deal amounts.
# It is suspended here, so the seven columns below are simply not produced;
# missing amounts stay missing and are filled by the imputation that already
# runs before model training.
#
#   TotalInvestedCapital_Est   deal level, R line 1010. Initialised to
#                              TotalInvestedCapital, then overwritten on the
#                              rows where the amount is missing, the synopsis
#                              matches the "undisclosed" regex, and every
#                              predictor is present; the prediction is capped
#                              at the third quartile of its DealTypeGrouped.
#   TotalRaised_Est            deals_panel, R line 1160. sum(na.rm = TRUE)
#   TotalRaised_Est_NA         deals_panel, R line 1162. NA if all NA
#   TotalRaised_Est_any        deals_panel, R line 1164. NA if any NA
#   TotalRaised_Est_cum        2_Arrange_Final.R:114, cumsum of the above
#   TotalRaised_Est_any_cum    2_Arrange_Final.R:113
#   TotalRaised_Est_NA_cum     2_Arrange_Final.R:115
#
# Why it is suspended, beyond the missing seed: the model is trained across all
# years and imputes a 2013 deal from 2020 patterns, it is fitted on the whole
# dataset before any train/test split, and `DealTypeGrouped` is among its
# predictors while the deal types are what determine the target. Measured on
# the published datasets, it supplied 19.5% of the rows of `TotalRaised_Est` in
# dataset_window, worth 29.7% of the feature's mass.
#
# The Zero_Invested rule below is a milder member of the same family and is
# kept, because it feeds `TotalRaised`, which is not an imputation.
# ---------------------------------------------------------------------------
RF_SUSPENDED = (
    "TotalInvestedCapital_Est",
    "TotalRaised_Est",
    "TotalRaised_Est_NA",
    "TotalRaised_Est_any",
    "TotalRaised_Est_cum",
    "TotalRaised_Est_any_cum",
    "TotalRaised_Est_NA_cum",
)

#: `1_Arrange_DB.R:861-871`.
INVESTOR_CATEGORY = {
    "Venture Capital": "Venture Capital",
    "Corporate Venture Capital": "Venture Capital",
    "Growth/Expansion": "Venture Capital",
    "Not-For-Profit Venture Capital": "Venture Capital",
    "VC-Backed Company": "Venture Capital",
    "Angel (individual)": "Angel",
    "Angel Group": "Angel",
    "Accelerator/Incubator": "Accelerator",
    "Corporation": "Corporate",
    "Corporate Development": "Corporate",
    "PE/Buyout": "Private Equity",
    "Family Office": "Private Equity",
    "PE-Backed Company": "Private Equity",
    "Holding Company": "Private Equity",
    "Merchant Banking Firm": "Private Equity",
    "Mezzanine": "Private Equity",
    "Secondary Buyer": "Private Equity",
    "Other Private Equity": "Private Equity",
    "Special Purpose Acquisition Company (SPAC)": "Private Equity",
    "Fundless Sponsor": "Private Equity",
    "Government": "Public Investor",
    "University": "Public Investor",
    "Sovereign Wealth Fund": "Public Investor",
    "Mutual Fund": "Public Investor",
}

CATEGORIES = [
    "Angel",
    "Corporate",
    "VentureCapital",
    "Accelerator",
    "PrivateEquity",
    "PublicInvestor",
]
CATEGORY_VALUES = {
    "Angel": "Angel",
    "Corporate": "Corporate",
    "VentureCapital": "Venture Capital",
    "Accelerator": "Accelerator",
    "PrivateEquity": "Private Equity",
    "PublicInvestor": "Public Investor",
}

#: `1_Arrange_DB.R:1122-1128` — the DealType lists behind the 15 flags.
PRESEED = [
    "Accelerator/Incubator",
    "Angel (individual)",
    "Grant",
    "Spin-Off",
    "Equity Crowdfunding",
    "Product Crowdfunding",
    "Capitalization",
]
MA_EXIT = [
    "Merger/Acquisition",
    "Buyout/LBO",
    "Debt - Acquisition",
    "Debt - Merger",
    "Merger of Equals",
    "Investor Buyout by Management",
    "Corporate Asset Purchase",
    "Reverse Merger",
]
PUBLIC_EXIT = [
    "IPO",
    "Secondary Transaction - Open Market",
    "Secondary Transaction - Stock Distribution",
    "Public Investment 2nd Offering",
    "PIPE",
]
FAILURE = [
    "Bankruptcy: Admin/Reorg",
    "Bankruptcy: Liquidation",
    "Out of Business",
    "Restart - Angel",
    "Restart - Early VC",
    "Restart - Later VC",
]
DEBT = [
    "Debt - General",
    "Debt Conversion",
    "Mezzanine",
    "Convertible Debt",
    "Debt Refinancing",
    "Debt - PPP",
    "Debt Repayment",
    "Dividend Recapitalization",
    "Exit Financing",
    "Project Financing",
    "Share Repurchase",
    "Leveraged Recapitalization",
]
PE_STATUSES = [
    "PE Growth/Expansion",
    "Secondary Transaction - Private",
    "Corporate",
    "Platform Creation",
    "GP Stakes",
    "General Corporate Purpose",
    "Capital Spending",
]
OTHER_STATUSES = [
    "Joint Venture",
    "Undetermined",
    "Vendor Loan",
    "Corporate Licensing",
    "Sale-Lease back facility",
]

BANKRUPTCY = ["Bankruptcy: Admin/Reorg", "Bankruptcy: Liquidation", "Out of Business"]
FIRST_ROUND = [
    "Accelerator/Incubator",
    "Angel (individual)",
    "Grant",
    "Capitalization",
    "Early Stage VC",
    "Seed Round",
    "Spin-Off",
]
ACQUIRED = ["Acquired/Merged", "Acquired/Merged (Operating Subsidiary)"]

DEAL_FLAGS = {
    "Is_Preseed": PRESEED,
    "Is_Seed": ["Seed Round"],
    "Is_EarlyVC": ["Early Stage VC"],
    "Is_LaterVC": ["Later Stage VC"],
    "Is_MA": MA_EXIT,
    "Is_Public_Exit": PUBLIC_EXIT,
    "Is_Out": FAILURE,
    "Is_Debt": DEBT,
    "Is_PE": PE_STATUSES,
    "Is_Grant": ["Grant"],
    "Is_SpinOff": ["Spin-Off"],
    "Is_CrowdFunding": ["Equity Crowdfunding", "Product Crowdfunding"],
    "Is_Accelerator": ["Accelerator/Incubator"],
    "Is_Angel": ["Angel (individual)"],
    "Other_Deal": OTHER_STATUSES,
}

INVESTOR_NUMERIC = [
    "TotalActivePortfolio",
    "TotalInvestments",
    "MedianRoundAmount",
    "MedianValuation",
]


def _r_ifelse(cond: pl.Expr, then: pl.Expr, otherwise) -> pl.Expr:
    """`ifelse` inside a summarise: NA condition gives NA."""
    return pl.when(cond.is_null()).then(None).when(cond).then(then).otherwise(otherwise)


def _build_deal_investors(cfg: PanelConfig) -> pl.DataFrame:
    """`1_Arrange_DB.R:830-904` — one row per DealID."""
    db22 = as_na(
        read_raw(
            cfg,
            "DealInvestorRelation",
            ["DealID", "InvestorID", "InvestorStatus", "IsLeadInvestor"],
        ),
        R_NA_NAN,
    )
    db32 = as_na(
        read_raw(
            cfg,
            "Investor",
            ["InvestorID", "PrimaryInvestorType", *INVESTOR_NUMERIC, "PreferredVerticals"],
        ),
        R_NA_NAN,
    ).with_columns(to_num(c) for c in INVESTOR_NUMERIC)

    db22 = db22.join(db32, on="InvestorID", how="left").with_columns(
        pl.col("PrimaryInvestorType")
        .replace_strict(INVESTOR_CATEGORY, default="Other")
        .alias("InvestorCategory")
    )

    new = pl.col("InvestorStatus") == "New Investor"
    lead = pl.col("IsLeadInvestor") == "Yes"
    # `any(InvestorStatus == "New Investor")` has no na.rm, so it is null when
    # nothing matches and something is missing; every aggregate gated on it
    # then becomes null too.
    gate = (
        pl.when(new.fill_null(False).any())
        .then(True)
        .when(new.is_null().any())
        .then(None)
        .otherwise(False)
    )

    def _mean_new(col: str) -> pl.Expr:
        return _r_ifelse(gate, pl.col(col).filter(new.fill_null(False)).mean(), None)

    def _has(category: str, mask: pl.Expr) -> pl.Expr:
        member = (
            pl.col("InvestorCategory").filter(mask.fill_null(False)) == CATEGORY_VALUES[category]
        )
        return _r_ifelse(gate, member.fill_null(False).any(), None)

    return db22.group_by("DealID").agg(
        new.fill_null(False).sum().alias("TotalInvestors"),
        _mean_new("TotalActivePortfolio").alias("MeanTotalActivePortfolio"),
        _mean_new("TotalInvestments").alias("MeanTotalInvestments"),
        _mean_new("MedianRoundAmount").alias("MeanMedianRoundAmount"),
        _mean_new("MedianValuation").alias("MeanMedianValuation"),
        _r_ifelse(
            gate,
            pl.col("PreferredVerticals")
            .filter(new.fill_null(False) & pl.col("PreferredVerticals").is_not_null())
            .unique(maintain_order=True)
            .str.join(", "),
            None,
        ).alias("PreferredVerticals"),
        *[_has(c, new).alias(f"has_{c}") for c in CATEGORIES],
        (new.fill_null(False) & lead.fill_null(False)).sum().alias("LeadInvestorCount"),
        # The lead block filters on IsLeadInvestor but keeps the New Investor
        # gate: the asymmetry is in the R and is reproduced.
        *[_has(c, lead).alias(f"has_{c}_Lead") for c in CATEGORIES],
    )


def _repair_deal_dates(deals: pl.DataFrame, *, threshold: int) -> pl.DataFrame:
    """`1_Arrange_DB.R:907-976` — four successive fills of a missing DealDate."""
    missing = pl.col("DealDate").is_null()
    own_date = pl.col("OwnershipStatusDate")
    deals = deals.with_columns(
        pl.when(
            missing
            & pl.col("DealType").is_in(BANKRUPTCY)
            & (pl.col("OwnershipStatus") == "Out of Business")
            & own_date.is_not_null()
        )
        .then(own_date)
        .otherwise(pl.col("DealDate"))
        .alias("DealDate")
    ).with_columns(
        pl.when(
            pl.col("DealDate").is_null()
            & (pl.col("DealType") == "Merger/Acquisition")
            & pl.col("OwnershipStatus").is_in(ACQUIRED)
            & own_date.is_not_null()
        )
        .then(own_date)
        .otherwise(pl.col("DealDate"))
        .alias("DealDate")
    )
    # Bug B1 again: the deal table cuts at > 2000 while stage 1 cut at > 1999.
    deals = deals.filter(pl.col("YearFounded") > threshold).with_columns(
        pl.when(
            pl.col("DealDate").is_null()
            & pl.col("DealType").is_in(FIRST_ROUND)
            & (pl.col("DealNo") == 1)
            & pl.col("YearFounded").is_not_null()
        )
        .then(pl.date(pl.col("YearFounded"), 1, 1))
        .otherwise(pl.col("DealDate"))
        .alias("DealDate")
    )
    # Remaining holes are set to the midpoint of the surrounding deal years,
    # rounded up, which needs lag/lead ordered by (CompanyID, DealNo).
    deals = deals.sort(["CompanyID", "DealNo"]).with_columns(
        pl.col("DealDate").dt.year().alias("_year")
    )
    prev_year = pl.col("_year").shift(1).over("CompanyID")
    next_year = pl.col("_year").shift(-1).over("CompanyID")
    return deals.with_columns(
        pl.when(
            pl.col("DealDate").is_null()
            & (pl.col("DealNo") > 1)
            & prev_year.is_not_null()
            & next_year.is_not_null()
        )
        .then(pl.date(((prev_year + next_year) / 2).ceil().cast(pl.Int64), 1, 1))
        .otherwise(pl.col("DealDate"))
        .alias("DealDate")
    ).drop("_year")


def build_deals(cfg: PanelConfig) -> pl.DataFrame:
    """`1_Arrange_DB.R:907-1145` — the deal-level frame."""
    deals = as_na(
        read_raw(
            cfg,
            "Deal",
            [
                "CompanyID",
                "DealID",
                "DealNo",
                "DealDate",
                "DealType",
                "PercentAcquired",
                "VCRound",
                "TotalInvestedCapital",
                "InvestorOwnership",
                "BusinessStatus",
                "FinancingStatus",
                "PremoneyValuation",
                "PostValuation",
                "CEOPBId",
                "DealSynopsis",
            ],
        ),
        R_NA_NAN,
    ).with_columns(
        pl.col("DealNo").cast(pl.Int64, strict=False),
        parse_date_r(pl.col("DealDate")).alias("DealDate"),
        *[
            to_num(c)
            for c in [
                "TotalInvestedCapital",
                "InvestorOwnership",
                "PremoneyValuation",
                "PostValuation",
                "PercentAcquired",
            ]
        ],
    )

    m1 = pl.read_parquet(cfg.interim("db_master_1.parquet")).select(
        "CompanyID",
        "YearFounded",
        "OwnershipStatus",
        "OwnershipStatusDate",
        "HQCountry",
        "PrimaryIndustrySector",
    )
    deals = deals.join(m1, on="CompanyID", how="left")

    threshold = 1999 if cfg.fix_founding_year_threshold else 2000
    deals = _repair_deal_dates(deals, threshold=threshold)

    deal_year = pl.col("DealDate").dt.year()
    deals = deals.with_columns(
        # Deals dated before the founding year are pulled onto it. R's pmax has
        # no na.rm here, so a deal that never got a date keeps Year_Delta NA and
        # then falls out of the panel entirely at the join; max_horizontal
        # ignores nulls instead and would park those deals on the founding year,
        # inventing deals in 7,335 company-years.
        pl.when(deal_year.is_null())
        .then(None)
        .otherwise(pl.max_horizontal(deal_year, pl.col("YearFounded")))
        .alias("Year_Delta"),
        pl.col("DealSynopsis")
        .str.contains("(?i)undisclosed amount|raised|received")
        .fill_null(False)
        .cast(pl.Int64)
        .alias("UndisclosedAmountFlag"),
    )
    deals = deals.join(_build_deal_investors(cfg), on="DealID", how="left")
    deals = as_na(deals, R_NA_NAN)

    # --- Zero_Invested, R lines 994-1019 -----------------------------------
    # Derived from statistics over the whole dataset, which is the same family
    # of problem as the suspended RandomForest, milder because it only says
    # "this deal type never discloses an amount, so call it zero".
    by_type = deals.group_by("DealType").agg(
        pl.len().alias("n_all"),
        pl.col("TotalInvestedCapital").is_null().sum().alias("n_missing"),
    )
    # tbl1 only lists deal types that appear with a missing amount at all.
    by_type = by_type.filter(pl.col("n_missing") > 0).with_columns(
        (pl.col("n_missing") / pl.col("n_all")).alias("perc")
    )
    cat0 = by_type.filter(pl.col("perc") > 0.9)["DealType"].to_list()
    cat_other = by_type.filter((pl.col("n_all") < 200) & (pl.col("perc") < 0.9))[
        "DealType"
    ].to_list()

    deals = deals.with_columns(
        pl.when(pl.col("DealType").is_in(cat0))
        .then(pl.lit("Zero_Invested"))
        .when(pl.col("DealType").is_in(cat_other))
        .then(pl.lit("Other"))
        .otherwise(pl.col("DealType"))
        .alias("DealTypeGrouped")
    ).with_columns(
        pl.when(
            pl.col("TotalInvestedCapital").is_null()
            & (pl.col("DealTypeGrouped") == "Zero_Invested")
        )
        .then(0.0)
        .otherwise(pl.col("TotalInvestedCapital"))
        .alias("TotalInvestedCapital")
    )

    # RandomForest imputation of TotalInvestedCapital_Est would run here; see
    # RF_SUSPENDED at the top of this module.

    return deals.with_columns(
        *[
            pl.col("DealType").is_in(values).fill_null(False).alias(flag)
            for flag, values in DEAL_FLAGS.items()
        ]
    )


def _any_narm(col: str) -> pl.Expr:
    return pl.col(col).fill_null(False).any()


def build_deals_panel(deals: pl.DataFrame) -> pl.DataFrame:
    """`1_Arrange_DB.R:1147-1209` — aggregate to (CompanyID, Year_Delta)."""
    tic = pl.col("TotalInvestedCapital")
    vcround = pl.col("VCRound")

    def _tail(col: str) -> pl.Expr:
        return tail_na_omit(col)

    return deals.group_by(["CompanyID", "Year_Delta"]).agg(
        pl.len().alias("N_Deal"),
        pl.col("DealType")
        .drop_nulls()
        .unique(maintain_order=True)
        .str.join("; ")
        .alias("DealType"),
        (vcround.is_not_null() & (vcround != "") & (vcround != "Angel")).sum().alias("N_VCround"),
        tic.fill_null(0.0).sum().alias("TotalRaised"),
        # Three different missing semantics on the same numbers, and the
        # difference is the point: sum-with-na.rm, null-if-all, null-if-any.
        pl.when(tic.is_null().all())
        .then(None)
        .otherwise(tic.fill_null(0.0).sum())
        .alias("TotalRaised_NA"),
        pl.when(tic.is_null().any()).then(None).otherwise(tic.sum()).alias("TotalRaised_any"),
        _tail("FinancingStatus").alias("FinancingStatus"),
        _tail("BusinessStatus").alias("BusinessStatus"),
        _tail("InvestorOwnership").alias("InvestorOwnership"),
        _tail("PremoneyValuation").alias("PremoneyValuation"),
        _tail("PostValuation").alias("PostValuation"),
        *[
            _any_narm(flag).alias(flag)
            for flag in DEAL_FLAGS
            if flag not in ("Is_Accelerator", "Is_Angel")
        ],
        (_any_narm("Is_Accelerator") | _any_narm("has_Accelerator")).alias("Is_Accelerator"),
        (_any_narm("Is_Angel") | _any_narm("has_Angel")).alias("Is_Angel"),
        pl.col("TotalInvestors").fill_null(0).sum().alias("TotalInvestors"),
        pl.col("MeanTotalActivePortfolio").mean().alias("MeanTotalActivePortfolio"),
        pl.col("MeanTotalInvestments").mean().alias("MeanTotalInvestments"),
        pl.col("MeanMedianRoundAmount").mean().alias("MeanMedianRoundAmount"),
        pl.col("MeanMedianValuation").mean().alias("MeanMedianValuation"),
        pl.col("PreferredVerticals").first().alias("PreferredVerticals"),
        *[
            _any_narm(f"has_{c}").alias(f"has_{c}")
            for c in ("Corporate", "VentureCapital", "PrivateEquity", "PublicInvestor")
        ],
        pl.col("LeadInvestorCount").fill_null(0).sum().alias("LeadInvestorCount"),
        *[_any_narm(f"has_{c}_Lead").alias(f"has_{c}_Lead") for c in CATEGORIES],
        _tail("CEOPBId").alias("CEO_ID"),
    )


def run(cfg: PanelConfig) -> None:
    deals = build_deals(cfg)
    deals_panel = build_deals_panel(deals)
    deals_panel.write_parquet(cfg.interim("deals_panel.parquet"))

    panel = pl.read_parquet(cfg.interim("db_master_2_relations.parquet")).rename(
        {"BusinessStatus": "CompanyBusinessStatus"}
    )
    panel = panel.join(deals_panel, on=["CompanyID", "Year_Delta"], how="left").with_columns(
        pl.coalesce("CompanyFinancingStatus", "FinancingStatus").alias("CompanyFinancingStatus"),
        pl.coalesce("CompanyBusinessStatus", "BusinessStatus").alias("CompanyBusinessStatus"),
        pl.coalesce("LastKnownValuation", "PostValuation").alias("LastKnownValuation"),
        # R tests all six TotalRaised variants; without the three _Est ones the
        # condition is the same, because they were null exactly when these are.
        pl.sum_horizontal(pl.col("TotalRaised", "TotalRaised_NA", "TotalRaised_any").is_null())
        .eq(3)
        .cast(pl.Int64)
        .alias("TR_D"),
    )
    panel = as_na(panel.drop("FinancingStatus", "BusinessStatus", "PostValuation"), R_NA_INF)
    panel.write_parquet(cfg.interim("db_master_2_deals.parquet"))


COLUMN_FINALISED_AT_STAGE.update(
    dict.fromkeys(
        ["DealType", "InvestorOwnership", "PremoneyValuation", "PreferredVerticals", "TR_D"], 4
    )
)
