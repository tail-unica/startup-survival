"""Stage 3 — competitors, employees, financials and news.

`run_competitors` transcribes `1_Arrange_DB.R:681-708` and completes
`db_master_1` (checkpoint B); `run_panel` transcribes lines 710-813 and adds
`EmployeeCount`, the financial fill and `N_News` to the panel.
"""

from __future__ import annotations

import polars as pl

from src.panel.config import PanelConfig
from src.panel.io import load_europe, read_raw, to_num
from src.panel.rutils import R_NA, R_NA_NAN, as_na, parse_date_r
from src.panel.validate import COLUMN_FINALISED_AT_STAGE

COMPETITOR_COLUMNS = [
    "SimilarityScoreMean",
    "SimilarityScoreMax",
    "N_Competitors",
    "Same_Country",
    "N_Europe",
    "N_Outside_Europe",
]

FINANCIAL_COLUMNS = ["Revenue", "GrossProfit", "NetIncome", "EnterpriseValue", "EBITDA", "EBIT"]


def _no_narm(agg: pl.Expr, col: pl.Expr) -> pl.Expr:
    """An R aggregate called without `na.rm`: one missing value nulls it all."""
    return pl.when(col.is_null().any()).then(None).otherwise(agg)


def run_competitors(cfg: PanelConfig) -> None:
    """`1_Arrange_DB.R:681-708`."""
    db17 = as_na(
        read_raw(
            cfg,
            "CompanySimilarRelation",
            [
                "CompanyID",
                "SimilarCompanyID",
                "SimilarityScore",
                "IsCompetitor",
                "SimilarCompanyHQCountry",
            ],
        ),
        R_NA_NAN,
    ).with_columns(to_num("SimilarityScore"))

    m1 = pl.read_parquet(cfg.interim("db_master_1_v1.parquet"))
    db17 = db17.join(m1.select("CompanyID", "HQCountry"), on="CompanyID", how="left")

    europe = load_europe()
    score = pl.col("SimilarityScore")
    same = pl.col("SimilarCompanyHQCountry") == pl.col("HQCountry")
    db17 = db17.with_columns(
        pl.col("SimilarCompanyHQCountry")
        .replace_strict(europe, default=None, return_dtype=pl.Boolean)
        .alias("_is_europe"),
        # R subsets with `score > 90`; a null score yields an NA element in the
        # subset rather than dropping the row, so it still reaches any().
        (score.is_null() | (score > 90)).alias("_keep"),
        pl.when(score.is_null()).then(None).otherwise(same).alias("_same"),
    )

    truth = pl.col("_same").filter(pl.col("_keep"))
    is_competitor = pl.col("IsCompetitor") == "Yes"
    db17_s = db17.group_by("CompanyID").agg(
        _no_narm(score.mean(), score).alias("SimilarityScoreMean"),
        _no_narm(score.max(), score).alias("SimilarityScoreMax"),
        _no_narm(is_competitor.sum(), pl.col("IsCompetitor")).alias("N_Competitors"),
        # Bug B4: any() without na.rm is null when nothing is true and
        # something is missing.
        pl.when(truth.fill_null(False).any())
        .then(True)
        .when(truth.is_null().any())
        .then(None)
        .otherwise(False)
        .alias("Same_Country"),
        # Bug B8: N_Europe counts every row, N_Outside_Europe only the rows
        # scoring above 90.
        pl.col("_is_europe").fill_null(False).sum().alias("N_Europe"),
        (~pl.col("_is_europe").fill_null(True) & (score > 90).fill_null(False))
        .sum()
        .alias("N_Outside_Europe"),
    )

    m1.join(db17_s, on="CompanyID", how="left").write_parquet(cfg.interim("db_master_1.parquet"))


def _latest_per_year(df: pl.DataFrame, date_col: str, year_col: str) -> pl.DataFrame:
    """`arrange(CompanyID, year, desc(Date))` then `distinct` — the last
    reading of each year wins, with missing dates sorted last."""
    return df.sort(
        ["CompanyID", year_col, date_col],
        descending=[False, False, True],
        nulls_last=True,
    ).unique(subset=["CompanyID", year_col], keep="first", maintain_order=True)


def run_panel(cfg: PanelConfig) -> None:
    """`1_Arrange_DB.R:710-813`."""
    panel = pl.read_parquet(cfg.interim("db_master_2_team.parquet"))

    # --- employees, lines 711-732 ------------------------------------------
    db6 = (
        as_na(
            read_raw(cfg, "CompanyEmployeeHistoryRelation", ["CompanyID", "Date", "EmployeeCount"]),
            R_NA,
        )
        .with_columns(parse_date_r(pl.col("Date")).alias("Date"), to_num("EmployeeCount"))
        .with_columns(pl.col("Date").dt.year().alias("Year"))
    )
    db6 = _latest_per_year(db6, "Date", "Year").select("CompanyID", "Year", "EmployeeCount")
    panel = panel.join(
        db6, left_on=["CompanyID", "Year_Delta"], right_on=["CompanyID", "Year"], how="left"
    )

    # --- financials, lines 736-774 -----------------------------------------
    db8 = (
        as_na(
            read_raw(
                cfg,
                "CompanyFinancialRelation",
                ["CompanyID", "PeriodEndDate", *FINANCIAL_COLUMNS],
            ),
            R_NA,
        )
        .with_columns(parse_date_r(pl.col("PeriodEndDate")).alias("Date"))
        .with_columns(
            pl.col("Date").dt.year().alias("Year_Delta"),
            *[to_num(c) for c in FINANCIAL_COLUMNS],
        )
    )
    db8 = _latest_per_year(db8, "Date", "Year_Delta").select(
        "CompanyID", "Year_Delta", *FINANCIAL_COLUMNS
    )
    panel = panel.join(db8, on=["CompanyID", "Year_Delta"], how="left", suffix="_db8").with_columns(
        # The financial table only fills cells that are already empty.
        pl.coalesce(pl.col(c), pl.col(f"{c}_db8")).alias(c)
        for c in FINANCIAL_COLUMNS
    )
    panel = panel.drop([f"{c}_db8" for c in FINANCIAL_COLUMNS])

    # --- news, lines 776-813 -----------------------------------------------
    db14 = (
        as_na(read_raw(cfg, "CompanyNewsRelation", ["CompanyID", "PublishDate"]), R_NA)
        .with_columns(parse_date_r(pl.col("PublishDate")).dt.year().alias("Year"))
        .group_by("CompanyID", "Year")
        .agg(pl.len().alias("N_News"))
    )
    panel = panel.join(
        db14, left_on=["CompanyID", "Year_Delta"], right_on=["CompanyID", "Year"], how="left"
    ).with_columns(pl.col("N_News").fill_null(0))

    panel.write_parquet(cfg.interim("db_master_2_relations.parquet"))


COLUMN_FINALISED_AT_STAGE.update(dict.fromkeys(["EmployeeCount", "N_News", *FINANCIAL_COLUMNS], 3))
