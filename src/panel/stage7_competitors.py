"""Stage 7 — year-by-year competitor columns.

Port of `notebook_temporizzazione_competitors.ipynb`, cells 4 and 6. The R
leaves the competitor columns static, counted once over a company's whole
similar-company list; this stage replaces them with a count of the competitors
actually alive in each panel year.

Three columns keep their R names and change meaning, which is worth knowing
before reading them as if they were the same variable:

- `Same_Country` was a boolean — "some competitor above 90 similarity sits in
  our country" — and becomes a **count** of same-country active competitors.
- `SimilarityScoreMean` is filled with 0 where no similar company was active
  that year. Zero is the *minimum* of the scale, not a neutral value, so a
  company with no live peers reads to a model as one whose peers are maximally
  dissimilar.
- `N_Competitors_All` is not the R's `N_Competitors`: it only counts
  competitors that have a usable life window, so it is the smaller of the two.

A fourth caveat belongs in the paper rather than here: `MaxYear` is the last
year with *data*, not the year a company died, so the timing systematically
keeps well-covered competitors alive longer than thinly-covered ones.
"""

from __future__ import annotations

import polars as pl

from src.panel.config import PanelConfig
from src.panel.io import read_raw, to_num
from src.panel.rutils import R_NA_NAN, as_na, parse_date_r
from src.panel.stage1_company import DATE_COLUMNS

COMPETITOR_COLUMNS = ["N_Competitors", "Same_Country", "SimilarityScoreMean"]
DROPPED = ["N_Europe", "N_Outside_Europe"]


def _company_life(cfg: PanelConfig) -> pl.DataFrame:
    """Founding year and last year with data, for **every** company.

    Not just the panel's cohorts: a competitor may be older than 2000 and still
    be a competitor. Dates go through `parse_date_r`, not the notebook's
    `%m/%d/%y` fallback — chrono reads a two-digit year 25-69 as 2025-2069
    while R's `cutoff_2000 = 24` reads it as 1925-1969, and this is the column
    that decides whether a competitor is alive.
    """
    cols = [*DATE_COLUMNS, "FiscalPeriod", "YearFounded", "HQCountry"]
    df = as_na(read_raw(cfg, "Company", ["CompanyID", *cols]), R_NA_NAN)
    quarter = pl.col("FiscalPeriod").str.extract(r"TTM (\d)Q\d{4}", 1).cast(pl.Int64, strict=False)
    fiscal_year = pl.col("FiscalPeriod").str.slice(-4).cast(pl.Int64, strict=False)
    df = df.with_columns(
        *[parse_date_r(pl.col(c)).alias(c) for c in DATE_COLUMNS],
        pl.date(fiscal_year, quarter * 3, 30).alias("FiscalDate"),
        pl.col("YearFounded").cast(pl.Int64, strict=False),
    )
    year_cols = [*DATE_COLUMNS, "FiscalDate"]
    return (
        df.with_columns(
            pl.max_horizontal([pl.col(c).dt.year() for c in year_cols]).alias("MaxYear")
        )
        .select("CompanyID", "YearFounded", "MaxYear", "HQCountry")
        .filter(pl.col("YearFounded").is_not_null() & pl.col("MaxYear").is_not_null())
    )


def _pairs(cfg: PanelConfig, panel_ids: pl.Series, life: pl.DataFrame):
    """The similar-company pairs, as declared: the relation is directed and the
    reverse direction is not added."""
    rel = as_na(
        read_raw(
            cfg,
            "CompanySimilarRelation",
            ["CompanyID", "SimilarCompanyID", "SimilarityScore", "IsCompetitor"],
        ),
        R_NA_NAN,
    ).with_columns(to_num("SimilarityScore"))
    rel = rel.filter(
        pl.col("CompanyID").is_in(panel_ids) & pl.col("SimilarCompanyID").is_in(life["CompanyID"])
    )
    window = life.select(
        pl.col("CompanyID").alias("SimilarCompanyID"),
        pl.col("YearFounded").alias("YF"),
        pl.col("MaxYear").alias("MY"),
        pl.col("HQCountry").alias("HQCountry_comp"),
    )
    similar_all = rel.select("CompanyID", "SimilarCompanyID", "SimilarityScore").join(
        window.select("SimilarCompanyID", "YF", "MY"), on="SimilarCompanyID", how="inner"
    )
    comp = (
        rel.filter(pl.col("IsCompetitor") == "Yes")
        .select("CompanyID", "SimilarCompanyID", "SimilarityScore")
        .join(window, on="SimilarCompanyID", how="inner")
        .join(
            life.select("CompanyID", pl.col("HQCountry").alias("HQCountry_panel")),
            on="CompanyID",
            how="left",
        )
        .with_columns(
            # Null when either country is unknown: excluded from the count
            # rather than counted as a mismatch.
            (pl.col("HQCountry_panel") == pl.col("HQCountry_comp")).alias("SameCountry")
        )
        .drop("HQCountry_panel", "HQCountry_comp")
    )
    return comp, similar_all


def _active(pairs: pl.DataFrame, panel_years: pl.DataFrame, extra: list[str]) -> pl.DataFrame:
    """Pairs restricted to the panel years in which the other company is alive."""
    return panel_years.join_where(
        pairs.select("CompanyID", "SimilarCompanyID", "YF", "MY", *extra),
        pl.col("CompanyID") == pl.col("CompanyID_right"),
        pl.col("Year_Delta") >= pl.col("YF"),
        pl.col("Year_Delta") <= pl.col("MY"),
    )


def run(cfg: PanelConfig) -> None:
    panel = pl.read_parquet(cfg.interim("db_master_panel.parquet"))
    life = _company_life(cfg)
    comp, similar_all = _pairs(cfg, panel["CompanyID"].unique(), life)

    panel_years = panel.select("CompanyID", "Year_Delta").unique()
    competitor_stats = (
        _active(comp, panel_years, ["SameCountry"])
        .group_by("CompanyID", "Year_Delta")
        .agg(
            pl.col("SimilarCompanyID").n_unique().alias("N_Competitors"),
            pl.col("SameCountry").drop_nulls().sum().cast(pl.Int64).alias("Same_Country"),
        )
    )
    similarity_stats = (
        _active(similar_all, panel_years, ["SimilarityScore"])
        .group_by("CompanyID", "Year_Delta")
        .agg(pl.col("SimilarityScore").mean().alias("SimilarityScoreMean"))
    )

    # The same aggregation without the year filter, for the no-window setting.
    static = comp.group_by("CompanyID").agg(
        pl.col("SimilarCompanyID").n_unique().alias("N_Competitors_All"),
        pl.col("SameCountry").drop_nulls().sum().cast(pl.Int64).alias("Same_Country_All"),
    )
    static_similarity = similar_all.group_by("CompanyID").agg(
        pl.col("SimilarityScore").mean().alias("SimilarityScoreMean_All")
    )

    out = (
        panel.drop(*COMPETITOR_COLUMNS, *DROPPED)
        .join(competitor_stats, on=["CompanyID", "Year_Delta"], how="left")
        .join(similarity_stats, on=["CompanyID", "Year_Delta"], how="left")
        .join(static, on="CompanyID", how="left")
        .join(static_similarity, on="CompanyID", how="left")
        .with_columns(
            pl.col(
                "N_Competitors", "Same_Country", "N_Competitors_All", "Same_Country_All"
            ).fill_null(0),
            pl.col("SimilarityScoreMean", "SimilarityScoreMean_All").fill_null(0.0),
        )
        .with_columns(
            # "Stay" means the company never leaves its current group.
            pl.when(pl.col("GrowthNextStageGroup") == "Stay")
            .then(pl.col("GrowthStageGroup"))
            .otherwise(pl.col("GrowthNextStageGroup"))
            .alias("GrowthNextStageGroup")
        )
    )

    # Last operation of the pipeline: it destroys every join back to the
    # reference files, so nothing may run after it.
    id_map = out.select("CompanyID").unique().sort("CompanyID").with_row_index("_new_id", offset=1)
    other = [c for c in out.columns if c != "CompanyID"]
    out = (
        out.join(id_map, on="CompanyID", how="left")
        .drop("CompanyID")
        .rename({"_new_id": "CompanyID"})
        .select(["CompanyID", *other])
    )

    out.write_parquet(cfg.interim("panel.parquet"))
    out.write_csv(cfg.interim("panel.csv.gz"), compression="gzip")
