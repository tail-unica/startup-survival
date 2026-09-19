"""Phase 7: the competitors, and the last shape of the panel.

Three columns — how many competitors, how many of them in the same country, the
mean similarity — plus the number of similar companies that mean is computed over.
They exist in one version, and ``timed`` decides its content:

- **on**: only the competitors *alive in that year* are counted, and a pair is kept
  only if the life window of the other company is known, which is read from the
  company table alone. It is the temporally clean version, and the price is that
  about a tenth of the declared pairs survive.
- **off**: every pair on record, with no window and no filter on the other company,
  including the similar companies outside the extraction. It is the baseline *with*
  the look-ahead.

The heaviest assumption of the phase is that the last year with data stands for
"still alive". It is not a closing date, and a company covered well comes out alive
for longer than one covered badly, so the count over-weights the large and the
well-documented. It is kept because that year is the end of life everywhere else in
the pipeline, and changing it here alone would introduce a worse inconsistency.
"""

from __future__ import annotations

import polars as pl

from src.panel.config import PanelConfig, PanelRules
from src.panel.expansions import active_pairs
from src.panel.expressions import MISSING_TOKENS, nullify
from src.panel.io import read_raw, to_num

#: The columns of the similarity table this phase reads.
SIMILAR_COLUMNS = [
    "CompanyID",
    "SimilarCompanyID",
    "SimilarityScore",
    "IsCompetitor",
    "SimilarCompanyHQCountry",
]


def similar_pairs(
    cfg: PanelConfig, panel: pl.DataFrame, life: pl.DataFrame, *, timed: bool
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The declared pairs, split into similar companies and competitors.

    Two different sets, because the two serve different columns: every pair feeds
    the mean similarity, and only the pairs flagged as competitors feed the counts.
    The relation is directed and reciprocal in a minority of cases: what is counted
    is whom the company declares, not who declares the company.

    The country of the other company is read from the relation table and not from
    the company table, so that the comparison stays possible when the other company
    is outside the extraction. A pair with either country missing is excluded from
    the count rather than counted as "a different country".

    :param cfg: Pipeline paths.
    :param panel: The panel, for the companies whose pairs are wanted.
    :param life: Output of :func:`src.panel.companies.company_life`, over **all**
        companies: a competitor can be older than the sample and is alive all the
        same.
    :param timed: Whether to keep only the pairs whose other company has a known
        life window.
    :return: The similar pairs and the competitor pairs.
    """
    raw = nullify(read_raw(cfg, "CompanySimilarRelation", SIMILAR_COLUMNS), MISSING_TOKENS)
    raw = raw.with_columns(to_num("SimilarityScore"))

    # The only filter that always applies: the company on the left is in the panel.
    ids = panel.select("CompanyID").unique().to_series()
    raw = raw.filter(pl.col("CompanyID").is_in(ids.implode()))

    raw = (
        raw.join(
            life.select("CompanyID", pl.col("HQCountry").alias("_own_country")),
            on="CompanyID",
            how="left",
        )
        .with_columns(
            (pl.col("_own_country") == pl.col("SimilarCompanyHQCountry")).alias("_same_country")
        )
        .drop("_own_country", "SimilarCompanyHQCountry")
    )

    similar = raw.select("CompanyID", "SimilarCompanyID", "SimilarityScore")
    competitors = raw.filter(pl.col("IsCompetitor") == "Yes").select(
        "CompanyID", "SimilarCompanyID", "SimilarityScore", "_same_country"
    )
    if not timed:
        return similar, competitors

    # Knowing *when* the other company was alive can only be read from the company
    # table, so the pairs whose counterpart is outside the extraction are lost: that
    # is the price of the timing.
    window = life.select(
        pl.col("CompanyID").alias("SimilarCompanyID"),
        pl.col("YearFounded").alias("YF"),
        pl.col("MaxYear").alias("MY"),
    )
    return (
        similar.join(window, on="SimilarCompanyID", how="inner"),
        competitors.join(window, on="SimilarCompanyID", how="inner"),
    )


def competitor_columns(
    panel: pl.DataFrame, similar: pl.DataFrame, competitors: pl.DataFrame, *, timed: bool
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Aggregate the pairs into the four columns, one row per company-year.

    Whatever the branch, the result has the same shape: with the timing off the
    value is simply the same for every year of a company.

    The count of competitors in the same country carries no similarity threshold.
    The number of similar companies is the *denominator* of the mean: without it a
    zero in the mean would be ambiguous, meaning either "no comparable company" or
    "comparable companies that are not alike at all", and no other column resolves
    it.

    :param panel: The truncated panel, for its company-years.
    :param similar: Similar pairs from :func:`similar_pairs`.
    :param competitors: Competitor pairs from :func:`similar_pairs`.
    :param timed: Whether to count only the companies alive in each year.
    :return: The competitor statistics and the similarity statistics.
    """
    panel_years = panel.select("CompanyID", "Year_Delta").unique()
    if timed:
        # A range join, run as an inequality join: the full cross product would not
        # fit in memory.
        active_competitors = active_pairs(panel_years, competitors, ["_same_country"])
        active_similar = active_pairs(panel_years, similar, ["SimilarityScore"])
        return (
            active_competitors.group_by("CompanyID", "Year_Delta").agg(
                pl.col("SimilarCompanyID").n_unique().alias("N_Competitors"),
                pl.col("_same_country").drop_nulls().sum().cast(pl.Int64).alias("Same_Country"),
            ),
            active_similar.group_by("CompanyID", "Year_Delta").agg(
                pl.col("SimilarityScore").mean().alias("SimilarityScoreMean"),
                pl.col("SimilarCompanyID").n_unique().alias("N_Similar"),
            ),
        )
    return (
        panel_years.join(
            competitors.group_by("CompanyID").agg(
                pl.col("SimilarCompanyID").n_unique().alias("N_Competitors"),
                pl.col("_same_country").drop_nulls().sum().cast(pl.Int64).alias("Same_Country"),
            ),
            on="CompanyID",
            how="inner",
        ),
        panel_years.join(
            similar.group_by("CompanyID").agg(
                pl.col("SimilarityScore").mean().alias("SimilarityScoreMean"),
                pl.col("SimilarCompanyID").n_unique().alias("N_Similar"),
            ),
            on="CompanyID",
            how="inner",
        ),
    )


def finalize(
    panel: pl.DataFrame,
    competitor_stats: pl.DataFrame,
    similarity_stats: pl.DataFrame,
    rules: PanelRules,
) -> pl.DataFrame:
    """Graft the competitor columns, close the target, and renumber the companies.

    No competitor means zero competitors, which is right for a count. No similar
    company means a mean similarity of zero, which is more debatable: zero is the
    bottom of the scale and not a neutral value, so a company with no comparable one
    looks to a model like a company whose comparables are maximally different. That
    is what the count of similar companies is there to disambiguate.

    The placeholder in the future stage means "this company does not leave the group
    it is in", so it becomes the current group.

    The renumbering is the last operation of the pipeline, and not by accident: from
    there on the original identifiers are gone, and nothing can be joined back to
    the source.

    :param panel: Output of :func:`src.panel.target.truncate_at_exit`.
    :param competitor_stats: First frame from :func:`competitor_columns`.
    :param similarity_stats: Second frame from :func:`competitor_columns`.
    :param rules: Domain rules; the panel schema is read.
    :return: The finished panel, in the declared column order.
    """
    from src.panel.target import STAY

    final = (
        panel.join(competitor_stats, on=["CompanyID", "Year_Delta"], how="left")
        .join(similarity_stats, on=["CompanyID", "Year_Delta"], how="left")
        .with_columns(
            pl.col("N_Competitors", "Same_Country", "N_Similar").fill_null(0),
            pl.col("SimilarityScoreMean").fill_null(0.0),
        )
        .with_columns(
            pl.when(pl.col("GrowthNextStageGroup") == STAY)
            .then(pl.col("GrowthStageGroup"))
            .otherwise(pl.col("GrowthNextStageGroup"))
            .alias("GrowthNextStageGroup")
        )
    )
    mapping = final.select("CompanyID").unique().sort("CompanyID").with_row_index("_new", offset=1)
    final = (
        final.join(mapping, on="CompanyID", how="left")
        .drop("CompanyID")
        .rename({"_new": "CompanyID"})
    )
    return final.select(rules.columns)
