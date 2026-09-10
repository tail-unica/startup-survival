"""The two heavy expansions, kept out of the notebook on purpose.

Both turn a compact table into a much larger one and then collapse it again,
and both peak near this machine's memory ceiling. Inside a function the
intermediates are freed on return; pasted into a notebook cell they would stay
in the kernel and kill the following stages.

They are also the two least interesting pieces to read line by line: the
substance is an `explode` and a join. Every *decision* — which threshold, which
aggregate, how a missing value is treated — stays in the notebook, next to the
comment that explains it.
"""

from __future__ import annotations

import polars as pl

from src.panel.rutils import r_seq


def expand_team(db3: pl.DataFrame, *, founding_year_threshold: int) -> pl.DataFrame:
    """`1_Arrange_DB.R:527-560` — one row per (company, year, person).

    Two grids joined. The first spans, for each company, every year from the
    earliest arrival to the latest departure of anyone on its team. The second
    spans, for each person, the years between their own arrival and departure.
    A left join of the first onto the second gives one row per person present
    in that company-year — and an empty row for a company-year nobody spans.

    `founding_year_threshold` is **bug B1** and the caller passes it
    explicitly: the R filters `YearFounded > 2000` here while the panel
    skeleton was cut at `> 1999`, so the whole 2000 founding cohort arrives
    with no team data at all. The filter is applied inside because it is what
    keeps the result down to a size that fits in memory.

    It also has a second, undocumented effect. `YearFounded` arrives from the
    person side, so a company-year that matched nobody has a null one and this
    filter removes it — which is why the aggregation downstream never sees the
    empty join row it would otherwise count as one person.
    """
    company_years = (
        db3.group_by("CompanyID")
        .agg(
            pl.col("DeltaStart").min().alias("_lo"),
            pl.col("DeltaEnd").max().alias("_hi"),
        )
        .with_columns(
            pl.when(pl.col("_lo").is_null() | pl.col("_hi").is_null())
            .then(None)
            .otherwise(r_seq(pl.col("_lo"), pl.col("_hi")))
            .alias("Years")
        )
        # unnest(keep_empty = TRUE): a company with no usable window keeps one
        # row with a null year.
        .explode("Years")
        .select("CompanyID", "Years")
    )
    person_years = (
        db3.filter(pl.col("DeltaStart").is_not_null())
        .with_columns(pl.col("DeltaEnd").fill_null(pl.col("DeltaStart")))
        .with_columns(r_seq(pl.col("DeltaStart"), pl.col("DeltaEnd")).alias("Years"))
        .explode("Years")
    )
    return company_years.join(person_years, on=["CompanyID", "Years"], how="left").filter(
        pl.col("YearFounded") > founding_year_threshold
    )


def active_pairs(panel_years: pl.DataFrame, pairs: pl.DataFrame, extra: list[str]) -> pl.DataFrame:
    """Pairs restricted to the panel years in which the other company is alive.

    A range join: for each (company, year) of the panel, every similar company
    whose `[YF, MY]` window contains that year. `join_where` runs it as an
    inequality join rather than materialising the full cross product, which is
    the only way it fits — the result is already several million rows.

    Kept out of the notebook for the same two reasons as `expand_team`: the
    intermediates are large and freed on return, and the substance is one
    join. Which pairs go in, and what is aggregated out, stays in the notebook.
    """
    return panel_years.join_where(
        pairs.select("CompanyID", "SimilarCompanyID", "YF", "MY", *extra),
        pl.col("CompanyID") == pl.col("CompanyID_right"),
        pl.col("Year_Delta") >= pl.col("YF"),
        pl.col("Year_Delta") <= pl.col("MY"),
    )
