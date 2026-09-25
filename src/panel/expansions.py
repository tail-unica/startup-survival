"""The two heavy expansions of the panel pipeline, kept out of the notebook.

Both turn a compact table into a much larger one and then collapse it again, and
both peak near this machine's memory ceiling. Inside a function the
intermediates are freed on return; pasted into a notebook cell they would stay
alive in the kernel and kill the following stages.

They are also the two least interesting pieces to read line by line: the
substance is an explode and a join. Every *decision* (which threshold, which
aggregate, how a missing value is treated) stays in the notebook, where the text
cells explain it.
"""

from __future__ import annotations

import polars as pl

from src.panel.expressions import seq_inclusive


def expand_team(db3: pl.DataFrame, *, min_founding_year: int) -> pl.DataFrame:
    """Expand the person-company table to one row per (company, year, person).

    Two grids joined. The first spans, for each company, every year from the
    earliest arrival to the latest departure of anyone on its team. The second
    spans, for each person, the years of their own window. A left join of the
    first onto the second gives one row per person present in that company-year,
    and an empty row for a company-year nobody spans.

    The founding-year filter is applied here, and not by the caller, because it
    is what keeps the result inside the available memory. It also has a second
    effect: ``YearFounded`` arrives from the person side, so a company-year that
    matched nobody has a null one and the filter removes it, which is why the
    aggregation downstream never sees an empty join row and counts it as one
    person.

    :param db3: One row per (company, person), with ``DeltaStart`` and
        ``DeltaEnd`` as ages of the company. A null ``DeltaStart`` means the
        arrival is unknown, and the person is left out of the expansion.
    :param min_founding_year: Oldest founding year admitted, inclusive.
    :return: One row per (company, year, person), the year as ``Years``.
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
            .otherwise(seq_inclusive(pl.col("_lo"), pl.col("_hi")))
            .alias("Years")
        )
        # A company with no usable window keeps one row, with a null year.
        .explode("Years")
        .select("CompanyID", "Years")
    )
    person_years = (
        db3.filter(pl.col("DeltaStart").is_not_null())
        .with_columns(pl.col("DeltaEnd").fill_null(pl.col("DeltaStart")))
        .with_columns(seq_inclusive(pl.col("DeltaStart"), pl.col("DeltaEnd")).alias("Years"))
        .explode("Years")
    )
    return company_years.join(person_years, on=["CompanyID", "Years"], how="left").filter(
        pl.col("YearFounded") >= min_founding_year
    )


def active_pairs(panel_years: pl.DataFrame, pairs: pl.DataFrame, extra: list[str]) -> pl.DataFrame:
    """Restrict company pairs to the panel years in which the other one is alive.

    A range join: for each (company, year) of the panel, every similar company
    whose ``[YF, MY]`` window contains that year. ``join_where`` runs it as an
    inequality join rather than materialising the full cross product, which is
    the only way it fits, the result is already several million rows.

    :param panel_years: Distinct (``CompanyID``, ``Year_Delta``) of the panel.
    :param pairs: Declared pairs, with the counterpart's ``YF`` and ``MY``.
    :param extra: Columns of ``pairs`` to carry through to the aggregation.
    :return: One row per surviving (company, year, counterpart).
    """
    return panel_years.join_where(
        pairs.select("CompanyID", "SimilarCompanyID", "YF", "MY", *extra),
        pl.col("CompanyID") == pl.col("CompanyID_right"),
        pl.col("Year_Delta") >= pl.col("YF"),
        pl.col("Year_Delta") <= pl.col("MY"),
    )
