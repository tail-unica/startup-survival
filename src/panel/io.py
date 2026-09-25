"""Readers for the raw PitchBook CSVs.

Everything is read as String and cast deliberately. Type inference is not used
anywhere: ``CompanyID`` looks numeric in some files and an inferred integer key
breaks joins silently.
"""

from __future__ import annotations

import polars as pl

from src.panel.config import PanelConfig

#: The five ``Company.csv`` date columns that feed MaxYear. Shared because two
#: places compute a MaxYear from them, the panel skeleton (phase 1) and the
#: competitor activity windows (phase 7), and if the two lists drift the two
#: MaxYear disagree.
COMPANY_DATE_COLUMNS = [
    "CompanyFinancingStatusDate",
    "BusinessStatusDate",
    "OwnershipStatusDate",
    "FirstFinancingDate",
    "LastKnownValuationDate",
]


def scan_raw(cfg: PanelConfig, table: str, columns: list[str]) -> pl.LazyFrame:
    """Lazily scan one raw table, projected to ``columns``, every column String.

    :param cfg: Pipeline paths.
    :param table: Table name without the ``.csv`` extension.
    :param columns: Columns to project.
    :return: Lazy frame of the projection.
    """
    return pl.scan_csv(
        cfg.raw(f"{table}.csv"),
        infer_schema_length=0,
        null_values=[],
        quote_char='"',
    ).select(columns)


def read_raw(cfg: PanelConfig, table: str, columns: list[str]) -> pl.DataFrame:
    """Collect a projected raw table in streaming mode.

    Streaming because the widest tables do not fit in memory at once.

    :param cfg: Pipeline paths.
    :param table: Table name without the ``.csv`` extension.
    :param columns: Columns to project.
    :return: Eager frame of the projection, every column String.
    """
    return scan_raw(cfg, table, columns).collect(engine="streaming")


def to_num(name: str) -> pl.Expr:
    """Cast a String column to Float64, turning anything non-numeric into null.

    :param name: Column name, kept on the result.
    :return: The cast expression.
    """
    return pl.col(name).cast(pl.Float64, strict=False).alias(name)
