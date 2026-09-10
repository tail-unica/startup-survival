"""Readers for the raw PitchBook CSVs.

Everything is read as String and cast deliberately. Type inference is not
used anywhere: `CompanyID` looks numeric in some files ("100020-70" does
not, but others do) and an inferred integer key breaks joins silently.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.panel.config import PanelConfig

#: Row count of each raw table, recorded once so that a parsing regression on
#: the quoted free-text columns fails loudly instead of skewing every count
#: downstream. Measured on the extraction that `scripts/check_extraction.py`
#: certifies as the one behind `data/reference/`.
RAW_ROW_COUNTS: dict[str, int] = {
    "Company": 134355,
    "CompanyAffiliateRelation": 129488,
    "CompanyBoardTeamRelation": 535568,
    "CompanyEmployeeHistoryRelation": 600805,
    "CompanyFinancialRelation": 559823,
    "CompanyNewsRelation": 1858,
    "CompanySimilarRelation": 1340950,
    "Deal": 385481,
    "DealInvestorRelation": 506641,
    "Investor": 92909,
    "Person": 1318626,
    "PersonEducationRelation": 1246054,
    "PersonPositionRelation": 1611914,
}


#: The five `Company.csv` date columns that feed MaxYear. Shared because two
#: places compute a MaxYear from them — the panel skeleton and the competitor
#: activity windows — and if the two lists drift the two MaxYear disagree.
COMPANY_DATE_COLUMNS = [
    "CompanyFinancingStatusDate",
    "BusinessStatusDate",
    "OwnershipStatusDate",
    "FirstFinancingDate",
    "LastKnownValuationDate",
]


def scan_raw(cfg: PanelConfig, table: str, columns: list[str]) -> pl.LazyFrame:
    """Lazy scan of one raw table, projected to `columns`, all String."""
    return pl.scan_csv(
        cfg.raw(f"{table}.csv"),
        infer_schema_length=0,
        null_values=[],
        quote_char='"',
    ).select(columns)


def read_raw(
    cfg: PanelConfig,
    table: str,
    columns: list[str],
    *,
    expect_rows: int | None = None,
) -> pl.DataFrame:
    """Collect a projected raw table in streaming mode, asserting its height."""
    df = scan_raw(cfg, table, columns).collect(engine="streaming")
    if expect_rows is not None and df.height != expect_rows:
        raise ValueError(
            f"row count mismatch for {table}: expected {expect_rows}, got {df.height}. "
            "A quoted free-text column is probably being mis-parsed."
        )
    return df


def to_num(name: str) -> pl.Expr:
    """Cast a String column to Float64; anything non-numeric becomes null."""
    return pl.col(name).cast(pl.Float64, strict=False).alias(name)


def to_int(name: str) -> pl.Expr:
    """Cast a String column to Int64; anything non-integer becomes null."""
    return pl.col(name).cast(pl.Int64, strict=False).alias(name)


#: Three classes, because `SimilarCompanyIsEurope` is three-valued in R:
#: countrycode returns NA for a name it cannot place, and NA is not the same as
#: FALSE downstream — `N_Europe` counts neither, `N_Outside_Europe` counts only
#: FALSE. Four names land here: Kosovo, Polynesia, Micronesia and British
#: Indian Ocean Territory.
_CONTINENT_CLASS = {"europe": True, "other": False, "unknown": None}


def load_europe() -> dict[str, bool | None]:
    """Frozen country -> is-Europe verdict, recovered from the R run.

    True is Europe, False is elsewhere, None is "countrycode could not place
    it". Derived once by scripts/derive_europe_mapping.py so the panel never
    depends on a third-party classification that could change between releases.
    """
    path = Path(__file__).parent / "data" / "europe.csv"
    df = pl.read_csv(path, infer_schema_length=0)
    return {r["country"]: _CONTINENT_CLASS[r["continent_class"]] for r in df.to_dicts()}
