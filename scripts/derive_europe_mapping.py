"""Solve the country->Europe mapping from N_Europe in db_master_1.csv.

R used countrycode(); rather than guess how that package classifies the
boundary cases, we recover the exact set the R run used and freeze it.

    uv run python scripts/derive_europe_mapping.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import polars as pl

RAW = Path("data/raw/pitchbook/CompanySimilarRelation.csv")
REF = Path("data/reference/db_master_1.csv")
OUT = Path("src/panel/data/europe.csv")

#: Countries the data cannot settle, because every row naming them belongs to a
#: company outside db_master_1 (YearFounded <= 1999). Verified: setting either
#: one True or False leaves zero companies inconsistent, so the choice cannot
#: affect any output. Classified the way countrycode does - Falkland Islands is
#: Americas, Northern Mariana Islands is Oceania - so neither is Europe.
UNOBSERVABLE = {"Falkland Islands": False, "Northern Mariana Islands": False}


def main() -> None:
    counts = (
        pl.scan_csv(RAW, infer_schema_length=0, null_values=[])
        .select(["CompanyID", "SimilarCompanyHQCountry"])
        .filter(
            pl.col("SimilarCompanyHQCountry").is_not_null()
            & (pl.col("SimilarCompanyHQCountry") != "")
        )
        .group_by(["CompanyID", "SimilarCompanyHQCountry"])
        .len()
        .collect(engine="streaming")
    )
    target = (
        pl.read_csv(REF, infer_schema_length=0, null_values=[])
        .select(["CompanyID", "N_Europe"])
        .with_columns(pl.col("N_Europe").cast(pl.Int64, strict=False))
        .drop_nulls()
    )
    countries = sorted(counts["SimilarCompanyHQCountry"].unique().to_list())
    known: dict[str, bool] = {}

    per_company = (
        counts.join(target, on="CompanyID", how="inner")
        .group_by("CompanyID")
        .agg(
            pl.col("SimilarCompanyHQCountry").alias("cs"),
            pl.col("len").alias("ns"),
            pl.col("N_Europe").first().alias("target"),
        )
    )
    rows = per_company.to_dicts()

    for _ in range(len(countries) + 2):  # iterate to a fixed point
        progress = False
        for r in rows:
            unknown = [(c, n) for c, n in zip(r["cs"], r["ns"], strict=True) if c not in known]
            if len(unknown) != 1:
                continue
            resolved = sum(
                n for c, n in zip(r["cs"], r["ns"], strict=True) if known.get(c)
            )
            c, n = unknown[0]
            remainder = r["target"] - resolved
            if remainder in (0, n):
                known[c] = remainder == n
                progress = True
        if not progress:
            break

    missing = [c for c in countries if c not in known]
    unresolved = [c for c in missing if c not in UNOBSERVABLE]
    if unresolved:
        raise SystemExit(f"undetermined countries: {unresolved}")
    for c in missing:
        known[c] = UNOBSERVABLE[c]
    if missing:
        print(f"non deducibili dai dati, fissati a mano: {sorted(missing)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["country", "is_europe"])
        for c in countries:
            w.writerow([c, "true" if known[c] else "false"])
    n_eu = sum(known.values())
    print(f"scritto {OUT}: {len(countries)} paesi, {n_eu} in Europa")


if __name__ == "__main__":
    main()
