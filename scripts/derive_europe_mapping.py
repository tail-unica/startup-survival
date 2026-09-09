"""Solve the country->continent classification from db_master_1.csv.

R used countrycode(); rather than guess how that package classifies the
boundary cases, we recover the exact verdict the R run used and freeze it.

Three classes, not two, because `SimilarCompanyIsEurope` is a three-valued
column: countrycode returns NA for a name it cannot place, and NA behaves
differently in the two aggregates that read it. `N_Europe` is
`sum(IsEurope, na.rm = TRUE)`, so NA and FALSE both count zero;
`N_Outside_Europe` is `sum(!IsEurope[...], na.rm = TRUE)`, where FALSE counts
one and NA still counts zero. A two-class mapping reproduces N_Europe and
overcounts N_Outside_Europe.

Pass 1 solves N_Europe for europe/other; pass 2 solves N_Outside_Europe over
the rows scoring above 90 and demotes to `unknown` every `other` country that
is never actually counted.

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
            resolved = sum(n for c, n in zip(r["cs"], r["ns"], strict=True) if known.get(c))
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

    # --- pass 2: which non-European countries countrycode could not place ---
    scored = (
        pl.scan_csv(RAW, infer_schema_length=0, null_values=[])
        .select(["CompanyID", "SimilarCompanyHQCountry", "SimilarityScore"])
        .with_columns(pl.col("SimilarityScore").cast(pl.Float64, strict=False))
        .filter(
            (pl.col("SimilarityScore") > 90)
            & pl.col("SimilarCompanyHQCountry").is_not_null()
            & (pl.col("SimilarCompanyHQCountry") != "")
        )
        .group_by(["CompanyID", "SimilarCompanyHQCountry"])
        .len()
        .collect(engine="streaming")
    )
    outside = (
        pl.read_csv(REF, infer_schema_length=0, null_values=[])
        .select(["CompanyID", "N_Outside_Europe"])
        .with_columns(pl.col("N_Outside_Europe").cast(pl.Int64, strict=False))
        .drop_nulls()
    )
    non_eu = {c for c, is_eu in known.items() if not is_eu}
    rows2 = (
        scored.join(outside, on="CompanyID", how="inner")
        .group_by("CompanyID")
        .agg(
            pl.col("SimilarCompanyHQCountry").alias("cs"),
            pl.col("len").alias("ns"),
            pl.col("N_Outside_Europe").first().alias("target"),
        )
        .to_dicts()
    )
    counted: dict[str, bool] = {c: False for c in known if known[c]}  # Europe: never counted
    for _ in range(len(countries) + 2):
        progress = False
        for r in rows2:
            unknown = [
                (c, n)
                for c, n in zip(r["cs"], r["ns"], strict=True)
                if c in non_eu and c not in counted
            ]
            if len(unknown) != 1:
                continue
            resolved = sum(n for c, n in zip(r["cs"], r["ns"], strict=True) if counted.get(c))
            c, n = unknown[0]
            remainder = r["target"] - resolved
            if remainder in (0, n):
                counted[c] = remainder == n
                progress = True
        if not progress:
            break

    def classify(c: str) -> str:
        if known[c]:
            return "europe"
        # Never counted as outside Europe -> countrycode returned NA for it.
        return "other" if counted.get(c, True) else "unknown"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["country", "continent_class"])
        for c in countries:
            w.writerow([c, classify(c)])
    n_eu = sum(1 for c in countries if classify(c) == "europe")
    unk = [c for c in countries if classify(c) == "unknown"]
    print(f"scritto {OUT}: {len(countries)} paesi, {n_eu} in Europa")
    print(f"continente ignoto per countrycode: {unk}")


if __name__ == "__main__":
    main()
