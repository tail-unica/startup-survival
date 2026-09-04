"""Dice se un'estrazione PitchBook e' quella che ha prodotto i CSV di riferimento.

Gli intermedi in config/RCode/DatiIntermedi/ sono la ground truth della pipeline
Python: se i grezzi non sono lo stesso vintage, nessun checkpoint puo' passare e
il problema non e' nel codice. Questo controllo lo verifica in pochi secondi,
leggendo due colonne da due file, invece di scoprirlo dopo una run intera.

    .venv/bin/python scripts/check_extraction.py "percorso/della/cartella"
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

REF = Path("config/RCode/DatiIntermedi")


def _scan(path: Path, cols: list[str]) -> pl.DataFrame:
    return (
        pl.scan_csv(path, infer_schema_length=0, null_values=[], quote_char='"')
        .select(cols)
        .collect(engine="streaming")
    )


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    raw = Path(sys.argv[1])
    if not (raw / "Company.csv").exists():
        print(f"ERRORE: {raw}/Company.csv non esiste")
        return 2

    ok = True

    m1 = _scan(REF / "db_master_1.csv", ["CompanyID"])
    comp = _scan(raw / "Company.csv", ["CompanyID", "YearFounded"])
    missing = m1.join(comp.select("CompanyID"), on="CompanyID", how="anti").height
    eligible = comp.with_columns(
        pl.col("YearFounded").cast(pl.Int64, strict=False)
    ).filter(pl.col("YearFounded") > 1999).height

    print(f"Company.csv                        : {comp.height:>9,} aziende")
    print(f"  con YearFounded > 1999           : {eligible:>9,}   (serve esattamente 116.920)")
    print(f"  aziende del riferimento mancanti : {missing:>9,}   (serve 0)")
    ok &= missing == 0 and eligible == 116_920

    bt_path = raw / "CompanyBoardTeamRelation.csv"
    if bt_path.exists():
        d3 = _scan(REF / "db3.csv", ["CompanyID", "PersonID"])
        bt = _scan(bt_path, ["CompanyID", "PersonID"]).unique()
        missing_pairs = d3.join(bt, on=["CompanyID", "PersonID"], how="anti").height
        print(f"\nCompanyBoardTeamRelation.csv       : {bt.height:>9,} coppie uniche")
        print(f"  coppie del riferimento mancanti  : {missing_pairs:>9,}   (serve 0)")
        ok &= missing_pairs == 0
    else:
        print("\nATTENZIONE: CompanyBoardTeamRelation.csv assente, controllo parziale")
        ok = False

    print()
    print("ESTRAZIONE CORRETTA" if ok else "NON E' L'ESTRAZIONE GIUSTA")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
