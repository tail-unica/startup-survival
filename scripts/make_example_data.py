"""Writes the synthetic PitchBook-shaped tables the example run reads.

The real extraction cannot be redistributed, so the repository ships a tiny
invented one: fourteen tables, only the columns the pipeline actually reads, and
a cast of eight companies chosen so that every branch of the panel construction
fires at least once. Nothing here describes a real company or person.

Each company's story is written above its rows, and that is the point: reading
this file next to ``1_panel_construction.ipynb`` shows what happens to a firm as
the phases go by.

    python scripts/make_example_data.py

The output lands in ``data/example/pitchbook/``, which the pipeline reads with
``--example`` on the command line, or with ``EXAMPLE = True`` in the notebook.

The rows are laid out one record per line and kept out of the formatter's hands
on purpose: they are meant to be read as a table.
"""

import csv
import sys
from pathlib import Path

USCITA = Path(__file__).resolve().parents[1] / "data" / "example" / "pitchbook"

# fmt: off

# ── Le aziende ──────────────────────────────────────────────────────────────
# C1  cresce: seed, early VC, later VC. Il team mescola un fondatore senza data
#     d'ingresso, un assunto datato e uno senza data, che non viene contato.
# C2  viene acquisita nel 2019, e il round di acquisizione non ha una data: la
#     prende dallo stato di proprieta'.
# C3  chiude nel 2018, e anche qui la data del round di fallimento arriva dallo
#     stato di proprieta'.
# C4  ha un solo round, di tipo iniziale e senza data: va all'anno di fondazione.
# C5  ha quattro round, il secondo e il terzo senza data: vengono distribuiti fra
#     il primo e il quarto.
# C6  e' fondata nel 1999, prima della soglia del campione: non entra nel panel.
# C7  ha date anteriori alla fondazione, quindi il suo panel finirebbe prima di
#     cominciare: resta con il solo anno zero.
# Ogni azienda ha un anno di fondazione diverso dalle altre: e' con quello che i
# test la ritrovano, perche' l'ultimo passaggio della pipeline rinumera gli ID.
# C8  non ha nessun round: sta nel panel senza stadio di crescita, e dichiara
#     concorrenti di cui uno fuori da questa estrazione.
# S1, S2, X1 compaiono solo come controparti o come datori di lavoro passati:
#     servono a mostrare che una relazione puo' puntare fuori dal campione e che
#     l'esperienza di una persona matura anche altrove.
COMPANY_COLS = [
    "CompanyID", "YearFounded", "HQCountry", "PrimaryIndustrySector", "OwnershipStatus",
    "OwnershipStatusDate", "CompanyFinancingStatusDate", "BusinessStatusDate",
    "FirstFinancingDate", "LastKnownValuationDate", "FiscalPeriod",
]
COMPANY = [
    ["C1", "2012", "Italy", "Information Technology", "Privately Held",
     "06/30/2020", "09/15/2019", "", "04/10/2013", "12/31/2019", "TTM 2Q2020"],
    ["C2", "2013", "Italy", "Healthcare", "Acquired/Merged",
     "03/20/2019", "03/20/2019", "03/20/2019", "05/05/2014", "", ""],
    ["C3", "2014", "Spain", "Information Technology", "Out of Business",
     "11/10/2018", "", "11/10/2018", "02/01/2015", "", ""],
    ["C4", "2015", "France", "Consumer Products and Services", "Privately Held",
     "01/31/2021", "01/31/2021", "", "", "", ""],
    ["C5", "2011", "Germany", "Energy", "Privately Held",
     "12/31/2019", "07/01/2018", "", "03/12/2012", "", "TTM 4Q2019"],
    ["C6", "1999", "Italy", "Materials and Resources", "Privately Held",
     "05/05/2015", "", "", "", "", ""],
    ["C7", "2016", "Italy", "Healthcare", "Privately Held",
     "08/08/2015", "", "", "", "", ""],
    ["C8", "2010", "Portugal", "Information Technology", "Privately Held",
     "04/04/2019", "", "", "", "", ""],
    ["S1", "2008", "Italy", "Information Technology", "Privately Held",
     "01/15/2021", "", "", "", "", ""],
    ["S2", "2009", "Germany", "Information Technology", "Out of Business",
     "06/30/2016", "", "", "", "", ""],
    ["X1", "2001", "Italy", "Information Technology", "Privately Held",
     "02/02/2012", "", "", "", "", ""],
]

# ── Le persone ──────────────────────────────────────────────────────────────
# P1  fondatrice e CEO di C1: nessuna data d'ingresso, quindi vale l'anno zero.
# P2  entra in C1 nel 2016 con data dichiarata: prima di quell'anno non conta.
# P3  entra in C1 senza data e non e' fondatrice: non conta in nessun anno.
# P4  ha il titolo nel nome ("Ph.D"), che vale in ogni anno.
# P5  si laurea nel 2017: prima di allora il suo titolo di studio e' piu' basso.
# P6  non dichiara l'anno di laurea, quindi il titolo conta da sempre.
# P7  e' fondatrice di C5 e ha un seggio senza data in un'altra azienda.
# P8  guida C3 e ha un fondo affiliato, con la sua annata.
# P9  non ha un genere dichiarato: conta fra le persone, non nel denominatore
#     della quota di donne.
CONTATORI = [
    "CurrentPositionsCount", "CurrentBoardSeatsCount", "CurrentAdvisoryRolesCount",
    "NumberOfAffiliatedFunds", "FormerPositionsCount", "FormerBoardSeatsCount",
    "FormerAdvisoryRolesCount", "AffiliatedDealsCount",
]
PERSON_COLS = ["PersonID", "FullName", "Gender", *CONTATORI]
PERSON = [
    ["P1", "Anna Rossi", "Female", "1", "1", "0", "0", "1", "0", "0", "1"],
    ["P2", "Marco Bianchi", "Male", "1", "0", "0", "0", "1", "0", "0", "0"],
    ["P3", "Chiara Verdi", "Female", "1", "0", "0", "0", "0", "0", "0", "0"],
    ["P4", "Luca Neri Ph.D", "Male", "0", "1", "1", "0", "0", "0", "0", "0"],
    ["P5", "Sara Gallo", "Female", "1", "0", "0", "0", "0", "0", "0", "0"],
    ["P6", "Paolo Conti", "Male", "1", "0", "0", "0", "1", "0", "0", "0"],
    ["P7", "Elena Ferri", "Female", "1", "0", "1", "0", "0", "1", "0", "0"],
    ["P8", "Davide Mori", "Male", "0", "0", "0", "1", "1", "0", "0", "0"],
    ["P9", "Kim Park", "", "1", "0", "0", "0", "0", "0", "0", "0"],
]

# ── Chi sta in quale azienda, e da quando ───────────────────────────────────
BOARD_COLS = [
    "CompanyID", "PersonID", "PersonName", "FullTitle", "IsCurrent", "StartDate", "EndDate",
    "LastUpdated",
]
BOARD = [
    ["C1", "P1", "Anna Rossi", "Co-Founder & CEO", "Yes", "", "", "01/10/2021"],
    ["C1", "P2", "Marco Bianchi", "Chief Technology Officer", "Yes",
     "07/01/2016", "", "01/10/2021"],
    ["C1", "P3", "Chiara Verdi", "Chief Financial Officer", "Yes", "", "", "01/10/2021"],
    # Lo stesso incarico registrato due volte, una con la data di uscita e una
    # come ancora in corso: le righe si fondono, e "in carica" vince.
    ["C1", "P4", "Luca Neri Ph.D", "Board Member", "No",
     "01/01/2015", "12/31/2018", "02/02/2019"],
    ["C1", "P4", "Luca Neri Ph.D", "Board Member", "Yes", "01/01/2015", "", "05/05/2020"],
    ["C2", "P5", "Sara Gallo", "Founder", "Yes", "", "", "03/03/2019"],
    ["C2", "P6", "Paolo Conti", "Chief Executive Officer", "Yes",
     "01/01/2014", "", "03/03/2019"],
    # Due ruoli diversi della stessa persona: si fondono nell'unione dei periodi,
    # e il titolo di fondatore sopravvive alla fusione.
    ["C3", "P8", "Davide Mori", "Founder", "No", "01/01/2014", "06/30/2016", "01/01/2019"],
    ["C3", "P8", "Davide Mori", "Chief Executive Officer", "No",
     "07/01/2016", "11/10/2018", "01/01/2019"],
    ["C3", "P9", "Kim Park", "Head of Operations", "Yes", "01/01/2015", "", "01/01/2019"],
    ["C4", "P6", "Paolo Conti", "Co-Founder", "Yes", "01/01/2015", "", "01/01/2021"],
    ["C5", "P7", "Elena Ferri", "Founder & Chairman", "Yes", "", "", "01/01/2020"],
    ["C5", "P2", "Marco Bianchi", "Advisor", "No", "01/01/2013", "12/31/2015", "01/01/2020"],
    ["C7", "P5", "Sara Gallo", "Founder", "Yes", "", "", "01/01/2017"],
    ["C8", "P1", "Anna Rossi", "Founder", "Yes", "", "", "01/01/2019"],
]

# ── L'istruzione ────────────────────────────────────────────────────────────
EDUCATION_COLS = ["PersonID", "Degree", "Major_Concentration", "GraduatingYear", "Institute"]
EDUCATION = [
    ["P1", "MSc", "Computer Science", "2010", "Politecnico di Milano"],
    ["P1", "MBA", "Business Administration", "2016", "Massachusetts Institute of Technology"],
    ["P2", "BSc", "Electrical Engineering", "2012", "Universita di Bologna"],
    ["P3", "Master", "Finance", "2011", "Bocconi"],
    ["P4", "PhD", "Robotics", "2009", "ETH Zurich"],
    ["P5", "Bachelor", "Biology", "2013", "Universita di Padova"],
    ["P5", "PhD", "Immunology", "2017", "Universita di Padova"],
    ["P6", "MBA", "Management", "", "INSEAD"],
    ["P7", "Master", "Law", "2005", "Universita di Roma"],
    ["P8", "MSc", "Data Science", "2013", "Universita di Cagliari"],
    ["P9", "Diploma", "", "2012", ""],
]

# ── L'esperienza: un ruolo per riga ─────────────────────────────────────────
POSITION_COLS = ["PersonID", "EntityID", "StartDate", "PositionLevel", "IsCurrent"]
POSITION = [
    ["P1", "C1", "", "Founder", "Yes"],
    ["P1", "X1", "05/01/2008", "Manager", "No"],
    ["P2", "C1", "07/01/2016", "Chief Technology Officer", "Yes"],
    ["P2", "C5", "01/01/2013", "Advisor", "No"],
    ["P3", "C1", "", "Chief Financial Officer", "Yes"],
    ["P5", "C2", "", "Founder", "Yes"],
    ["P6", "C4", "01/01/2015", "Founder", "Yes"],
    # Una posizione senza data in un'entita' che non sta da nessuna parte: l'anno
    # si risolve con il primo anno noto della persona.
    ["P6", "Z9", "", "Advisor", "No"],
    ["P6", "C2", "01/01/2014", "Chief Executive Officer", "Yes"],
    ["P7", "C5", "", "Founder", "Yes"],
    ["P8", "C3", "01/01/2014", "Founder", "No"],
    ["P9", "C3", "01/01/2015", "Head of Operations", "Yes"],
]
BOARD_SEAT_COLS = ["PersonID", "CompanyID", "StartDate", "IsCurrent"]
BOARD_SEAT = [
    ["P1", "S1", "03/01/2017", "Yes"],
    ["P4", "C1", "01/01/2015", "Yes"],
    # Un seggio senza data: vale l'anno di fondazione dell'azienda, che e' il solo
    # limite inferiore disponibile.
    ["P7", "S2", "", "No"],
]
ADVISORY_COLS = ["PersonID", "EntityID", "StartDate", "IsCurrent"]
ADVISORY = [
    ["P4", "X1", "06/01/2014", "Yes"],
    ["P7", "X1", "01/01/2010", "Yes"],
]
AFFILIATED_DEAL_COLS = ["PersonID", "CompanyID", "DealDate"]
AFFILIATED_DEAL = [
    ["P1", "C1", "06/01/2015"],
]
AFFILIATED_FUND_COLS = ["PersonID", "FundID", "InvestorID"]
AFFILIATED_FUND = [
    ["P8", "F1", "I1"],
]
FUND_COLS = ["FundID", "Vintage"]
FUND = [
    ["F1", "2016"],
]

# ── I round ─────────────────────────────────────────────────────────────────
DEAL_COLS = [
    "CompanyID", "DealID", "DealNo", "DealDate", "DealType", "TotalInvestedCapital", "CEOPBId",
    "DealSize",
]
DEAL = [
    ["C1", "D1", "1", "04/10/2013", "Seed Round", "0.8", "P1", "0.8"],
    ["C1", "D2", "2", "06/01/2015", "Early Stage VC", "3.5", "P1", "3.5"],
    # Importo non dichiarato: alza la quota non dichiarata di quell'anno.
    ["C1", "D3", "3", "09/15/2018", "Later Stage VC", "", "P2", ""],
    ["C2", "D4", "1", "05/05/2014", "Seed Round", "0.5", "P6", "0.5"],
    ["C2", "D5", "2", "", "Merger/Acquisition", "12.0", "P6", "12.0"],
    ["C3", "D6", "1", "02/01/2015", "Accelerator/Incubator", "0.05", "P8", "0.05"],
    ["C3", "D7", "2", "", "Bankruptcy: Liquidation", "", "P8", ""],
    ["C4", "D8", "1", "", "Grant", "0.1", "P6", "0.1"],
    ["C5", "D9", "1", "03/12/2012", "Angel (individual)", "0.2", "P7", "0.2"],
    ["C5", "D10", "2", "", "Seed Round", "0.4", "P7", "0.4"],
    ["C5", "D11", "3", "", "Early Stage VC", "1.2", "P7", "1.2"],
    ["C5", "D12", "4", "07/01/2018", "Later Stage VC", "6.0", "P7", "6.0"],
    ["C6", "D13", "1", "01/01/2005", "Seed Round", "0.3", "", "0.3"],
    ["C7", "D14", "1", "01/01/2016", "Grant", "0.05", "P5", "0.05"],
]
DEAL_INVESTOR_COLS = ["DealID", "InvestorID", "InvestorStatus", "IsLeadInvestor"]
DEAL_INVESTOR = [
    ["D1", "I2", "New Investor", "Yes"],
    ["D2", "I1", "New Investor", "Yes"],
    ["D2", "I2", "Follow-on Investor", "No"],
    ["D3", "I1", "Follow-on Investor", "Yes"],
    ["D3", "I4", "New Investor", "No"],
    ["D4", "I3", "New Investor", "Yes"],
    ["D6", "I2", "New Investor", "Yes"],
    ["D9", "I3", "New Investor", "Yes"],
    ["D11", "I1", "New Investor", "Yes"],
    ["D12", "I1", "Follow-on Investor", "Yes"],
    # Uno stato non dichiarato rende incerta la domanda "c'e' un nuovo
    # investitore?", e gli aggregati di quel round restano vuoti.
    ["D14", "I4", "", "No"],
]
INVESTOR_COLS = [
    "InvestorID", "PrimaryInvestorType", "TotalInvestments", "MedianRoundAmount", "YearFounded",
]
INVESTOR = [
    ["I1", "Venture Capital", "120", "4.0", "2005"],
    ["I2", "Accelerator/Incubator", "300", "0.1", "2010"],
    ["I3", "Angel (individual)", "8", "0.3", ""],
    ["I4", "Corporation", "25", "10.0", "1998"],
]

# ── Chi somiglia a chi ──────────────────────────────────────────────────────
# S1 e' viva per tutta la vita di C1 e sta nel suo stesso paese; S2 chiude nel
# 2016, quindi dopo non conta piu'; S3 non sta in questa estrazione, quindi con la
# temporizzazione accesa la coppia si perde e spenta conta comunque; X1 e' simile
# ma non concorrente, quindi entra nella media di similarita' e non nei conteggi.
SIMILAR_COLS = [
    "CompanyID", "SimilarCompanyID", "SimilarityScore", "IsCompetitor", "SimilarCompanyHQCountry",
]
SIMILAR = [
    ["C1", "S1", "90", "Yes", "Italy"],
    ["C1", "S2", "70", "Yes", "Germany"],
    ["C1", "S3", "60", "Yes", "Italy"],
    ["C1", "X1", "40", "No", "Italy"],
    ["C3", "S1", "55", "Yes", "Italy"],
    ["C8", "S3", "80", "Yes", "Portugal"],
]

# fmt: on

TABELLE = {
    "Company": (COMPANY_COLS, COMPANY),
    "CompanyBoardTeamRelation": (BOARD_COLS, BOARD),
    "Person": (PERSON_COLS, PERSON),
    "PersonEducationRelation": (EDUCATION_COLS, EDUCATION),
    "PersonPositionRelation": (POSITION_COLS, POSITION),
    "PersonBoardSeatRelation": (BOARD_SEAT_COLS, BOARD_SEAT),
    "PersonAdvisoryRelation": (ADVISORY_COLS, ADVISORY),
    "PersonAffiliatedDealRelation": (AFFILIATED_DEAL_COLS, AFFILIATED_DEAL),
    "PersonAffiliatedFundRelation": (AFFILIATED_FUND_COLS, AFFILIATED_FUND),
    "Fund": (FUND_COLS, FUND),
    "Deal": (DEAL_COLS, DEAL),
    "DealInvestorRelation": (DEAL_INVESTOR_COLS, DEAL_INVESTOR),
    "Investor": (INVESTOR_COLS, INVESTOR),
    "CompanySimilarRelation": (SIMILAR_COLS, SIMILAR),
}


def main() -> int:
    """Write every table, checking that each row matches its header.

    :return: Process exit code.
    """
    USCITA.mkdir(parents=True, exist_ok=True)
    for nome, (intestazione, righe) in TABELLE.items():
        for r in righe:
            if len(r) != len(intestazione):
                raise ValueError(f"{nome}: riga con {len(r)} valori invece di {len(intestazione)}")
        with (USCITA / f"{nome}.csv").open("w", newline="") as f:
            scrittore = csv.writer(f)
            scrittore.writerow(intestazione)
            scrittore.writerows(righe)
        print(f"  {nome}.csv: {len(righe)} righe x {len(intestazione)} colonne")
    print(f"{len(TABELLE)} tabelle in {USCITA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
