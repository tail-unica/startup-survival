"""Writes the synthetic PitchBook-shaped tables the example run reads.

The real extraction cannot be redistributed, so the repository ships a tiny
invented one: fourteen tables, only the columns the pipeline actually reads, and
a cast of eight companies chosen so that every branch of the panel construction
fires at least once. Nothing here describes a real company or person.

    python scripts/make_example_data.py

The output lands in ``data/example/pitchbook/``, which the pipeline reads with
``--example`` on the command line, or with ``EXAMPLE = True`` in the notebook.
"""

import csv
import sys
from pathlib import Path

OUTPUT = Path(__file__).resolve().parents[1] / "data" / "example" / "pitchbook"

# fmt: off

# The companies
# The identifiers are zero-padded, and the companies the pipeline drops carry the
# highest ones. The last step of the pipeline renumbers the surviving companies
# with consecutive integers, in the order of their identifiers: numbered this way,
# every company keeps in the panel the number it has here, and that is what the
# tests address it by. A new company goes at the end, or the match breaks.
#
# 01  grows: seed, early VC, later VC. Its team mixes a founder with no start
#     date, a dated hire, and an undated one who is never counted.
# 02  is acquired in 2019, and the acquisition round carries no date: it takes
#     the one of the ownership change.
# 03  goes out of business in 2018, and here too the date of the bankruptcy round
#     comes from the ownership status.
# 04  has one round only, of an initial kind and undated: it lands on the
#     founding year.
# 05  has four rounds, the second and the third undated: they are spread between
#     the first and the fourth.
# 06  carries dates older than its founding, so its panel would end before it
#     begins: it is left with year zero alone.
# 07  has no round at all: it sits in the panel with no growth stage, and it
#     declares competitors, one of them outside this extraction.
# 08, 09, 10 appear only as counterparts or as past employers: they are there to
#     show that a relation can point outside the sample, and that a person's
#     experience also matures elsewhere.
# 11  is founded in 1999, before the threshold of the sample: it never enters the
#     panel, which is why it carries the last number.
COMPANY_COLS = [
    "CompanyID", "YearFounded", "HQCountry", "PrimaryIndustrySector", "OwnershipStatus",
    "OwnershipStatusDate", "CompanyFinancingStatusDate", "BusinessStatusDate",
    "FirstFinancingDate", "LastKnownValuationDate", "FiscalPeriod",
]
COMPANY = [
    ["01", "2012", "Italy", "Information Technology", "Privately Held",
     "06/30/2020", "09/15/2019", "", "04/10/2013", "12/31/2019", "TTM 2Q2020"],
    ["02", "2013", "Italy", "Healthcare", "Acquired/Merged",
     "03/20/2019", "03/20/2019", "03/20/2019", "05/05/2014", "", ""],
    ["03", "2014", "Spain", "Information Technology", "Out of Business",
     "11/10/2018", "", "11/10/2018", "02/01/2015", "", ""],
    ["04", "2015", "France", "Consumer Products and Services", "Privately Held",
     "01/31/2021", "01/31/2021", "", "", "", ""],
    ["05", "2011", "Germany", "Energy", "Privately Held",
     "12/31/2019", "07/01/2018", "", "03/12/2012", "", "TTM 4Q2019"],
    ["06", "2016", "Italy", "Healthcare", "Privately Held",
     "08/08/2015", "", "", "", "", ""],
    ["07", "2010", "Portugal", "Information Technology", "Privately Held",
     "04/04/2019", "", "", "", "", ""],
    ["08", "2008", "Italy", "Information Technology", "Privately Held",
     "01/15/2021", "", "", "", "", ""],
    ["09", "2009", "Germany", "Information Technology", "Out of Business",
     "06/30/2016", "", "", "", "", ""],
    ["10", "2001", "Italy", "Information Technology", "Privately Held",
     "02/02/2012", "", "", "", "", ""],
    ["11", "1999", "Italy", "Materials and Resources", "Privately Held",
     "05/05/2015", "", "", "", "", ""],
]

# The people
# P1  founder and chief executive of 01: no start date, so year zero applies.
# P2  joins 01 in 2016 with a declared date: before that year she does not count.
# P3  joins 01 with no date and is not a founder: she never counts.
# P4  carries the degree in his name ("Ph.D"), which holds in every year.
# P5  graduates in 2017: before then her highest degree is a lower one.
# P6  declares no graduating year, so his degree counts from the beginning.
# P7  founded 05 and holds an undated board seat in another company.
# P8  runs 03 and has an affiliated fund, with its own vintage.
# P9  declares no gender: counted among the people, not in the denominator of the
#     share of women.
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

# Who sits in which company, and since when
BOARD_COLS = [
    "CompanyID", "PersonID", "PersonName", "FullTitle", "IsCurrent", "StartDate", "EndDate",
    "LastUpdated",
]
BOARD = [
    ["01", "P1", "Anna Rossi", "Co-Founder & CEO", "Yes", "", "", "01/10/2021"],
    ["01", "P2", "Marco Bianchi", "Chief Technology Officer", "Yes",
     "07/01/2016", "", "01/10/2021"],
    ["01", "P3", "Chiara Verdi", "Chief Financial Officer", "Yes", "", "", "01/10/2021"],
    # The same appointment recorded twice, once with a leaving date and once as
    # still in office: the rows merge, and "in office" wins.
    ["01", "P4", "Luca Neri Ph.D", "Board Member", "No",
     "01/01/2015", "12/31/2018", "02/02/2019"],
    ["01", "P4", "Luca Neri Ph.D", "Board Member", "Yes", "01/01/2015", "", "05/05/2020"],
    ["02", "P5", "Sara Gallo", "Founder", "Yes", "", "", "03/03/2019"],
    ["02", "P6", "Paolo Conti", "Chief Executive Officer", "Yes",
     "01/01/2014", "", "03/03/2019"],
    # Two different roles of one person: they merge into the union of their
    # periods, and the founder title survives the merge.
    ["03", "P8", "Davide Mori", "Founder", "No", "01/01/2014", "06/30/2016", "01/01/2019"],
    ["03", "P8", "Davide Mori", "Chief Executive Officer", "No",
     "07/01/2016", "11/10/2018", "01/01/2019"],
    ["03", "P9", "Kim Park", "Head of Operations", "Yes", "01/01/2015", "", "01/01/2019"],
    ["04", "P6", "Paolo Conti", "Co-Founder", "Yes", "01/01/2015", "", "01/01/2021"],
    ["05", "P7", "Elena Ferri", "Founder & Chairman", "Yes", "", "", "01/01/2020"],
    ["05", "P2", "Marco Bianchi", "Advisor", "No", "01/01/2013", "12/31/2015", "01/01/2020"],
    ["06", "P5", "Sara Gallo", "Founder", "Yes", "", "", "01/01/2017"],
    ["07", "P1", "Anna Rossi", "Founder", "Yes", "", "", "01/01/2019"],
]

# The education
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

# The experience: one role per row
POSITION_COLS = ["PersonID", "EntityID", "StartDate", "PositionLevel", "IsCurrent"]
POSITION = [
    ["P1", "01", "", "Founder", "Yes"],
    ["P1", "10", "05/01/2008", "Manager", "No"],
    ["P2", "01", "07/01/2016", "Chief Technology Officer", "Yes"],
    ["P2", "05", "01/01/2013", "Advisor", "No"],
    ["P3", "01", "", "Chief Financial Officer", "Yes"],
    ["P5", "02", "", "Founder", "Yes"],
    ["P6", "04", "01/01/2015", "Founder", "Yes"],
    # An undated position in an entity that appears nowhere else: the year is
    # resolved with the first year known for that person.
    ["P6", "13", "", "Advisor", "No"],
    ["P6", "02", "01/01/2014", "Chief Executive Officer", "Yes"],
    ["P7", "05", "", "Founder", "Yes"],
    ["P8", "03", "01/01/2014", "Founder", "No"],
    ["P9", "03", "01/01/2015", "Head of Operations", "Yes"],
]
BOARD_SEAT_COLS = ["PersonID", "CompanyID", "StartDate", "IsCurrent"]
BOARD_SEAT = [
    ["P1", "08", "03/01/2017", "Yes"],
    ["P4", "01", "01/01/2015", "Yes"],
    # An undated board seat: the founding year of the company applies, which is
    # the only lower bound available.
    ["P7", "09", "", "No"],
]
ADVISORY_COLS = ["PersonID", "EntityID", "StartDate", "IsCurrent"]
ADVISORY = [
    ["P4", "10", "06/01/2014", "Yes"],
    ["P7", "10", "01/01/2010", "Yes"],
]
AFFILIATED_DEAL_COLS = ["PersonID", "CompanyID", "DealDate"]
AFFILIATED_DEAL = [
    ["P1", "01", "06/01/2015"],
]
AFFILIATED_FUND_COLS = ["PersonID", "FundID", "InvestorID"]
AFFILIATED_FUND = [
    ["P8", "F1", "I1"],
]
FUND_COLS = ["FundID", "Vintage"]
FUND = [
    ["F1", "2016"],
]

# The rounds
DEAL_COLS = [
    "CompanyID", "DealID", "DealNo", "DealDate", "DealType", "TotalInvestedCapital", "CEOPBId",
    "DealSize",
]
DEAL = [
    ["01", "D1", "1", "04/10/2013", "Seed Round", "0.8", "P1", "0.8"],
    ["01", "D2", "2", "06/01/2015", "Early Stage VC", "3.5", "P1", "3.5"],
    # An undeclared amount: it raises the undisclosed share of that year.
    ["01", "D3", "3", "09/15/2018", "Later Stage VC", "", "P2", ""],
    ["02", "D4", "1", "05/05/2014", "Seed Round", "0.5", "P6", "0.5"],
    ["02", "D5", "2", "", "Merger/Acquisition", "12.0", "P6", "12.0"],
    ["03", "D6", "1", "02/01/2015", "Accelerator/Incubator", "0.05", "P8", "0.05"],
    ["03", "D7", "2", "", "Bankruptcy: Liquidation", "", "P8", ""],
    ["04", "D8", "1", "", "Grant", "0.1", "P6", "0.1"],
    ["05", "D9", "1", "03/12/2012", "Angel (individual)", "0.2", "P7", "0.2"],
    ["05", "D10", "2", "", "Seed Round", "0.4", "P7", "0.4"],
    ["05", "D11", "3", "", "Early Stage VC", "1.2", "P7", "1.2"],
    ["05", "D12", "4", "07/01/2018", "Later Stage VC", "6.0", "P7", "6.0"],
    ["11", "D13", "1", "01/01/2005", "Seed Round", "0.3", "", "0.3"],
    ["06", "D14", "1", "01/01/2016", "Grant", "0.05", "P5", "0.05"],
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
    # An undeclared status makes the question "is there a new investor?"
    # unanswerable, and the aggregates of that round stay empty.
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

# Who resembles whom
# 08 is alive for the whole life of 01 and sits in its own country; 09 closes in
# 2016, so after that it stops counting; 12 is outside this extraction, so with
# the timing on the pair is lost and with it off it counts anyway; 10 is similar
# without being a competitor, so it enters the mean similarity and not the counts.
SIMILAR_COLS = [
    "CompanyID", "SimilarCompanyID", "SimilarityScore", "IsCompetitor", "SimilarCompanyHQCountry",
]
SIMILAR = [
    ["01", "08", "90", "Yes", "Italy"],
    ["01", "09", "70", "Yes", "Germany"],
    ["01", "12", "60", "Yes", "Italy"],
    ["01", "10", "40", "No", "Italy"],
    ["03", "08", "55", "Yes", "Italy"],
    ["07", "12", "80", "Yes", "Portugal"],
]

# fmt: on

TABLES = {
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
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, (header, rows) in TABLES.items():
        for r in rows:
            if len(r) != len(header):
                raise ValueError(f"{name}: row with {len(r)} values instead of {len(header)}")
        with (OUTPUT / f"{name}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"  {name}.csv: {len(rows)} rows x {len(header)} columns")
    print(f"{len(TABLES)} tables in {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
