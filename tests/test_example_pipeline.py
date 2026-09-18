"""The panel pipeline, end to end, on the synthetic extraction.

``scripts/make_example_data.py`` writes fourteen invented tables, each company
built to fire one branch of ``build_panel.ipynb``. This module runs the notebook
over them and states what the panel must say about each company: not a golden
file compared blindly, but the expected value of the rules that matter.

The run takes about twenty seconds and needs no PitchBook data.
"""

import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "build_panel.ipynb"


def _esegui(tmp_path: Path) -> pl.DataFrame:
    """Run the notebook in example mode and return the panel it writes.

    The stage outputs go to a temporary directory, so a test run never touches
    ``data/``. The notebook resolves its own paths from the working directory,
    hence the copy is executed from the repository root.

    :param tmp_path: Directory for the stage outputs and the copy.
    :return: The final panel.
    """
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    tocca = 0
    for cella in nb["cells"]:
        sorgente = "".join(cella["source"])
        if "ESEMPIO = False" in sorgente:
            sorgente = sorgente.replace("ESEMPIO = False", "ESEMPIO = True").replace(
                'interim_dir=Path("data/example/interim")', f'interim_dir=Path("{tmp_path}")'
            )
            cella["source"] = sorgente.splitlines(keepends=True)
            tocca += 1
    assert tocca == 1, "la cella di import non si trova: il test va riallineato al notebook"

    copia = ROOT / "_test_esempio.ipynb"
    copia.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        esito = subprocess.run(
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--execute",
                "--to",
                "notebook",
                "--ExecutePreprocessor.timeout=-1",
                "--output-dir",
                str(tmp_path),
                "--output",
                "eseguito.ipynb",
                copia.name,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    finally:
        copia.unlink(missing_ok=True)
    if esito.returncode != 0:
        pytest.fail(f"il notebook non arriva in fondo:\n{esito.stderr[-3000:]}")
    return pl.read_parquet(tmp_path / "panel.parquet")


@pytest.fixture(scope="module")
def panel(tmp_path_factory) -> pl.DataFrame:
    """The panel of the synthetic extraction, built once for the whole module."""
    pytest.importorskip("nbconvert", reason="serve a eseguire il notebook")
    return _esegui(tmp_path_factory.mktemp("esempio"))


def _azienda(panel: pl.DataFrame, anno_fondazione: int) -> pl.DataFrame:
    """The rows of the only company founded in that year, sorted by age.

    The panel renumbers the companies as its last step, so the fixture's own
    identifiers are gone by then; the founding years are distinct on purpose, and
    they are what the tests address a company by.

    :param panel: The panel.
    :param anno_fondazione: Founding year of the wanted company.
    :return: Its rows.
    """
    righe = panel.filter(pl.col("YearFounded") == anno_fondazione).sort("Age")
    quante = righe["CompanyID"].n_unique()
    assert quante == 1, f"aziende fondate nel {anno_fondazione}: {quante}, invece di una"
    return righe


# ── la forma del panel ─────────────────────────────────────────────────────


def test_the_panel_has_one_row_per_company_year(panel):
    assert panel.height == 79
    assert panel["CompanyID"].n_unique() == 10
    assert panel.select("CompanyID", "Age").n_unique() == panel.height


def test_no_row_precedes_the_founding_year(panel):
    assert panel.filter(pl.col("Age") < 0).height == 0


def test_the_company_founded_before_the_sample_threshold_is_absent(panel):
    # C6 is founded in 1999, and the sample starts in 2000.
    assert panel.filter(pl.col("YearFounded") == 1999).height == 0


def test_a_company_whose_data_predates_its_founding_keeps_only_year_zero(panel):
    # C7 declares 2016 and carries a 2015 date: the panel cannot end before it
    # begins, so the company is left with its first year alone.
    c7 = _azienda(panel, 2016)
    assert c7.height == 1
    assert c7["Age"].to_list() == [0]


# ── le date dei round ──────────────────────────────────────────────────────


def test_an_undated_acquisition_takes_the_date_of_the_ownership_change(panel):
    # C2 is acquired in 2019 and its acquisition round has no date: the panel
    # stops the year before, and the exit survives as the next stage.
    c2 = _azienda(panel, 2013)
    assert c2["Age"].max() == 5  # 2018, the year before the acquisition
    assert c2["GrowthNextStageGroup"].to_list()[-1] == "Exit"


def test_an_undated_bankruptcy_takes_the_date_of_the_ownership_change(panel):
    # C3 closes in 2018, and its liquidation round has no date either.
    c3 = _azienda(panel, 2014)
    assert c3["Age"].max() == 3  # 2017, the year before the closure
    assert c3["GrowthNextStageGroup"].to_list()[-1] == "Out"


def test_an_undated_first_round_of_an_initial_type_goes_to_the_founding_year(panel):
    # C4's only round is a grant with no date: it lands on age zero, which is why
    # the company already has a stage in its first year.
    c4 = _azienda(panel, 2015)
    assert c4["GrowthStageGroup"].to_list()[0] == "Early"
    assert c4["N_Deal"].to_list()[0] == 1


def test_undated_rounds_between_two_dated_ones_are_spread_over_the_gap(panel):
    # C5 has rounds in 2012 and 2018 with two undated ones in between: they are
    # placed at 2014 and 2016, so the count grows one step at a time.
    c5 = _azienda(panel, 2011)
    assert c5.select("Age", "N_Deal").rows() == [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 2),
        (4, 2),
        (5, 3),
        (6, 3),
        (7, 4),
        (8, 4),
    ]


def test_the_stage_of_a_company_never_goes_backwards(panel):
    ordine = {"Early": 0, "Later": 1, "Exit": 2, "Out": 2}
    for (_,), g in panel.sort("Age").group_by("CompanyID", maintain_order=True):
        stadi = [ordine[s] for s in g["GrowthStageGroup"].drop_nulls()]
        assert stadi == sorted(stadi)


# ── il team ────────────────────────────────────────────────────────────────


def test_a_founder_counts_from_the_founding_year(panel):
    # P1 founded C1 and declares no start date.
    c1 = _azienda(panel, 2012)
    assert c1["Total_People"].to_list()[0] == 1
    assert c1["Total_Founders"].to_list()[0] == 1


def test_a_person_with_a_declared_start_does_not_count_before_it(panel):
    # P4 joins C1's board in 2015 (age 3) and P2 in 2016 (age 4).
    c1 = _azienda(panel, 2012)
    assert c1.select("Age", "Total_People").rows()[:5] == [(0, 1), (1, 1), (2, 1), (3, 2), (4, 3)]


def test_a_non_founder_without_a_start_date_is_never_counted(panel):
    # P3 is C1's CFO with no start date: the team never reaches four people, even
    # though the raw table lists four.
    c1 = _azienda(panel, 2012)
    assert c1["Total_People"].max() == 3


def test_a_company_with_no_team_has_no_team_columns(panel):
    # S1 and the other counterparts have nobody on their board.
    senza = panel.filter(pl.col("Total_People").is_null())
    assert senza.height == 79 - 46
    assert senza["Percent_Females"].null_count() == senza.height


def test_the_share_of_women_ignores_the_people_with_no_gender(panel):
    # C3's team is P8 (male) and P9 (no gender): the share is 0, over one person
    # of known gender, and not 0 over two.
    c3 = _azienda(panel, 2014)
    assert c3["Total_People"].max() == 2
    assert c3["Percent_Females"].drop_nulls().to_list() == [0.0, 0.0, 0.0, 0.0]


def test_the_education_of_a_person_changes_when_the_degree_is_awarded(panel):
    # P5 founded C2 with a bachelor's and earns a PhD in 2017, after C2's panel
    # ends; in C7, founded in 2016, the same person still carries the bachelor's.
    c7 = _azienda(panel, 2016)
    assert c7["Highest_Degree_Mean"].to_list() == [3.0]


# ── i concorrenti ──────────────────────────────────────────────────────────


def test_only_the_competitors_alive_that_year_are_counted(panel):
    # C1 declares three competitors: S1 alive throughout, S2 until 2016, S3
    # outside this extraction. Timed, the count is two until 2016 and one after.
    c1 = _azienda(panel, 2012)
    assert c1.select("Age", "N_Competitors").rows() == [
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 2),
        (4, 2),
        (5, 1),
        (6, 1),
        (7, 1),
        (8, 1),
    ]


def test_a_similar_company_that_is_not_a_competitor_enters_the_mean_only(panel):
    # X1 is similar to C1 (score 40) without being a competitor, and it is alive
    # until 2012: in that year the mean covers S1, S2 and X1 but the count is two.
    c1 = _azienda(panel, 2012)
    assert c1["N_Similar"].to_list()[0] == 3
    assert c1["N_Competitors"].to_list()[0] == 2
    assert c1["SimilarityScoreMean"].to_list()[0] == pytest.approx((90 + 70 + 40) / 3)


def test_the_competitors_in_the_same_country_are_counted_apart(panel):
    # Of C1's two live competitors only S1 is Italian, like C1.
    c1 = _azienda(panel, 2012)
    assert c1["Same_Country"].to_list()[0] == 1


def test_a_company_whose_only_competitor_is_outside_the_extraction_counts_none(panel):
    # C8 declares S3 alone, and S3 has no life window in this extraction.
    c8 = _azienda(panel, 2010)
    assert c8["N_Competitors"].sum() == 0
    assert c8["SimilarityScoreMean"].sum() == 0.0
    # It does have a team, so what is missing is the counterpart, not the company.
    assert c8["Total_People"].is_not_null().all()


# ── il capitale ────────────────────────────────────────────────────────────


def test_the_capital_of_a_year_without_rounds_is_zero(panel):
    c1 = _azienda(panel, 2012)
    assert c1["TotalRaised"].to_list()[0] == 0.0
    assert c1["UndisclosedAmountShare"].to_list()[0] == 0.0


def test_a_round_with_no_amount_declares_it(panel):
    # C1's 2018 round has no amount: the capital of that year is zero and the
    # undisclosed share is one, which is what tells the two apart.
    c1 = _azienda(panel, 2012)
    riga = c1.filter(pl.col("Age") == 6)
    assert riga["TotalRaised"].to_list() == [0.0]
    assert riga["UndisclosedAmountShare"].to_list() == [1.0]
