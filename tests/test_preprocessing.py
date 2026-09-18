"""Target engineering and missing-value policy, on hand-written panels.

These are the paper's invariants: the target is read at a fixed horizon from the
starting age and never later, a feature is computed only from what the panel
carries up to that row, and nothing is decided by looking at the whole dataset
before the split. Every fixture is a few rows written here, so a test failing
means the logic changed, not the extraction.
"""

import polars as pl
import pytest

from src.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    build_full_history_dataset,
    build_windowed_dataset,
    handle_missing_values,
    preprocess_dataset,
)

WINDOW = 7
LAST_YEAR = 2024


def _panel(righe: list[dict]) -> pl.DataFrame:
    """A panel with the columns the target logic reads, plus the given rows.

    :param righe: One dict per company-year; missing keys take the defaults.
    :return: Panel frame.
    """
    default = {
        "CompanyID": 1,
        "Age": 0,
        "YearFounded": 2012,
        "GrowthStageGroup": "Early",
        "GrowthNextStageGroup": "Early",
        "TimeNextStageGroup": 0,
    }
    return pl.DataFrame([{**default, **r} for r in righe])


def _storia(company: int, anni: int, *, yf: int = 2012, next_stage: str = "Later", quando: int = 3):
    """One company followed for ``anni`` years, changing stage at ``quando``.

    :param company: Company id.
    :param anni: Number of company-years.
    :param yf: Founding year.
    :param next_stage: Stage reached at ``quando``.
    :param quando: Age at which the next stage arrives.
    :return: List of row dicts for :func:`_panel`.
    """
    return [
        {
            "CompanyID": company,
            "Age": age,
            "YearFounded": yf,
            "GrowthStageGroup": "Early",
            "GrowthNextStageGroup": next_stage,
            "TimeNextStageGroup": max(quando - age, 0),
        }
        for age in range(anni)
    ]


# ── build_windowed_dataset ──────────────────────────────────────────────────


def test_one_row_per_company_at_its_starting_age():
    out = build_windowed_dataset(_panel(_storia(1, 5)), WINDOW, LAST_YEAR)
    assert out.height == 1
    assert out["Age"].to_list() == [0]
    assert out["StartingAge"].to_list() == [0]


def test_a_firm_that_is_early_only_after_age_two_is_out_of_the_sample():
    # The paper predicts from the first early-stage year, and only for firms that
    # reach it within two years of founding.
    tardi = [
        {"CompanyID": 1, "Age": age, "GrowthStageGroup": None if age < 3 else "Early"}
        for age in range(6)
    ]
    assert build_windowed_dataset(_panel(tardi), WINDOW, LAST_YEAR).height == 0


def test_a_firm_that_is_never_early_is_out_of_the_sample():
    righe = [{"CompanyID": 1, "Age": age, "GrowthStageGroup": "Later"} for age in range(4)]
    assert build_windowed_dataset(_panel(righe), WINDOW, LAST_YEAR).height == 0


def test_the_target_is_the_next_stage_when_it_arrives_inside_the_window():
    out = build_windowed_dataset(_panel(_storia(1, 9, quando=3)), WINDOW, LAST_YEAR)
    assert out["Target"].to_list() == ["Later"]


def test_the_target_stays_the_current_stage_when_the_change_is_too_late():
    # The change happens, but after StartingAge + 7: at the decision point it is
    # not knowable, so the firm counts as not having moved.
    tardiva = _storia(1, 12, quando=11)
    out = build_windowed_dataset(_panel(tardiva), WINDOW, LAST_YEAR)
    assert out["Target"].to_list() == ["Early"]


def test_the_target_is_read_at_the_last_age_when_the_panel_is_shorter_than_the_window():
    corta = _storia(1, 3, quando=2)
    out = build_windowed_dataset(_panel(corta), WINDOW, LAST_YEAR)
    assert out["TargetAge"].to_list() == [2]
    assert out["Target"].to_list() == ["Later"]


def test_rows_without_a_stage_do_not_define_the_starting_age():
    righe = [
        {"CompanyID": 1, "Age": 0, "GrowthStageGroup": None},
        {"CompanyID": 1, "Age": 1, "GrowthStageGroup": "Early"},
        {"CompanyID": 1, "Age": 2, "GrowthStageGroup": "Early"},
    ]
    out = build_windowed_dataset(_panel(righe), WINDOW, LAST_YEAR)
    assert out["StartingAge"].to_list() == [1]


def test_a_decision_year_that_leaves_less_than_a_window_of_future_is_dropped():
    # With a 7-year window and data up to 2024, a firm deciding in 2018 could not
    # be observed for the full horizon.
    recente = _storia(1, 5, yf=2018)
    assert build_windowed_dataset(_panel(recente), WINDOW, LAST_YEAR).height == 0


def test_a_decision_year_before_2010_is_dropped():
    vecchia = _storia(1, 9, yf=2005)
    assert build_windowed_dataset(_panel(vecchia), WINDOW, LAST_YEAR).height == 0


def test_two_companies_keep_their_own_starting_age_and_target():
    panel = _panel(_storia(1, 9, quando=2) + _storia(2, 9, next_stage="Exit", quando=8))
    out = build_windowed_dataset(panel, WINDOW, LAST_YEAR).sort("CompanyID")
    assert out["Target"].to_list() == ["Later", "Early"]


# ── build_full_history_dataset ─────────────────────────────────────────────


def _panel_storia() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "CompanyID": [1, 1, 1],
            "Age": [0, 1, 2],
            "YearFounded": [2012] * 3,
            "GrowthStageGroup": ["Early", "Early", "Later"],
            "GrowthNextStageGroup": ["Later", "Later", "Later"],
            "TimeNextStageGroup": [2, 1, 0],
            "Total_People": [2, 3, 4],
            "TotalRaised": [1.0, 2.0, 4.0],
            "UndisclosedAmountShare": [0.0, 1.0, 0.5],
            "WorkExp_Idx_Mean": [0.0, 1.0, 2.0],
            "Highest_Degree_Mean": [3.0, 3.0, 4.0],
            "Avg_Earliest_Year": [2000.0, 2002.0, 2004.0],
            "N_Deal": [1, 2, 3],
            "N_Competitors": [0, 1, 2],
            "Same_Country": [0, 0, 1],
            "SimilarityScoreMean": [10.0, 20.0, 30.0],
        }
    )


def test_full_history_sums_the_capital_and_averages_the_shares():
    window = pl.DataFrame({"CompanyID": [1]})
    out = build_full_history_dataset(_panel_storia(), window)
    assert out["TotalRaised"].to_list() == [7.0]
    assert out["UndisclosedAmountShare"].to_list() == [0.5]
    assert out["WorkExp_Idx_Mean"].to_list() == [1.0]
    assert out["Avg_Earliest_Year"].to_list() == [2002.0]


def test_full_history_takes_the_last_year_of_every_other_column():
    out = build_full_history_dataset(_panel_storia(), pl.DataFrame({"CompanyID": [1]}))
    assert out["N_Deal"].to_list() == [3]
    assert out["Total_People"].to_list() == [4]
    assert out["Target"].to_list() == ["Later"]


def test_full_history_keeps_only_the_firms_of_the_windowed_dataset():
    panel = pl.concat(
        [_panel_storia(), _panel_storia().with_columns(pl.lit(2, pl.Int64).alias("CompanyID"))]
    )
    out = build_full_history_dataset(panel, pl.DataFrame({"CompanyID": [1]}))
    assert out["CompanyID"].to_list() == [1]


def test_full_history_skips_the_years_without_a_stage_or_without_a_team():
    panel = _panel_storia().with_columns(
        pl.when(pl.col("Age") == 2)
        .then(None)
        .otherwise(pl.col("Total_People"))
        .alias("Total_People")
    )
    out = build_full_history_dataset(panel, pl.DataFrame({"CompanyID": [1]}))
    # The last usable year is age 1: the capital sums only the years kept.
    assert out["Age"].to_list() == [1]
    assert out["TotalRaised"].to_list() == [3.0]


# ── handle_missing_values ──────────────────────────────────────────────────


def _dataset_mancanti(n=100, nulli_avg=0):
    """A dataset whose ``Avg_Earliest_Year`` is null on ``nulli_avg`` rows."""
    return pl.DataFrame(
        {
            "CompanyID": list(range(n)),
            "Target": [i % 2 for i in range(n)],
            "Total_People": [3] * n,
            "Total_Founders": [1] * n,
            "YearFounded": [2012] * n,
            "Age": [0] * n,
            "Is_Eco": [True] * n,
            "Is_Eng": [False] * n,
            "Is_NS": [False] * n,
            "Is_Hum": [False] * n,
            "Is_SS": [False] * n,
            "Is_Med": [False] * n,
            "Is_Law": [False] * n,
            "Is_IT": [True] * n,
            "N_Competitors": [None] * n,
            "Same_Country": [None] * n,
            "SimilarityScoreMean": [50.0] * (n - 1) + [None],
            "Avg_Earliest_Year": [None] * nulli_avg + [2000.0] * (n - nulli_avg),
        }
    )


def test_counts_with_no_competitor_become_zero_not_missing():
    out = handle_missing_values(_dataset_mancanti(), flag_no_time_window=False)
    assert out["N_Competitors"].to_list() == [0] * 100
    assert out["Same_Country"].to_list() == [0] * 100


def test_a_missing_similarity_takes_the_mean_of_the_column():
    out = handle_missing_values(_dataset_mancanti(), flag_no_time_window=False)
    assert out["SimilarityScoreMean"].to_list() == [50.0] * 100


def test_a_column_missing_on_more_than_half_the_rows_is_dropped():
    tenuta = handle_missing_values(_dataset_mancanti(nulli_avg=49), flag_no_time_window=False)
    scartata = handle_missing_values(_dataset_mancanti(nulli_avg=51), flag_no_time_window=False)
    assert "Avg_Earliest_Year" in tenuta.columns
    assert "Avg_Earliest_Year" not in scartata.columns


def test_the_rows_without_a_team_are_dropped_only_in_the_windowed_dataset():
    dataset = _dataset_mancanti().with_columns(
        pl.when(pl.col("CompanyID") == 0)
        .then(None)
        .otherwise(pl.col("Total_People"))
        .alias("Total_People")
    )
    finestra = handle_missing_values(dataset, flag_no_time_window=False)
    storia = handle_missing_values(dataset, flag_no_time_window=True)
    assert finestra.height == 99
    assert storia.height == 100


def test_a_row_missing_six_of_its_features_is_dropped():
    # The budget is per row, and it goes with the column threshold: five missing
    # values out of ~49 is a row the imputation can still complete, six is not.
    dataset = _dataset_mancanti()
    vuote = ["Is_Eco", "Is_Eng", "Is_NS", "Is_Hum", "Is_SS", "Is_Med"]
    with_five = dataset.with_columns(
        [
            pl.when(pl.col("CompanyID") == 0).then(None).otherwise(pl.col(c)).alias(c)
            for c in vuote[:5]
        ]
    )
    with_six = dataset.with_columns(
        [pl.when(pl.col("CompanyID") == 0).then(None).otherwise(pl.col(c)).alias(c) for c in vuote]
    )
    assert handle_missing_values(with_five, flag_no_time_window=False).height == 100
    assert handle_missing_values(with_six, flag_no_time_window=False).height == 99


def test_the_no_window_dataset_keeps_every_column_so_the_two_stay_comparable():
    out = handle_missing_values(_dataset_mancanti(nulli_avg=90), flag_no_time_window=True)
    assert "Avg_Earliest_Year" in out.columns


# ── preprocess_dataset ────────────────────────────────────────────────────


@pytest.fixture
def ranking(tmp_path):
    """A QS ranking file with one top university."""
    path = tmp_path / "qs.csv"
    path.write_text(
        "University,Year,Overall Score\n"
        "Massachusetts Institute of Technology (MIT),2024,100\n"
        "Universita di Cagliari,2024,20\n"  # in the file, so also in its top 50
    )
    return str(path)


def _dataset_completo(n=60, **override):
    """A dataset carrying every column ``preprocess_dataset`` selects.

    :param n: Number of rows.
    :param override: Columns to replace, as full lists of length ``n``.
    :return: Dataset frame.
    """

    def ciclo(valori):
        return [valori[i % len(valori)] for i in range(n)]

    righe = {c: [0] * n for c in FEATURE_COLUMNS}
    righe.update(
        {
            "CompanyID": list(range(n)),
            "Target": ciclo(["Later", "Exit", "Early", "Out"]),
            # The first half names a ranked university, the second one an
            # institute sharing no word with any of them: a fixture with two
            # ranked names would have both inside its own top 50.
            "Institute": ["Massachusetts Institute of Technology; Altro"] * (n // 2)
            + ["Politecnico Zurigo"] * (n - n // 2),
            "Gender_CEO": ciclo(["Female", "Male", None]),
            "HQCountry": ["USA"] * n,
            "PrimaryIndustrySector": ["Software"] * n,
            "Total_People": [3] * n,
            "YearFounded": [2012] * n,
            "SimilarityScoreMean": [50.0] * n,
        }
    )
    righe.update(override)
    return pl.DataFrame(righe)


def test_the_target_becomes_one_for_later_and_exit_and_zero_otherwise(ranking):
    out = preprocess_dataset(_dataset_completo(), ranking)
    atteso = {"Later": 1, "Exit": 1, "Early": 0, "Out": 0}
    originale = _dataset_completo()
    coppie = dict(zip(originale["CompanyID"].to_list(), originale["Target"].to_list(), strict=True))
    for cid, target in zip(out["CompanyID"].to_list(), out["Target"].to_list(), strict=True):
        assert target == atteso[coppie[cid]]


def test_the_institute_list_becomes_a_single_top_fifty_flag(ranking):
    out = preprocess_dataset(_dataset_completo(), ranking)
    assert "Institute" not in out.columns
    # handle_missing_values casts the booleans to 0/1 before the models see them.
    assert out["HasTop50Institute"].to_list()[0] == 1
    assert out["HasTop50Institute"].to_list()[-1] == 0


def test_an_institute_sharing_half_its_words_with_a_ranked_one_counts_as_top_fifty(ranking):
    # Documented on purpose, because it is loose: the match is a word overlap of
    # 0.5, so "Universita di Sassari" passes for "Universita di Cagliari" once
    # the latter is in the ranking. Two words in common out of four are enough.
    dataset = _dataset_completo(n=4, Institute=["Universita di Sassari"] * 4)
    out = preprocess_dataset(dataset, ranking)
    assert out["HasTop50Institute"].to_list() == [1, 1, 1, 1]


def test_an_empty_institute_list_is_not_a_top_fifty(ranking):
    dataset = _dataset_completo(n=4, Institute=[None, "", "Ignota", "Altro Ateneo"])
    out = preprocess_dataset(dataset, ranking)
    assert out["HasTop50Institute"].to_list() == [0, 0, 0, 0]


def test_the_ceo_gender_becomes_one_indicator_column(ranking):
    out = preprocess_dataset(_dataset_completo(), ranking)
    assert "Gender_CEO_Female" in out.columns
    assert "Gender_CEO_Male" not in out.columns
    assert "Gender_CEO_null" not in out.columns


def test_the_categories_that_are_frequency_encoded_per_split_stay_raw(ranking):
    # Encoding them here would fit on the whole dataset, before the split, and
    # let a held-out row contribute to its own encoding.
    out = preprocess_dataset(_dataset_completo(), ranking)
    assert out["HQCountry"].dtype == pl.String
    assert out["PrimaryIndustrySector"].dtype == pl.String


def test_preprocess_dataset_reads_only_the_declared_columns(ranking):
    dataset = _dataset_completo().with_columns(pl.lit("da ignorare").alias("ColonnaDiTroppo"))
    out = preprocess_dataset(dataset, ranking)
    assert "ColonnaDiTroppo" not in out.columns


def test_the_column_lists_do_not_overlap_and_carry_the_keys():
    # FEATURE_COLUMNS feeds the models, TARGET_COLUMNS places the firm in time.
    assert "CompanyID" in TARGET_COLUMNS and "CompanyID" not in FEATURE_COLUMNS
    assert "Target" not in FEATURE_COLUMNS
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)
