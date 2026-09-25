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
    build_full_history_dataset,
    build_windowed_dataset,
    handle_missing_values,
    preprocess_dataset,
)

# The horizon these tests pin. The pipeline reads T from config.yaml; here it is
# fixed so the expected rows do not move when the configured value changes.
T = 7
LAST_YEAR = 2024
FIRST_DECISION_YEAR = 2010


def _panel(rows: list[dict]) -> pl.DataFrame:
    """A panel with the columns the target logic reads, plus the given rows.

    :param rows: One dict per company-year; missing keys take the defaults.
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
    return pl.DataFrame([{**default, **r} for r in rows])


def _history(
    company: int, years: int, *, yf: int = 2012, next_stage: str = "Later", at_age: int = 3
):
    """One company followed for ``years`` company-years, changing stage at ``at_age``.

    :param company: Company id.
    :param years: Number of company-years.
    :param yf: Founding year.
    :param next_stage: Stage reached at ``at_age``.
    :param at_age: Age at which the next stage arrives.
    :return: List of row dicts for :func:`_panel`.
    """
    return [
        {
            "CompanyID": company,
            "Age": age,
            "YearFounded": yf,
            "GrowthStageGroup": "Early",
            "GrowthNextStageGroup": next_stage,
            "TimeNextStageGroup": max(at_age - age, 0),
        }
        for age in range(years)
    ]


# build_windowed_dataset


def test_one_row_per_company_at_its_starting_age():
    out = build_windowed_dataset(_panel(_history(1, 5)), T, LAST_YEAR)
    assert out.height == 1
    assert out["Age"].to_list() == [0]
    assert out["StartingAge"].to_list() == [0]


def test_a_firm_that_is_early_only_after_age_two_is_out_of_the_sample():
    # The paper predicts from the first early-stage year, and only for firms that
    # reach it within two years of founding.
    late = [
        {"CompanyID": 1, "Age": age, "GrowthStageGroup": None if age < 3 else "Early"}
        for age in range(6)
    ]
    assert build_windowed_dataset(_panel(late), T, LAST_YEAR).height == 0
    # The threshold is a parameter, not a constant: raising it keeps the same firm.
    kept = build_windowed_dataset(_panel(late), T, LAST_YEAR, FIRST_DECISION_YEAR, 3)
    assert kept.height == 1


def test_a_firm_that_is_never_early_is_out_of_the_sample():
    rows = [{"CompanyID": 1, "Age": age, "GrowthStageGroup": "Later"} for age in range(4)]
    assert build_windowed_dataset(_panel(rows), T, LAST_YEAR).height == 0


def test_the_target_is_the_next_stage_when_it_arrives_inside_the_window():
    out = build_windowed_dataset(_panel(_history(1, 9, at_age=3)), T, LAST_YEAR)
    assert out["Target"].to_list() == ["Later"]


def test_the_target_stays_the_current_stage_when_the_change_is_too_late():
    # The change happens, but after StartingAge + T: at the decision point it is
    # not knowable, so the firm counts as not having moved.
    late_change = _history(1, 12, at_age=11)
    out = build_windowed_dataset(_panel(late_change), T, LAST_YEAR)
    assert out["Target"].to_list() == ["Early"]


def test_the_target_is_read_at_the_last_age_when_the_panel_is_shorter_than_the_window():
    short = _history(1, 3, at_age=2)
    out = build_windowed_dataset(_panel(short), T, LAST_YEAR)
    assert out["TargetAge"].to_list() == [2]
    assert out["Target"].to_list() == ["Later"]


def test_rows_without_a_stage_do_not_define_the_starting_age():
    rows = [
        {"CompanyID": 1, "Age": 0, "GrowthStageGroup": None},
        {"CompanyID": 1, "Age": 1, "GrowthStageGroup": "Early"},
        {"CompanyID": 1, "Age": 2, "GrowthStageGroup": "Early"},
    ]
    out = build_windowed_dataset(_panel(rows), T, LAST_YEAR)
    assert out["StartingAge"].to_list() == [1]


def test_a_decision_year_that_leaves_less_than_a_window_of_future_is_dropped():
    # With a T-year window and data up to 2024, a firm deciding in 2018 could not
    # be observed for the full horizon.
    recent = _history(1, 5, yf=2018)
    assert build_windowed_dataset(_panel(recent), T, LAST_YEAR).height == 0


def test_a_decision_year_before_the_threshold_is_dropped():
    old = _history(1, 9, yf=2005)
    assert build_windowed_dataset(_panel(old), T, LAST_YEAR, FIRST_DECISION_YEAR).height == 0
    # The threshold is a parameter, not a constant: lowering it keeps the same firm.
    assert build_windowed_dataset(_panel(old), T, LAST_YEAR, 2000).height == 1


def test_two_companies_keep_their_own_starting_age_and_target():
    panel = _panel(_history(1, 9, at_age=2) + _history(2, 9, next_stage="Exit", at_age=8))
    out = build_windowed_dataset(panel, T, LAST_YEAR).sort("CompanyID")
    assert out["Target"].to_list() == ["Later", "Early"]


# build_full_history_dataset


def _panel_history() -> pl.DataFrame:
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
            "N_Similar": [1, 2, 4],
            "Same_Country": [0, 0, 1],
            "SimilarityScoreMean": [10.0, 20.0, 30.0],
        }
    )


def test_full_history_sums_the_capital_and_averages_the_shares():
    controlled = pl.DataFrame({"CompanyID": [1]})
    out = build_full_history_dataset(_panel_history(), controlled)
    assert out["TotalRaised"].to_list() == [7.0]
    assert out["UndisclosedAmountShare"].to_list() == [0.5]
    assert out["WorkExp_Idx_Mean"].to_list() == [1.0]
    assert out["Avg_Earliest_Year"].to_list() == [2002.0]


def test_full_history_takes_the_last_year_of_every_other_column():
    out = build_full_history_dataset(_panel_history(), pl.DataFrame({"CompanyID": [1]}))
    assert out["N_Deal"].to_list() == [3]
    assert out["Total_People"].to_list() == [4]
    assert out["Target"].to_list() == ["Later"]


def test_full_history_keeps_only_the_firms_of_the_windowed_dataset():
    panel = pl.concat(
        [_panel_history(), _panel_history().with_columns(pl.lit(2, pl.Int64).alias("CompanyID"))]
    )
    out = build_full_history_dataset(panel, pl.DataFrame({"CompanyID": [1]}))
    assert out["CompanyID"].to_list() == [1]


def test_full_history_skips_the_years_without_a_stage_or_without_a_team():
    panel = _panel_history().with_columns(
        pl.when(pl.col("Age") == 2)
        .then(None)
        .otherwise(pl.col("Total_People"))
        .alias("Total_People")
    )
    out = build_full_history_dataset(panel, pl.DataFrame({"CompanyID": [1]}))
    # The last usable year is age 1: the capital sums only the years kept.
    assert out["Age"].to_list() == [1]
    assert out["TotalRaised"].to_list() == [3.0]


def test_full_history_follows_the_order_of_the_windowed_dataset():
    panel = pl.concat(
        [_panel_history(), _panel_history().with_columns(pl.lit(2, pl.Int64).alias("CompanyID"))]
    )
    out = build_full_history_dataset(panel, pl.DataFrame({"CompanyID": [2, 1]}))
    assert out["CompanyID"].to_list() == [2, 1]


def test_full_history_refuses_a_firm_of_the_windowed_dataset_it_cannot_describe():
    panel = _panel_history().with_columns(pl.lit(None, pl.Int64).alias("Total_People"))
    with pytest.raises(ValueError):
        build_full_history_dataset(panel, pl.DataFrame({"CompanyID": [1]}))


# handle_missing_values


def _dataset_with_missing(n=100, null_avg_rows=0):
    """A dataset whose ``Avg_Earliest_Year`` is null on ``null_avg_rows`` rows."""
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
            "N_Similar": [None] * n,
            "Same_Country": [None] * n,
            "SimilarityScoreMean": [50.0] * (n - 1) + [None],
            "Avg_Earliest_Year": [None] * null_avg_rows + [2000.0] * (n - null_avg_rows),
        }
    )


def test_counts_with_no_competitor_become_zero_not_missing():
    out = handle_missing_values(_dataset_with_missing(), flag_no_time_window=False)
    assert out["N_Competitors"].to_list() == [0] * 100
    assert out["N_Similar"].to_list() == [0] * 100
    assert out["Same_Country"].to_list() == [0] * 100


def test_a_missing_similarity_becomes_zero_and_not_a_statistic_of_the_sample():
    # Every other row of the fixture scores 50, so the mean of the column is 50:
    # imputing it would carry the rest of the sample into the row, and the row that
    # has no comparable company would look like an average one.
    out = handle_missing_values(_dataset_with_missing(), flag_no_time_window=False)
    assert out["SimilarityScoreMean"].to_list() == [50.0] * 99 + [0.0]


def test_a_column_missing_on_more_than_half_the_rows_is_dropped():
    kept = handle_missing_values(_dataset_with_missing(null_avg_rows=49), flag_no_time_window=False)
    dropped = handle_missing_values(
        _dataset_with_missing(null_avg_rows=51), flag_no_time_window=False
    )
    assert "Avg_Earliest_Year" in kept.columns
    assert "Avg_Earliest_Year" not in dropped.columns


def test_the_rows_without_a_team_are_dropped_only_in_the_windowed_dataset():
    dataset = _dataset_with_missing().with_columns(
        pl.when(pl.col("CompanyID") == 0)
        .then(None)
        .otherwise(pl.col("Total_People"))
        .alias("Total_People")
    )
    windowed = handle_missing_values(dataset, flag_no_time_window=False)
    history = handle_missing_values(dataset, flag_no_time_window=True)
    assert windowed.height == 99
    assert history.height == 100


def test_a_row_missing_six_of_its_features_is_dropped():
    # The budget is per row, and it goes with the column threshold: five missing
    # values out of ~49 is a row the imputation can still complete, six is not.
    dataset = _dataset_with_missing()
    empty = ["Is_Eco", "Is_Eng", "Is_NS", "Is_Hum", "Is_SS", "Is_Med"]
    with_five = dataset.with_columns(
        [
            pl.when(pl.col("CompanyID") == 0).then(None).otherwise(pl.col(c)).alias(c)
            for c in empty[:5]
        ]
    )
    with_six = dataset.with_columns(
        [pl.when(pl.col("CompanyID") == 0).then(None).otherwise(pl.col(c)).alias(c) for c in empty]
    )
    assert handle_missing_values(with_five, flag_no_time_window=False).height == 100
    assert handle_missing_values(with_six, flag_no_time_window=False).height == 99


def test_the_no_window_dataset_keeps_every_column_so_the_two_stay_comparable():
    out = handle_missing_values(_dataset_with_missing(null_avg_rows=90), flag_no_time_window=True)
    assert "Avg_Earliest_Year" in out.columns


# preprocess_dataset


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


def _full_dataset(n=60, **override):
    """A dataset carrying every column ``preprocess_dataset`` selects.

    :param n: Number of rows.
    :param override: Columns to replace, as full lists of length ``n``.
    :return: Dataset frame.
    """

    def cycle(values):
        return [values[i % len(values)] for i in range(n)]

    rows: dict[str, list] = {c: [0] * n for c in FEATURE_COLUMNS}
    rows.update(
        {
            "CompanyID": list(range(n)),
            "Target": cycle(["Later", "Exit", "Early", "Out"]),
            # The first half names a ranked university, the second one an
            # institute sharing no word with any of them: a fixture with two
            # ranked names would have both inside its own top 50.
            "Institute": ["Massachusetts Institute of Technology; Altro"] * (n // 2)
            + ["Politecnico Zurigo"] * (n - n // 2),
            "Gender_CEO": cycle(["Female", "Male", None]),
            "HQCountry": ["USA"] * n,
            "PrimaryIndustrySector": ["Software"] * n,
            "Total_People": [3] * n,
            "YearFounded": [2012] * n,
            "SimilarityScoreMean": [50.0] * n,
        }
    )
    rows.update(override)
    return pl.DataFrame(rows)


def test_the_target_becomes_one_for_later_and_exit_and_zero_otherwise(ranking):
    out = preprocess_dataset(_full_dataset(), ranking)
    expected = {"Later": 1, "Exit": 1, "Early": 0, "Out": 0}
    original = _full_dataset()
    pairs = dict(zip(original["CompanyID"].to_list(), original["Target"].to_list(), strict=True))
    for cid, target in zip(out["CompanyID"].to_list(), out["Target"].to_list(), strict=True):
        assert target == expected[pairs[cid]]


def test_the_institute_list_becomes_a_single_top_fifty_flag(ranking):
    out = preprocess_dataset(_full_dataset(), ranking)
    assert "Institute" not in out.columns
    # handle_missing_values casts the booleans to 0/1 before the models see them.
    assert out["HasTop50Institute"].to_list()[0] == 1
    assert out["HasTop50Institute"].to_list()[-1] == 0


def test_an_institute_sharing_half_its_words_with_a_ranked_one_counts_as_top_fifty(ranking):
    # Documented on purpose, because it is loose: the match is a word overlap of
    # 0.5, so "Universita di Sassari" passes for "Universita di Cagliari" once
    # the latter is in the ranking. Two words in common out of four are enough.
    dataset = _full_dataset(n=4, Institute=["Universita di Sassari"] * 4)
    out = preprocess_dataset(dataset, ranking)
    assert out["HasTop50Institute"].to_list() == [1, 1, 1, 1]


def test_an_empty_institute_list_is_not_a_top_fifty(ranking):
    dataset = _full_dataset(n=4, Institute=[None, "", "Ignota", "Altro Ateneo"])
    out = preprocess_dataset(dataset, ranking)
    assert out["HasTop50Institute"].to_list() == [0, 0, 0, 0]


def test_the_ceo_gender_becomes_one_indicator_column(ranking):
    out = preprocess_dataset(_full_dataset(), ranking)
    assert "Gender_CEO_Female" in out.columns
    assert "Gender_CEO_Male" not in out.columns
    assert "Gender_CEO_null" not in out.columns


def test_the_categories_that_are_frequency_encoded_per_split_stay_raw(ranking):
    # Encoding them here would fit on the whole dataset, before the split, and
    # let a held-out row contribute to its own encoding.
    out = preprocess_dataset(_full_dataset(), ranking)
    assert out["HQCountry"].dtype == pl.String
    assert out["PrimaryIndustrySector"].dtype == pl.String
