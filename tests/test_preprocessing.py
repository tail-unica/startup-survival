import polars as pl

from src.preprocessing import handleCategoricalVariables


def _dataset(n_usa=1000, n_ita=10, sector_nulls=0):
    """Minimal frame with the columns handleCategoricalVariables touches."""
    countries = ["USA"] * n_usa + ["ITA"] * n_ita
    n = len(countries)
    sectors = ["Software"] * (n - sector_nulls) + [None] * sector_nulls
    # Gender_CEO is mostly null in the real panel, and the dummy step drops that
    # level explicitly, so the fixture has to carry one.
    genders = [["Female", "Male", None][i % 3] for i in range(n)]
    return pl.DataFrame({
        "CompanyID": list(range(n)),
        "HQCountry": countries,
        "PrimaryIndustrySector": sectors,
        "Gender_CEO": genders,
        "Target": [i % 2 for i in range(n)],
    })


def test_raw_categorical_columns_survive_preprocessing():
    """The frequency encoding moved past the train/test split, so the processed
    CSV has to carry the categories themselves, not their global frequency."""
    out = handleCategoricalVariables(_dataset())

    assert "HQCountry" in out.columns
    assert "PrimaryIndustrySector" in out.columns


def test_no_frequency_columns_are_produced():
    out = handleCategoricalVariables(_dataset())

    assert "HQCountryFreq" not in out.columns
    assert "PrimaryIndustrySectorFreq" not in out.columns


def test_rare_categories_are_not_collapsed_before_the_split():
    """Collapsing here would use the whole dataset to decide what is rare."""
    out = handleCategoricalVariables(_dataset(n_usa=1000, n_ita=10))

    assert set(out["HQCountry"].unique()) == {"USA", "ITA"}


def test_rows_with_a_missing_sector_are_kept():
    """The old inner join dropped them silently; they now reach the encoder,
    which sends them to Others."""
    out = handleCategoricalVariables(_dataset(n_usa=1000, n_ita=10, sector_nulls=7))

    assert len(out) == 1010
    assert out["PrimaryIndustrySector"].null_count() == 7


def test_ceo_gender_is_still_a_single_indicator():
    """Dummies are not frequency encoding: no leakage, so they stay put."""
    out = handleCategoricalVariables(_dataset())

    assert "Gender_CEO_Female" in out.columns
    assert "Gender_CEO_Male" not in out.columns
    assert "Gender_CEO" not in out.columns
