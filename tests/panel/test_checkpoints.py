import polars as pl

from src.panel.config import PanelConfig
from src.panel.validate import (
    BASELINE_KEYS,
    CHECKPOINTS,
    EST_COLUMNS,
    KEPT_EXTRA,
    load_reference,
)


def test_six_checkpoints_are_registered_with_the_expected_shapes():
    assert set(CHECKPOINTS) == {"A", "B", "C", "D", "E", "F"}
    assert CHECKPOINTS["A"].reference == "db3.csv"
    assert CHECKPOINTS["A"].key == ["CompanyID", "PersonID"]
    assert CHECKPOINTS["A"].expect_rows == 534_851
    assert CHECKPOINTS["B"].reference == "db_master_1.csv"
    assert CHECKPOINTS["B"].key == ["CompanyID"]
    assert CHECKPOINTS["B"].expect_rows == 116_920
    assert CHECKPOINTS["C"].reference == "db_master_2.csv"
    assert CHECKPOINTS["C"].key == ["CompanyID", "Year_Delta"]
    assert CHECKPOINTS["C"].expect_rows == 1_001_625
    assert CHECKPOINTS["D"].reference == "db_selected.csv"
    assert CHECKPOINTS["D"].expect_rows == 1_001_625
    assert CHECKPOINTS["E"].reference == "db_master_panel.csv.gz"
    assert CHECKPOINTS["E"].expect_rows == 882_324
    assert CHECKPOINTS["F"].reference == "data/raw/panel.csv.gz"
    assert CHECKPOINTS["F"].expect_rows == 882_324


def test_db_master_csv_is_not_a_reference():
    assert all(cp.reference != "db_master.csv" for cp in CHECKPOINTS.values())


def test_est_columns_are_expected_missing_from_db_master_2_onwards():
    assert CHECKPOINTS["A"].expected_missing == frozenset()
    assert CHECKPOINTS["B"].expected_missing == frozenset()
    for letter in "CDEF":
        assert CHECKPOINTS[letter].expected_missing == EST_COLUMNS


def test_tr_d_is_expected_extra_only_after_db_master_2():
    # TR_D exists in db_master_2.csv, so C compares it; vars_selected drops it,
    # so from db_selected onwards it is a column we add.
    assert CHECKPOINTS["C"].expected_extra == frozenset()
    for letter in "DEF":
        assert CHECKPOINTS[letter].expected_extra == KEPT_EXTRA


def test_only_db_master_panel_uses_the_r_na_token():
    assert CHECKPOINTS["E"].na_token == "NA"
    assert all(CHECKPOINTS[k].na_token is None for k in "ABCDF")


def test_load_reference_keeps_every_column_as_text():
    cfg = PanelConfig()
    df = load_reference(cfg, "db_master_1.csv").head(50)
    assert set(df.schema.values()) == {pl.String}


def test_load_reference_leaves_the_r_na_token_as_a_string():
    cfg = PanelConfig()
    df = load_reference(cfg, "db_master_panel.csv.gz").head(50)
    assert df.null_count().sum_horizontal().item() == 0


def test_declared_missing_and_extra_columns_exist_where_claimed():
    # A typo here would silently excuse nothing and make a checkpoint fail
    # for a reason the report cannot explain.
    cfg = PanelConfig()
    m2 = set(load_reference(cfg, "db_master_2.csv").head(1).columns)
    sel = set(load_reference(cfg, "db_selected.csv").head(1).columns)
    assert EST_COLUMNS <= m2
    assert EST_COLUMNS <= sel
    assert KEPT_EXTRA <= m2  # TR_D survives into db_master_2
    assert not (KEPT_EXTRA & sel)  # and is dropped by vars_selected


def test_every_checkpoint_has_a_baseline_counterpart():
    """The baseline comparison must cover at least what the checkpoints do,
    since it is the stronger of the two: it excuses no column."""
    for cp in CHECKPOINTS.values():
        name = cp.interim.removesuffix(".parquet")
        assert name in BASELINE_KEYS, name
        assert BASELINE_KEYS[name] == cp.key, name
