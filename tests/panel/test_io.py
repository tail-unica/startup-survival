import polars as pl
import pytest

from src.panel.config import PanelConfig
from src.panel.io import RAW_ROW_COUNTS, read_raw, scan_raw, to_int, to_num


def test_scan_raw_returns_all_string_columns():
    cfg = PanelConfig()
    lf = scan_raw(cfg, "CompanyAffiliateRelation", ["CompanyID", "AffiliateType"])
    assert set(lf.collect_schema().values()) == {pl.String}


def test_read_raw_projects_only_requested_columns():
    cfg = PanelConfig()
    df = read_raw(cfg, "CompanyAffiliateRelation", ["CompanyID", "AffiliateType"])
    assert df.columns == ["CompanyID", "AffiliateType"]


def test_read_raw_row_count_matches_recorded_constant():
    cfg = PanelConfig()
    table = "CompanyAffiliateRelation"
    df = read_raw(cfg, table, ["CompanyID"], expect_rows=RAW_ROW_COUNTS[table])
    assert df.height == RAW_ROW_COUNTS[table]


def test_read_raw_raises_on_row_count_mismatch():
    cfg = PanelConfig()
    with pytest.raises(ValueError, match="row count mismatch"):
        read_raw(cfg, "CompanyAffiliateRelation", ["CompanyID"], expect_rows=1)


def test_to_num_and_to_int_null_out_non_numeric():
    df = pl.DataFrame({"a": ["1.5", "NA", "", None, "x"]})
    assert df.select(to_num("a"))["a"].to_list() == [1.5, None, None, None, None]
    assert df.select(to_int("a"))["a"].to_list() == [None, None, None, None, None]
    df2 = pl.DataFrame({"a": ["3", "NA"]})
    assert df2.select(to_int("a"))["a"].to_list() == [3, None]
