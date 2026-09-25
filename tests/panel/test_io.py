"""The readers, on CSVs written by the test itself."""

import polars as pl
import pytest

from src.panel.config import PanelConfig
from src.panel.io import COMPANY_DATE_COLUMNS, read_raw, scan_raw, to_num


@pytest.fixture
def cfg(tmp_path):
    """A pipeline whose raw directory holds one small table."""
    (tmp_path / "Company.csv").write_text(
        "CompanyID,YearFounded,Note\n"
        "000001-01,2013,plain\n"
        '2000,2000,"a note, with a comma"\n'
        '3,NA,"row con\nun ritorno a capo"\n'
    )
    return PanelConfig(raw_dir=tmp_path, interim_dir=tmp_path / "interim")


def test_every_column_is_read_as_string(cfg):
    lf = scan_raw(cfg, "Company", ["CompanyID", "YearFounded"])
    assert set(lf.collect_schema().values()) == {pl.String}


def test_a_numeric_looking_key_keeps_its_text_form(cfg):
    # "2000" inferred as an integer would silently fail to join with "2000" read
    # as text from another table: that is why inference is off everywhere.
    ids = read_raw(cfg, "Company", ["CompanyID"])["CompanyID"].to_list()
    assert ids == ["000001-01", "2000", "3"]


def test_read_raw_projects_only_the_requested_columns_in_order(cfg):
    df = read_raw(cfg, "Company", ["YearFounded", "CompanyID"])
    assert df.columns == ["YearFounded", "CompanyID"]


def test_quoted_free_text_does_not_break_the_row_count(cfg):
    # A comma and a newline inside quotes: mis-parsing them would add rows and
    # skew every count downstream.
    assert read_raw(cfg, "Company", ["CompanyID"]).height == 3


def test_missing_tokens_are_left_to_nullify(cfg):
    # read_raw passes null_values=[] on purpose: what counts as missing is one
    # decision, taken once by nullify, and not spread over the readers.
    assert read_raw(cfg, "Company", ["YearFounded"])["YearFounded"].to_list() == [
        "2013",
        "2000",
        "NA",
    ]


def test_to_num_nulls_out_anything_non_numeric_and_keeps_the_name():
    df = pl.DataFrame({"a": ["1.5", "NA", "", None, "x", "3"]})
    out = df.select(to_num("a"))
    assert out.columns == ["a"]
    assert out["a"].to_list() == [1.5, None, None, None, None, 3.0]


def test_company_date_columns_is_the_single_source_of_maxyear():
    # Phase 1 and phase 7 both build a MaxYear out of these; two lists would
    # give two different ends of life for the same company.
    assert COMPANY_DATE_COLUMNS == [
        "CompanyFinancingStatusDate",
        "BusinessStatusDate",
        "OwnershipStatusDate",
        "FirstFinancingDate",
        "LastKnownValuationDate",
    ]
