"""Phase 7: what the last operation of the pipeline does to the identifiers."""

from __future__ import annotations

import dataclasses

import polars as pl
import yaml

from src.panel.competitors import finalize
from src.panel.config import PanelRules

RULES = PanelRules.from_config(yaml.safe_load(open("config/config.yaml")))


def test_finalize_replaces_the_source_identifiers_with_consecutive_integers():
    """The renumbering is what allows the panel to be released.

    It is the last operation of the pipeline on purpose: once it has run, no
    identifier of the extraction survives and nothing can be joined back to the
    source. The example run cannot show this (its identifiers are already the
    integers the renumbering assigns), so it is stated here.
    """
    panel = pl.DataFrame(
        {
            "CompanyID": ["PB10", "PB2", "PB7"],
            "Year_Delta": [0, 0, 0],
            "GrowthStageGroup": ["Early", "Early", "Later"],
            "GrowthNextStageGroup": ["Early", "Later", "Later"],
        }
    )
    empty = pl.DataFrame(schema={"CompanyID": pl.String, "Year_Delta": pl.Int64})
    rules = dataclasses.replace(RULES, columns=["CompanyID", "Year_Delta"])

    final = finalize(
        panel,
        empty.with_columns(N_Competitors=pl.lit(0), Same_Country=pl.lit(0)),
        empty.with_columns(N_Similar=pl.lit(0), SimilarityScoreMean=pl.lit(0.0)),
        rules,
    )

    assert sorted(final["CompanyID"].to_list()) == [1, 2, 3]
    assert final["CompanyID"].dtype != pl.String
