"""Phase 6, and the checks a finished panel has to pass.

The target of the whole study is one column, and it is built by two steps whose
order is a constraint. The checks are what would catch that order being swapped.
"""

from __future__ import annotations

import polars as pl
import pytest
import yaml

from src.panel import checks
from src.panel.config import PanelRules
from src.panel.target import STAY, group_stages, next_stage, truncate_at_exit

RULES = PanelRules.from_config(yaml.safe_load(open("config/config.yaml").read()))


def _panel(stages: list[str | None], company: str = "A") -> pl.DataFrame:
    """One company with a growth stage per year."""
    return pl.DataFrame(
        {
            "CompanyID": [company] * len(stages),
            "Year_Delta": [2015 + i for i in range(len(stages))],
            "Age": list(range(len(stages))),
            "GrowthStage": stages,
        }
    )


# the groups


def test_the_seven_stages_collapse_into_four_groups():
    panel = _panel(["Preseed", "Seed", "EarlyVC", "LaterVC_or_Other", "Exit_M&A"])
    grouped = group_stages(panel, RULES)
    assert grouped["GrowthStageGroup"].to_list() == ["Early", "Early", "Early", "Later", "Exit"]
    assert "GrowthStage" not in grouped.columns


def test_a_null_stage_stays_null_rather_than_becoming_a_category():
    grouped = group_stages(_panel([None, "Seed"]), RULES)
    assert grouped["GrowthStageGroup"].to_list() == [None, "Early"]


# the future stage


def test_the_future_stage_is_the_next_different_group_and_its_distance():
    panel = next_stage(group_stages(_panel(["Seed", "EarlyVC", "LaterVC_or_Other"]), RULES))
    assert panel["GrowthNextStageGroup"].to_list() == ["Later", "Later", STAY]
    assert panel["TimeNextStageGroup"].to_list()[:2] == [2, 1]


def test_a_company_that_never_changes_group_carries_the_placeholder():
    panel = next_stage(group_stages(_panel(["Seed", "EarlyVC"]), RULES))
    assert panel["GrowthNextStageGroup"].to_list() == [STAY, STAY]
    # And as a distance, the years left in its panel.
    assert panel["TimeNextStageGroup"].to_list() == [1, 0]


def test_the_exit_is_reachable_as_a_future_stage_because_it_is_read_before_the_cut():
    panel = next_stage(group_stages(_panel(["Seed", "EarlyVC", "Exit_M&A"]), RULES))
    assert panel["GrowthNextStageGroup"].to_list()[:2] == ["Exit", "Exit"]


def test_the_future_stage_never_crosses_into_another_company():
    panel = pl.concat([_panel(["Seed", "EarlyVC"], "A"), _panel(["Exit_M&A"], "B")])
    out = next_stage(group_stages(panel, RULES))
    assert out.filter(pl.col("CompanyID") == "A")["GrowthNextStageGroup"].to_list() == [STAY, STAY]


# the truncation


def test_the_panel_is_cut_at_the_exit_the_year_of_the_exit_included():
    panel = next_stage(group_stages(_panel(["Seed", "EarlyVC", "Exit_M&A"]), RULES))
    cut = truncate_at_exit(panel, RULES)
    assert cut["Age"].to_list() == [0, 1]
    # The outcome survives in the future stage of the last row left.
    assert cut["GrowthNextStageGroup"].to_list()[-1] == "Exit"


def test_a_non_terminal_year_after_an_exit_goes_as_well():
    panel = next_stage(group_stages(_panel(["Seed", "Exit_M&A", "LaterVC_or_Other"]), RULES))
    assert truncate_at_exit(panel, RULES)["Age"].to_list() == [0]


def test_a_company_whose_first_year_is_terminal_leaves_the_panel_entirely():
    panel = next_stage(group_stages(_panel(["Out"]), RULES))
    assert truncate_at_exit(panel, RULES).height == 0


def test_a_company_that_never_exits_keeps_every_year():
    panel = next_stage(group_stages(_panel(["Seed", "EarlyVC", "LaterVC_or_Other"]), RULES))
    assert truncate_at_exit(panel, RULES).height == 3


# the checks


def _finished(**overrides) -> pl.DataFrame:
    """A minimal panel that passes every structural check."""
    rows = {
        "CompanyID": [1, 1, 2],
        "Age": [0, 1, 0],
        "GrowthStageGroup": ["Early", "Early", "Early"],
        "GrowthNextStageGroup": ["Later", "Later", "Early"],
        "N_Deal": [1, 2, 1],
        "TotalInvestors": [1, 3, 0],
        "Total_People": [2, 2, None],
    }
    rows.update(overrides)
    frame = pl.DataFrame(rows)
    # The schema check compares the full column list, so the rest is filled in.
    missing = [c for c in RULES.columns if c not in frame.columns]
    return frame.with_columns([pl.lit(None).alias(c) for c in missing]).select(RULES.columns)


def test_a_sound_panel_passes():
    checks.check_structure(_finished(), RULES)


def test_a_missing_column_is_named():
    panel = _finished().drop("N_Similar")
    with pytest.raises(ValueError, match="N_Similar"):
        checks.check_structure(panel, RULES)


def test_a_year_before_the_founding_year_is_caught():
    with pytest.raises(ValueError, match="precede the founding year"):
        checks.check_structure(_finished(Age=[-1, 1, 0]), RULES)


def test_a_duplicated_company_year_is_caught():
    with pytest.raises(ValueError, match="more than one row"):
        checks.check_structure(_finished(Age=[0, 0, 0]), RULES)


def test_a_surviving_terminal_stage_is_caught():
    # It is what a truncation that did not run, or ran too early, looks like.
    panel = _finished(GrowthStageGroup=["Early", "Exit", "Early"])
    with pytest.raises(ValueError, match="terminal stage"):
        checks.check_structure(panel, RULES)


def test_a_left_over_placeholder_is_caught():
    panel = _finished(GrowthNextStageGroup=["Later", STAY, "Early"])
    with pytest.raises(ValueError, match="placeholder"):
        checks.check_structure(panel, RULES)


def test_a_cumulative_column_going_backwards_is_caught():
    with pytest.raises(ValueError, match=r"cumulative: \['N_Deal'\]"):
        checks.check_structure(_finished(N_Deal=[2, 1, 1]), RULES)


def test_a_flag_that_switches_back_off_is_caught():
    # The round flags accumulate as well: a company that has closed a debt round has
    # closed one for good, so the flag cannot return to false the year after.
    with pytest.raises(ValueError, match=r"cumulative: \['Is_Debt'\]"):
        checks.check_structure(_finished(Is_Debt=[True, False, False]), RULES)
