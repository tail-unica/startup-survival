"""The seven phases, chained.

The notebooks call the phases one function at a time, with the explanation of each
step beside it; the command line calls :func:`build_panel`, which calls exactly the
same functions in the same order. There is therefore one definition of the
pipeline, and no way for the two routes to drift apart.

Every stage writes its parquet under the interim directory and the next one reads
the frame it is handed, so re-running one phase does not force the others and a
kernel can be restarted without losing work.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from src.panel import companies, competitors, deals, people, stages, target, team
from src.panel.config import PanelConfig, PanelRules

#: The file name of the panel, by switch configuration. The name says which of the
#: two a file holds, so the two can live side by side.
PANEL_NAMES = {True: "panel", False: "panel_snapshot"}


@dataclass(frozen=True)
class PanelReport:
    """What one run of the pipeline has to say about itself.

    :param rows: Rows of the finished panel.
    :param companies: Distinct companies in it.
    :param with_team: Rows carrying team data.
    :param with_stage: Rows carrying a growth stage.
    :param ceo_filled: Rows whose chief executive was filled from the board.
    :param ceo_corrected: Rows whose chief executive was corrected.
    :param rounds_without_date: Rounds that carry no year and leave the panel.
    """

    rows: int
    companies: int
    with_team: int
    with_stage: int
    ceo_filled: int
    ceo_corrected: int
    rounds_without_date: int


def panel_name(*, timed: bool) -> str:
    """The file name the panel of that configuration is written under.

    :param timed: Whether the attributes are those of each row's own year.
    :return: The name, without extension.
    """
    return PANEL_NAMES[timed]


def build_panel(
    cfg: PanelConfig,
    rules: PanelRules,
    *,
    min_founding_year: int,
    timed: bool = True,
    write: bool = True,
) -> tuple[pl.DataFrame, PanelReport]:
    """Build the panel from the raw tables, phase by phase.

    :param cfg: Pipeline paths.
    :param rules: Domain rules, from ``config.yaml``.
    :param min_founding_year: Oldest founding year admitted, inclusive.
    :param timed: With ``True`` every attribute is the one of the row's own year;
        with ``False`` they are the ones declared at extraction time, which is the
        panel that carries the look-ahead.
    :param write: Whether to write the panel and the per-phase parquets.
    :return: The finished panel and the report of the run.
    """
    # ── Phase 1: the skeleton ──────────────────────────────────────────────
    all_companies = companies.read_companies(cfg)
    life = companies.company_life(all_companies)
    skeleton = companies.attach_ownership_status(
        companies.build_skeleton(life, min_founding_year=min_founding_year), all_companies
    )
    registry = companies.company_registry(all_companies, min_founding_year=min_founding_year)
    del all_companies

    # ── Phase 2: one row per (company, person) ─────────────────────────────
    years = people.company_years(skeleton)
    board = people.merge_appointments(people.read_board(cfg, skeleton))
    ceo_roles = people.ceo_roles(board, years, rules)
    pairs = people.presence_window(people.person_attributes(board, cfg, rules), years)
    # The two per-person tables cover everyone who appears on a board of the panel,
    # which is a slightly wider set than the pairs that survive the cut at the
    # company's last year.
    experience = people.experience_events(cfg, board)
    education = people.education_by_year(people.classify_studies(cfg, rules), board, rules)
    del board

    # ── Phase 3: the team columns ──────────────────────────────────────────
    expanded = team.expand_team(pairs, min_founding_year=min_founding_year)
    expanded = team.attach_person_attributes(expanded, experience, education, rules, timed=timed)
    parameters = team.experience_parameters(expanded, timed=timed)
    expanded = team.experience_index(expanded, parameters, rules, timed=timed)
    panel = team.join_skeleton(skeleton, team.aggregate_team(expanded, rules))
    del expanded, skeleton

    # ── Phase 4: the funding rounds ────────────────────────────────────────
    by_deal = deals.aggregate_by_deal(deals.investor_participations(cfg, rules, timed=timed), rules)
    rounds = deals.repair_deal_dates(
        deals.read_deals(cfg, registry), rules, min_founding_year=min_founding_year
    )
    rounds = deals.place_deals_in_years(rounds, by_deal)
    lost = deals.companies_losing_rounds(rounds, rules)
    panel = deals.attach_deals(
        panel, deals.aggregate_by_company_year(deals.add_deal_flags(rounds, rules), rules)
    )
    del by_deal, rounds

    # ── Phase 5: stages, cumulative totals, chief executive ────────────────
    panel = stages.cumulative_totals(stages.growth_stage(stages.cumulate_flags(panel, rules)))
    panel, ceo_filled, ceo_corrected = stages.resolve_ceo(panel, ceo_roles)
    panel = stages.ceo_attributes(
        panel, pairs, experience, education, parameters, rules, timed=timed
    )
    panel = stages.attach_registry(panel, registry)
    del pairs, experience, education, parameters, registry, ceo_roles

    # ── Phase 6: the groups, the future stage, the truncation ──────────────
    panel = target.truncate_at_exit(target.next_stage(target.group_stages(panel, rules)), rules)

    # ── Phase 7: the competitors, and the final shape ──────────────────────
    similar, competitor_pairs = competitors.similar_pairs(cfg, panel, life, timed=timed)
    competitor_stats, similarity_stats = competitors.competitor_columns(
        panel, similar, competitor_pairs, timed=timed
    )
    final = competitors.finalize(panel, competitor_stats, similarity_stats, rules)

    report = PanelReport(
        rows=final.height,
        companies=final["CompanyID"].n_unique(),
        with_team=int(final["Total_People"].is_not_null().sum()),
        with_stage=int(final["GrowthStageGroup"].is_not_null().sum()),
        ceo_filled=ceo_filled,
        ceo_corrected=ceo_corrected,
        rounds_without_date=int(lost["rounds_without_date"].sum()),
    )
    if write:
        name = panel_name(timed=timed)
        final.write_parquet(cfg.interim(f"{name}.parquet"))
        final.write_csv(cfg.interim(f"{name}.csv.gz"), compression="gzip")
        lost.write_parquet(cfg.interim("companies_losing_rounds.parquet"))
    return final, report
