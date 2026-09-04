"""Configuration for the panel pipeline.

Every ``fix_*`` flag defaults to ``False``, and ``False`` means "reproduce
the R behaviour, bug included". Fixing a defect is always opt-in: see
section 6 of the design spec for the registry and the measured impact of
each one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Every bug flag, in the order of the spec's registry (B1..B10).
BUG_FLAGS: tuple[str, ...] = (
    "fix_founding_year_threshold",       # B1
    "fix_is_other_label",                # B2
    "fix_stageblock_na",                 # B3
    "fix_same_country_narm",             # B4
    "fix_permanenza_media_per_company",  # B5
    "fix_negative_delta",                # B6
    "fix_institute_na_literal",          # B7
    "fix_europe_asymmetry",              # B8
    "fix_dup_coalesce",                  # B9
    "fix_is_out_na",                     # B10
)

IMPUTATION_STRATEGIES: tuple[str, ...] = ("r_legacy", "r_injected", "leakage_free")


@dataclass(frozen=True)
class PanelConfig:
    """Paths, bug flags and imputation strategy for one pipeline run."""

    raw_dir: Path = Path("config/RCode/DB pulito")
    ref_dir: Path = Path("config/RCode/DatiIntermedi")
    interim_dir: Path = Path("data/interim")

    seed: int = 12
    imputation_strategy: str = "r_legacy"
    ignore_failed_checks: bool = False
    n_examples: int = 10
    rtol: float = 1e-9

    fix_founding_year_threshold: bool = False
    fix_is_other_label: bool = False
    fix_stageblock_na: bool = False
    fix_same_country_narm: bool = False
    fix_permanenza_media_per_company: bool = False
    fix_negative_delta: bool = False
    fix_institute_na_literal: bool = False
    fix_europe_asymmetry: bool = False
    fix_dup_coalesce: bool = False
    fix_is_out_na: bool = False

    def __post_init__(self) -> None:
        if self.imputation_strategy not in IMPUTATION_STRATEGIES:
            raise ValueError(
                f"unknown imputation strategy {self.imputation_strategy!r}; "
                f"expected one of {IMPUTATION_STRATEGIES}"
            )

    def interim(self, name: str) -> Path:
        """Path of an interim parquet, creating the directory if needed."""
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        return self.interim_dir / name

    def reference(self, name: str) -> Path:
        return self.ref_dir / name

    def raw(self, name: str) -> Path:
        return self.raw_dir / name

    def active_fixes(self) -> tuple[str, ...]:
        """Flags currently enabled — printed by the notebook on every run."""
        return tuple(f for f in BUG_FLAGS if getattr(self, f))
