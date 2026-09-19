"""Paths of the panel pipeline.

One frozen dataclass, instantiated once per run. There is
no seed and no imputation setting: the pipeline is deterministic, and no missing
value is filled by a fitted model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PanelConfig:
    """Where the pipeline reads from and writes to.

    :param raw_dir: Directory of the raw PitchBook CSVs.
    :param interim_dir: Directory of the per-stage parquet outputs.
    """

    raw_dir: Path = Path("data/raw/pitchbook")
    interim_dir: Path = Path("data/interim")

    def __post_init__(self) -> None:
        """Accept plain strings too, since the command line hands them over."""
        object.__setattr__(self, "raw_dir", Path(self.raw_dir))
        object.__setattr__(self, "interim_dir", Path(self.interim_dir))

    def raw(self, name: str) -> Path:
        """Path of a raw CSV.

        :param name: File name, extension included.
        :return: Path inside :attr:`raw_dir`.
        """
        return self.raw_dir / name

    def interim(self, name: str) -> Path:
        """Path of an interim parquet, creating the directory if needed.

        :param name: File name, extension included.
        :return: Path inside :attr:`interim_dir`.
        """
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        return self.interim_dir / name


@dataclass(frozen=True)
class PanelRules:
    """The domain rules of the panel, as ``config.yaml`` declares them.

    They are research choices — what counts as a founder, how a degree is ranked,
    which field a subject belongs to — so they live in the configuration and are
    passed to the functions that apply them. A test can therefore state a rule of
    its own instead of depending on the project's.
    """

    degree_rules: list[tuple[str, str]]
    degree_hierarchy: list[str]
    field_rules: list[tuple[str, str]]
    field_flags: dict[str, str]
    founder_pattern: str
    founder_assistant_pattern: str
    ceo_pattern: str
    name_degree_patterns: dict[str, str]
    investor_categories: dict[str, str]
    investor_flags: dict[str, str]
    deal_flags: dict[str, list[str]]
    date_repair: dict[str, object]
    venture_types: list[str]
    stage_groups: dict[str, str]
    terminal_groups: list[str]
    columns: list[str]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> PanelRules:
        """Read the rules from a parsed ``config.yaml``.

        :param config: The whole configuration, or just its ``panel`` section.
        :return: The rules.
        """
        panel = config.get("panel", config)
        return cls(
            degree_rules=[(p, v) for p, v in panel["degree_rules"]],
            degree_hierarchy=list(panel["degree_hierarchy"]),
            field_rules=[(p, v) for p, v in panel["field_rules"]],
            field_flags=dict(panel["field_flags"]),
            founder_pattern=panel["founder_pattern"],
            founder_assistant_pattern=panel["founder_assistant_pattern"],
            ceo_pattern=panel["ceo_pattern"],
            name_degree_patterns=dict(panel["name_degree_patterns"]),
            investor_categories=dict(panel["investor_categories"]),
            investor_flags=dict(panel["investor_flags"]),
            deal_flags={k: list(v) for k, v in panel["deal_flags"].items()},
            date_repair=dict(panel["date_repair"]),
            venture_types=list(panel["venture_types"]),
            stage_groups=dict(panel["stage_groups"]),
            terminal_groups=list(panel["terminal_groups"]),
            columns=list(panel["columns"]),
        )
