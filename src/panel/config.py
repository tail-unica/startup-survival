"""Paths of the panel pipeline.

One frozen dataclass, instantiated once by ``build_panel_light.ipynb``. There is
no seed and no imputation setting: the pipeline is deterministic, and no missing
value is filled by a fitted model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PanelConfig:
    """Where the pipeline reads from and writes to.

    :param raw_dir: Directory of the raw PitchBook CSVs.
    :param interim_dir: Directory of the per-stage parquet outputs.
    """

    raw_dir: Path = Path("data/raw/pitchbook")
    interim_dir: Path = Path("data/interim")

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
