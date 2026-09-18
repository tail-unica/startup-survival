"""The contract between the two notebooks.

``build_panel.ipynb`` writes the panel; ``notebook.ipynb`` reads it through
``src/preprocessing.py``. The contract is a list of column names, and nothing
enforces it at runtime: a column renamed in the notebook would surface as a
polars error in the middle of a sweep, or worse as a silently missing feature.

The test reads the notebook source, not its output, so it runs without the
PitchBook extraction and without executing anything.
"""

import json
import re
from pathlib import Path

import pytest
import yaml

from src.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
PANEL_NOTEBOOK = ROOT / "build_panel.ipynb"
MODEL_NOTEBOOK = ROOT / "notebook.ipynb"


def _codice(notebook: Path) -> str:
    """Concatenate the code cells of a notebook.

    :param notebook: Path of the ``.ipynb`` file.
    :return: Their source, as one string.
    """
    celle = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    return "\n".join("".join(c["source"]) for c in celle if c["cell_type"] == "code")


def _lista(codice: str, nome: str) -> list[str]:
    """Read a list of string literals assigned to ``nome`` in ``codice``.

    :param codice: Notebook source.
    :param nome: Name of the assignment, e.g. ``COLONNE_FINALI``.
    :return: The string literals, in order.
    """
    m = re.search(rf"^{nome} = \[(.*?)^\]", codice, re.S | re.M)
    assert m, f"{nome} non si trova nel notebook"
    return re.findall(r'"([^"]+)"', m.group(1))


@pytest.fixture(scope="module")
def colonne_panel() -> list[str]:
    """The columns the panel notebook writes, in order."""
    return _lista(_codice(PANEL_NOTEBOOK), "COLONNE_FINALI")


def test_the_panel_carries_every_feature_the_models_read(colonne_panel):
    mancanti = [c for c in FEATURE_COLUMNS if c not in colonne_panel]
    assert not mancanti, f"il panel non produce: {mancanti}"


def test_the_panel_carries_the_columns_the_target_is_built_from(colonne_panel):
    mancanti = [c for c in TARGET_COLUMNS if c not in colonne_panel]
    assert not mancanti, f"il panel non produce: {mancanti}"


def test_the_panel_has_no_duplicate_column():
    colonne = _lista(_codice(PANEL_NOTEBOOK), "COLONNE_FINALI")
    assert len(colonne) == len(set(colonne))


def test_the_panel_columns_that_no_model_reads_are_the_declared_ones(colonne_panel):
    # A column the panel carries and the models ignore is legitimate, but it has
    # to be a declared one. Today that is `N_Similar`, the denominator of
    # `SimilarityScoreMean`, added so that "no comparable firm" and "comparable
    # but dissimilar firms" stop being the same zero: whether it becomes a
    # feature is a modelling decision, still open.
    inutilizzate = set(colonne_panel) - set(FEATURE_COLUMNS) - set(TARGET_COLUMNS)
    assert inutilizzate == {"N_Similar"}


def test_the_two_notebooks_agree_on_the_sample_threshold():
    # The panel keeps the firms founded from `first_year` on, and the datasets
    # keep the decision years from 2010 on: both read config.yaml, and the panel
    # notebook must not hard-code a different one.
    codice = _codice(PANEL_NOTEBOOK)
    assert 'ANNO_MIN_FONDAZIONE = int(CONFIG["first_year"])' in codice
    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
    assert config["first_year"] <= 2010


def test_the_model_notebook_reads_the_processed_datasets_from_the_config():
    codice = _codice(MODEL_NOTEBOOK)
    assert 'config["paths"]["dataset_window"]' in codice
    assert 'config["paths"]["dataset_nowindow"]' in codice


def test_the_dataset_builder_reads_the_two_panels_the_notebook_writes():
    # One panel per switch configuration, and the datasets read one each: a name
    # that drifts here would rebuild them from the wrong file, silently.
    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
    codice = _codice(PANEL_NOTEBOOK)
    attesi = {
        Path(config["paths"]["panel_timed"]).name: 'NOME_PANEL = "panel"',
        Path(config["paths"]["panel_snapshot"]).name: 'else "panel_snapshot"',
    }
    for nome, riga in attesi.items():
        assert nome.endswith(".csv.gz"), nome
        assert riga in codice, f"il notebook non scrive {nome}"
    assert 'cfg.interim(f"{NOME_PANEL}.csv.gz")' in codice


def test_the_panel_notebook_admits_only_all_on_or_all_off():
    # The two datasets differ in the switch configuration, so a panel built with
    # a mixed one would belong to neither.
    codice = _codice(PANEL_NOTEBOOK)
    assert "assert all(INTERRUTTORI) or not any(INTERRUTTORI)" in codice
