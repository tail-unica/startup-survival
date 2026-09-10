"""Every stage module must at least import.

Nothing else in the suite imports them: the stage tests read the parquet a
stage produced, so a module left with a broken import passes the whole suite
in silence. That happened once, deleting stage1_company while
stage7_competitors still imported a constant from it.

The file shrinks as phases migrate into build_panel.ipynb, and disappears
with the last one.
"""

import importlib

import pytest

STAGE_MODULES = [
    # le fasi da 1 a 4 sono migrate in build_panel.ipynb
    "src.panel.expansions",
    "src.panel.stage5_final",
    "src.panel.stage6_panel",
    "src.panel.stage7_competitors",
]


@pytest.mark.parametrize("name", STAGE_MODULES)
def test_stage_module_imports(name):
    assert importlib.import_module(name) is not None
