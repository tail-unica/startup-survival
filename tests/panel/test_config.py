from pathlib import Path

from src.panel.config import BUG_FLAGS, PanelConfig


def test_all_bug_flags_default_to_r_behaviour():
    cfg = PanelConfig()
    for flag in BUG_FLAGS:
        assert getattr(cfg, flag) is False, f"{flag} must default to False (R behaviour)"


def test_bug_flag_registry_matches_dataclass_fields():
    cfg = PanelConfig()
    declared = {f for f in vars(cfg) if f.startswith("fix_")}
    assert declared == set(BUG_FLAGS)


def test_paths_are_relative_to_repo_root():
    cfg = PanelConfig()
    assert cfg.raw_dir == Path("data/raw/pitchbook")
    assert cfg.ref_dir == Path("data/reference")
    assert cfg.interim_dir == Path("data/interim")


def test_default_imputation_strategy_is_standalone():
    assert PanelConfig().imputation_strategy == "r_legacy"


def test_rejects_unknown_imputation_strategy():
    import pytest

    with pytest.raises(ValueError, match="unknown imputation strategy"):
        PanelConfig(imputation_strategy="magic")
