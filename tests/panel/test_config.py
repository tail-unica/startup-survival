from pathlib import Path

from src.panel.config import PanelConfig


def test_default_paths_are_relative_to_the_repo_root():
    cfg = PanelConfig()
    assert cfg.raw_dir == Path("data/raw/pitchbook")
    assert cfg.interim_dir == Path("data/interim")


def test_raw_and_interim_join_the_file_name_to_their_directory(tmp_path):
    cfg = PanelConfig(raw_dir=tmp_path / "raw", interim_dir=tmp_path / "interim")
    assert cfg.raw("Company.csv") == tmp_path / "raw" / "Company.csv"
    assert cfg.interim("panel.parquet") == tmp_path / "interim" / "panel.parquet"


def test_interim_creates_its_directory_so_a_stage_can_write_straight_away(tmp_path):
    cfg = PanelConfig(interim_dir=tmp_path / "new" / "interim")
    assert not cfg.interim_dir.exists()
    path = cfg.interim("panel.parquet")
    assert path.parent.is_dir()


def test_raw_does_not_create_anything_because_it_only_reads(tmp_path):
    cfg = PanelConfig(raw_dir=tmp_path / "missing")
    cfg.raw("Company.csv")
    assert not (tmp_path / "missing").exists()


def test_config_is_frozen_so_a_stage_cannot_repoint_the_pipeline():
    cfg = PanelConfig()
    try:
        cfg.interim_dir = Path("/elsewhere")  # ty: ignore[invalid-assignment]
    except Exception as exc:  # dataclasses.FrozenInstanceError
        assert "frozen" in type(exc).__name__.lower() or "frozen" in str(exc).lower()
    else:
        raise AssertionError("PanelConfig must be immutable")
