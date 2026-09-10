"""Builds the startup panel from the raw PitchBook CSVs.

This is the Python port of the two R scripts that used to produce it, plus the
two post-processing steps that followed them. Seven stages, each reading the
previous one's parquet from data/interim/ and writing its own, so a stage can
be re-run without redoing the ones before it.

Six of the stages have a reference file to check against; `--verify` runs those
checks and stops at the first one that fails.

**Phases already migrated to `build_panel.ipynb` are not here.** As the
review moves through the pipeline the notebook becomes the only place that
stage's code lives, and this script shrinks; it will be deleted when the last
phase migrates. A stage missing from STAGES therefore needs its parquet to
already exist in data/interim/, produced by running that phase in the
notebook.

    uv run python scripts/build_panel.py
    uv run python scripts/build_panel.py --verify
    uv run python scripts/build_panel.py --from 4          # resume at stage 4

Nothing here writes to data/raw/ or data/reference/. The finished panel lands
in data/interim/panel.csv.gz; data/raw/panel.csv.gz is the reference it is
compared against, and stays untouched.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.panel.config import PanelConfig  # noqa: E402

#: (stage number, label, module, function, checkpoints verifiable after it).
#:
#: Each stage runs in its own interpreter. Run end to end in one process the
#: pipeline peaks past the 7 GB on this machine and the kernel kills it partway
#: through stage 6; a subprocess per stage hands the memory back every time,
#: and the stages already communicate through parquet files rather than memory.
STAGES = [
    # fasi 1 e 2: migrate in build_panel.ipynb
    (3, "competitor", "stage3_relations", "run_competitors", ["B"]),
    (3, "dipendenti, financials, news", "stage3_relations", "run_panel", []),
    (4, "deal e investitori", "stage4_deals", "run", []),
    (5, "finalizzazione", "stage5_final", "run", ["C", "D"]),
    (6, "raggruppamento stadi e troncamento", "stage6_panel", "run", ["E"]),
    (7, "temporizzazione competitor", "stage7_competitors", "run", ["F"]),
]


def _run_stage(module: str, function: str) -> None:
    code = (
        f"from src.panel import {module}; from src.panel.config import PanelConfig; "
        f"{module}.{function}(PanelConfig())"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def _run_checkpoint(letter: str) -> bool:
    """Also a subprocess: a checkpoint reads a whole reference CSV as text,
    which on db3.csv and db_master_2.csv is several gigabytes the driver would
    otherwise hold for the rest of the run."""
    done = subprocess.run(
        [sys.executable, "-m", "src.panel.validate", "--checkpoint", letter], cwd=ROOT, check=False
    )
    return done.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="run each checkpoint after its stage")
    parser.add_argument("--from", dest="start", type=int, default=1, help="first stage to run")
    args = parser.parse_args()

    cfg = PanelConfig()
    fixes = cfg.active_fixes()
    print(f"correzioni attive: {list(fixes) if fixes else 'nessuna (comportamento R)'}\n")

    for number, label, module, function, checkpoints in STAGES:
        if number < args.start:
            continue
        started = time.perf_counter()
        print(f"stadio {number}: {label} ...", flush=True)
        _run_stage(module, function)
        print(f"  fatto in {time.perf_counter() - started:.0f}s")
        if not args.verify:
            continue
        for letter in checkpoints:
            if not _run_checkpoint(letter) and not cfg.ignore_failed_checks:
                print(f"\ncheckpoint {letter} fallito: mi fermo.")
                return 1
    print(f"\npanel scritto in {cfg.interim('panel.csv.gz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
