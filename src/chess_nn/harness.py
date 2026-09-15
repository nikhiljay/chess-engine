"""Paths into the training repo and the surrounding chess-challenge harness."""

import importlib.util
from pathlib import Path
from types import ModuleType

TRAINING_ROOT = Path(__file__).resolve().parents[2]
HARNESS_ROOT = TRAINING_ROOT.parent

DATA_DIR = TRAINING_ROOT / "data"
SHARDS_DIR = DATA_DIR / "shards"
DEV_OPENINGS = DATA_DIR / "dev_openings.txt"
RUNS_DIR = TRAINING_ROOT / "runs"

OFFICIAL_OPENINGS = HARNESS_ROOT / "data" / "openings.txt"
COMPETE_BIN = HARNESS_ROOT / "target" / "release" / "compete"
DUMP_ENCODING_BIN = HARNESS_ROOT / "target" / "release" / "dump-encoding"
REFERENCE_ENCODING = HARNESS_ROOT / "tools" / "encoding.py"
VALIDATE_MODEL = HARNESS_ROOT / "tools" / "validate_model.py"


def load_harness_tool(path: Path, name: str) -> ModuleType:
    """Import a standalone script from the harness tools/ directory."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_opening_fens(path: Path) -> list[str]:
    """Read FENs in the harness openings format (# comments, blank lines skipped)."""
    lines = (line.strip() for line in path.read_text().splitlines())
    return [line for line in lines if line and not line.startswith("#")]
